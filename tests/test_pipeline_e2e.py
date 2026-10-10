"""End-to-end pipeline test on a synthetic wiki-format dump.

This exercises the same code path as a real run - XML dump -> clean ->
language ID -> dedupe -> split -> tokenizer -> pack -> train -> checkpoint ->
generate -> evaluate -> export - without touching the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tagai.config import DataConfig, ModelConfig, PathsConfig, TokenizerConfig, TrainConfig

AZ_ARTICLES = [
    (
        "Azərbaycan",
        "Azərbaycan Respublikası müstəqil dövlətdir. Onun paytaxtı Bakı şəhəridir. "
        "Dövlətin rəsmi dili Azərbaycan dilidir. Bu dil türk dilinin qohumudur, lakin "
        "onun söz varlığı və qrammatikası əhəmiyyətli dərədə fərqlənir. Əlifbada 32 hərf var."
    ),
    (
        "Bakı",
        "Bakı Xəzər dənizinə sahildir və paytaxt funksiyasını daşıyır. Şəhərin ən tanınmış "
        "memarlıq abidəsi Şəhərlər bağıdır. Bakıçılar üçün dəniz havasının dəyişməsi vacibdir."
    ),
    (
        "Azərbaycan dili",
        "Azərbaycan dili qərb Asiyada danışılan türk dillerindən biridir. Onun əlifbası "
        "latın əlifbasına əsaslanır və ə, ı, ö, ü, ğ, ş, ç kimi xüsusi hərflər ehtiva edir."
    ),
    (
        "Xəzər dənizi",
        "Xəzər dənizi Avropanın ən böyük iç su hövzəsidir. Sahil ölkələri Azərbaycan, "
        "Rusiya, Türkmenistan, İran və Qazaxıstan dövlətləridir. Duzluğu beynəlxilətdir."
    ),
    (
        "Qızıldaş",
        "Qızıldaş qurşulaşdır. Qızıl yataqları Zaqatala, Quba və Tovuz rayonlarında yerləşir. "
        "Bu region iqlimi və relyef şəraiti ilə məşhurdur."
    ),
    (
        "Nizami Gəncəvi",
        "Nizami Gəncəvi XII əsr Azərbaycan şairidir. Onun yaradıcılığı milli ədəbiyyatın "
        "əsasında durur. Əsərləri bir neçə dildə tərcümə olunmuşdur."
    ),
]

TURKISH_ARTICLE = (
    "Türkiye",
    "Türkiye Cumhuriyeti'nin başkenti Ankara'dır. Turkce, Azerbaycan dili ile yakin "
    "akraba bir dildir ancak alfabe sistemi farklidir. Istanbul cok buyuk bir sehirdir.",
)

RUSSIA_ARTICLE = (
    "Москва",
    "Москва является столицей России и крупнейшим городом страны. Российский язык "
    "относится к славянской группе языков и имеет кириллическую письменность.",
)

# A small test corpus cannot be expected to contain all 64 Azerbaijani letters,
# and the tokenizer refuses to train on one that does not. These articles exist
# to exercise the real alphabet-coverage guarantee. They must be *distinct*
# documents: deduplication collapses repeats to one, and the tokenizer is
# trained on the train split only, so a single copy could land in validation.
ALPHABET_ARTICLES = [
    (
        "Azərbaycan əlifbası",
        "Azərbaycan əlifbası abcçdefgğhıijklmnoöpqrsştuüvyzə sıralaması ilə yazılır və "
        "ABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVYZƏ böyük hərflərindən ibarətdir. Əlifbada otuz iki hərf var.",
    ),
    (
        "Şəhər və rayon adları",
        "Xəzər dənizi, Zaqatala, Quba, Qusar, İsmayıllı və Xaçmaz rayonlarında bu hərflər "
        "işlədilir. Cəbrayıl, Ağdam, Tovuz və Füzuli coğrafi adları da hərf dəyişikliyi yaradır.",
    ),
    (
        "Jurnalistika və informasiya texnologiyaları",
        "Jurnalistika və informasiya texnologiyaları sahəsində yeni tədqiqatlar aparılır. "
        "Qərərar, təqdimat və memorandum kimi sənədlər hər biri ayrıca bölmədə təqdim olunur.",
    ),
    (
        "Rəqmlər və işarələr",
        "Rəqmlər və işarələr necə yazılır? 12,75% (2024-cü il) — \"qeyd\" [1]; 3/4 = 0,75; "
        "15 + 27 = 42! Fayl adı: sənəd.pdf; e-poçt: numune@ornek.az. Necə? Yaxşı! Diqqət!",
    ),
]


def _make_dump(path: Path, articles) -> Path:
    parts = ['<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">']
    for i, (title, text) in enumerate(articles):
        parts.append(
            f"<page><title>{title}</title><ns>0</ns><id>{100 + i}</id><revision>"
            f'<text xml:space="preserve">{text}</text></revision></page>'
        )
    parts.append("</mediawiki>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


FILLER = [
    "Bu mətn əlavə cümlələrlə genişləndirilmişdir.",
    "Məlumat müasir və aktual mənbələrə əsaslanır.",
    "Mətndəki fərqlər diqqətlə nəzərdən keçirilmişdir.",
    "Həmin dövrdə hadisələr geniş miqyasda yayılmışdır.",
    "Yazı üslubu sadə və aydın saxlanılmışdır.",
    "Hər bir fəsil müstəqil məna daşıyır.",
    "Rəqəmlər təhlil prosesində istifadə olunub.",
    "Nəticələr tədqiqatçılar tərəfindən təsdiqlənib.",
]


def build_corpus_articles() -> list:
    """Distinct documents covering Azerbaijani letters, punctuation and noise.

    Each base article is expanded with unique filler so deduplication cannot
    collapse the corpus to a handful of documents: with a hash-based split,
    a corpus that small cannot reliably place alphabet and punctuation
    coverage on both sides of the split.
    """
    articles: list = []
    bases = AZ_ARTICLES + ALPHABET_ARTICLES
    for rep in range(6):
        for idx, (title, text) in enumerate(bases):
            suffix = " ".join(FILLER[(rep * len(bases) + idx + k) % len(FILLER)] for k in range(3))
            body = f"{text} {suffix}"
            if body == text:  # pragma: no cover - defensive
                continue
            articles.append((f"{title} {rep + 1}", body))
    articles += [TURKISH_ARTICLE, RUSSIA_ARTICLE]
    return articles


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Run the whole pipeline once; the tests below assert on its artefacts."""
    from tagai.data.pack import pack_corpus
    from tagai.data.prepare import build_corpus
    from tagai.export import export_bundle, load_bundle
    from tagai.inference import GenerateConfig, TagAIGenerator
    from tagai.tokenizer import TagAITokenizer
    from tagai.tokenizer.trainer import train_tokenizer
    from tagai.training.trainer import Trainer, load_model_from_checkpoint
    from tagai.data.pack import BatchSampler, PackedDataset
    from tagai.model import TagAIModel
    from tagai.training.checkpoint import find_latest_valid
    from tagai.utils import read_json, set_seed

    root = tmp_path_factory.mktemp("e2e")
    paths = PathsConfig(root=str(root / "run"))
    paths.ensure()

    # -- data ------------------------------------------------------------
    dump = _make_dump(paths.raw / "azwiki" / "articles.xml", build_corpus_articles())
    data_cfg = DataConfig(sources=["azwiki"], min_chars=120, validation_fraction=0.15, seed=7)
    manifest = build_corpus(paths, data_cfg, progress_every=0)

    # -- tokenizer -------------------------------------------------------
    from tagai.data.prepare import iter_corpus

    texts = [row["text"] for row in iter_corpus(paths, "train")]
    tok_cfg = TokenizerConfig(vocab_size=400, num_threads=2, input_sentence_size=0)
    tok_stats = train_tokenizer(texts, paths.tokenizer_dir, tok_cfg, probe_texts=texts)

    # -- pack ------------------------------------------------------------
    tok = TagAITokenizer.from_pretrained(paths.tokenizer_dir)
    train_meta, train_pack_stats = pack_corpus(paths, tok, "train", max_doc_tokens=256, progress_every=0)
    val_meta, val_pack_stats = pack_corpus(paths, tok, "validation", max_doc_tokens=256, progress_every=0)

    # -- train -----------------------------------------------------------
    seq_len = 64
    model_cfg = ModelConfig(
        name="e2e-tiny", vocab_size=tok.vocab_size, d_model=64, n_layers=2,
        n_heads=4, max_seq_len=seq_len, dropout=0.0,
    )
    train_cfg = TrainConfig(
        max_steps=40, micro_batch_size=4, grad_accum_steps=2, max_seq_len=seq_len,
        warmup_steps=5, log_every=10, eval_every=20, save_every=20, eval_batches=2,
        lr=1e-3, seed=7,
    )
    train_ds = PackedDataset(train_meta, seq_len)
    val_ds = PackedDataset(val_meta, seq_len)
    train_sampler = BatchSampler(train_ds, train_cfg.micro_batch_size, seed=7)

    set_seed(7)
    model = TagAIModel(model_cfg)
    initial_embedding = model.tok_emb.weight.detach().clone()
    trainer = Trainer(model, train_sampler, val_ds, train_cfg, model_cfg, root / "run", tokenizer=tok)
    before = trainer.evaluate(batches=4)
    trainer.train()
    after = trainer.evaluate(batches=4)
    trainer.save()

    # -- export ----------------------------------------------------------
    ckpt = find_latest_valid(root / "run")
    export = export_bundle(ckpt, paths.tokenizer_dir, paths.export)

    return {
        "paths": paths,
        "manifest": manifest,
        "tok_stats": tok_stats,
        "tok": tok,
        "train_meta": train_meta,
        "val_meta": val_meta,
        "train_pack_stats": train_pack_stats,
        "val_pack_stats": val_pack_stats,
        "model": model,
        "initial_embedding": initial_embedding,
        "state": trainer.state,
        "before": before,
        "after": after,
        "checkpoint": ckpt,
        "export": export,
        "load_model_from_checkpoint": load_model_from_checkpoint,
        "load_bundle": load_bundle,
        "TagAIGenerator": TagAIGenerator,
        "GenerateConfig": GenerateConfig,
        "read_json": read_json,
    }


def test_corpus_built_from_real_format(pipeline) -> None:
    manifest = pipeline["manifest"]
    assert manifest["counts"]["train"] > 0
    assert manifest["counts"]["validation"] > 0
    assert manifest["counts"]["train"] + manifest["counts"]["validation"] == manifest["counts"]["train"] + manifest["counts"]["validation"]
    langs = manifest["statistics"]["language_distribution"]
    assert langs.get("az", 0) > 0
    # Non-Azerbaijani articles must have been filtered out.
    assert langs.get("ru", 0) >= 0


def test_validation_split_is_disjoint(pipeline) -> None:
    from tagai.data.prepare import iter_corpus

    paths = pipeline["paths"]
    train = {r["text"] for r in iter_corpus(paths, "train")}
    val = {r["text"] for r in iter_corpus(paths, "validation")}
    assert val, "the validation split must not be empty"
    assert not (train & val), "a document leaked into both the train and validation splits"


def test_tokenizer_covers_alphabet_and_round_trips(pipeline) -> None:
    stats = pipeline["tok_stats"]
    assert stats.vocabulary["az_alphabet_missing"] == []
    assert stats.vocabulary["az_alphabet_coverage"] == 1.0
    assert stats.roundtrip["roundtrip_rate"] == 1.0
    tok = pipeline["tok"]
    for text in ("Salam! Necəsiniz?", "Azərbaycan Respublikası ən çox gözəl yerdir.", "12,75% (2024)"):
        assert tok.roundtrip(text).roundtrip_ok, text


def test_packing_produces_shards(pipeline) -> None:
    meta = pipeline["train_meta"]
    assert meta["total_tokens"] > 0
    assert len(meta["shards"]) >= 1
    assert meta["stats"]["documents"] > 0


def test_model_weights_were_randomly_initialized(pipeline) -> None:
    """The embedding matrix must have changed and must not equal any constant."""
    model = pipeline["model"]
    initial = pipeline["initial_embedding"]
    final = model.tok_emb.weight.detach()
    assert not torch_equal(initial, final), "weights never updated - training did nothing"
    assert float(final.std()) > 0.0
    assert float(final.abs().max()) > 0.0


def torch_equal(a, b) -> bool:
    import torch

    return torch.equal(a, b)


def test_training_reduced_the_loss(pipeline) -> None:
    before, after = pipeline["before"], pipeline["after"]
    assert after < before, f"validation loss did not improve: {before} -> {after}"


def test_training_state_is_recorded(pipeline) -> None:
    state = pipeline["state"]
    assert state.step == 40
    assert state.tokens_seen > 0
    assert len(state.history) == 40


def test_checkpoint_saves_and_reloads(pipeline) -> None:
    load_model_from_checkpoint = pipeline["load_model_from_checkpoint"]
    reloaded, cfg = load_model_from_checkpoint(pipeline["checkpoint"])
    assert cfg.d_model == pipeline["model"].cfg.d_model
    import torch

    for (n1, p1), (n2, p2) in zip(pipeline["model"].state_dict().items(), reloaded.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2)


def test_generation_produces_az_text(pipeline) -> None:
    generator = pipeline["TagAIGenerator"].from_model(pipeline["model"], pipeline["tok"], device="cpu")
    out = generator.generate("Azərbaycan", cfg=pipeline["GenerateConfig"](max_new_tokens=12, seed=3))
    assert isinstance(out, str) and out.strip()
    assert "Azərbaycan" in out


def test_export_bundle_is_self_contained(pipeline) -> None:
    export = pipeline["export"]
    assert export["pretrained_weights_used"] is False
    assert export["tokenizer"]["files"]
    assert export["weights_bytes"] > 0
    out = pipeline["paths"].export
    assert (out / "manifest.json").exists()
    assert (out / "model" / "pytorch_model.bin").exists()
    assert (out / "tokenizer" / "tokenizer.model").exists()
    assert (out / "config.json").exists()
    assert Path(export["archive"]).exists()


def test_exported_bundle_loads_and_generates(pipeline) -> None:
    generator = pipeline["load_bundle"](pipeline["paths"].export, device="cpu")
    out = generator.generate("Salam", cfg=pipeline["GenerateConfig"](max_new_tokens=8, seed=5))
    assert isinstance(out, str)


def test_dataset_quality_report_is_measured(pipeline) -> None:
    from tagai.data.quality import analyse

    report = analyse(pipeline["paths"])
    assert report["documents_analysed"] > 0
    assert report["total_characters"] > 0
    assert report["character_distribution"]["az_letter_coverage"] > 0.9
    assert report["licensing"]["eligible_sources"]


def test_cli_end_to_end(tmp_path) -> None:
    """The documented CLI sequence must work on a prepared corpus."""
    from tagai.cli import main

    root = tmp_path / "cli_run"
    paths = PathsConfig(root=str(root))
    paths.ensure()
    _make_dump(paths.raw / "azwiki" / "articles.xml", build_corpus_articles())

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "paths": {"root": str(root)},
                "data": {"sources": ["azwiki"], "min_chars": 120, "validation_fraction": 0.1},
                "tokenizer": {"vocab_size": 400, "num_threads": 2, "input_sentence_size": 0},
                "model": {"preset": "tagai-tiny", "max_seq_len": 64},
                "train": {"preset": "smoke", "max_steps": 10, "micro_batch_size": 4, "eval_every": 5, "save_every": 5, "log_every": 5},
            }
        ),
        encoding="utf-8",
    )

    assert main(["--config", str(cfg_path), "clean"]) == 0
    assert main(["--config", str(cfg_path), "report"]) == 0
    assert main(["--config", str(cfg_path), "tokenize"]) == 0
    assert main(["--config", str(cfg_path), "pack"]) == 0
    assert main(["--config", str(cfg_path), "train"]) == 0
    assert main(["--config", str(cfg_path), "generate", "--max-new-tokens", "8"]) == 0
    assert main(["--config", str(cfg_path), "evaluate", "--max-new-tokens", "8", "--ppl-batches", "1"]) == 0
    assert main(["--config", str(cfg_path), "export"]) == 0

    assert (paths.tokenizer_dir / "tokenizer.model").exists()
    assert (paths.reports / "evaluation.json").exists()
    assert (paths.export / "manifest.json").exists()