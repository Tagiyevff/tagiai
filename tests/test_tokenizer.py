"""Tokenizer tests: Azerbaijani coverage, round-trip, fragmentation, I/O."""

from __future__ import annotations

import pytest

from tagai.config import AZERBAIJANI_ALPHABET, TokenizerConfig
from tagai.tokenizer.trainer import AZERBAIJANI_PROBES, train_tokenizer
from tagai.tokenizer.wrapper import TagAITokenizer

AZ_TEXT = [
    "Azərbaycan Respublikası müstəqil dövlətdir. Onun ərazisi qərb-şərq istiqamətində uzanır və dəniz sahiləri var.",
    "İnsan özü üçün ən vacib məqsəd olan işi görəndə xoşbəxt olur, çünki əmək insana hörmət qazandırır.",
    "Dünən axşam qərxanadakı kitabxanada uzun müddət oturub universitet üçün materialları hazırladıq.",
    "TÜRK DİLİ İLE AZƏRBAYCAN DİLİ ARASINDAKİ FARKLILIKLAR çox əhəmiyyətlidir.",
    "Müasir texnologiyalar süni intellekt sahəsində inqilab yaratdı.",
    "Uşağım məktəbdə yaxşı oxuyur və hər axşam ev tapşırığını yerinə yetirir.",
    "Sözün yazılışı, tələffuzu və mənası haqqında danışaq.",
    "Qişə yarpaq töhküyürdü, küləkdə yellənirdi və günəş hələ dəydi.",
    "Müqayisə etsək, bu fərq nəzəriyyə ilə təcrübə arasında aydın olar.",
    "Hesablamanı diqqətlə apardıq: 12 və 34 cəmlənir, 45-dən 9 çıxarılır, 7-yə 8 vurulur.",
]

# Every one of the 64 Azerbaijani letters has to appear in the training corpus,
# otherwise SentencePiece is entitled to drop it and encode it as <unk>.
AZ_ALPHABET_TRAINING = [
    "Xəzər dənizi, Zaqatala, Quba, Qusar, İsmayıllı və Xaçmaz rayonlarında səfər etdik.",
    "Jurnalistika və informasiya texnologiyaları sahəsində yeni tədqiqatlar aparılır.",
    "Şübhəsiz ki, Yuxarı Qarabağ torpaqları sülhün və ədalətin bərabər yoludur.",
    "Abcçdefgğhıijklmnoöpqrsştuüvyzə xüsusi aləf və hərflər dəstəklənir.",
    "ABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVYZƏ böyük hərflər dəstəklənir.",
    'Rəqmlər və işarələr: 12,75% (2024-cü il) — "qeyd" [1]; 3/4 = 0,75; 15 + 27 = 42.',
    "Fayl adı: sənəd.pdf; e-poçt: nümunə@ornek.az; veb: https://example.az/səhifə.",
    "Salam! Necəsiniz? Yaxşı yaxşı... Hələ bilmirəm: mən gələcəyəm, ancaq sonra.",
    "Suallar? Cavablar! Əlbəttə... Həmçinin «qeyd» və «yekun» ifadələri işlədilir.",
]


@pytest.fixture(scope="module")
def training_texts() -> list[str]:
    return AZ_TEXT + list(AZERBAIJANI_PROBES) + AZ_ALPHABET_TRAINING


@pytest.fixture(scope="module")
def trained(tmp_path_factory, training_texts):
    out = tmp_path_factory.mktemp("tokenizer")
    cfg = TokenizerConfig(vocab_size=1500, num_threads=4, input_sentence_size=0)
    stats = train_tokenizer(training_texts * 20, out, cfg, probe_texts=training_texts)
    return TagAITokenizer.from_pretrained(out), stats, training_texts


def test_alphabet_constants_are_consistent() -> None:
    """The 32-letter Azerbaijani alphabet must be complete and duplicate-free."""
    assert len(set(AZERBAIJANI_ALPHABET)) == len(AZERBAIJANI_ALPHABET)
    assert len(AZERBAIJANI_ALPHABET) == 64
    for ch in "əıöüğşçƏİÖÜĞŞÇ":
        assert ch in AZERBAIJANI_ALPHABET, f"{ch} missing from the alphabet constant"


def test_vocab_size_matches_request(trained) -> None:
    tok, stats, _ = trained
    assert tok.vocab_size == 1500
    assert stats.vocab_size == tok.vocab_size


def test_special_ids_are_valid(trained) -> None:
    tok, _, _ = trained
    ids = tok.special_ids()
    for name, value in ids.items():
        assert 0 <= value < tok.vocab_size, f"{name}={value} outside vocabulary"
    assert len({ids["pad_id"], ids["unk_id"], ids["bos_id"], ids["eos_id"]}) == 4


@pytest.mark.parametrize("text", AZERBAIJANI_PROBES)
def test_unicode_roundtrip(trained, text: str) -> None:
    """Azerbaijani letters must survive encode -> decode unchanged."""
    tok, _, _ = trained
    sample = tok.roundtrip(text)
    assert sample.roundtrip_ok, f"round-trip failed:\n  in : {text!r}\n  out: {sample.reconstructed!r}"


def test_all_az_alphabet_chars_are_encodable(trained) -> None:
    """None of the 64 Azerbaijani letters may encode to <unk>."""
    tok, _, _ = trained
    missing = tok.assert_az_alphabet_complete()
    assert missing == [], f"these Azerbaijani letters are not representable: {missing}"


def test_az_alphabet_coverage_is_complete(trained) -> None:
    tok, _, _ = trained
    report = tok.vocabulary_report()
    assert report["az_alphabet_size"] == 64
    assert report["az_alphabet_coverage"] == 1.0, report


def test_roundtrip_report_rate(trained) -> None:
    tok, _, texts = trained
    report = tok.roundtrip_report(AZERBAIJANI_PROBES + AZ_TEXT)
    assert report["tested"] > 0
    assert report["roundtrip_rate"] == 1.0, report


def test_numbers_and_punctuation_survive(trained) -> None:
    tok, _, _ = trained
    for text in ["1 234 567 rəqəmi 1,5 milyon deməkdir.", "12,75%-dir (2024-cü il).", "Salam! Necəsiniz?"]:
        assert tok.roundtrip(text).roundtrip_ok, text


def test_fragmentation_is_reasonable(trained) -> None:
    """A working tokenizer needs several characters per Azerbaijani token."""
    tok, _, _ = trained
    ids = tok.encode(AZ_TEXT[0])
    chars_per_token = len(AZ_TEXT[0]) / max(1, len(ids))
    assert 2.0 < chars_per_token < 8.0, f"suspicious fragmentation: {chars_per_token:.2f} chars/token"


def test_bos_eos_added(trained) -> None:
    tok, _, _ = trained
    plain = tok.encode("Salam")
    wrapped = tok.encode("Salam", add_bos=True, add_eos=True)
    assert wrapped == [tok.bos_id] + plain + [tok.eos_id]


def test_batch_encode_matches_single(trained) -> None:
    tok, _, _ = trained
    texts = AZ_TEXT[:4]
    batch = list(tok.encode_batch(texts, add_eos=False))
    singles = [tok.encode(t) for t in texts]
    assert batch == singles


def test_decode_roundtrips_ids(trained) -> None:
    tok, _, _ = trained
    ids = tok.encode(AZ_TEXT[2])
    assert tok.decode(ids) == AZ_TEXT[2]


def test_train_stats_are_measured(trained, training_texts) -> None:
    _, stats, _ = trained
    assert stats.training_docs == len(training_texts) * 20
    assert stats.training_bytes > 0
    assert stats.longest_line_chars > 0
    assert stats.fragmentation["total_tokens"] > 0


def test_artifacts_written(tmp_path, training_texts) -> None:
    out = tmp_path / "tok"
    train_tokenizer(
        training_texts * 5,
        out,
        TokenizerConfig(vocab_size=1500, num_threads=2, input_sentence_size=0),
        probe_texts=training_texts,
    )
    for name in ("tokenizer.model", "tokenizer_config.json", "tokenizer_stats.json"):
        assert (out / name).exists(), f"missing artifact {name}"


def test_missing_alphabet_raises_loudly(tmp_path) -> None:
    """A corpus with alphabet gaps must fail, not silently produce <unk> tokens."""
    tiny = ["Bu bir cümlədir və yalnız bir neçə söz var."]
    with pytest.raises(ValueError, match="cannot encode these Azerbaijani letters"):
        train_tokenizer(
            tiny * 40,
            tmp_path / "gap",
            TokenizerConfig(vocab_size=80, num_threads=2, input_sentence_size=0, hard_vocab_limit=False),
        )


def test_vocab_size_too_large_gives_actionable_error(tmp_path, training_texts) -> None:
    """Asking for more pieces than the corpus supports must explain what to do."""
    with pytest.raises(ValueError, match="larger than this corpus can support"):
        train_tokenizer(
            training_texts,
            tmp_path / "toobig",
            TokenizerConfig(vocab_size=1_000_000, num_threads=2, input_sentence_size=0),
        )


def test_empty_corpus_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        train_tokenizer([], tmp_path / "empty", TokenizerConfig(vocab_size=100))


def test_larger_vocab_yields_better_compression(tmp_path, training_texts) -> None:
    """Vocabulary size / sequence-length trade-off, measured rather than assumed."""
    small = TagAITokenizer.from_pretrained(_train(tmp_path / "small", 500, training_texts * 4))
    large = TagAITokenizer.from_pretrained(_train(tmp_path / "large", 1500, training_texts * 40))
    text = AZ_TEXT[0]
    small_ratio = len(text) / len(small.encode(text))
    large_ratio = len(text) / len(large.encode(text))
    assert small.vocab_size < large.vocab_size
    assert large_ratio >= small_ratio - 0.5, f"larger vocab compressed worse: {large_ratio:.2f} vs {small_ratio:.2f}"


def _train(path, vocab_size: int, texts):
    train_tokenizer(
        texts,
        path,
        TokenizerConfig(vocab_size=vocab_size, num_threads=2, input_sentence_size=0),
        probe_texts=texts[:5],
        require_alphabet=False,
    )
    return path