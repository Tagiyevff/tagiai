"""Train a SentencePiece BPE tokenizer on the cleaned Azerbaijani corpus."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from ..config import TokenizerConfig
from ..utils import atomic_write_json, get_logger
from .wrapper import TagAITokenizer

log = get_logger("tokenizer.trainer")

USER_DEFINED_SYMBOLS: list[str] = []  # Azerbaijani is whitespace-tokenised; no extra symbols needed.


@dataclass
class TokenizerStats:
    vocab_size: int = 0
    training_docs: int = 0
    training_bytes: int = 0
    training_sentences: int = 0
    corpus_lines_used: int = 0
    average_chars_per_line: float = 0.0
    longest_line_chars: int = 0
    fragmentation: dict = field(default_factory=dict)
    vocabulary: dict = field(default_factory=dict)
    roundtrip: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def write_corpus_file(texts: Iterable[str], out_path: str | os.PathLike[str]) -> dict:
    """Write one document per line with newlines collapsed (SentencePiece format)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    docs = 0
    total_bytes = 0
    longest = 0
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for text in texts:
            line = " ".join(text.split())
            if not line:
                continue
            fh.write(line + "\n")
            docs += 1
            total_bytes += len(line.encode("utf-8"))
            longest = max(longest, len(line))
    return {"documents": docs, "bytes": total_bytes, "longest_line_chars": longest, "path": str(out)}


def _train_sentencepiece(spm, corpus_path: Path, prefix: Path, cfg: TokenizerConfig, character_coverage: float) -> None:
    try:
        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=str(prefix),
            vocab_size=cfg.vocab_size,
            model_type=cfg.model_type,
            character_coverage=character_coverage,
            input_sentence_size=cfg.input_sentence_size,
            shuffle_input_sentence=cfg.shuffle_input_sentence,
            num_threads=cfg.num_threads,
            pad_id=cfg.pad_id,
            unk_id=cfg.unk_id,
            bos_id=cfg.bos_id,
            eos_id=cfg.eos_id,
            user_defined_symbols=USER_DEFINED_SYMBOLS,
            hard_vocab_limit=cfg.hard_vocab_limit,
            normalization_rule_name="nmt_nfkc",
            remove_extra_whitespaces=True,
            max_sentence_length=cfg.max_sentence_length,
            train_extremely_large_corpus=cfg.train_extremely_large_corpus,
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            pad_piece="<pad>",
            minloglevel=1,
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "Vocabulary size too high" in msg:
            max_size = msg.rsplit("<=", 1)[-1].rstrip(".").strip()
            raise ValueError(
                f"tokenizer.vocab_size={cfg.vocab_size} is larger than this corpus can support "
                f"(maximum {max_size}). Either lower vocab_size, add more training text, or set "
                f"tokenizer.hard_vocab_limit=false to accept a smaller vocabulary."
            ) from exc
        raise


def train_tokenizer(
    texts: list[str] | Iterable[str],
    out_dir: str | os.PathLike[str],
    cfg: TokenizerConfig,
    *,
    extra_characters: str | None = None,
    probe_texts: list[str] | None = None,
    require_alphabet: bool = True,
) -> TokenizerStats:
    """Train SentencePiece and write tokenizer artifacts + statistics to ``out_dir``.

    After training, every one of the 64 Azerbaijani letters is checked. If any
    would encode to ``<unk>`` the tokenizer is retrained at full character
    coverage; if letters are still missing, the run raises. A corpus with gaps
    in the alphabet produces a model that cannot spell, and that must not be a
    silent failure discovered months later.
    """
    try:
        import sentencepiece as spm
    except ImportError as exc:  # pragma: no cover
        raise ImportError("sentencepiece is required: pip install sentencepiece") from exc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    corpus_path = out / "tokenizer_corpus.txt"
    corpus_info = write_corpus_file(texts, corpus_path)
    if corpus_info["documents"] == 0:
        raise ValueError("cannot train a tokenizer on an empty corpus")

    prefix = out / "tokenizer"  # produces tokenizer.model / tokenizer.vocab
    coverage_used = cfg.character_coverage

    _train_sentencepiece(spm, corpus_path, prefix, cfg, coverage_used)
    tok = TagAITokenizer(Path(str(prefix) + ".model"))

    missing = tok.assert_az_alphabet_complete()
    if missing and coverage_used < 1.0:
        # A rare letter fell outside the coverage threshold. Retraining at 100%
        # coverage keeps it at the cost of a few rare noise characters.
        log.warning(
            "Azerbaijani letters missing at character_coverage=%s (%s); retraining at 1.0",
            coverage_used,
            " ".join(missing),
        )
        _train_sentencepiece(spm, corpus_path, prefix, cfg, 1.0)
        tok = TagAITokenizer(Path(str(prefix) + ".model"))
        missing = tok.assert_az_alphabet_complete()
        coverage_used = 1.0

    if missing and require_alphabet:
        raise ValueError(
            "the trained tokenizer cannot encode these Azerbaijani letters: "
            + " ".join(missing)
            + ". They are probably absent from the corpus. Add documents that use them "
            "(or raise tokenizer.vocab_size), then retrain."
        )
    if missing:
        log.warning("Azerbaijani letters missing from the vocabulary: %s", " ".join(missing))

    stats = TokenizerStats(
        vocab_size=tok.vocab_size,
        training_docs=corpus_info["documents"],
        training_bytes=corpus_info["bytes"],
        corpus_lines_used=corpus_info["documents"],
        average_chars_per_line=corpus_info["bytes"] / max(1, corpus_info["documents"]),
        longest_line_chars=corpus_info["longest_line_chars"],
        config={**{k: v for k, v in asdict(cfg).items()}, "character_coverage_used": coverage_used},
    )
    stats.vocabulary = tok.vocabulary_report()

    probes = probe_texts if probe_texts is not None else AZERBAIJANI_PROBES
    if probes:
        stats.roundtrip = tok.roundtrip_report(probes)
        stats.fragmentation = fragmentation_report(tok, probes)
    else:
        stats.roundtrip = {"tested": 0, "failures": 0, "roundtrip_rate": 1.0}
        stats.fragmentation = {}

    tok.save_meta(out)
    # The raw training corpus is large; keep it only when it is small enough to help.
    if corpus_info["bytes"] > 50 * 1024 * 1024:
        corpus_path.unlink(missing_ok=True)

    atomic_write_json(out / "tokenizer_stats.json", stats.to_dict())
    log.info(
        "tokenizer trained: vocab=%d docs=%d roundtrip_rate=%.5f az_alphabet_coverage=%.3f",
        stats.vocab_size,
        stats.training_docs,
        stats.roundtrip.get("roundtrip_rate", float("nan")),
        stats.vocabulary.get("az_alphabet_coverage", 0.0),
    )
    return stats


def fragmentation_report(tok: TagAITokenizer, texts: list[str], top_k: int = 15) -> dict:
    """How many tokens does Azerbaijani text actually cost?"""
    total_chars = 0
    total_tokens = 0
    per_text = []
    worst: list[tuple[float, str, list[str]]] = []
    for text in texts:
        ids = tok.encode(text)
        n = max(1, len(ids))
        per_text.append(n)
        total_chars += len(text)
        total_tokens += n
        pieces = [tok.id_to_piece(i) for i in ids]
        worst.append((n / max(1, len(text)), text[:60], pieces[:20]))
    worst.sort(reverse=True)
    return {
        "samples": len(texts),
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "chars_per_token": round(total_chars / max(1, total_tokens), 4),
        "mean_tokens_per_text": round(sum(per_text) / max(1, len(per_text)), 2),
        "max_tokens_per_text": max(per_text, default=0),
        "worst_fragmentation": [
            {"ratio": round(r, 3), "text": t, "pieces": p} for r, t, p in worst[:top_k]
        ],
    }


AZERBAIJANI_PROBES: list[str] = [
    "Azərbaycan Respublikası müstəqil dövlətdir və onun pay payı dağlar, dənizlər və düzənliklərdən ibarətdir.",
    "Salam! Mən Azərbaycan dilində yazmaq və oxumaq bacarıram.",
    "İnsan özü üçün ən vacib məqsəd olan işi görəndə xoşbəxt olur.",
    "Kitabı oxudum, amma çox sən demək istədim ki, o, çox maraqlı idi.",
    "Dünən axşam qərxanadakı kitabxanada uzun müddət oturub universitet üçün materialları hazırladıq.",
    "TÜRK DİLİ İLE AZƏRBAYCAN DİLİ ARASINDAKİ FARKLILIKLAR çox əhəmiyyətlidir.",
    "Müasir texnologiyalar süni intellekt sahəsində inqilab yaratdı.",
    "1 234 567 rəqəmi 1,5 milyon deməkdir; faiz dərəcəsi isə 12,75%-dir.",
    "birbaşa, haqqında, üzrə, görə, gedən, güllə, öyrənmək, açıq, doğru, çox, özü, ki, ki, də, lakin, ancaq.",
    "Çox sağ ol, mənə kömək etdiyiniz üçün. Yəqin, sabah yenidən görüşərik.",
]