"""Thin, dependency-light wrapper around a trained SentencePiece model."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from ..utils import atomic_write_json, get_logger

log = get_logger("tokenizer")

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<s>", "</s>"


@dataclass
class EncodingSample:
    text: str
    ids: list[int]
    pieces: list[str]
    roundtrip_ok: bool
    reconstructed: str


class TagAITokenizer:
    """Encode/decode + Azerbaijani-specific integrity checks.

    The model interface intentionally matches what the trainer needs:
    ``encode`` returns plain python ints so it can be used without torch.
    """

    def __init__(self, model_path: str | os.PathLike[str]) -> None:
        try:
            import sentencepiece as spm
        except ImportError as exc:  # pragma: no cover
            raise ImportError("sentencepiece is required: pip install sentencepiece") from exc
        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else 0
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

    # -- properties ------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def piece_size_bytes(self) -> int:
        return self.sp.get_piece_size()

    def id_to_piece(self, idx: int) -> str:
        return self.sp.id_to_piece(idx)

    def piece_to_id(self, piece: str) -> int:
        return self.sp.piece_to_id(piece)

    def special_ids(self) -> dict[str, int]:
        return {"pad_id": self.pad_id, "unk_id": self.unk_id, "bos_id": self.bos_id, "eos_id": self.eos_id}

    # -- encode / decode -------------------------------------------------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(self, texts: Sequence[str], num_threads: int = 8, add_eos: bool = True) -> Iterator[list[int]]:
        results = self.sp.encode(list(texts), num_threads=num_threads, out_type=int)
        if add_eos:
            results = [r + [self.eos_id] for r in results]
        yield from results

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        """Decode token ids back to text.

        ``skip_special=False`` (default) keeps a literal round trip, which is
        what the integrity tests need.
        """
        id_list = [int(i) for i in ids]
        if skip_special:
            id_list = [i for i in id_list if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.decode(id_list)

    def pieces(self, text: str) -> list[str]:
        return self.sp.encode(text, out_type=str)

    # -- Azerbaijani integrity -------------------------------------------
    def roundtrip(self, text: str) -> EncodingSample:
        ids = self.encode(text)
        pieces = [self.sp.id_to_piece(i) for i in ids]
        reconstructed = self.decode(ids)
        # SentencePiece prefixes a U+2581 word-boundary marker; normalise whitespace
        # and the leading marker before comparing.
        ok = _normalize_ws(reconstructed) == _normalize_ws(text)
        return EncodingSample(text=text, ids=ids, pieces=pieces, roundtrip_ok=ok, reconstructed=reconstructed)

    def roundtrip_report(self, texts: Sequence[str]) -> dict:
        total = 0
        failures = 0
        char_loss_total = 0
        examples: list[dict] = []
        for text in texts:
            sample = self.roundtrip(text)
            total += 1
            if not sample.roundtrip_ok:
                failures += 1
                char_loss_total += abs(len(text) - len(sample.reconstructed))
                if len(examples) < 20:
                    examples.append({"input": text[:120], "decoded": sample.reconstructed[:120]})
        return {
            "tested": total,
            "failures": failures,
            "roundtrip_rate": round(1 - failures / total, 6) if total else 1.0,
            "char_delta_total": char_loss_total,
            "examples": examples,
        }

    def vocabulary_report(self) -> dict:
        """Summarize vocabulary composition and Azerbaijani-letter coverage."""
        from ..config import AZERBAIJANI_ALPHABET

        pieces = [self.sp.id_to_piece(i) for i in range(self.vocab_size)]
        piece_set = set(pieces)
        letter_pieces = [p for p in pieces if p.startswith("▁") and len(p) > 1 and p[1:].isalpha()]
        az_pieces = [p for p in pieces if any(c in AZERBAIJANI_ALPHABET for c in p)]

        # Coverage must be measured by *encodability*, not by substring search:
        # a letter can appear inside a multi-character piece without having its
        # own single-character piece, which is what SentencePiece needs to
        # guarantee lossless encoding of arbitrary text.
        missing: list[str] = []
        for ch in AZERBAIJANI_ALPHABET:
            probe = f"x{ch}x"
            ids = self.sp.encode(probe, out_type=int)
            if self.unk_id in ids:
                missing.append(ch)
        return {
            "vocab_size": self.vocab_size,
            "az_alphabet_size": len(AZERBAIJANI_ALPHABET),
            "az_alphabet_missing": missing,
            "az_alphabet_coverage": round(1 - len(missing) / max(1, len(AZERBAIJANI_ALPHABET)), 4),
            "multichar_az_pieces": len(az_pieces),
            "multichar_word_pieces": len(letter_pieces),
            "avg_piece_chars": round(sum(len(p) for p in pieces) / max(1, len(pieces)), 3),
            "pieces": len(piece_set),
        }

    def assert_az_alphabet_complete(self) -> list[str]:
        """Return the Azerbaijani letters that would encode to <unk>.

        Called after training so a coverage failure is a loud error rather
        than a corpus full of silent unknowns discovered months later.
        """
        report = self.vocabulary_report()
        return report["az_alphabet_missing"]

    # -- persistence -----------------------------------------------------
    def save_meta(self, out_dir: str | os.PathLike[str]) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_file": Path(self.model_path).name,
            "vocab_size": self.vocab_size,
            **self.special_ids(),
        }
        atomic_write_json(out / "tokenizer_config.json", payload)
        return out / "tokenizer_config.json"

    @classmethod
    def from_pretrained(cls, model_dir: str | os.PathLike[str]) -> "TagAITokenizer":
        d = Path(model_dir)
        cand = d / "tokenizer.model"
        if not cand.exists():
            found = sorted(d.glob("*.model"))
            if not found:
                raise FileNotFoundError(f"no *.model SentencePiece file in {d}")
            cand = found[0]
        return cls(cand)

    def __len__(self) -> int:
        return self.vocab_size


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())