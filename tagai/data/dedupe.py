"""Exact and near-duplicate detection.

Exact duplicates use a 64-bit blake2b digest. Near duplicates use a
character-level MinHash/LSH band index, which is cheap in memory and catches
the "same article with a slightly different intro" pattern that Wikipedia
dumps are full of.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from ..utils import get_logger

log = get_logger("data.dedupe")

_NUM_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_for_dedupe(text: str) -> str:
    """Lowercase, strip punctuation/digits, collapse whitespace.

    Numbers and punctuation are removed so that "1.230" vs "1230" or
    "Bələdi," vs "Bələdi" collapse to the same key.
    """
    text = text.lower()
    text = _NUM_RE.sub("", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def exact_key(text: str) -> str:
    return hashlib.blake2b(normalize_for_dedupe(text).encode("utf-8"), digest_size=8).hexdigest()


def shingles(text: str, k: int = 5) -> set[str]:
    """Character k-gram set (works for any language without tokenization)."""
    norm = normalize_for_dedupe(text)
    if len(norm) < k:
        return {norm} if norm else set()
    return {norm[i : i + k] for i in range(0, len(norm) - k + 1, 2)}


def minhash(sh: set[str], num_perm: int = 64, seed: int = 1337) -> tuple[int, ...]:
    import numpy as np

    if not sh:
        return tuple([0] * num_perm)
    rng = np.random.default_rng(seed)
    salts = rng.integers(0, 2**31 - 1, size=num_perm, dtype=np.int64)
    a = rng.integers(1, 2**31 - 1, size=num_perm, dtype=np.int64)
    b = rng.integers(0, 2**31 - 1, size=num_perm, dtype=np.int64)
    vals = np.array([int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little") % (2**31) for s in sh], dtype=np.int64)
    if vals.size == 0:
        return tuple([0] * num_perm)
    hashes = (a[None, :] * vals[:, None] + b[None, :]) % (2**31 - 1)
    sig = hashes.min(axis=0)
    salts = (salts ^ sig).astype(np.int64)
    return tuple(int(x) for x in salts)


@dataclass
class DedupeResult:
    kept: list[str] = field(default_factory=list)
    exact_duplicates: int = 0
    near_duplicates: int = 0
    comparisons: int = 0

    @property
    def removed(self) -> int:
        return self.exact_duplicates + self.near_duplicates

    def summary(self, total: int) -> dict:
        return {
            "input_documents": total,
            "unique_documents": len(self.kept),
            "exact_duplicates_removed": self.exact_duplicates,
            "near_duplicates_removed": self.near_duplicates,
            "duplicate_rate": round(self.removed / max(1, total), 5),
        }


class Deduplicator:
    """Streaming exact + MinHash/LSH near-duplicate filter.

    Memory is bounded by the LSH buckets, not by the corpus: only a signature
    per retained document is stored.
    """

    def __init__(self, threshold: float = 0.90, num_perm: int = 64, buckets: int = 32, k: int = 5, seed: int = 1337) -> None:
        self.threshold = threshold
        self.num_perm = num_perm
        self.buckets = buckets
        self.k = k
        self.seed = seed
        self._exact: set[str] = set()
        self._buckets: dict[tuple[int, int], list[tuple[int, ...]]] = {}
        self.stats = DedupeResult()

    def _bands(self) -> int:
        return max(1, min(self.buckets, self.num_perm))

    def is_duplicate(self, text: str) -> tuple[bool, str]:
        key = exact_key(text)
        if key in self._exact:
            self.stats.exact_duplicates += 1
            return True, "exact"

        sig = minhash(shingles(text, self.k), self.num_perm, self.seed)
        band_size = max(1, self.num_perm // self._bands())
        for b in range(self._bands()):
            band = tuple(sig[b * band_size : (b + 1) * band_size])
            bucket_key = (b, hash(band))
            candidates = self._buckets.get(bucket_key, [])
            for cand_sig in candidates:
                self.stats.comparisons += 1
                if _sig_similarity(sig, cand_sig) >= self.threshold:
                    self.stats.near_duplicates += 1
                    return True, "near"
            self._buckets.setdefault(bucket_key, []).append(sig)
        self._exact.add(key)
        return False, "unique"

    def filter(self, texts: Iterable[str]) -> DedupeResult:
        """Collect the unique texts; counters are maintained by ``is_duplicate``."""
        result = DedupeResult()
        for text in texts:
            dup, _ = self.is_duplicate(text)
            if dup:
                continue
            result.kept.append(text)
        result.exact_duplicates = self.stats.exact_duplicates
        result.near_duplicates = self.stats.near_duplicates
        result.comparisons = self.stats.comparisons
        return result

    def __len__(self) -> int:
        return len(self._exact)


def _sig_similarity(a: Sequence[int], b: Sequence[int]) -> float:
    if not a or not b:
        return 0.0
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / len(a)


def paragraph_repetition(text: str) -> dict:
    """Detect artificial inflation via repeated paragraphs within a document."""
    paras = [" ".join(p.split()) for p in text.split("\n\n") if p.strip()]
    if len(paras) < 3:
        return {"paragraphs": len(paras), "unique": len(set(paras)), "repeat_ratio": 0.0}
    uniq = len(set(paras))
    return {
        "paragraphs": len(paras),
        "unique": uniq,
        "repeat_ratio": round(1 - uniq / len(paras), 4),
    }


def dedupe_texts(texts: Iterable[str], threshold: float = 0.90, buckets: int = 32) -> Iterator[tuple[str, bool, str]]:
    """Generator yielding ``(text, is_duplicate, kind)`` for streaming callers."""
    dd = Deduplicator(threshold=threshold, buckets=buckets)
    for text in texts:
        dup, kind = dd.is_duplicate(text)
        yield text, dup, kind