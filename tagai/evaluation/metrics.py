"""Language-quality metrics that work without human raters."""

from __future__ import annotations

import math
import re
from typing import Sequence

AZ_SENTENCE_END = re.compile(r"[.!?…]+\s")
_WORD = re.compile(r"\w+", re.UNICODE)


def token_statistics(text: str, tokenizer=None) -> dict:
    words = _WORD.findall(text)
    tokens = tokenizer.encode(text) if tokenizer is not None else None
    return {
        "chars": len(text),
        "words": len(words),
        "tokens": len(tokens) if tokens is not None else None,
        "chars_per_token": round(len(text) / max(1, len(tokens)), 3) if tokens else None,
        "unique_word_ratio": round(len(set(w.lower() for w in words)) / max(1, len(words)), 4),
    }


def repetition_ratio(text: str, n: int = 4) -> float:
    """Fraction of repeated n-grams. High values mean degenerate output."""
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return round(1 - len(set(grams)) / len(grams), 4)


def distinct_n(texts: Sequence[str], n: int = 3) -> float:
    """Distinct-n across a batch: the standard diversity metric."""
    total, unique = 0, set()
    for t in texts:
        words = t.split()
        for i in range(max(0, len(words) - n + 1)):
            grams = tuple(words[i : i + n])
            total += 1
            unique.add(grams)
    return round(len(unique) / total, 4) if total else 0.0


def sentence_count(text: str) -> int:
    return len([s for s in AZ_SENTENCE_END.split(text.strip()) if s])


def mean_sentence_length(text: str) -> float:
    words = _WORD.findall(text)
    sents = sentence_count(text)
    return round(len(words) / max(1, sents), 2)


def az_char_ratio(text: str) -> float:
    az = set("abcçdefgğhıijklmnoöpqrsştuüvyzəABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVYZƏ")
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return round(sum(1 for c in letters if c in az) / len(letters), 4)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance normalised by reference length."""
    ref, hyp = reference, hypothesis
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return round(prev[-1] / len(ref), 4)


def perplexity_from_nll(nll: float) -> float:
    return float(math.exp(min(20.0, nll)))


def bits_per_character(nll: float, chars: int) -> float:
    """Scale-free quality signal: fewer bits/char means a better model of text."""
    return round(nll / max(1, chars) / math.log(2), 5)


def summarize_generations(texts: Sequence[str], tokenizer=None) -> dict:
    if not texts:
        return {"count": 0}
    reps = [repetition_ratio(t) for t in texts]
    return {
        "count": len(texts),
        "mean_chars": round(sum(len(t) for t in texts) / len(texts), 1),
        "mean_sentence_length": round(sum(mean_sentence_length(t) for t in texts) / len(texts), 2),
        "mean_az_char_ratio": round(sum(az_char_ratio(t) for t in texts) / len(texts), 4),
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
        "mean_repetition_ratio": round(sum(reps) / len(reps), 4),
        "degenerate_rate": round(sum(1 for r in reps if r > 0.5) / len(reps), 4),
        "empty_rate": round(sum(1 for t in texts if not t.strip()) / len(texts), 4),
        "token_stats_mean": _mean_token_stats(texts, tokenizer),
    }


def _mean_token_stats(texts: Sequence[str], tokenizer) -> dict | None:
    if tokenizer is None:
        return None
    stats = [token_statistics(t, tokenizer) for t in texts]
    return {
        "mean_words": round(sum(s["words"] for s in stats) / len(stats), 2),
        "mean_tokens": round(sum(s["tokens"] for s in stats) / len(stats), 2),
        "mean_chars_per_token": round(sum(s["chars_per_token"] for s in stats) / len(stats), 3),
    }