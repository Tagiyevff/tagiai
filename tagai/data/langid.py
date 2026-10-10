"""Lightweight language identification tuned for Azerbaijani vs Turkish.

`langdetect` is used when installed because it is already present in most
notebook images, but TagAI also ships a character-profile classifier. The
profile is deliberately conservative: Azerbaijani and Turkish share most of
their alphabet, so the discriminator is the set of letters that only Azerbaijani
normally uses in running text (ə, ğ outside loanwords, plus the ğ/ğ vs ğ
contrast with Turkish) combined with script checks.

Keeping a dependency-free fallback means the pipeline still runs in an offline
kernel where `pip install langdetect` fails.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

from ..utils import get_logger

log = get_logger("data.langid")

AZ_CORE = set("abcçdefgğhıijklmnoöpqrsştuüvyzə")
AZ_UPPER = set("ABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVYZƏ")
TR_ONLY = set("çğıöşü")  # also valid Azerbaijani letters
AZ_ONLY = set("ə")  # the discriminating letter
MATHY = re.compile(r"[<>{}\[\]|\\^`~*#$\\]")
URLISH = re.compile(r"https?://|www\.|\.com|\.org|\.az\b", re.IGNORECASE)


@functools.lru_cache(maxsize=1)
def _langdetect_available() -> bool:
    try:
        import langdetect  # noqa: F401

        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _langdetect_factory():
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    # Deterministic results across runs.
    DetectorFactory.seed = 0
    return detect_langs, LangDetectException


def char_profile(text: str) -> dict:
    total = sum(1 for c in text if not c.isspace()) or 1
    az_only = sum(1 for c in text if c in "əƏ")
    az_upper_only = sum(1 for c in text if c in "Ə")
    return {
        "az_only_ratio": (az_only + az_upper_only) / total,
        "tr_only_ratio": sum(1 for c in text if c in TR_ONLY) / total,
        "az_letters_ratio": sum(1 for c in text if c in AZ_CORE or c in AZ_UPPER) / total,
        "mathy_ratio": len(MATHY.findall(text)) / total,
        "urlish": bool(URLISH.search(text)),
        "total_chars": total,
    }


def detect_language(text: str, sample_chars: int = 4000) -> dict:
    """Return a language decision with the evidence attached.

    Keys: ``lang``, ``confidence``, ``method``, ``az_ratio``, ``profile``.
    """
    snippet = text[:sample_chars]
    profile = char_profile(snippet)
    az_ratio = profile["az_letters_ratio"]
    cyr = sum(1 for c in snippet if "\u0400" <= c <= "\u04FF") / max(1, len(snippet))

    if cyr > 0.20:
        return {
            "lang": "ru",
            "confidence": round(min(0.99, cyr), 4),
            "method": "script",
            "az_ratio": az_ratio,
            "profile": profile,
        }

    lang, conf, method = "unknown", 0.0, "char_profile"
    if _langdetect_available() and len(snippet) >= 120:
        detect_langs, LangDetectException = _langdetect_factory()
        try:
            best = detect_langs(snippet)
            if best:
                lang, conf = best[0].lang, float(best[0].prob)
                method = "langdetect+char_profile"
        except LangDetectException:
            lang, conf, method = "unknown", 0.0, "char_profile"
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("langdetect failed: %s", exc)

    # Character evidence overrides the statistical guess: ə does not exist in
    # modern Turkish orthography at all.
    if profile["az_only_ratio"] > 0.004:
        lang = "az"
        conf = max(conf, 0.9)
        method = "char_profile:schwa"
    elif lang == "tr" and profile["az_only_ratio"] <= 0.0005:
        lang = "tr"
        conf = max(conf, 0.5)
    elif lang == "unknown" and az_ratio > 0.55:
        lang = "az"
        conf = 0.5
        method = "char_profile:fallback"

    return {"lang": lang, "confidence": round(conf, 4), "method": method, "az_ratio": round(az_ratio, 4), "profile": profile}


def is_azerbaijani(text: str, min_confidence: float = 0.55) -> tuple[bool, dict]:
    res = detect_language(text)
    ok = res["lang"] == "az" and res["confidence"] >= min_confidence
    return ok, res


@dataclass
class LanguageFilter:
    """Streaming filter that keeps Azerbaijani and reports per-document verdicts."""

    min_confidence: float = 0.55
    keep_langs: tuple[str, ...] = ("az",)

    def accept(self, text: str) -> tuple[bool, dict]:
        res = detect_language(text)
        ok = res["lang"] in self.keep_langs and res["confidence"] >= self.min_confidence
        return ok, res