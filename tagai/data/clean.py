"""Unicode normalisation, Azerbaijani-aware cleaning and boilerplate removal."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from ..utils import get_logger

log = get_logger("data.clean")

# Azerbaijani letters that must survive normalisation untouched.
AZ_PROTECTED = "əıöüğşçƏİÖÜĞŞÇ"
_CYRILLIC = re.compile(r"[\u0400-\u04FF\u0500-\u052F]")

_WS_RUN = re.compile(r"[ \t\u00a0\u2000-\u200b\u3000]+")
_NL_RUN = re.compile(r"\n{3,}")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SOFT_HYPHEN = re.compile(r"\u00ad")
_SURROGATE = re.compile(r"[\ud800-\udfff]")
_REPLACEMENT = re.compile(r"\ufffd+")
_MISC_APOSTROPHE = re.compile(r"[\u2018\u2019\u02bc\u02bb\u055a\u055b\u201b\u2032\u00b4\u0060]")
_DASH = re.compile(r"[\u2010-\u2015\u2212]")
_QUOTES = {
    "\u201c": '"',
    "\u201d": '"',
    "\u00ab": '"',
    "\u00bb": '"',
    "\u2033": '"',
    "\u2036": "'",
    "\u2039": "'",
}
_QUOTE_TABLE = str.maketrans(_QUOTES)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# No length cap on the link body: a single wikilink can exceed any fixed
# limit, and a failed match leaves the literal "[[" and "]]" in the corpus.
_WIKILINK_RE = re.compile(r"\[\[([^\]|]*)(?:\|[^\]]*)?\]\]")
_EXTLINK_RE = re.compile(r"\[(?:https?://|www\.)[^\s\]]+\s+([^\]]*)\]")
_REF_MARK_RE = re.compile(r"\[[0-9]{1,3}\]|\[\^[\w\s]{0,40}\]|^:[0-9]{1,3}\s", re.MULTILINE)
_LIST_BULLET_RE = re.compile(r"^[\*\u2022\u00b7\-\u2013\u2014o]\s+", re.MULTILINE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]{0,300}>")
_HEADING_RE = re.compile(r"^\s*={2,6}\s*(.+?)\s*={2,6}\s*$", re.MULTILINE)
_TAG_RE = re.compile(r"<[^>]{1,40}>")
# Links into a namespace (File:, Category:, Template:) are navigation chrome.
# The target is a page name, not prose, so the whole link is dropped rather
# than unlinked the way an ordinary article link is.
_NAMESPACE_LINK_RE = re.compile(
    r"^(Special|Talk|User|Wikipedia|File|Image|Template|Category|MediaWiki|Module|Draft|Portal|Help|Image|"
    r"İnterfeys|İnterfeys dəyişikliyi|MediaVİKİ|MediaViKi|Şablon|Hərəkət|Vikipediya|Dostluq|Xəbər|"
    r"Məqalə|Modul|Portal|Danışma|Qovluq|Qeyd|Fayl|Şəkəl|Kateqoriya|Yardım|Redaktə|Mətn|"
    r"İstifadəçi|Media|qovluq|danışma|fayl|şablon|kateqoriya)\s*:",
    re.IGNORECASE,
)


def _wikilink_sub(match: "re.Match[str]") -> str:
    target = (match.group(1) or "").strip()
    if not target:
        return " "
    if _NAMESPACE_LINK_RE.match(target):
        return " "  # navigation link: no linguistic content
    return target
# MediaWiki templates ({{dil|az}}, {{Infobox person}}) and table markup carry no
# prose, only presentation instructions, so they are removed outright.
_TEMPLATE_RE = re.compile(r"\{\{[^}]{0,500}\}\}")
_TABLE_RE = re.compile(r"\{\|[\s\S]{0,400}?\|\}")
_PARAM_RE = re.compile(r"\{\|[^}]{0,300}\}")
# Infobox rows survive a naive {{...}} strip because templates nest.
# A line like "| url = https://..." or "! header" is presentation, not prose.
_WIKI_FIELD_ROW_RE = re.compile(r"^\s*[|!]{1,2}\s*[^\n]{0,80}?\s*=[^\n]{0,200}$", re.MULTILINE)
_WIKI_LIST_ROW_RE = re.compile(r"^\s*[|!]{1,2}\s*$", re.MULTILINE)
# Emphasis markers are presentation, not punctuation; leaving "**" and "'''"
# in the corpus teaches the model to emit stray stars.
_WIKI_EMPHASIS_RE = re.compile(r"'{2,5}")
_MD_EMPHASIS_RE = re.compile(r"\*{1,3}")
# Section headings that survived the =...= pass.
_HEADING_LEFTOVER_RE = re.compile(r"^\s*=+\s*", re.MULTILINE)


def strip_nested_templates(text: str, max_depth: int = 40) -> str:
    """Remove ``{{ ... }}`` spans, honouring nested braces.

    A regex cannot do this: ``{{Infobox|child={{WikiTemplate}}}}`` has inner
    closing braces that a non-greedy ``\\{\\{[^}]*\\}\\}`` matches against,
    leaving the outer template's parameters behind as stray ``| key =`` text.
    """
    out: list[str] = []
    i, n = 0, len(text)
    depth = 0
    start = 0
    while i < n and depth < max_depth:
        if text.startswith("{{", i):
            if depth == 0:
                start = i
            depth += 1
            i += 2
            continue
        if text.startswith("}}", i) and depth > 0:
            depth -= 1
            i += 2
            if depth == 0:
                out.append(" ")
            continue
        if depth == 0:
            out.append(text[i])
        i += 1
    if depth > 0:  # unbalanced braces: keep the remainder rather than losing text
        out.append(text[start:])
    return "".join(out)

REPLACEMENT_CHAR_WARN = "\ufffd"


@dataclass
class CleanStats:
    """Counters that make each cleaning decision auditable."""

    seen: int = 0
    kept: int = 0
    dropped_short: int = 0
    dropped_long: int = 0
    dropped_empty: int = 0
    dropped_boilerplate: int = 0
    dropped_url_heavy: int = 0
    dropped_repetitive: int = 0
    dropped_control: int = 0
    dropped_cyrillic_heavy: int = 0
    replacement_chars_fixed: int = 0
    wikilinks_removed: int = 0
    urls_removed: int = 0
    chars_before: int = 0
    chars_after: int = 0
    reasons: dict = field(default_factory=dict)

    def drop(self, reason: str, n: int = 1) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + n

    def to_dict(self) -> dict:
        d = {
            "documents_seen": self.seen,
            "documents_kept": self.kept,
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "dropped_total": self.seen - self.kept,
            "drop_rates": {k: round(v / max(1, self.seen), 5) for k, v in sorted(self.reasons.items())},
        }
        return d

    def summary(self) -> dict:
        return {
            "documents_seen": self.seen,
            "documents_kept": self.kept,
            "retention_rate": round(self.kept / max(1, self.seen), 5),
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "char_retention_rate": round(self.chars_after / max(1, self.chars_before), 5),
            "reasons": dict(sorted(self.reasons.items())),
        }


# Script and character statistics -----------------------------------------
_AZ_LETTERS = set("abcçdefgğhıijklmnoöpqrsştuüvyzə")
_AZ_UPPER = set("ABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVYZƏ")
_LATIN_EXTRA = set("âäîôöûùçşğ")
_DIGITS = set("0123456789")
_PUNCT = set(" .,;:!?()[]{}\"'`~@#$%^&*-+=_|<>/\\n\t")
_TR_SUSPECT = set("şĞİı")  # present in Turkish orthography, must exist in Azerbaijani too


def char_stats(text: str) -> dict:
    total = len(text) or 1
    az = sum(1 for c in text if c in _AZ_LETTERS)
    azu = sum(1 for c in text if c in _AZ_UPPER)
    latin = az + azu + sum(1 for c in text if c in _LATIN_EXTRA and c not in "öÖ")
    cyr = len(_CYRILLIC.findall(text))
    return {
        "len": len(text),
        "az_ratio": (az + azu) / total,
        "latin_ratio": latin / total,
        "cyrillic_ratio": cyr / total,
        "digit_ratio": sum(1 for c in text if c in _DIGITS) / total,
        "punct_ratio": sum(1 for c in text if c in _PUNCT) / total,
    }


def has_az_specials(text: str) -> bool:
    return any(c in text for c in "əıöüğşçƏİÖÜĞŞÇ")


def normalize_unicode(text: str) -> str:
    """NFKC normalise but keep Azerbaijani letter distinctions intact.

    NFKC folds e.g. full-width forms and compatibility ligatures. It does not
    merge ə/ä or ı/i, so Azerbaijani spelling survives. Casing is preserved.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_QUOTE_TABLE)
    text = _DASH.sub("-", text)
    text = _MISC_APOSTROPHE.sub("'", text)
    return text


def strip_control(text: str) -> str:
    text = _CTRL.sub(" ", text)
    text = _SOFT_HYPHEN.sub("", text)
    text = _SURROGATE.sub("", text)
    return text


def clean_markup(text: str) -> str:
    """Remove MediaWiki/HTML markup that survives a naive dump parse."""
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = _TABLE_RE.sub(" ", text)
    text = strip_nested_templates(text)
    text = _TEMPLATE_RE.sub(" ", text)  # any unbalanced leftovers
    text = _WIKI_FIELD_ROW_RE.sub(" ", text)
    text = _WIKI_LIST_ROW_RE.sub(" ", text)
    text = _WIKILINK_RE.sub(_wikilink_sub, text)
    text = _EXTLINK_RE.sub(lambda m: m.group(1) or " ", text)
    text = _HEADING_RE.sub(lambda m: m.group(1).strip(), text)
    text = _REF_MARK_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _WIKI_EMPHASIS_RE.sub("", text)
    text = _MD_EMPHASIS_RE.sub("", text)
    text = _HEADING_LEFTOVER_RE.sub("", text)
    return text


def strip_urls(text: str, keep_text: bool = True) -> str:
    if keep_text:
        return _URL_RE.sub(" ", _EMAIL_RE.sub(" ", text))
    return _URL_RE.sub("", _EMAIL_RE.sub("", text))


def collapse_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RUN.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _NL_RUN.sub("\n\n", text)
    return text.strip()


def max_line_overrun_ratio(text: str, limit: int = 250) -> float:
    lines = text.split("\n")
    if not lines:
        return 1.0
    return sum(1 for ln in lines if len(ln) > limit) / len(lines)


def looks_like_table(text: str, limit: int = 250, min_lines: int = 5, ratio: float = 0.6) -> bool:
    """Detect leftover tables/lists of links rather than prose.

    The minimum line count matters: a single long paragraph is legitimate
    text, while a document where most lines run past 250 characters is
    almost always residual markup.
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return len(lines) >= min_lines and max_line_overrun_ratio(text, limit) > ratio


def repeated_token_ratio(text: str, n: int = 5) -> float:
    """Fraction of n-grams that are duplicates - detects padding/inflation."""
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def distinct_line_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 3:
        return 1.0
    return len(set(lines)) / len(lines)


BOILERPLATE_MARKERS = (
    "bu səhifə redaktə olunub",
    "bu səhifə son dəfə",
    "məlumat dəyişdirilməyib",
    "kənardan bu mətnin",
    "vikipediya",
    "wikipedia",
    "qısa məqalə",
    "redaktə et",
    "məqalə haqqında",
    "dostluqlar qovluğu",
    "şərh",
    "kateqoriya",
    "xarici keçidlər",
    "bax:",
    "qeyd:",
    "mənbə:",
    "istinad",
    "nümunə göstər",
    "redirect",
    "bölmələr",
    "menyu",
    "yazarlar",
    "tərcümə",
    "proyekt",
    "istifadəçi",
    "fayl:",
    "şablon:",
    "media:",
    "modul:",
    "qovluq:",
    "danışma:",
    "hərəkət:",
    "yardım:",
    "qısaməqalə",
)

BOILERPLATE_LINES_EXACT = {
    "bələdgi xarici keçidlər",
    "bələdgi maddələr",
    "həmçinin bax",
    "qeydlər",
    "istinadlar",
    "mənbələr",
    "xarici keçidlər",
    "həmçinin",
    "qalıq",
    "həmçinin baxın",
    "qısaməqalə",
    "görüntü",
    "şərh",
    "qeyd",
}


def boilerplate_score(text: str) -> float:
    """Fraction of lines that look like wiki navigation/boilerplate."""
    lines = [ln.strip().lower() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 1.0
    bad = 0
    for ln in lines:
        if ln in BOILERPLATE_LINES_EXACT:
            bad += 1
            continue
        if any(ln.startswith(m) for m in BOILERPLATE_MARKERS):
            bad += 1
    return bad / len(lines)


def fix_mojibake(text: str) -> str:
    """Repair text that was UTF-8 bytes decoded as a single-byte code page.

    Several code pages are tried, most specific first. ``cp1254`` matters most
    here: it is the Windows code page for Azerbaijani/Turkish, so legacy pages
    damaged by it carry ``ə`` and ``ş`` as mojibake, and only that codec can
    round-trip them. ``cp1252`` and ``latin-1`` cover Western European damage.
    """
    if not any(marker in text for marker in ("Ã", "â€", "Ä", "Å", "Å", "Å")):
        return text
    for _ in range(2):
        for codec in ("cp1254", "cp1252", "latin-1"):
            try:
                candidate = text.encode(codec, errors="strict").decode("utf-8", errors="strict")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if candidate != text:
                text = candidate
                break
        else:
            break
    return text


@dataclass
class CleanOutcome:
    text: str
    keep: bool
    reason: str = ""
    stats: dict = field(default_factory=dict)


def clean_document(
    raw: str,
    *,
    min_chars: int = 200,
    max_chars: int = 50_000,
    min_az_ratio: float = 0.70,
    min_latin_ratio: float = 0.60,
    max_repeat_ratio: float = 0.35,
    max_boilerplate_ratio: float = 0.30,
    remove_urls: bool = True,
) -> CleanOutcome:
    """Full cleaning pipeline for one document, returning a keep/drop decision."""
    stats_counter: dict = {}
    original_len = len(raw)
    if original_len > max_chars * 6:  # hopeless: markup-dominated
        return CleanOutcome("", False, "too_long", stats_counter)

    text = fix_mojibake(raw)
    text = normalize_unicode(text)
    text = strip_control(text)
    text = _REPLACEMENT.sub("\ufffd", text)
    text = clean_markup(text)
    if remove_urls:
        text = strip_urls(text, keep_text=True)
    text = collapse_whitespace(text)
    text = _LIST_BULLET_RE.sub("", text)

    if not text:
        return CleanOutcome("", False, "empty_after_clean", stats_counter)
    if len(text) < min_chars:
        return CleanOutcome("", False, "too_short", stats_counter)
    if len(text) > max_chars:
        return CleanOutcome("", False, "too_long", stats_counter)

    cs = char_stats(text)
    if cs["latin_ratio"] < min_latin_ratio:
        return CleanOutcome("", False, "not_latin", cs)
    if cs["cyrillic_ratio"] > 0.02:
        return CleanOutcome("", False, "cyrillic_heavy", cs)
    if cs["az_ratio"] < min_az_ratio * 0.55:
        return CleanOutcome("", False, "low_az_letters", cs)

    if boilerplate_score(text) > max_boilerplate_ratio:
        return CleanOutcome("", False, "boilerplate", cs)
    if repeated_token_ratio(text, 5) > max_repeat_ratio:
        return CleanOutcome("", False, "repetitive", cs)
    if distinct_line_ratio(text) < 0.5:
        return CleanOutcome("", False, "duplicate_lines", cs)
    if looks_like_table(text):
        return CleanOutcome("", False, "long_lines", cs)
    if text.count("http") > 2 or text.count("www.") > 2:
        return CleanOutcome("", False, "url_heavy", cs)

    cs["replacement_chars"] = text.count(REPLACEMENT_CHAR_WARN)
    cs["has_az_specials"] = has_az_specials(text)
    return CleanOutcome(text, True, "kept", cs)