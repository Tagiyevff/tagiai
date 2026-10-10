"""Streaming MediaWiki XML dump reader.

The stdlib ``xml.etree.ElementTree.iterparse`` is used deliberately: it works
in every notebook image without a compiled dependency, and dump pages are
processed one at a time so a 100M-tag file needs only a few hundred MB of RAM.

For very large dumps prefer the ``*.multistream.xml.bz2`` file plus ``bz2``,
because the multistream layout lets many independent readers work in parallel
and each stream index tells us the byte range for a given article.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
from xml.etree import ElementTree as ET

from ..utils import get_logger

log = get_logger("data.wikitext")

MEDIAWIKI_NS = "{http://www.mediawiki.org/xml/export-0.11/}"
# Azerbaijani Wikipedia uses Azerbaijani namespace names ("Fayl:", "Şablon:"),
# so filtering only the English spellings silently lets every File/Categoría
# page into the corpus.
SKIP_TITLES_RE = re.compile(
    r"^(Special:|Talk:|User:|User talk:|Wikipedia:|Wikipedia talk:|File:|File talk:|"
    r"Template:|Template talk:|Help:|Help talk:|Category:|Category talk:|MediaWiki:|"
    r"MediaWiki talk:|Module:|Draft:|Portal:|"
    # Azerbaijani namespace names
    r"İnterfeys|İnterfeys dəyişikliyi|MediaVİKİ|MediaViKi|Məhsul dəyişikliyi|"
    r"Şablon|Şablon dəyişikliyi|Hərəkət|Vikipediya|Vikipediya dəyişikliyi|"
    r"Dostluq|Dostluq dəyişikliyi|Xəbər|Məqalə|Modul|Portal|"
    r"Danışma|Qovluq|Qeyd|Viki|Fayl|Şəkəl|Media|Kateqoriya|Kateqoriya dəyişikliyi|"
    r"Yardım|Help|Module|Draft|Redaktə|Mətn|İstifadəçi|İstifadəçi dəyişikliyi|"
    r"Xəbər dəyişikliyi|Danışma dəyişikliyi|Qovluq dəyişikliyi|Hərəkət dəyişikliyi)",
    re.IGNORECASE,
)
SKIP_SUFFIXES = (
    "/dbpedia",
    "/Portal",
    "/Şablon",
    "/Vikipediya",
    "/Dostluq",
    "/Xəbər",
    "/Danışma",
    "/Qovluq",
    "/Hərəkət",
    "/Kateqoriya",
    "/Media",
    "/Modul",
    "/İnterfeys",
)


def open_dump(path: str | Path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".bz2":
        return bz2.open(path, "rb")
    if suffix == ".gz":
        return gzip.open(path, "rb")
    if suffix in (".xz", ".lzma"):
        return lzma.open(path, "rb")
    return open(path, "rb")


@dataclass
class WikiPage:
    title: str
    page_id: int
    namespace_id: int
    text: str
    is_redirect: bool = False

    @property
    def is_content(self) -> bool:
        return self.namespace_id == 0


@dataclass
class WikiStats:
    pages_seen: int = 0
    articles_seen: int = 0
    articles_kept: int = 0
    redirects_skipped: int = 0
    namespace_skipped: int = 0
    title_skipped: int = 0
    empty_skipped: int = 0
    chars: int = 0
    reasons: dict = field(default_factory=dict)

    def note(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def to_dict(self) -> dict:
        return {
            "pages_seen": self.pages_seen,
            "articles_seen": self.articles_seen,
            "articles_kept": self.articles_kept,
            "redirects_skipped": self.redirects_skipped,
            "namespace_skipped": self.namespace_skipped,
            "title_skipped": self.title_skipped,
            "empty_skipped": self.empty_skipped,
            "article_chars": self.chars,
            "skip_reasons": self.reasons,
        }


REDIRECT_RE = re.compile(r"^\s*#\s*(yönləndirmə|redirect|redireksiy[əa])\b", re.IGNORECASE)


def iter_wiki_pages(path: str | Path, stats: Optional[WikiStats] = None) -> Iterator[WikiPage]:
    """Yield every ``<page>`` element's title/text from a MediaWiki XML dump.

    Namespace-agnostic: official dumps declare the export namespace, but
    extracts and API-derived fragments often do not, and silently yielding
    nothing would be indistinguishable from an empty corpus.
    """
    stats = stats or WikiStats()
    context = ET.iterparse(open_dump(path), events=("end",))
    for event, elem in context:
        if _local(elem.tag) != "page":
            continue
        stats.pages_seen += 1
        title = _find_text(elem, "title")
        ns = _find_text(elem, "ns") or "0"
        try:
            ns_id = int(ns)
        except ValueError:
            ns_id = 0
        page_id_raw = _find_text(elem, "id") or "0"
        try:
            page_id = int(page_id_raw)
        except ValueError:
            page_id = 0
        text = _find_revision_text(elem)

        if ns_id != 0:
            stats.namespace_skipped += 1
            stats.note(f"ns_{ns_id}")
        elif _should_skip_title(title):
            stats.title_skipped += 1
        elif not text or not text.strip():
            stats.empty_skipped += 1
        elif REDIRECT_RE.match(text):
            stats.redirects_skipped += 1
        else:
            stats.articles_seen += 1
            stats.articles_kept += 1
            stats.chars += len(text)
            yield WikiPage(title=title, page_id=page_id, namespace_id=ns_id, text=text)

        elem.clear()


def _should_skip_title(title: str) -> bool:
    if not title:
        return True
    if SKIP_TITLES_RE.match(title):
        return True
    return any(title.endswith(s) for s in SKIP_SUFFIXES)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _find_text(elem: ET.Element, name: str) -> Optional[str]:
    for child in elem:
        if _local(child.tag) == name:
            return child.text
    return None


def _find_revision_text(elem: ET.Element) -> str:
    for child in elem:
        if _local(child.tag) == "revision":
            for sub in child:
                if _local(sub.tag) == "text":
                    return sub.text or ""
    return ""


def parse_xml_stream_articles(path: str | Path, limit: Optional[int] = None) -> Iterator[WikiPage]:
    """Convenience wrapper: only main-namespace, non-redirect articles."""
    yield from iter_wiki_pages(path)


def count_articles(path: str | Path) -> int:
    stats = WikiStats()
    for _ in iter_wiki_pages(path, stats):
        pass
    return stats.articles_kept