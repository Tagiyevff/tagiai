"""Resumable HTTP downloads with bounded exponential backoff and integrity checks."""

from __future__ import annotations

import bz2
import gzip
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..utils import atomic_write_json, get_logger, retry_call

log = get_logger("data.download")

USER_AGENT = "TagAI/0.1 (Azerbaijani LM research; contact: project owner)"
MEDIAWIKI_NS = "http://www.mediawiki.org/xml/export-0.11/"
_RETRYABLE = (urllib.error.URLError, TimeoutError, ConnectionError, OSError)


@dataclass
class DownloadResult:
    path: Path
    bytes: int
    from_cache: bool
    url: str
    status: str = "ok"
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "path": str(self.path),
            "bytes": self.bytes,
            "from_cache": self.from_cache,
            "status": self.status,
            "error": self.error,
        }


def download_file(
    url: str,
    dest: str | os.PathLike[str],
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    resume: bool = True,
    max_attempts: int = 6,
    timeout: int = 120,
    chunk: int = 1 << 20,
) -> DownloadResult:
    """Download ``url`` to ``dest``, skipping the transfer when already complete.

    A partially downloaded file is resumed with a Range request. A finished
    file with the wrong size is re-fetched from scratch rather than trusted.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists():
        size = dest.stat().st_size
        if expected_bytes is None or size == expected_bytes:
            if expected_sha256 is None:
                log.info("cache hit: %s (%d bytes)", dest.name, size)
                return DownloadResult(dest, size, True, url)
        else:
            log.warning("cached %s has wrong size (%d != %d); re-downloading", dest.name, size, expected_bytes)
            dest.unlink(missing_ok=True)

    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    start = 0
    if resume and part.exists():
        start = part.stat().st_size
        log.info("resuming %s from byte %d", dest.name, start)
        headers["Range"] = f"bytes={start}-"

    def _attempt() -> Path:
        nonlocal start
        with urllib.request.urlopen(urllib.request.Request(url, headers=dict(headers)), timeout=timeout) as resp:
            mode = "ab" if start and resp.status == 206 else "wb"
            if mode == "wb":
                # The server ignored our Range header; restart from zero.
                start = 0
                headers.pop("Range", None)
            with open(part, mode) as fh:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    fh.write(block)
        os.replace(part, dest)
        return dest

    try:
        retry_call(_attempt, attempts=max_attempts, base_delay=3.0, max_delay=90.0, exceptions=_RETRYABLE, label=f"download {url}")
    except RuntimeError as exc:
        log.error("download failed permanently: %s", exc)
        return DownloadResult(dest, part.stat().st_size if part.exists() else 0, False, url, status="failed", error=str(exc))

    size = dest.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        log.error("size mismatch for %s: got %d want %d", url, size, expected_bytes)
        return DownloadResult(dest, size, False, url, status="size_mismatch", error=f"expected {expected_bytes} got {size}")
    return DownloadResult(dest, size, False, url)


def download_text(url: str, *, timeout: int = 60) -> str:
    def _get() -> str:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")

    return retry_call(_get, attempts=4, base_delay=2.0, label=f"fetch {url}")


def list_wiki_dump_files(dump_dir_url: str, project: str | None = None) -> list[dict]:
    """Parse a Wikimedia dump directory listing into downloadable files.

    ``dump_dir_url`` is the directory that holds the dump files, e.g.
    ``https://dumps.wikimedia.org/azwiki/latest/``. Only the XML article dump,
    the multistream index and the article-title list are useful for TagAI;
    history and discussion dumps are skipped to save bandwidth.
    """
    html = download_text(dump_dir_url)
    wanted = (
        "pages-articles-multistream.xml.bz2",
        "pages-articles.xml.bz2",
        "pages-articles-multistream.xml.gz",
        "pages-articles.xml.gz",
        "pages-articles-multistream-index.txt.bz2",
    )
    files: list[dict] = []
    for name in wanted:
        m = re.search(rf'href="([^"]*{re.escape(name)})"', html)
        if m:
            files.append({"name": name, "url": m.group(1), "bytes": _size_near(html, m.group(1))})
    if not files:
        raise RuntimeError(f"no article dump found at {dump_dir_url}")
    return files


def _size_near(html: str, href: str) -> int | None:
    idx = html.find(href)
    if idx < 0:
        return None
    window = html[max(0, idx - 400) : idx + 400]
    sizes = re.findall(r"(\d+(?:\.\d+)?)\s*(B|KB|MB|GB)", window)
    if not sizes:
        return None
    val, unit = sizes[-1]
    mult = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}[unit]
    return int(float(val) * mult)


def read_multistream_index(index_path: str | os.PathLike[str]) -> list[int]:
    """Read the byte offsets of every bz2 stream in a multistream dump.

    The ``pages-articles-multistream-index.txt.bz2`` file lists one line per
    article as ``<stream_offset>:<page_id>:<sequence>``. Many articles share an
    offset because a single bz2 stream holds a batch of pages (100 in the
    azwiki dump), so the unique offsets are the stream boundaries.

    Each boundary is independently decompressible, which is what lets us pull
    a small real sample out of a ~1 GB dump for pipeline validation.
    """
    index_path = Path(index_path)
    opener = bz2.open if index_path.suffix == ".bz2" else (gzip.open if index_path.suffix == ".gz" else open)
    offsets: set[int] = set()
    with opener(index_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line or line.startswith("offset:"):
                continue
            head = line.split(":", 1)[0]
            try:
                offsets.add(int(head))
            except ValueError:
                continue
    return sorted(offsets)


def extract_wiki_streams(
    dump_url: str,
    offsets: Sequence[int],
    *,
    max_streams: int = 10,
    timeout: int = 240,
    progress: bool = True,
) -> str:
    """Download whole bz2 streams and return them as one XML fragment.

    Every stream is a self-contained bz2 member containing a complete set of
    ``<page>`` elements, so streams can be fetched by byte range and simply
    concatenated into a valid ``<mediawiki>`` document.
    """
    usable = list(offsets)[:max_streams]
    if not usable:
        raise ValueError("no stream offsets supplied")
    last = usable[-1]

    parts: list[str] = [f'<mediawiki xmlns="{MEDIAWIKI_NS}">']
    pages = 0
    for i, offset in enumerate(usable):
        end = (usable[i + 1] - 1) if i + 1 < len(usable) else (last + 4 * 1024 * 1024)
        try:
            blob = download_range(dump_url, offset, end, timeout=timeout)
            raw = bz2.decompress(blob).decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - a bad range must not kill the sample
            log.warning("stream at offset %d could not be read (%s); skipping", offset, exc)
            continue
        start = raw.find("<page>")
        end_tag = raw.rfind("</page>")
        if start < 0 or end_tag < 0:
            continue
        fragment = raw[start : end_tag + len("</page>")]
        parts.append(fragment)
        pages += fragment.count("<page>")
        if progress:
            log.info("stream %d/%d at offset %d: %d pages (%d bytes compressed)", i + 1, len(usable), offset, fragment.count("<page>"), len(blob))
    parts.append("</mediawiki>")
    log.info("sampled %d pages from %d streams", pages, len(usable))
    return "\n".join(parts)


def download_range(url: str, start: int, end: int, timeout: int = 120) -> bytes:
    """Fetch a byte range with HTTP Range; used for multistream extraction."""
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity", "Range": f"bytes={start}-{end}"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:
        return resp.read()


def _decompress_page(blob: bytes) -> str:
    """Decompress one bz2 stream and return just its ``<page>`` element."""
    try:
        raw = bz2.decompress(blob).decode("utf-8", errors="replace")
    except (OSError, ValueError) as exc:
        log.warning("could not decompress a dump stream: %s", exc)
        return ""
    start = raw.find("<page>")
    end = raw.find("</page>")
    if start < 0 or end < 0:
        return ""
    return raw[start : end + len("</page>")]


def free_disk_gb(path: str | os.PathLike[str] = ".") -> float:
    usage = shutil_usage(path)
    return usage


def shutil_usage(path: str | os.PathLike[str] = ".") -> float:
    import shutil

    return shutil.disk_usage(str(path)).free / 1024**3


def ensure_space(path: str | os.PathLike[str], needed_bytes: int, factor: float = 1.3) -> tuple[bool, str]:
    """Check free space with headroom before an expensive operation."""
    free = shutil_usage(path) * 1024**3
    if free < needed_bytes * factor:
        return False, f"insufficient space at {path}: {free/1024**3:.2f} GB free, need ~{needed_bytes*factor/1024**3:.2f} GB"
    return True, f"{free/1024**3:.2f} GB free"


def save_manifest(path: str | os.PathLike[str], entries: list[dict]) -> Path:
    return atomic_write_json(path, {"files": entries})