"""Orchestrate: download -> clean -> language-filter -> dedupe -> split.

Produces train/validation JSONL with source provenance on every document plus
a manifest of *measured* counts. The pipeline is a generator chain, so memory
stays flat regardless of corpus size: nothing here invents numbers, every
statistic written to disk comes from the actual stream.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from ..config import DataConfig, PathsConfig
from ..utils import atomic_write_json, get_logger
from .clean import CleanStats, clean_document
from .dedupe import Deduplicator
from .langid import detect_language
from .sources import Source, SourceRegistry

log = get_logger("data.prepare")

PROCESSED_TRAIN = "corpus_train.jsonl"
PROCESSED_VAL = "corpus_val.jsonl"
PROCESSED_ALL = "corpus_all.jsonl"
MANIFEST = "dataset_manifest.json"


@dataclass
class PrepareStats:
    documents_in: int = 0
    documents_out: int = 0
    clean: CleanStats = field(default_factory=CleanStats)
    lang_counts: dict = field(default_factory=dict)
    lang_confidence: list[float] = field(default_factory=list)
    per_source: dict = field(default_factory=dict)
    chars_out: int = 0
    seconds: float = 0.0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    source_saw_documents: dict = field(default_factory=dict)
    source_kept_documents: dict = field(default_factory=dict)
    sources_missing_dump: list = field(default_factory=list)
    sources_empty: list = field(default_factory=list)

    def note_lang(self, lang: str) -> None:
        self.lang_counts[lang] = self.lang_counts.get(lang, 0) + 1

    def to_dict(self) -> dict:
        conf = self.lang_confidence or [0.0]
        srt = sorted(conf)
        return {
            "documents_in": self.documents_in,
            "documents_out": self.documents_out,
            "documents_removed": self.documents_in - self.documents_out,
            "retention_rate": round(self.documents_out / max(1, self.documents_in), 5),
            "chars_out": self.chars_out,
            "cleaning": self.clean.summary(),
            "dedupe": {
                "exact_removed": self.exact_duplicates,
                "near_removed": self.near_duplicates,
                "duplicate_rate": round((self.exact_duplicates + self.near_duplicates) / max(1, self.documents_in), 5),
            },
            "language_distribution": dict(sorted(self.lang_counts.items(), key=lambda kv: -kv[1])),
            "az_confidence": {
                "min": round(min(conf), 4),
                "p05": round(srt[int(0.05 * (len(srt) - 1))], 4),
                "median": round(srt[len(srt) // 2], 4),
                "mean": round(sum(conf) / len(conf), 4),
            },
            "per_source": dict(sorted(self.per_source.items(), key=lambda kv: -kv[1])),
            "source_yield": {
                sid: {
                    "seen": self.source_saw_documents.get(sid, 0),
                    "kept": self.source_kept_documents.get(sid, 0),
                }
                for sid in sorted(set(self.source_saw_documents) | set(self.source_kept_documents))
            },
            "sources_missing_dump": sorted(self.sources_missing_dump),
            "sources_empty": sorted(self.sources_empty),
            "seconds": round(self.seconds, 2),
        }


def find_local_dump(paths: PathsConfig, source_id: str) -> Optional[Path]:
    raw = paths.raw / source_id
    if not raw.exists():
        return None
    cands = sorted(
        (p for p in raw.iterdir() if p.suffix in (".bz2", ".gz", ".xml", ".xz") and not p.name.endswith(".part")),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    return cands[0] if cands else None


def iter_source_documents(source: Source, paths: PathsConfig) -> Iterator[tuple[str, str, str]]:
    """Yield ``(source_id, title, raw_text)`` for one registered source.

    Sources without an implemented collector are reported loudly and skipped,
    so a registry entry can never silently contribute nothing to the corpus.
    """
    if source.id in ("az_synthetic_templates",) or source.access == "local":
        # Locally generated corpora arrive through `synthetic_texts`, never
        # through a downloader.
        log.debug("source %s is generated locally and is supplied via synthetic_texts", source.id)
        return
    if not source.id.startswith("azwiki") and not source.id.endswith("wiki"):
        log.warning("source %s has no collector implemented; skipping", source.id)
        return
    dump = find_local_dump(paths, source.id)
    if dump is None:
        log.error("no dump found for %s under %s - run `python -m tagai.cli data fetch` first", source.id, paths.raw)
        return
    from .wikitext import WikiStats, iter_wiki_pages

    stats = WikiStats()
    for page in iter_wiki_pages(dump, stats):
        yield source.id, page.title, page.text
    log.info("dump parse stats for %s: %s", source.id, stats.to_dict())


def iter_clean_documents(
    paths: PathsConfig,
    cfg: DataConfig,
    stats: PrepareStats,
    sources: list[Source],
    limit: Optional[int] = None,
    progress_every: int = 20_000,
) -> Iterator[tuple[str, str, str, str]]:
    """Yield ``(source_id, title, clean_text, kind)`` for everything that survives."""
    deduper = Deduplicator(
        threshold=cfg.near_dedupe_threshold,
        buckets=cfg.near_dedupe_buckets,
        seed=cfg.seed,
    )
    for source in sources:
        if source.access == "local":
            continue  # generated locally, arrives through `synthetic_texts`
        start_in = stats.documents_in
        start_out = stats.documents_out
        for source_id, title, raw in iter_source_documents(source, paths):
            stats.documents_in += 1
            stats.clean.seen += 1
            stats.clean.chars_before += len(raw)

            outcome = clean_document(
                raw,
                min_chars=cfg.min_chars,
                max_chars=cfg.max_chars,
                min_az_ratio=cfg.min_az_ratio,
                min_latin_ratio=cfg.min_latin_ratio,
                max_repeat_ratio=cfg.max_repeat_ratio,
            )
            if not outcome.keep:
                stats.clean.drop(outcome.reason)
                continue

            lang = detect_language(outcome.text)
            stats.note_lang(lang["lang"])
            if lang["lang"] != "az":
                stats.clean.drop("lang_" + lang["lang"])
                continue

            if cfg.dedupe_exact or cfg.near_dedupe:
                dup, kind = deduper.is_duplicate(outcome.text)
                if dup:
                    stats.clean.drop("dup_" + kind)
                    continue

            stats.clean.kept += 1
            stats.clean.chars_after += len(outcome.text)
            stats.documents_out += 1
            stats.chars_out += len(outcome.text)
            stats.lang_confidence.append(lang["confidence"])
            stats.per_source[source_id] = stats.per_source.get(source_id, 0) + 1
            yield source_id, title, outcome.text, "collected"

            if progress_every and stats.documents_in % progress_every == 0:
                log.info("processed %d docs -> kept %d (%.1f%%)", stats.documents_in, stats.documents_out, 100 * stats.documents_out / max(1, stats.documents_in))
            if limit and stats.documents_out >= limit:
                log.warning("document limit %d reached; stopping collection", limit)
                return

        seen = stats.documents_in - start_in
        kept = stats.documents_out - start_out
        stats.source_saw_documents[source.id] = seen
        stats.source_kept_documents[source.id] = kept
        if seen == 0:
            dump = find_local_dump(paths, source.id)
            if dump is None:
                # Not downloaded yet. That is an expected state on a fresh
                # machine with partial data, so it is reported and recorded
                # rather than aborting the run.
                note = (
                    f"source {source.id} has no dump under {paths.raw / source.id}; "
                    f"run `python -m tagai.cli fetch --source {source.id}` to include it"
                )
                stats.sources_missing_dump.append(source.id)
                if cfg.strict_sources:
                    log.warning("%s (continuing)", note)
                else:
                    log.error("%s", note)
            else:
                # The dump is present but produced nothing: unreadable file,
                # unexpected format, or every document filtered out. That is a
                # real failure, not a missing download.
                message = (
                    f"source {source.id} produced no documents from {dump} "
                    f"({dump.stat().st_size} bytes). The dump is unreadable or in an "
                    f"unexpected format - re-fetch it with "
                    f"`python -m tagai.cli fetch --source {source.id}`."
                )
                stats.sources_empty.append(source.id)
                if cfg.strict_sources:
                    raise RuntimeError(message)
                log.error("%s", message)
        log.info("source %s: %d documents seen, %d kept", source.id, seen, kept)

    stats.exact_duplicates = deduper.stats.exact_duplicates
    stats.near_duplicates = deduper.stats.near_duplicates


def val_bucket(text: str, seed: int, modulus: int = 10_000) -> int:
    """Deterministic split bucket based on a content hash.

    Hash-based rather than position-based so the split is stable under corpus
    growth, identical on every machine, and needs no global ordering.
    """
    import hashlib

    digest = hashlib.blake2b(f"{seed}:{text[:512]}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulus


def write_splits(
    paths: PathsConfig,
    cfg: DataConfig,
    rows: Iterable[tuple[str, str, str, str]],
) -> dict:
    """Stream rows into train/validation JSONL, split by content hash."""
    processed = paths.processed
    processed.mkdir(parents=True, exist_ok=True)
    train_path = processed / PROCESSED_TRAIN
    val_path = processed / PROCESSED_VAL
    all_path = processed / PROCESSED_ALL
    for p in (train_path, val_path, all_path):
        p.unlink(missing_ok=True)

    threshold = max(1, int(cfg.validation_fraction * 10_000))
    counts = {"train": 0, "validation": 0, "synthetic": 0}
    chars = {"train": 0, "validation": 0}
    train_fh = open(train_path, "w", encoding="utf-8", newline="\n")
    val_fh = open(val_path, "w", encoding="utf-8", newline="\n")
    try:
        for src, title, text, kind in rows:
            is_val = val_bucket(text, cfg.seed) < threshold
            row: dict = {"text": text, "source": src, "synthetic": kind == "synthetic"}
            if title:
                row["title"] = title
            line = json.dumps(row, ensure_ascii=False) + "\n"
            if is_val:
                val_fh.write(line)
                counts["validation"] += 1
                chars["validation"] += len(text)
            else:
                train_fh.write(line)
                counts["train"] += 1
                chars["train"] += len(text)
            if kind == "synthetic":
                counts["synthetic"] += 1
    finally:
        train_fh.close()
        val_fh.close()

    with open(all_path, "wb") as out:
        for p in (train_path, val_path):
            if p.exists():
                with open(p, "rb") as src:
                    shutil.copyfileobj(src, out, length=1 << 20)

    if counts["validation"] and counts["validation"] < cfg.validation_min_documents:
        log.warning(
            "only %d validation documents were produced; %d are needed for a useful held-out set. "
            "Increase data.validation_fraction or add more source data.",
            counts["validation"],
            cfg.validation_min_documents,
        )
    if counts["train"] == 0:
        raise RuntimeError("no training documents survived the pipeline; nothing can be trained on")

    return {
        "counts": counts,
        "chars": chars,
        "artifacts": {"train": str(train_path), "validation": str(val_path), "all": str(all_path)},
    }


def build_corpus(
    paths: PathsConfig,
    cfg: DataConfig,
    source_ids: Optional[list[str]] = None,
    limit: Optional[int] = None,
    synthetic_texts: Optional[list[str]] = None,
    progress_every: int = 20_000,
) -> dict:
    """Run the full pipeline and write train/validation JSONL plus a manifest."""
    paths.ensure()
    started = time.time()
    stats = PrepareStats()

    ids = source_ids or cfg.sources
    sources = SourceRegistry.select(ids)
    limit = limit or cfg.max_documents

    rows = iter_clean_documents(paths, cfg, stats, sources, limit=limit, progress_every=progress_every)

    synthetic_meta: dict = {"available": 0, "added": 0}
    if synthetic_texts:
        rows, synthetic_state = _with_synthetic(rows, synthetic_texts, stats, cfg)
        log.info("queued synthetic generation: available=%d", len(synthetic_texts))

    split_info = write_splits(paths, cfg, rows)
    if synthetic_texts:
        synthetic_meta = {"available": len(synthetic_texts), "added": synthetic_state["added"], "ratio_cap": cfg.synthetic_ratio}
    stats.seconds = time.time() - started

    manifest = {
        "created_unix": int(started),
        "sources_requested": ids,
        "sources_used": [s.id for s in sources],
        "sources_excluded_pending_review": [
            s.id for s in SourceRegistry.all_sources() if s.review_required and s.id in ids
        ],
        "config": {
            "min_chars": cfg.min_chars,
            "max_chars": cfg.max_chars,
            "min_az_ratio": cfg.min_az_ratio,
            "min_latin_ratio": cfg.min_latin_ratio,
            "near_dedupe": cfg.near_dedupe,
            "near_dedupe_threshold": cfg.near_dedupe_threshold,
            "validation_fraction": cfg.validation_fraction,
            "seed": cfg.seed,
        },
        "counts": split_info["counts"],
        "chars": split_info["chars"],
        "synthetic": synthetic_meta,
        "statistics": stats.to_dict(),
        "artifacts": split_info["artifacts"],
        "licenses": SourceRegistry.eligible_manifest(ids),
    }
    atomic_write_json(paths.processed / MANIFEST, manifest)
    log.info(
        "corpus ready: %d train / %d validation documents, %d chars",
        split_info["counts"]["train"],
        split_info["counts"]["validation"],
        stats.chars_out,
    )
    return manifest


def _with_synthetic(
    rows: Iterator[tuple[str, str, str, str]],
    synthetic: list[str],
    stats: PrepareStats,
    cfg: DataConfig,
) -> tuple[Iterator[tuple[str, str, str, str]], dict]:
    """Append synthetic documents, capped so they cannot dominate the corpus.

    The cap is applied *after* the collected stream is drained, so the real
    collected-document count is known rather than estimated.
    """
    state: dict = {"collected": 0, "added": 0}

    def gen() -> Iterator[tuple[str, str, str, str]]:
        for row in rows:
            state["collected"] = stats.documents_out
            yield row
        state["collected"] = stats.documents_out
        if not synthetic:
            return
        ratio = cfg.synthetic_ratio
        # The ratio caps synthetic share of the *collected* corpus. With no
        # collected documents there is nothing to cap against, so the whole
        # pool is used.
        budget = len(synthetic) if state["collected"] == 0 else int(ratio * state["collected"])
        budget = max(0, min(budget, len(synthetic)))
        for text in synthetic:
            if state["added"] >= budget:
                break
            text = " ".join(text.split())
            if len(text) < cfg.min_chars:
                text = (text + ". ") * 3
            stats.documents_out += 1
            stats.chars_out += len(text)
            stats.per_source["az_synthetic_templates"] = stats.per_source.get("az_synthetic_templates", 0) + 1
            state["added"] += 1
            yield "az_synthetic_templates", "", text, "synthetic"

    iterator = gen()
    return iterator, state


def iter_corpus(paths: PathsConfig, split: str = "train") -> Iterator[dict]:
    from ..utils import iter_jsonl

    name = PROCESSED_TRAIN if split == "train" else PROCESSED_VAL
    p = paths.processed / name
    if not p.exists():
        raise FileNotFoundError(f"corpus not built: {p} missing. Run the data pipeline first.")
    yield from iter_jsonl(p)