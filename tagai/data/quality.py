"""Dataset quality measurement.

Everything in the report is computed from the files on disk. If a number
cannot be measured, it is reported as null rather than guessed.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

from ..config import AZERBAIJANI_ALPHABET
from ..utils import atomic_write_json, get_logger, human_bytes, iter_jsonl, read_json
from .clean import char_stats
from .dedupe import paragraph_repetition
from .langid import detect_language
from .prepare import MANIFEST, PROCESSED_TRAIN, PROCESSED_VAL
from .sources import SourceRegistry

log = get_logger("data.quality")

AZ_ALL = set(AZERBAIJANI_ALPHABET)


def _iter_docs(paths: PathsConfig, split: str, limit: Optional[int] = None) -> Iterator[dict]:
    name = PROCESSED_TRAIN if split == "train" else PROCESSED_VAL
    p = paths.processed / name
    if not p.exists():
        return
    for i, row in enumerate(iter_jsonl(p)):
        if limit and i >= limit:
            return
        yield row


def analyse(paths: PathsConfig, limit: Optional[int] = None, deep_sample: int = 2_000) -> dict:
    """Compute a full dataset quality report over the prepared corpus."""
    report: dict = {"generated_by": "tagai.data.quality", "split_stats": {}}
    if not (paths.processed / PROCESSED_TRAIN).exists():
        report["error"] = "no prepared corpus found; run `python -m tagai.cli data prepare` first"
        atomic_write_json(paths.reports / "dataset_quality.json", report)
        return report

    manifest_path = paths.processed / MANIFEST
    if manifest_path.exists():
        report["manifest"] = read_json(manifest_path)

    for split in ("train", "validation"):
        report["split_stats"][split] = _split_stats(paths, split)

    combined = _combined_stats(paths, limit)
    report.update(combined)

    tok_meta = paths.tokenizer_dir / "tokenizer_stats.json"
    if tok_meta.exists():
        report["tokenizer"] = read_json(tok_meta)

    report["licensing"] = {
        "eligible_sources": [s.id for s in SourceRegistry.default_sources()],
        "excluded_pending_review": [s.id for s in SourceRegistry.all_sources() if s.review_required],
        "full_manifest": SourceRegistry.license_manifest(),
    }
    report["limitations"] = _limitations(report)

    atomic_write_json(paths.reports / "dataset_quality.json", report)
    atomic_write_text_report(paths, report)
    log.info("dataset quality report written to %s", paths.reports / "dataset_quality.json")
    return report


def _split_stats(paths: PathsConfig, split: str) -> dict:
    name = PROCESSED_TRAIN if split == "train" else PROCESSED_VAL
    p = paths.processed / name
    if not p.exists():
        return {"exists": False}
    docs = 0
    chars = 0
    synth = 0
    sources: Counter = Counter()
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = __import__("json").loads(line)
            except Exception:
                continue
            text = row.get("text", "")
            docs += 1
            chars += len(text)
            sources[row.get("source", "unknown")] += 1
            if row.get("synthetic"):
                synth += 1
    return {
        "exists": True,
        "path": str(p),
        "file_bytes": p.stat().st_size,
        "documents": docs,
        "characters": chars,
        "synthetic_documents": synth,
        "synthetic_share": round(synth / max(1, docs), 5),
        "source_distribution": dict(sources.most_common()),
        "mean_chars_per_doc": round(chars / max(1, docs), 1),
    }


def _combined_stats(paths: PathsConfig, limit: Optional[int]) -> dict:
    lengths: list[int] = []
    char_counts: Counter = Counter()
    missing_az_chars: Counter = Counter()
    az_conf: list[float] = []
    lang_counts: Counter = Counter()
    para_repeats: list[float] = []
    replacement_docs = 0
    source_chars: Counter = Counter()
    docs = 0
    total_chars = 0
    cyrillic_docs = 0

    for row in _iter_docs(paths, "train", limit):
        text = row.get("text", "")
        if not text:
            continue
        docs += 1
        n = len(text)
        lengths.append(n)
        total_chars += n
        source_chars[row.get("source", "unknown")] += n
        char_counts.update(text)
        if "\ufffd" in text:
            replacement_docs += 1
        cs = char_stats(text)
        if cs["cyrillic_ratio"] > 0.02:
            cyrillic_docs += 1
        for c in AZ_ALL:
            if c not in text:
                missing_az_chars[c] += 1
        if docs <= 2_000:  # language ID is the slowest step: sample it
            res = detect_language(text)
            lang_counts[res["lang"]] += 1
            az_conf.append(res["confidence"])
        if docs <= 5_000:
            para_repeats.append(paragraph_repetition(text)["repeat_ratio"])

    lengths_sorted = sorted(lengths)

    def pct(p: float) -> int:
        if not lengths_sorted:
            return 0
        return lengths_sorted[min(len(lengths_sorted) - 1, int(p * (len(lengths_sorted) - 1)))]

    never_seen = sorted(c for c in AZ_ALL if missing_az_chars.get(c, 0) >= docs)
    total_vocab = sum(char_counts.values()) or 1
    top_chars = char_counts.most_common(40)
    conf_sorted = sorted(az_conf)

    return {
        "documents_analysed": docs,
        "total_characters": total_chars,
        "total_characters_human": human_bytes(total_chars),
        "unique_characters": len(char_counts),
        "document_length": {
            "mean": round(sum(lengths) / max(1, len(lengths)), 1),
            "p05": pct(0.05),
            "p25": pct(0.25),
            "median": pct(0.50),
            "p75": pct(0.75),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": lengths_sorted[-1] if lengths_sorted else 0,
        },
        "character_distribution": {
            "az_alphabet_size": len(AZ_ALL),
            # Coverage = how much of the alphabet appears at least once.
            "az_letter_coverage": round(1 - len(never_seen) / max(1, len(AZ_ALL)), 4),
            "az_letters_never_seen": never_seen,
            "az_letter_document_frequency": {
                c: docs - missing_az_chars.get(c, 0) for c in sorted(AZ_ALL)
            },
            "az_letter_document_frequency_pct": {
                c: round(100 * (docs - missing_az_chars.get(c, 0)) / max(1, docs), 2) for c in sorted(AZ_ALL)
            },
            "replacement_char_documents": replacement_docs,
            "replacement_char_rate": round(replacement_docs / max(1, docs), 5),
            "cyrillic_heavy_documents": cyrillic_docs,
            "top_characters": [{"char": c, "count": n, "share": round(n / total_vocab, 6)} for c, n in top_chars],
        },
        "language_distribution_sample": dict(lang_counts.most_common()),
        "az_confidence_sample": {
            "n": len(conf_sorted),
            "min": round(min(conf_sorted), 4) if conf_sorted else None,
            "median": round(conf_sorted[len(conf_sorted) // 2], 4) if conf_sorted else None,
            "mean": round(sum(conf_sorted) / len(conf_sorted), 4) if conf_sorted else None,
        },
        "paragraph_repetition_sample": {
            "mean": round(sum(para_repeats) / max(1, len(para_repeats)), 4),
            "docs_with_repeats": sum(1 for r in para_repeats if r > 0.2),
            "sampled": len(para_repeats),
        },
        "chars_by_source": dict(source_chars.most_common()),
    }


def _limitations(report: dict) -> list[str]:
    out: list[str] = []
    analysed = report.get("documents_analysed", 0)
    if analysed == 0:
        out.append("No documents available to analyse.")
    elif report.get("manifest", {}).get("counts", {}).get("train", 0) > analysed:
        out.append(
            f"Statistics were computed on a sample of {analysed} documents; the corpus is larger."
        )
    if not report.get("tokenizer"):
        out.append("Tokenizer statistics unavailable: the tokenizer has not been trained yet.")
    excluded = report.get("licensing", {}).get("excluded_pending_review", [])
    if excluded:
        out.append(
            "Sources excluded pending licence review: " + ", ".join(excluded) + ". "
            "The corpus is therefore narrower than the full Azerbaijani web."
        )
    lang_dist = report.get("language_distribution_sample", {})
    az_share = lang_dist.get("az", 0) / max(1, sum(lang_dist.values()))
    if lang_dist and az_share < 0.95:
        out.append(f"Only {az_share:.1%} of the sampled documents were identified as Azerbaijani.")
    return out


def atomic_write_text_report(paths: PathsConfig, report: dict) -> Path:
    """Human-readable markdown companion to the JSON report."""
    lines = ["# TagAI Dataset Quality Report", ""]
    lines.append("All numbers below are measured from the files on disk.")
    lines.append("")
    stats = report.get("split_stats", {})
    lines.append("## Splits")
    lines.append("")
    lines.append("| split | documents | characters | synthetic | synthetic share |")
    lines.append("|---|---|---|---|---|")
    for split in ("train", "validation"):
        s = stats.get(split, {})
        if not s.get("exists"):
            lines.append(f"| {split} | - | - | - | - |")
            continue
        lines.append(
            f"| {split} | {s['documents']} | {s['characters']} | {s.get('synthetic_documents', 0)} | {s.get('synthetic_share', 0)} |"
        )
    lines.append("")

    if "total_characters" in report:
        lines.append("## Character and length distribution")
        lines.append("")
        dl = report["document_length"]
        lines.append(
            f"- Documents analysed: {report['documents_analysed']}\n"
            f"- Total characters: {report['total_characters']} ({report.get('total_characters_human', '')})\n"
            f"- Unique characters: {report['unique_characters']}\n"
            f"- Doc length mean/median/p95: {dl['mean']} / {dl['median']} / {dl['p95']}\n"
        )
        cd = report["character_distribution"]
        freq = cd.get("az_letter_document_frequency_pct", {})
        rarest = sorted(freq.items(), key=lambda kv: kv[1])[:5] if freq else []
        lines.append(
            f"- Azerbaijani alphabet letters present in the corpus: "
            f"{cd['az_alphabet_size'] - len(cd['az_letters_never_seen'])}/{cd['az_alphabet_size']} "
            f"(coverage {cd['az_letter_coverage']})\n"
            f"- Letters never seen: {', '.join(cd['az_letters_never_seen']) or '(none)'}\n"
            + (f"- Rarest letters (share of documents): {', '.join(f'{c}={v}%' for c, v in rarest)}\n" if rarest else "")
            + f"- Documents containing U+FFFD: {cd['replacement_char_documents']}\n"
        )
        lines.append("")

    if "tokenizer" in report:
        tok = report["tokenizer"]
        lines.append("## Tokenizer")
        lines.append("")
        lines.append(f"- Vocabulary size: {tok.get('vocab_size')}\n- Documents used: {tok.get('training_docs')}\n")
        rt = tok.get("roundtrip", {})
        if rt:
            lines.append(f"- Unicode round-trip rate: {rt.get('roundtrip_rate')} ({rt.get('tested')} probes)\n")
        lines.append("")

    lines.append("## Licensing")
    lines.append("")
    lic = report.get("licensing", {})
    lines.append(f"- Eligible sources: {', '.join(lic.get('eligible_sources', []))}\n")
    lines.append(f"- Excluded pending licence review: {', '.join(lic.get('excluded_pending_review', [])) or '(none)'}\n")
    lines.append("")

    lim = report.get("limitations", [])
    if lim:
        lines.append("## Known limitations")
        lines.append("")
        for item in lim:
            lines.append(f"- {item}")
        lines.append("")

    path = paths.reports / "dataset_quality.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path