#!/usr/bin/env python
"""Reproduce the whole verified TagAI run from a clean checkout.

    python scripts/run_smoke_pipeline.py --root runs/demo

Fetches a real sample of Azerbaijani Wikipedia, builds the corpus, trains the
tokenizer, packs, trains from random weights, generates, evaluates and exports.
No pretrained weights are downloaded at any point.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tagai.cli import main as cli_main  # noqa: E402
from tagai.utils import configure_stdio  # noqa: E402

CONFIG = str(ROOT / "configs" / "tagai_smoke.json")


def stage(name: str, argv: list[str], started: list[float]) -> None:
    print(f"\n{'=' * 70}\n== {name}\n{'=' * 70}", flush=True)
    t0 = time.time()
    rc = cli_main(["--config", CONFIG, *argv])
    started.append((name, time.time() - t0))
    if rc != 0:
        raise SystemExit(f"stage {name!r} failed with exit code {rc}")


def main() -> int:
    configure_stdio()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/demo")
    ap.add_argument("--streams", type=int, default=25, help="bz2 streams to fetch (100 articles each)")
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    root = ["--root", args.root]
    timings: list[tuple[str, float]] = []

    if not args.skip_download:
        stage("fetch sample", root + ["sample", "--streams", str(args.streams)], timings)

    stage("clean", root + ["clean", "--min-chars", "300", "--synthetic-ratio", "0.03", "--synthetic-n", "200"], timings)
    stage("quality report", root + ["report"], timings)
    stage("train tokenizer", root + ["tokenize", "--vocab-size", str(args.vocab_size)], timings)
    stage("pack tokens", root + ["pack"], timings)
    stage("pretrain", root + ["train", "--max-steps", str(args.max_steps)], timings)
    stage("generate", root + ["generate", "--max-new-tokens", "48"], timings)
    stage("evaluate", root + ["evaluate", "--max-new-tokens", "32", "--ppl-batches", "10"], timings)
    stage("export", root + ["export"], timings)

    print(f"\n{'=' * 70}\n== timing\n{'=' * 70}")
    for name, secs in timings:
        print(f"  {name:<18s} {secs:8.1f}s")
    print(f"  {'TOTAL':<18s} {sum(s for _, s in timings):8.1f}s")
    print(f"\nArtefacts under {args.root}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())