"""Create a single uploadable ZIP with everything TagAI needs to run.

    python scripts/make_deploy_bundle.py

Produces ``tagai_deploy.zip`` containing only ``tagai/`` and ``configs/`` - the
minimal set required to train. Tests, notebooks, docs and scripts are left
out, because a Colab upload is the slowest part of the workflow and none of
them are imported at runtime.

Upload that one file to Colab and extract it; you do not need a GitHub account
this way.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INCLUDE_DIRS = ["tagai", "configs"]
INCLUDE_FILES = ["requirements.txt", "README.md"]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "runs"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="tagai_deploy.zip")
    ap.add_argument("--with-docs", action="store_true", help="also include docs/ and scripts/")
    args = ap.parse_args()

    out = ROOT / args.out
    if out.exists():
        out.unlink()

    written = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in INCLUDE_FILES:
            p = ROOT / name
            if p.exists():
                zf.write(p, name)
                written += 1
        for d in INCLUDE_DIRS:
            base = ROOT / d
            if not base.exists():
                print(f"missing {d}/", file=sys.stderr)
                return 1
            for p in sorted(base.rglob("*")):
                if not p.is_file():
                    continue
                if any(part in EXCLUDE_PARTS for part in p.parts):
                    continue
                zf.write(p, p.relative_to(ROOT))
                written += 1
        if args.with_docs:
            for d in ("docs", "scripts"):
                base = ROOT / d
                if not base.exists():
                    continue
                for p in sorted(base.rglob("*")):
                    if p.is_file() and not any(part in EXCLUDE_PARTS for part in p.parts):
                        zf.write(p, p.relative_to(ROOT))
                        written += 1

    size_mb = out.stat().st_size / 1024**2
    print(f"{out.name}: {written} files, {size_mb:.2f} MB")
    print(f"contains: {', '.join(INCLUDE_DIRS)} (+ {', '.join(INCLUDE_FILES)})")
    if args.with_docs:
        print("also contains: docs/, scripts/")
    print("\nColab'da kullanım:")
    print("  unzip -q tagai_deploy.zip -d /content")
    print("  cd /content  &&  sys.path.insert(0, '/content')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())