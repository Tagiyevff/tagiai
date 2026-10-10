"""Repository integrity: what git tracks must match what exists on disk.

A `.gitignore` pattern that is too broad is invisible locally - the code works
because it is on disk - and only fails after a clone. The first Colab run of
TagAI died with `ModuleNotFoundError: No module named 'tagai.data'` for exactly
this reason, so the invariant is now a test.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_PACKAGE_MODULES = [
    "tagai/__init__.py",
    "tagai/__main__.py",
    "tagai/cli.py",
    "tagai/config.py",
    "tagai/export.py",
    "tagai/utils.py",
    "tagai/model/__init__.py",
    "tagai/model/transformer.py",
    "tagai/tokenizer/__init__.py",
    "tagai/tokenizer/trainer.py",
    "tagai/tokenizer/wrapper.py",
    "tagai/data/__init__.py",
    "tagai/data/clean.py",
    "tagai/data/dedupe.py",
    "tagai/data/download.py",
    "tagai/data/langid.py",
    "tagai/data/pack.py",
    "tagai/data/prepare.py",
    "tagai/data/quality.py",
    "tagai/data/sources.py",
    "tagai/data/synthetic.py",
    "tagai/data/wikitext.py",
    "tagai/training/__init__.py",
    "tagai/training/checkpoint.py",
    "tagai/training/optimizer.py",
    "tagai/training/trainer.py",
    "tagai/evaluation/__init__.py",
    "tagai/evaluation/metrics.py",
    "tagai/evaluation/perplexity.py",
    "tagai/evaluation/suite.py",
    "tagai/inference/__init__.py",
    "tagai/inference/generate.py",
]

REQUIRED_ROOT_FILES = [
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    ".gitignore",
    "configs/tagai_smoke.json",
    "configs/tagai_pilot.json",
    "configs/tagai_main.json",
    "notebooks/tagai_from_scratch_colab.ipynb",
    "notebooks/tagai_from_scratch_kaggle.ipynb",
    "scripts/build_notebooks.py",
    "scripts/build_training_report.py",
    "scripts/make_deploy_bundle.py",
    "scripts/run_smoke_pipeline.py",
    "docs/architecture.md",
    "docs/dataset_provenance.md",
    "docs/licensing.md",
    "docs/not_covered.md",
    "docs/training_report.md",
]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def tracked_files() -> set[str]:
    return set(git("ls-files").splitlines())


def has_git_repo() -> bool:
    return (ROOT / ".git").exists()


requires_repo = pytest.mark.skipif(not has_git_repo(), reason="not a git repository")


@pytest.mark.parametrize("rel", REQUIRED_PACKAGE_MODULES)
def test_package_module_exists_on_disk(rel: str) -> None:
    assert (ROOT / rel).is_file(), f"missing module: {rel}"


@pytest.mark.parametrize("rel", REQUIRED_ROOT_FILES)
def test_required_file_exists_on_disk(rel: str) -> None:
    assert (ROOT / rel).is_file(), f"missing file: {rel}"


@pytest.mark.parametrize("rel", REQUIRED_PACKAGE_MODULES + REQUIRED_ROOT_FILES)
@requires_repo
def test_required_file_is_tracked_by_git(rel: str) -> None:
    """Every module a clone needs must be committed.

    A clone that is missing a submodule fails at import time on Colab, long
    after the local run that produced the files succeeded.
    """
    assert rel in tracked_files(), f"{rel} exists on disk but is not committed - check .gitignore"


@requires_repo
def test_no_source_file_is_silently_ignored() -> None:
    """Nothing under tagai/, tests/, configs/, scripts/, docs/ may be ignored."""
    result = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    ignored = [
        line for line in result.stdout.splitlines()
        if line.startswith(("tagai/", "tests/", "configs/", "scripts/", "docs/", "notebooks/"))
        and "__pycache__" not in line
    ]
    assert not ignored, "source files excluded by .gitignore:\n  " + "\n  ".join(ignored)


@requires_repo
def test_tracked_files_match_disk() -> None:
    """No uncommitted source file, and no tracked file missing from disk."""
    status = git("status", "--porcelain")
    uncommitted = [line for line in status.splitlines() if line.strip()]
    assert not uncommitted, "uncommitted changes:\n  " + "\n  ".join(uncommitted)


@requires_repo
def test_no_run_artefacts_are_tracked() -> None:
    """Checkpoints, token shards and bundles must never enter the repository."""
    tracked = tracked_files()
    bad = [
        f for f in tracked
        if f.startswith("runs/") or f.endswith((".pt", ".safetensors", ".bin", ".ckpt", ".zip"))
        or f.endswith(".jsonl")
    ]
    assert not bad, "artefacts are tracked:\n  " + "\n  ".join(bad)


def test_gitignore_does_not_exclude_source_directories() -> None:
    """An unanchored directory pattern matches at any depth.

    A bare `data/` matched `tagai/data/` and silently removed the whole
    subpackage from the repository. Any pattern naming a directory this project
    uses for source code must therefore be anchored with a leading slash.
    """
    source_dirs = {"data", "tagai", "tests", "configs", "scripts", "docs", "notebooks", "tokenizer"}
    lines = [
        ln.strip()
        for ln in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    offenders = [
        ln for ln in lines
        if not ln.startswith("/") and not ln.startswith("*") and not ln.startswith("!")
        and ln.rstrip("/").strip("*") in source_dirs
    ]
    assert not offenders, (
        "unanchored .gitignore patterns would match source directories: " + ", ".join(offenders)
    )


def test_every_package_subpackage_is_importable() -> None:
    """Guards the same class of bug without git: no empty subpackage."""
    import importlib

    for mod in (
        "tagai", "tagai.cli", "tagai.config", "tagai.export", "tagai.utils",
        "tagai.model", "tagai.tokenizer", "tagai.data", "tagai.training",
        "tagai.evaluation", "tagai.inference",
        "tagai.data.clean", "tagai.data.dedupe", "tagai.data.download",
        "tagai.data.langid", "tagai.data.pack", "tagai.data.prepare",
        "tagai.data.quality", "tagai.data.sources", "tagai.data.synthetic",
        "tagai.data.wikitext",
    ):
        importlib.import_module(mod)


@requires_repo
def test_remote_points_at_the_expected_repository() -> None:
    remotes = git("remote", "-v")
    if not remotes:
        pytest.skip("no remote configured")
    assert "github.com/Tagiyevff/tagiai" in remotes, f"unexpected remote:\n{remotes}"


@requires_repo
def test_notebooks_reference_the_real_repository() -> None:
    """The clone URL in the notebooks must match the actual remote."""
    import json

    for name in ("colab", "kaggle"):
        path = ROOT / "notebooks" / f"tagai_from_scratch_{name}.ipynb"
        nb = json.loads(path.read_text(encoding="utf-8"))
        text = "\n".join("".join(c["source"]) for c in nb["cells"])
        assert 'REPO = "tagiai"' in text, f"{path.name} does not point at the tagiai repository"