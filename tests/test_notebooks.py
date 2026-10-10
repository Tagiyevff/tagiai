"""Notebook validation: the generated .ipynb files must be executable.

These notebooks are a deliverable, and a notebook with a syntax error in cell
7 is worse than no notebook at all. Magics (``!pip``, ``%%bash``) and Colab
comments (``#@title``) are stripped before compiling.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
COLAB = NOTEBOOK_DIR / "tagai_from_scratch_colab.ipynb"
KAGGLE = NOTEBOOK_DIR / "tagai_from_scratch_kaggle.ipynb"

SHELL_MAGIC = re.compile(r"^\s*(!|%%)")


def cell_source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def python_source(cell: dict) -> str:
    """Drop IPython magics so the cell can be compiled as plain Python."""
    lines = []
    for line in cell_source(cell).splitlines():
        if SHELL_MAGIC.match(line):
            continue
        if line.strip().startswith("#@"):
            continue
        lines.append(line)
    return "\n".join(lines)


def load(path: Path) -> dict:
    assert path.exists(), f"missing notebook: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", [COLAB, KAGGLE], ids=lambda p: p.name)
def test_notebook_is_valid_nbformat(path: Path) -> None:
    nb = load(path)
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list) and nb["cells"]
    for cell in nb["cells"]:
        assert cell["cell_type"] in {"code", "markdown"}
        assert "source" in cell
        if cell["cell_type"] == "code":
            assert cell["source"], "a code cell must not be empty"


@pytest.mark.parametrize("path", [COLAB, KAGGLE], ids=lambda p: p.name)
def test_every_code_cell_compiles(path: Path) -> None:
    nb = load(path)
    failures = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = python_source(cell)
        try:
            ast.parse(src)
        except SyntaxError as exc:
            failures.append(f"cell {i} ({(cell['metadata'].get('tags') or ['untagged'])[0]}): {exc}")
    assert not failures, "syntax errors:\n" + "\n".join(failures)


@pytest.mark.parametrize("path", [COLAB, KAGGLE], ids=lambda p: p.name)
def test_cells_are_ordered_with_markdown_headers(path: Path) -> None:
    """A reader must be able to follow the notebook top to bottom."""
    nb = load(path)
    kinds = [c["cell_type"] for c in nb["cells"]]
    assert kinds[0] == "markdown", "the notebook must open with an explanation"
    # Every code cell should be preceded by a markdown cell explaining it.
    for i, kind in enumerate(kinds):
        if kind == "code" and i > 0:
            assert kinds[i - 1] == "markdown" or any(
                c["cell_type"] == "markdown" for c in nb["cells"][max(0, i - 2) : i]
            ), f"code cell {i} has no preceding markdown"


def test_colab_deletes_a_stale_clone_before_cloning() -> None:
    """`git clone` will not write into an existing directory.

    An incomplete clone left over from a previous run therefore survives a
    re-run and fails again at import time. The cell must remove it first.
    """
    nb = load(COLAB)
    text = "\n".join(cell_source(c) for c in nb["cells"])
    assert "shutil.rmtree" in text, "cell 3 must delete a stale clone directory"
    assert 'REPO = "tagiai"' in text, "cell 3 must clone the real repository"


def test_colab_verifies_the_clone_is_complete() -> None:
    """A missing subpackage must fail at cell 3 with a clear file list."""
    nb = load(COLAB)
    text = "\n".join(cell_source(c) for c in nb["cells"])
    assert "required = [" in text and "missing" in text
    assert "tagai\" / \"data\"" in text.replace("'", '"') or '"data"' in text
    assert "import tagai.data" in text, "cell 3 must import the data subpackage to prove it exists"


def test_colab_covers_every_required_stage() -> None:
    """The specification lists 20 required Colab cells; each must be present."""
    nb = load(COLAB)
    text = "\n".join(cell_source(c) for c in nb["cells"]).lower()
    required = {
        "gpu detection": "nvidia-smi",
        "dependency install": "pip -q install",
        "persistent storage": "drive.mount",
        "dataset download": "extract_wiki_streams",
        "cleaning and filtering": "build_corpus",
        "deduplication": "deduplicator",
        "tokenizer training": "train_tokenizer",
        "model creation": "tagaimodel(",
        "parameter verification": "num_parameters",
        "smoke test": "smoke test",
        "training configuration": "micro_batch_size = pick_batch",
        "pretraining": "trainer.train()",
        "checkpoint listing": "list_checkpoints",
        "text generation": "tagaigenerator",
        "evaluation": "evalsuite",
        "export": "export_bundle",
        "training report": "training_report",
        "packaging": "zipfile.zipfile",
    }
    missing = [name for name, needle in required.items() if needle.lower() not in text]
    assert not missing, f"Colab notebook is missing stages: {missing}"


def test_colab_states_no_pretrained_weights() -> None:
    nb = load(COLAB)
    text = "\n".join(cell_source(c) for c in nb["cells"])
    assert "pretrained_weights_used" in text or "pretrained_weights_used\": false" in text
    assert "heç bir əvvəlcədən təlim edilmiş" in text or "No pretrained weights" in text


def test_colab_respects_the_drive_storage_budget() -> None:
    """A free Drive account is 15 GB; the notebook must not fill it.

    The packaging step must exclude token shards, and the durable store must
    be configured to keep a single checkpoint.
    """
    nb = load(COLAB)
    text = "\n".join(cell_source(c) for c in nb["cells"])

    assert "durable_keep_last" in text, "the notebook must set a durable checkpoint budget"
    assert "durable_on_improvement_only" in text, "durable copies should be tied to validation improvement"
    assert "drive.mount" in text, "Drive mounting is a required stage"

    # The old behaviour zipped the entire run directory, including multi-GB
    # token shards, straight into Drive.
    assert "make_archive(str(paths.root_path" not in text
    assert "make_archive(str(paths.root_path" not in text.replace("shutil.", "")
    assert "tokens" in text and "ASLA konmaz" in text or "not Drive" in text


def test_kaggle_reuses_the_same_package() -> None:
    nb = load(KAGGLE)
    text = "\n".join(cell_source(c) for c in nb["cells"])
    for needle in ("import tagai", "TagAIModel", "train_tokenizer", "export_bundle", "Trainer"):
        assert needle in text, f"Kaggle notebook must reuse {needle}"


def test_notebooks_have_no_execution_outputs() -> None:
    """A notebook shipped with stale outputs misleads readers about what ran."""
    for path in (COLAB, KAGGLE):
        nb = load(path)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                assert not cell.get("outputs"), f"{path.name} has committed outputs"


def test_generator_script_is_idempotent() -> None:
    """Regenerating the notebooks must reproduce them byte for byte."""
    import subprocess
    import sys

    before = {p.name: p.read_bytes() for p in (COLAB, KAGGLE)}
    subprocess.run([sys.executable, str(NOTEBOOK_DIR.parent / "scripts" / "build_notebooks.py")],
                   check=True, capture_output=True)
    for name, data in before.items():
        assert (NOTEBOOK_DIR / name).read_bytes() == data, f"{name} is not reproducible"