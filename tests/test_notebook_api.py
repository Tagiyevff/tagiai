"""Verify every tagai API referenced by the notebooks actually exists.

A notebook that imports a name the library does not export fails only after the
user has paid for a GPU session. This test closes that gap offline.
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS = [NOTEBOOK_DIR / "tagai_from_scratch_colab.ipynb", NOTEBOOK_DIR / "tagai_from_scratch_kaggle.ipynb"]

IMPORT_RE = re.compile(r"^\s*from\s+(tagai[\w.]*)\s+import\s+(.+)$")


def notebook_imports(path: Path) -> list[tuple[str, str]]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    out: list[tuple[str, str]] = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        # Join parenthesised multi-line imports so "import (\n  a,\n  b,\n)"
        # is parsed as a single logical statement.
        flat = re.sub(r"import\s*\(([^)]*)\)", lambda m: "import " + " ".join(m.group(1).split()), src)
        for line in flat.splitlines():
            m = IMPORT_RE.match(line)
            if not m:
                continue
            module, names = m.group(1), m.group(2)
            for name in names.split(","):
                name = name.strip().strip("()")
                if not name or not name.isidentifier():
                    continue
                alias = name.split(" as ")[0].strip()
                out.append((module, alias))
    return out


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_imports_resolve(path: Path) -> None:
    failures = []
    for module_name, symbol in notebook_imports(path):
        try:
            mod = importlib.import_module(module_name)
        except ImportError as exc:
            failures.append(f"cannot import module {module_name}: {exc}")
            continue
        if not hasattr(mod, symbol):
            failures.append(f"{module_name} has no attribute {symbol!r}")
    assert not failures, "notebook imports that do not resolve:\n" + "\n".join(failures)


NOTEBOOK_FUNCTIONS = [
    ("tagai.config", "build_config"),
    ("tagai.config", "ModelConfig"),
    ("tagai.data.prepare", "build_corpus"),
    ("tagai.data.prepare", "iter_corpus"),
    ("tagai.data.sources", "SourceRegistry"),
    ("tagai.data.synthetic", "generate"),
    ("tagai.data.dedupe", "Deduplicator"),
    ("tagai.data.download", "download_file"),
    ("tagai.data.download", "list_wiki_dump_files"),
    ("tagai.data.download", "read_multistream_index"),
    ("tagai.data.download", "extract_wiki_streams"),
    ("tagai.data.download", "ensure_space"),
    ("tagai.data.pack", "pack_corpus"),
    ("tagai.data.pack", "PackedDataset"),
    ("tagai.data.pack", "BatchSampler"),
    ("tagai.tokenizer.trainer", "train_tokenizer"),
    ("tagai.tokenizer", "TagAITokenizer"),
    ("tagai.model", "TagAIModel"),
    ("tagai.training.trainer", "Trainer"),
    ("tagai.training.checkpoint", "find_latest_valid"),
    ("tagai.training.checkpoint", "list_checkpoints"),
    ("tagai.inference", "TagAIGenerator"),
    ("tagai.inference", "GenerateConfig"),
    ("tagai.evaluation", "EvalSuite"),
    ("tagai.evaluation", "evaluate_ppl"),
    ("tagai.evaluation.perplexity", "plot_loss_curve"),
    ("tagai.export", "export_bundle"),
    ("tagai.utils", "atomic_write_json"),
    ("tagai.utils", "read_json"),
    ("tagai.utils", "human_bytes"),
    ("tagai.utils", "set_seed"),
]


@pytest.mark.parametrize("module_name,symbol", NOTEBOOK_FUNCTIONS)
def test_library_surface_used_by_notebooks_exists(module_name: str, symbol: str) -> None:
    """Every entry point a notebook calls must exist in the shipped package."""
    mod = importlib.import_module(module_name)
    assert hasattr(mod, symbol), f"{module_name}.{symbol} is missing"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_only_load_the_local_tokenizer(path: Path) -> None:
    """A from-scratch project must never pull pretrained weights.

    ``TagAITokenizer.from_pretrained`` is allowed only when it points at the
    run's own tokenizer directory; a Hugging Face repository id would mean an
    external model had entered the pipeline.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"])

    for forbidden in ("AutoModelForCausalLM", "AutoTokenizer", "transformers", "huggingface_hub", "hf_hub_download"):
        assert forbidden not in src, f"notebook must not reference {forbidden}"

    for call in re.findall(r"\w+\.from_pretrained\(([^)]*)\)", src):
        assert "paths.tokenizer_dir" in call, f"from_pretrained must load the local tokenizer, got: {call}"
    for call in re.findall(r"from_pretrained\(([^)]*)\)", src):
        assert "paths.tokenizer_dir" in call, f"from_pretrained must load the local tokenizer, got: {call}"

    assert "TagAITokenizer.from_pretrained" in src, "the notebook should load the tokenizer it trained"