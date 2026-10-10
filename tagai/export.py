"""Package a trained checkpoint into a self-contained, portable bundle."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from .training.checkpoint import validate_checkpoint
from .config import ModelConfig
from .utils import atomic_write_json, get_logger, human_bytes

log = get_logger("export")

BUNDLE_NAME = "tagai"


def export_bundle(
    checkpoint_path: str | Path,
    tokenizer_dir: str | Path,
    out_dir: str | Path,
    *,
    training_report: Optional[dict] = None,
    evaluation_report: Optional[dict] = None,
    extra_files: Optional[list[str | Path]] = None,
) -> dict:
    """Write model weights, tokenizer, config and reports into ``out_dir``.

    The bundle is a plain directory plus a zip so it can be re-uploaded to a
    new notebook without re-running training.
    """
    ckpt = Path(checkpoint_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model").mkdir(exist_ok=True)
    (out / "tokenizer").mkdir(exist_ok=True)

    ok, why = validate_checkpoint(ckpt)
    if not ok:
        raise RuntimeError(f"cannot export an invalid checkpoint: {why}")

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg_dict = payload.get("model_config") or {}
    model_cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})

    state = payload["model_state"]
    weights_path = out / "model" / "pytorch_model.bin"
    torch.save({"state_dict": state, "config": model_cfg.__dict__}, weights_path)

    # With tied embeddings `lm_head.weight is tok_emb.weight`. safetensors
    # refuses shared storage because it would duplicate the tensor on disk, so
    # the tied alias is dropped and restored from `tie_embeddings` on load.
    safetensors_state = _dedupe_tied_weights(state, tie_embeddings=model_cfg.tie_embeddings)
    safetensors_path = None
    try:
        from safetensors.torch import save_file

        target = out / "model" / "model.safetensors"
        save_file({k: v.detach().cpu().contiguous() for k, v in safetensors_state.items()}, str(target))
        safetensors_path = target
    except ImportError:
        log.info("safetensors not installed; shipping pytorch_model.bin only")
    except Exception as exc:  # noqa: BLE001 - a format we cannot write must not fail the export
        log.warning("safetensors export failed (%s); shipping pytorch_model.bin only", exc)
        (out / "model" / "model.safetensors").unlink(missing_ok=True)
        safetensors_path = None

    tok_src = Path(tokenizer_dir)
    copied = []
    for name in ("tokenizer.model", "tokenizer_config.json", "tokenizer_stats.json"):
        p = tok_src / name
        if p.exists():
            shutil.copyfile(p, out / "tokenizer" / name)
            copied.append(name)

    from . import __version__ as tagai_version

    manifest = {
        "name": BUNDLE_NAME,
        "format_version": 1,
        "exported_unix": int(time.time()),
        "tagai_version": tagai_version,
        "source_checkpoint": str(ckpt),
        "checkpoint_step": payload.get("step"),
        "tokens_seen": payload.get("tokens_seen"),
        "best_val_loss": payload.get("best_val_loss"),
        "model": {
            "architecture": "decoder-only transformer (from scratch)",
            # Count unique tensors: with tied embeddings the state dict lists
            # lm_head.weight and tok_emb.weight as separate entries backed by
            # one storage, and summing them would overstate the model size.
            "parameters": sum(t.numel() for t in safetensors_state.values()),
            "parameter_tensors": len(safetensors_state),
            "config": asdict(model_cfg),
        },
        "tokenizer": {
            "type": "sentencepiece_bpe",
            "files": copied,
            "vocab_size": (json.loads((out / "tokenizer" / "tokenizer_config.json").read_text())["vocab_size"] if "tokenizer_config.json" in copied else None),
        },
        "pretrained_weights_used": False,
        "initialization": "random (N(0, initializer_range)) at every run; no external checkpoint is loaded",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "files": {
            "weights": str(weights_path.relative_to(out)),
            "safetensors": str(safetensors_path.relative_to(out)) if safetensors_path else None,
            "tokenizer": copied,
        },
        "weights_bytes": weights_path.stat().st_size,
        "weights_human": human_bytes(weights_path.stat().st_size),
    }
    if training_report:
        atomic_write_json(out / "training_report.json", training_report)
        manifest["files"]["training_report"] = "training_report.json"
    if evaluation_report:
        atomic_write_json(out / "evaluation_report.json", evaluation_report)
        manifest["files"]["evaluation_report"] = "evaluation_report.json"

    atomic_write_json(out / "manifest.json", manifest)
    _write_readme(out, manifest)
    (out / "config.json").write_text(json.dumps(asdict(model_cfg), indent=2), encoding="utf-8")

    for src in extra_files or []:
        p = Path(src)
        if p.exists():
            shutil.copyfile(p, out / p.name)

    zip_path = shutil.make_archive(str(out.with_suffix("")), "zip", root_dir=str(out))
    manifest["archive"] = zip_path
    atomic_write_json(out / "manifest.json", manifest)
    log.info("bundle exported to %s (%s, archive %.1f MB)", out, manifest["weights_human"], Path(zip_path).stat().st_size / 1024**2)
    return manifest


def _dedupe_tied_weights(state: dict, *, tie_embeddings: bool) -> dict:
    """Drop weight entries that alias another tensor's storage.

    A checkpoint stores tied parameters as two separate objects backed by one
    storage, so identity comparison is not enough - the storage pointer is what
    reveals the alias.
    """
    out: dict = {}
    seen_ptr: dict[int, str] = {}
    for name, tensor in state.items():
        if tie_embeddings and name == "lm_head.weight":
            continue  # restored from tok_emb.weight on load
        try:
            ptr = tensor.untyped_storage().data_ptr()
        except Exception:  # pragma: no cover - exotic tensor types
            ptr = id(tensor)
        if ptr in seen_ptr:
            log.debug("skipping aliased weight %s (same storage as %s)", name, seen_ptr[ptr])
            continue
        seen_ptr[ptr] = name
        out[name] = tensor
    return out


def _write_readme(out: Path, manifest: dict) -> None:
    text = f"""# {manifest['name']} model bundle

- Parameters: {manifest['model']['parameters']:,}
- Context length: {manifest['model']['config']['max_seq_len']}
- Vocabulary: {manifest['tokenizer']['vocab_size']}
- Trained from randomly initialized weights: {not manifest['pretrained_weights_used']}
- Exported at step {manifest['checkpoint_step']}

## Load

```python
import torch
from tagai.model import TagAIModel
from tagai.tokenizer import TagAITokenizer
from tagai.config import ModelConfig

manifest = json.load(open("manifest.json"))
cfg = ModelConfig(**manifest["model"]["config"])
model = TagAIModel(cfg)
model.load_state_dict(torch.load("model/pytorch_model.bin", map_location="cpu")["state_dict"])
tok = TagAITokenizer.from_pretrained("tokenizer")

from tagai.inference import TagAIGenerator
gen = TagAIGenerator.from_model(model, tok)
print(gen.generate("Salam, mənə kömək edə bilərsinizmi?"))
```
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def load_bundle(bundle_dir: str | Path, device: str = "cpu"):
    """Rebuild a runnable generator from an exported bundle."""
    from .inference import TagAIGenerator
    from .model import TagAIModel
    from .tokenizer import TagAITokenizer
    from .utils import read_json

    d = Path(bundle_dir)
    manifest = read_json(d / "manifest.json")
    cfg = ModelConfig(**manifest["model"]["config"])
    model = TagAIModel(cfg)
    weights = d / "model" / "model.safetensors"
    if weights.exists():
        from safetensors.torch import load_file

        state = load_file(str(weights))
        if cfg.tie_embeddings and "lm_head.weight" not in state and "tok_emb.weight" in state:
            # The tied alias is not stored; restore it before loading.
            state["lm_head.weight"] = state["tok_emb.weight"]
        model.load_state_dict(state)
    else:
        model.load_state_dict(torch.load(d / "model" / "pytorch_model.bin", map_location="cpu")["state_dict"])
    tok = TagAITokenizer.from_pretrained(d / "tokenizer")
    return TagAIGenerator.from_model(model, tok, device=device)