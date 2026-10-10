"""Perplexity evaluation over the held-out split."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..utils import get_logger

log = get_logger("evaluation.perplexity")


def _iter_batches(dataset, batch_size: int, max_batches: int):
    """Accept a PackedDataset, a BatchSampler, or any object exposing ``batches``."""
    from ..data.pack import BatchSampler, PackedDataset

    sampler = dataset
    source = None
    if isinstance(dataset, BatchSampler):
        source = dataset.dataset
        sampler = dataset
    elif isinstance(dataset, PackedDataset):
        source = dataset
        sampler = BatchSampler(dataset, batch_size, seed=0, shuffle=False)
    elif hasattr(dataset, "batches"):
        source = getattr(dataset, "dataset", None)

    if sampler is not None and source is not None and hasattr(sampler, "batches"):
        for i, idx_batch in enumerate(sampler.batches()):
            if i >= max_batches:
                return
            yield np.stack([source.get(int(k)) for k in idx_batch])
        return

    for i, window in enumerate(dataset):
        if i >= max_batches:
            return
        yield np.asarray(window)


@torch.no_grad()
def evaluate_ppl(model, dataset, device: Optional[torch.device] = None, max_batches: int = 50, micro_batch: int = 8) -> dict:
    """Token-level perplexity with the standard length normalisation.

    Reports PPL computed over all predicted tokens (the standard number) and
    the bits-per-token of the same data, which is comparable across tokenizer
    changes.
    """
    device = device or next(model.parameters()).device
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    batches = 0
    for stack in _iter_batches(dataset, micro_batch, max_batches):
        windows = torch.from_numpy(np.ascontiguousarray(stack.astype(np.int64)))
        for row in windows:
            x = row[:-1].unsqueeze(0).to(device)
            y = row[1:].unsqueeze(0).to(device)
            _, loss = model(x, y)
            n = int(y.numel())
            if not torch.isfinite(loss):
                log.error("non-finite loss during perplexity evaluation")
                return {"perplexity": None, "nll": None, "tokens": total_tokens}
            total_nll += float(loss.item()) * n
            total_tokens += n
        batches += 1
    if total_tokens == 0:
        return {"perplexity": None, "nll": None, "tokens": 0, "reason": "no evaluation windows available"}
    nll = total_nll / total_tokens
    return {
        "nll": round(nll, 5),
        "perplexity": round(math.exp(min(30.0, nll)), 4),
        "tokens": total_tokens,
        "batches": batches,
        "bits_per_token": round(nll / math.log(2), 5),
    }


@torch.no_grad()
def evaluate_on_texts(model, tokenizer, texts: list[str], device: Optional[torch.device] = None, max_len: int = 512) -> dict:
    """Perplexity over arbitrary held-out strings."""
    device = device or next(model.parameters()).device
    model.eval()
    total_nll, total_tokens, total_chars = 0.0, 0, 0
    for text in texts:
        ids = tokenizer.encode(text)[: max_len * 2]
        if len(ids) < 2:
            continue
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        _, loss = model(x, y)
        total_nll += float(loss.item()) * y.numel()
        total_tokens += int(y.numel())
        total_chars += len(text)
    if total_tokens == 0:
        return {"perplexity": None, "tokens": 0, "texts": len(texts)}
    nll = total_nll / total_tokens
    return {
        "nll": round(nll, 5),
        "perplexity": round(math.exp(min(30.0, nll)), 4),
        "tokens": total_tokens,
        "chars": total_chars,
        "bits_per_character": round(nll / max(2, total_chars) / math.log(2), 6),
        "texts": len(texts),
    }


def loss_curve(history: list[dict], key: str = "loss") -> list[tuple[int, float]]:
    return [(h["step"], h[key]) for h in history if key in h]


def plot_loss_curve(history: list[dict], out_path: str | Path) -> Optional[Path]:
    """Save a loss curve PNG. Returns None if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping loss plot")
        return None
    steps = [h["step"] for h in history if "loss" in h]
    losses = [h["loss"] for h in history if "loss" in h]
    if not steps:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, losses, label="train loss", linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy (nats/token)")
    ax.set_title("TagAI training loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p