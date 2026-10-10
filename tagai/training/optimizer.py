"""Optimizer and LR schedule construction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import TrainConfig
from ..utils import get_logger

log = get_logger("training.optimizer")


def build_param_groups(model: nn.Module, cfg: TrainConfig) -> list[dict]:
    """Decay 2-D weight matrices only; leave biases and norms undecayed."""
    no_decay = list(cfg.betas_wd_exclude)
    decay, no_decay_params = [], []
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:  # tied embeddings appear twice in named_parameters
            continue
        seen.add(id(param))
        if param.ndim < 2 or name.endswith(".bias") or any(t in name.lower() for t in no_decay) or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay, "group_name": "decay"},
        {"params": no_decay_params, "weight_decay": 0.0, "group_name": "no_decay"},
    ]
    return [g for g in groups if g["params"]]


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    groups = build_param_groups(model, cfg)
    fused = cfg.precision in ("bf16", "fp16") and torch.cuda.is_available()
    kwargs = dict(lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=1e-8)
    if fused:
        try:
            return torch.optim.AdamW(groups, fused=True, **kwargs)
        except (RuntimeError, TypeError):
            log.debug("fused AdamW unavailable; falling back to the standard implementation")
    return torch.optim.AdamW(groups, **kwargs)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to ``min_lr_ratio * lr``.

    Defined in closed form (not as a scheduler object) so resuming from a
    checkpoint reproduces the identical schedule without storing state.
    """
    warmup = max(0, cfg.warmup_steps)
    if warmup and step < warmup:
        return cfg.lr * (step + 1) / warmup
    decay_steps = max(1, cfg.max_steps - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = cfg.lr * cfg.min_lr_ratio
    return min_lr + (cfg.lr - min_lr) * cosine


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


@dataclass
class GradDiagnostics:
    finite: bool = True
    grad_norm: float = 0.0
    nonfinite_params: int = 0
    update_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "finite": self.finite,
            "grad_norm": round(self.grad_norm, 6),
            "nonfinite_params": self.nonfinite_params,
            "update_ratio": round(self.update_ratio, 8),
        }


def check_gradients(model: nn.Module, clip: float = 1.0) -> GradDiagnostics:
    """Clip and diagnose gradients.

    A non-finite gradient normally means the run has already diverged; the
    diagnostic reports it so the trainer can abort instead of writing a
    poisoned checkpoint.
    """
    total = torch.zeros((), dtype=torch.float32)
    nonfinite = 0
    for p in model.parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                nonfinite += 1
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                total += p.grad.detach().float().norm() ** 2
    norm = float(total.sqrt())
    if torch.isfinite(torch.tensor(norm)):
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    return GradDiagnostics(finite=(nonfinite == 0), grad_norm=norm, nonfinite_params=nonfinite)


def warmup_and_clip(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    step: int,
    cfg: TrainConfig,
) -> GradDiagnostics:
    set_lr(optimizer, lr_at(step, cfg))
    return check_gradients(model, cfg.grad_clip)