"""The pretraining loop: next-token prediction with autograd + AMP + resume."""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from ..config import ModelConfig, TrainConfig
from ..utils import capture_rng_state, get_logger, restore_rng_state, setup_logging
from . import checkpoint as ckpt_mod
from .optimizer import build_optimizer, check_gradients, lr_at, set_lr

log = get_logger("training.trainer")

CSV_HEADER = ["step", "loss", "val_loss", "val_ppl", "lr", "grad_norm", "tokens", "seconds", "dt"]


@dataclass
class TrainState:
    step: int = 0
    tokens_seen: int = 0
    best_val_loss: float = float("inf")
    epoch: int = 0
    history: list = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "best_val_loss": self.best_val_loss,
            "epoch": self.epoch,
            "history": self.history[-500:],
            "start_time": self.start_time,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainState":
        return cls(
            step=d.get("step", 0),
            tokens_seen=d.get("tokens_seen", 0),
            best_val_loss=d.get("best_val_loss", float("inf")),
            epoch=d.get("epoch", 0),
            history=d.get("history", []),
            start_time=d.get("start_time", time.time()),
        )


class Trainer:
    """Owns the model, optimizer, data iteration, logging and recovery."""

    def __init__(
        self,
        model: nn.Module,
        train_data,
        val_data,
        cfg: TrainConfig,
        model_cfg: ModelConfig,
        run_dir: str | os.PathLike[str],
        *,
        tokenizer=None,
        device: Optional[torch.device] = None,
        on_eval: Optional[Callable[[int, "Trainer"], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.train_data = train_data
        self.val_data = val_data
        self.tokenizer = tokenizer
        self.on_eval = on_eval

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = build_optimizer(self.model, cfg)
        self.state = TrainState()
        self.amp_dtype = self._resolve_precision()
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.amp_dtype == torch.float16 and self.device.type == "cuda"))
        self._val_iterator = None
        self._train_iterator = None
        self._last_durable_step: Optional[int] = None
        self._best_val_at_last_durable = float("inf")
        self.log_path = self.run_dir / "train_log.csv"
        self._write_csv_header()
        setup_logging(log_file=self.run_dir / "tagai.log")

    # -- setup helpers ---------------------------------------------------
    def _resolve_precision(self) -> torch.dtype:
        pref = self.cfg.precision
        if self.device.type != "cuda":
            return torch.float32
        if pref == "fp32":
            return torch.float32
        if pref == "fp16":
            return torch.float16
        if pref == "bf16":
            return torch.bfloat16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def _write_csv_header(self) -> None:
        if not self.log_path.exists():
            self.log_path.write_text(",".join(CSV_HEADER) + "\n", encoding="utf-8")

    def _log_row(self, row: dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(",".join(str(row.get(k, "")) for k in CSV_HEADER) + "\n")

    # -- data ------------------------------------------------------------
    def _to_device(self, arr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn a ``(B, seq_len + 1)`` window stack into next-token pairs.

        Inputs and targets are shifted along the *sequence* axis, never the
        batch axis: ``x = w[:, :-1]`` predicts ``y = w[:, 1:]``.
        """
        window = torch.from_numpy(np.ascontiguousarray(arr.astype(np.int64)))
        if window.dim() == 1:
            window = window.unsqueeze(0)
        x = window[:, :-1].contiguous()
        y = window[:, 1:].contiguous()
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def _sampler(self, dataset, batch_size: int):
        """Accept a BatchSampler, a PackedDataset, or a plain window sequence."""
        from ..data.pack import BatchSampler, PackedDataset

        if hasattr(dataset, "batches") and hasattr(dataset, "dataset"):
            return dataset, dataset.dataset
        if isinstance(dataset, PackedDataset):
            sampler = BatchSampler(dataset, batch_size, seed=self.cfg.seed + self.state.epoch)
            return sampler, dataset
        return None, None

    def _batches(self, dataset, batch_size: int):
        """Yield stacked ``(B, seq_len + 1)`` windows for one epoch."""
        sampler, source = self._sampler(dataset, batch_size)
        if sampler is not None:
            for idx_batch in sampler.batches():
                yield np.stack([source.get(int(i)) for i in idx_batch])
            return
        if hasattr(dataset, "get"):  # window provider without a sampler
            n = len(dataset)
            order = np.random.default_rng(self.cfg.seed + self.state.epoch).permutation(n)
            for start in range(0, n - batch_size + 1, batch_size):
                yield np.stack([dataset.get(int(i)) for i in order[start : start + batch_size]])
            return
        for start in range(0, len(dataset) - batch_size + 1, batch_size):
            yield np.stack(list(dataset[start : start + batch_size]))

    # -- loss ------------------------------------------------------------
    def _compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
            _, loss = self.model(x, y)
        if loss is None:
            raise RuntimeError("model returned no loss")
        return loss

    # -- validation ------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, batches: Optional[int] = None) -> float:
        """Mean validation loss, or NaN when nothing could be evaluated.

        Returning 0.0 for "no data" would be indistinguishable from a
        perfect model, so an unusable validation set is reported as NaN.
        """
        self.model.eval()
        if self.val_data is None or len(self.val_data) == 0:
            return float("nan")
        n = batches or self.cfg.eval_batches
        total, count = 0.0, 0
        for i, stack in enumerate(self._batches(self.val_data, self.cfg.micro_batch_size)):
            if i >= n:
                break
            x, y = self._to_device(stack)
            loss = self._compute_loss(x, y)
            if not math.isfinite(float(loss)):
                log.error("non-finite validation loss at batch %d", i)
                return float("nan")
            total += float(loss.item())
            count += 1
        self.model.train()
        if count == 0:
            log.warning(
                "validation produced no batches: %d windows < batch size %d. "
                "Increase data.validation_fraction or lower train.micro_batch_size.",
                len(self.val_data),
                self.cfg.micro_batch_size,
            )
            return float("nan")
        return total / count

    # -- main loop -------------------------------------------------------
    def train(self) -> TrainState:
        cfg = self.cfg
        log.info(
            "training start: device=%s precision=%s params=%d effective_batch=%d seq_len=%d max_steps=%d",
            self.device,
            self.amp_dtype,
            self.model.num_parameters(),
            cfg.micro_batch_size * cfg.grad_accum_steps,
            cfg.max_seq_len,
            cfg.max_steps,
        )
        self.model.train()
        t_start = time.time()
        last_step_time = t_start
        step = self.state.step
        micro_in_accum = 0

        train_iter = iter(self._batches(self.train_data, cfg.micro_batch_size))
        epoch = self.state.epoch
        while step < cfg.max_steps:
            if cfg.max_tokens is not None and self.state.tokens_seen >= cfg.max_tokens:
                log.info("token budget %d reached; stopping", cfg.max_tokens)
                break

            try:
                stack = next(train_iter)
            except StopIteration:
                epoch += 1
                if hasattr(self.train_data, "step"):
                    self.train_data.step()
                if len(self.train_data) < cfg.micro_batch_size:
                    log.error("training set smaller than one batch (%d < %d)", len(self.train_data), cfg.micro_batch_size)
                    break
                train_iter = iter(self._batches(self.train_data, cfg.micro_batch_size))
                log.info("epoch %d complete (%d windows)", epoch, len(self.train_data))
                continue

            x, y = self._to_device(stack)
            set_lr(self.optimizer, lr_at(step, cfg))
            loss = self._compute_loss(x, y)
            self.scaler.scale(loss / cfg.grad_accum_steps).backward()
            micro_in_accum += 1
            self.state.tokens_seen += int(x.numel())

            if micro_in_accum < cfg.grad_accum_steps:
                continue
            micro_in_accum = 0

            diag = check_gradients(self.model, cfg.grad_clip)
            if not diag.finite:
                log.error("non-finite gradients at step %d; aborting to protect the checkpoint", step)
                self.save(step)
                break
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            step += 1
            self.state.step = step
            self.state.epoch = epoch
            self.state.history.append(
                {"step": step, "loss": float(loss.item()), "lr": lr_at(step, cfg), "tokens": self.state.tokens_seen, "grad_norm": diag.grad_norm}
            )

            if step % cfg.log_every == 0:
                now = time.time()
                loss_avg = float(np.mean([h["loss"] for h in self.state.history[-cfg.log_every :]]))
                row = {
                    "step": step,
                    "loss": round(loss_avg, 5),
                    "lr": round(lr_at(step, cfg), 8),
                    "grad_norm": round(diag.grad_norm, 4),
                    "tokens": self.state.tokens_seen,
                    "seconds": round(now - t_start, 1),
                    "dt": round(now - last_step_time, 3),
                }
                self._log_row(row)
                log.info(
                    "step %d/%d | loss %.4f | ppl %.2f | lr %.2e | gnorm %.2f | %.1f tok/s | elapsed %.0fs",
                    step,
                    cfg.max_steps,
                    loss_avg,
                    math.exp(min(20, loss_avg)),
                    lr_at(step, cfg),
                    diag.grad_norm,
                    self.state.tokens_seen / max(1e-6, now - t_start),
                    now - t_start,
                )
                last_step_time = now

            if cfg.eval_every and step % cfg.eval_every == 0:
                val = self.evaluate()
                if math.isfinite(val):
                    self.state.best_val_loss = min(self.state.best_val_loss, val)
                    self._log_row({"step": step, "val_loss": round(val, 5), "val_ppl": round(math.exp(min(20, val)), 3)})
                    log.info("step %d | val_loss %.4f | val_ppl %.2f | best %.4f", step, val, math.exp(min(20, val)), self.state.best_val_loss)
                else:
                    log.warning("step %d | validation unavailable (see the warning above)", step)
                if self.on_eval:
                    self.on_eval(step, self)

            if cfg.save_every and step % cfg.save_every == 0:
                self.save(step)

        if micro_in_accum:  # flush a partial accumulation window
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        log.info("training finished at step %d (%.0fs)", self.state.step, time.time() - t_start)
        return self.state

    # -- persistence -----------------------------------------------------
    def save(self, step: Optional[int] = None) -> Path:
        """Save locally, and mirror to the durable directory if configured."""
        step = step if step is not None else self.state.step
        extra = {
            "tokens_seen": self.state.tokens_seen,
            "best_val_loss": self.state.best_val_loss,
            "epoch": self.state.epoch,
            "rng_state": capture_rng_state(),
            "train_state": self.state.to_dict(),
            "train_config": asdict(self.cfg),
            "config": {"model": asdict(self.model_cfg), "train": asdict(self.cfg)},
            "data_fingerprint": self._data_fingerprint(),
        }
        path = ckpt_mod.save_checkpoint(
            self.run_dir,
            step,
            self.model,
            self.optimizer,
            extra=extra,
            config=asdict(self.model_cfg),
            keep_last=self.cfg.keep_last_checkpoints,
            cpu_optimizer_state=self.cfg.save_cpu_optimizer_state,
        )
        self._last_local_step = step
        self._maybe_save_durable(step, extra)
        return path

    def _maybe_save_durable(self, step: int, extra: dict) -> None:
        """Mirror one checkpoint to the durable directory, within a strict budget.

        A durable copy is expensive both to write and to keep: a 135 M
        parameter model with optimizer state is ~1.6 GB, and a free Drive
        account holds 15 GB in total. So the copy is made only when validation
        has actually improved, never more often than
        ``durable_min_steps_between``, without optimizer state when asked, and
        only if enough free space is left on the volume.
        """
        if not self.cfg.durable_dir:
            return
        durable_dir = Path(self.cfg.durable_dir)

        last_step = self._last_durable_step
        if last_step is not None and step - last_step < self.cfg.durable_min_steps_between:
            return

        if self.cfg.durable_on_improvement_only:
            best = self.state.best_val_loss
            if not math.isfinite(best):
                return
            # Only mirror a step that produced a *new* best validation loss.
            if last_step is not None and best >= self._best_val_at_last_durable:
                return

        free_gb = ckpt_mod.free_space_gb(durable_dir)
        if free_gb < self.cfg.durable_min_free_gb:
            log.warning(
                "durable save skipped: %.1f GB free at %s, need at least %.1f GB",
                free_gb, durable_dir, self.cfg.durable_min_free_gb,
            )
            return

        try:
            path = ckpt_mod.save_checkpoint(
                self.run_dir,
                step,
                self.model,
                self.optimizer,
                extra=extra,
                config=asdict(self.model_cfg),
                save_optimizer=self.cfg.durable_save_optimizer,
                keep_last=max(1, self.cfg.durable_keep_last),
                cpu_optimizer_state=self.cfg.save_cpu_optimizer_state,
                dest_dir=durable_dir,
            )
        except OSError as exc:
            log.error("durable save failed (%s); the local checkpoint is unaffected", exc)
            return

        self._last_durable_step = step
        self._best_val_at_last_durable = self.state.best_val_loss
        log.info(
            "durable checkpoint written at step %d (%.1f MB) | %.1f GB free on the durable volume",
            step, path.stat().st_size / 1024**2, ckpt_mod.free_space_gb(durable_dir),
        )

    def storage_report(self) -> dict:
        """Current checkpoint footprint, so the storage budget stays visible."""
        local = ckpt_mod.list_checkpoints(self.run_dir)
        out = {
            "local_checkpoints": len(local),
            "local_gb": round(sum(c.bytes for c in local) / 1024**3, 3),
            "local_free_gb": round(ckpt_mod.free_space_gb(self.run_dir), 2),
        }
        if self.cfg.durable_dir:
            durable = ckpt_mod.list_checkpoints_in(self.cfg.durable_dir)
            out.update(
                {
                    "durable_dir": str(self.cfg.durable_dir),
                    "durable_checkpoints": len(durable),
                    "durable_gb": round(sum(c.bytes for c in durable) / 1024**3, 3),
                    "durable_free_gb": round(ckpt_mod.free_space_gb(self.cfg.durable_dir), 2),
                }
            )
        return out

    def _data_fingerprint(self) -> Optional[str]:
        meta = getattr(self.train_data, "meta", None)
        if not meta:
            return None
        import hashlib

        parts = [f"{s['path']}:{s['tokens']}" for s in meta.get("shards", [])]
        return hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=8).hexdigest()

    def resume(self, path: str | os.PathLike[str], *, strict_data: bool = True) -> int:
        """Restore model/optimizer/RNG/step. Returns the step training resumes from.

        The compatibility check runs *before* any weights are touched, so a
        mismatched resume fails with an actionable message instead of a wall
        of PyTorch shape errors.
        """
        from ..training.checkpoint import validate_checkpoint

        ok, why = validate_checkpoint(Path(path))
        if not ok:
            raise RuntimeError(f"cannot resume: {why}")

        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        tok_vocab = self.tokenizer.vocab_size if self.tokenizer is not None else self.model_cfg.vocab_size
        compatible, problems = ckpt_mod.compatibility_check(
            {"model_config": payload.get("model_config"), "data_fingerprint": payload.get("data_fingerprint")},
            asdict(self.model_cfg),
            tok_vocab,
            self._data_fingerprint(),
        )
        if not compatible:
            msg = "refusing to resume: " + "; ".join(problems)
            if strict_data:
                raise RuntimeError(msg)
            log.warning("%s (continuing anyway because strict_data=False)", msg)
        del payload

        meta = ckpt_mod.load_checkpoint(path, self.model, self.optimizer, map_location=str(self.device))
        ts = meta.get("train_state")
        if ts:
            self.state = TrainState.from_dict(ts)
        else:
            self.state.step = meta.get("step", 0)
            self.state.tokens_seen = meta.get("tokens_seen", 0) or 0
            self.state.best_val_loss = meta.get("best_val_loss", float("inf"))
        if "rng_state" in meta:
            try:
                restore_rng_state(meta["rng_state"])
            except Exception as exc:  # pragma: no cover - torch version drift
                log.warning("could not restore RNG state: %s", exc)

        log.info("resumed from %s at step %d (%d tokens seen)", path, self.state.step, self.state.tokens_seen)
        return self.state.step

    def resolve_resume_path(self) -> Optional[Path]:
        """Find a checkpoint to resume from.

        Local checkpoints win; the durable directory is the fallback, which is
        what makes a fresh Colab session able to continue a run whose local
        disk was wiped.
        """
        mode = self.cfg.resume
        if mode == "none":
            return None
        if mode == "auto":
            local = ckpt_mod.find_latest_valid(self.run_dir)
            if local is not None:
                return local
            if self.cfg.durable_dir:
                durable = ckpt_mod.find_latest_valid(self.run_dir, ck_dir=self.cfg.durable_dir)
                if durable is not None:
                    log.info("resuming from the durable checkpoint at %s", durable)
                return durable
            return None
        p = Path(mode)
        if not p.is_absolute():
            candidate = p if p.exists() else p.parent
            p = candidate
        return p if p.exists() else None


def load_model_from_checkpoint(path: str | os.PathLike[str], device: str = "cpu") -> tuple[nn.Module, ModelConfig]:
    """Rebuild a model from a checkpoint using the config stored inside it."""
    from ..model.transformer import TagAIModel

    p = Path(path)
    ok, why = ckpt_mod.validate_checkpoint(p)
    if not ok:
        raise RuntimeError(why)
    payload = torch.load(p, map_location=device, weights_only=False)
    cfg_dict = payload.get("model_config")
    if not cfg_dict:
        raise RuntimeError(f"checkpoint {p} does not store a model_config")
    model_cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})
    model = TagAIModel(model_cfg)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return model, model_cfg