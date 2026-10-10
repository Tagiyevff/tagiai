"""Atomic, resumable checkpointing with corruption detection.

Rules that protect a long free-GPU run:

* write to ``<name>.tmp`` then ``os.replace`` - a crash never leaves a
  half-written checkpoint where a valid one used to be;
* write a SHA-256 sidecar so corruption is detected on load, not discovered
  when training continues on garbage weights;
* never delete the newest valid checkpoint, whatever the retention policy
  says;
* refuse to resume into a run whose model/tokenizer/packed data disagree.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from ..utils import atomic_write_json, get_logger, read_json, sha256_file

log = get_logger("training.checkpoint")

CHECKPOINT_RE = re.compile(r"^step(\d+)\.pt$")
LATEST_NAME = "latest.pt"
MANIFEST_NAME = "checkpoints_manifest.json"


def checkpoint_dir(run_dir: str | os.PathLike[str]) -> Path:
    return Path(run_dir) / "checkpoints"


def save_checkpoint(
    run_dir: str | os.PathLike[str],
    step: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    extra: Optional[dict[str, Any]] = None,
    config: Optional[dict[str, Any]] = None,
    save_optimizer: bool = True,
    keep_last: int = 3,
    cpu_optimizer_state: bool = True,
    dest_dir: Optional[str | os.PathLike[str]] = None,
    quiet: bool = False,
) -> Path:
    """Write ``step{step}.pt`` atomically, refresh ``latest.pt`` and the manifest.

    ``dest_dir`` redirects the write to an arbitrary directory (used for the
    durable copy on Drive, which must be budgeted separately from the fast
    local disk) and prunes within that directory instead of the run's own.
    """
    durable = dest_dir is not None
    ck_dir = Path(dest_dir) if durable else checkpoint_dir(run_dir)
    ck_dir.mkdir(parents=True, exist_ok=True)
    target = ck_dir / f"step{step:06d}.pt"

    payload: dict[str, Any] = {
        "step": step,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": config or getattr(model, "config_dict", dict)(),
        "timestamp": time.time(),
    }
    if optimizer is not None and save_optimizer:
        opt_state = optimizer.state_dict()
        if cpu_optimizer_state:
            opt_state = _optimizer_to_cpu(opt_state)
        payload["optimizer_state"] = opt_state
    if extra:
        payload.update(extra)

    tmp = target.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)  # atomic on the same filesystem

    # `latest.pt` is a copy, not a symlink: Colab/Drive mounts frequently
    # break symlinks across filesystems.
    latest = ck_dir / LATEST_NAME
    tmp_latest = ck_dir / (LATEST_NAME + ".tmp")
    shutil.copyfile(target, tmp_latest)
    os.replace(tmp_latest, latest)

    atomic_write_json(ck_dir / (target.stem + ".sha256"), {"sha256": sha256_file(target), "bytes": target.stat().st_size})

    _prune(ck_dir, keep_last)
    if not durable:
        write_manifest(run_dir)
    size_mb = target.stat().st_size / 1024**2
    if not quiet:
        where = "durable" if durable else "local"
        log.info("checkpoint saved [%s]: %s (%.1f MB)", where, target.name, size_mb)
    return target


def _optimizer_to_cpu(opt_state: dict) -> dict:
    out = dict(opt_state)
    if "state" in out:
        out["state"] = {
            k: {kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v
            for k, v in out["state"].items()
        }
    return out


def _prune(ck_dir: Path, keep_last: int) -> None:
    """Delete old checkpoints, always protecting the newest valid one."""
    if keep_last <= 0:
        return
    steps = sorted(
        (int(m.group(1)), p) for p in ck_dir.iterdir() if (m := CHECKPOINT_RE.match(p.name)) and not p.name.endswith(".tmp")
    )
    if len(steps) <= keep_last:
        return
    for _, path in steps[: len(steps) - keep_last]:
        try:
            path.unlink()
            path.with_suffix(".sha256").unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover
            log.warning("could not delete old checkpoint %s: %s", path.name, exc)


@dataclass
class CheckpointInfo:
    step: int
    path: Path
    bytes: int
    sha256_ok: Optional[bool] = None

    def to_dict(self) -> dict:
        return {"step": self.step, "path": str(self.path), "bytes": self.bytes, "sha256_ok": self.sha256_ok}


def list_checkpoints(run_dir: str | os.PathLike[str]) -> list[CheckpointInfo]:
    return list_checkpoints_in(checkpoint_dir(run_dir))


def list_checkpoints_in(ck_dir: str | os.PathLike[str]) -> list[CheckpointInfo]:
    """List checkpoints in an explicit directory (used for durable storage)."""
    ck_dir = Path(ck_dir)
    if not ck_dir.exists():
        return []
    out: list[CheckpointInfo] = []
    for p in ck_dir.iterdir():
        m = CHECKPOINT_RE.match(p.name)
        if not m or p.name.endswith(".tmp"):
            continue
        sidecar = p.with_suffix(".sha256")
        ok: Optional[bool] = None
        if sidecar.exists():
            expected = read_json(sidecar).get("sha256")
            ok = expected == sha256_file(p)
        out.append(CheckpointInfo(step=int(m.group(1)), path=p, bytes=p.stat().st_size, sha256_ok=ok))
    out.sort(key=lambda c: c.step)
    return out


def write_manifest(run_dir: str | os.PathLike[str]) -> Path:
    ck_dir = checkpoint_dir(run_dir)
    entries = [c.to_dict() for c in list_checkpoints(run_dir)]
    latest = ck_dir / LATEST_NAME
    return atomic_write_json(
        ck_dir / MANIFEST_NAME,
        {
            "latest": str(latest) if latest.exists() else None,
            "latest_step": entries[-1]["step"] if entries else None,
            "checkpoint_count": len(entries),
            "checkpoints": entries,
            "updated": time.time(),
        },
    )


def validate_checkpoint(path: str | os.PathLike[str]) -> tuple[bool, str]:
    """Verify a checkpoint is loadable and not corrupted."""
    p = Path(path)
    if not p.exists():
        return False, f"missing: {p}"
    if p.stat().st_size == 0:
        return False, f"empty file: {p}"
    sidecar = p.with_suffix(".sha256")
    if sidecar.exists():
        expected = read_json(sidecar).get("sha256")
        if expected and sha256_file(p) != expected:
            return False, f"checksum mismatch: {p}"
    try:
        payload = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, f"unreadable torch file ({exc}): {p}"
    if not isinstance(payload, dict) or "model_state" not in payload:
        return False, f"missing model_state: {p}"
    return True, "ok"


def find_latest_valid(run_dir: str | os.PathLike[str], *, ck_dir: Optional[str | os.PathLike[str]] = None) -> Optional[Path]:
    """Newest checkpoint that actually loads. Corrupted files are skipped."""
    directory = Path(ck_dir) if ck_dir is not None else checkpoint_dir(run_dir)
    candidates = sorted(
        (p for p in directory.glob("step*.pt") if not p.name.endswith(".tmp")),
        key=lambda p: int(CHECKPOINT_RE.match(p.name).group(1)) if CHECKPOINT_RE.match(p.name) else 0,
        reverse=True,
    ) if directory.exists() else []
    for p in candidates:
        ok, why = validate_checkpoint(p)
        if ok:
            return p
        log.warning("ignoring unusable checkpoint %s: %s", p.name, why)
    return None


def load_checkpoint(
    path: str | os.PathLike[str],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    map_location: str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint into ``model``/``optimizer`` and return its metadata."""
    p = Path(path)
    ok, why = validate_checkpoint(p)
    if not ok:
        raise RuntimeError(f"cannot load checkpoint: {why}")
    payload = torch.load(p, map_location=map_location, weights_only=False)

    if model is not None:
        state = payload["model_state"]
        if any(k.startswith("_orig_mod.") for k in state):  # tolerate DDP/compile prefixes
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if missing or unexpected:
            log.warning("state_dict mismatch: missing=%s unexpected=%s", missing[:5], unexpected[:5])

    if optimizer is not None and "optimizer_state" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer_state"])
        except (ValueError, KeyError) as exc:
            log.warning("optimizer state incompatible (%s); starting with a fresh optimizer", exc)

    meta = {
        "step": payload.get("step", 0),
        "timestamp": payload.get("timestamp"),
        "model_config": payload.get("model_config"),
        "tokens_seen": payload.get("tokens_seen"),
        "best_val_loss": payload.get("best_val_loss"),
        "path": str(p),
    }
    for key in ("rng_state", "data_state", "config", "history", "generation"):
        if key in payload:
            meta[key] = payload[key]
    return meta


def compatibility_check(
    checkpoint_meta: dict,
    model_config: dict,
    tokenizer_vocab: int,
    data_fingerprint: Optional[str] = None,
) -> tuple[bool, list[str]]:
    """Refuse to resume a run whose definitions drifted apart."""
    problems: list[str] = []
    ck_cfg = checkpoint_meta.get("model_config") or {}
    for key in ("vocab_size", "d_model", "n_layers", "n_heads", "max_seq_len", "n_kv_heads", "d_ff"):
        want = model_config.get(key)
        have = ck_cfg.get(key)
        if want is not None and have is not None and want != have:
            problems.append(f"model.{key} differs: checkpoint={have} current={want}")
    if ck_cfg.get("vocab_size") and ck_cfg["vocab_size"] != tokenizer_vocab:
        problems.append(f"tokenizer vocab differs: checkpoint={ck_cfg['vocab_size']} current={tokenizer_vocab}")
    ck_fp = checkpoint_meta.get("data_fingerprint")
    if data_fingerprint and ck_fp and ck_fp != data_fingerprint:
        problems.append(f"packed data changed since the checkpoint (fingerprint {ck_fp[:12]} -> {data_fingerprint[:12]})")
    return (not problems), problems


def free_space_gb(path: str | os.PathLike[str]) -> float:
    """Free space on the volume holding ``path``.

    The directory itself may not exist yet - a Drive folder that has never
    been written to has no local path until it is created - so walk up to the
    nearest existing ancestor instead of raising.
    """
    p = Path(path)
    for candidate in (p, *p.parents):
        try:
            return shutil.disk_usage(str(candidate)).free / 1024**3
        except (FileNotFoundError, NotADirectoryError, OSError):
            continue
    return 0.0


def needs_space(path: str | os.PathLike[str], expected_checkpoint_bytes: int, headroom: float = 1.5) -> bool:
    """Refuse to start an expensive step that cannot be checkpointed."""
    return free_space_gb(path) * 1024**3 < expected_checkpoint_bytes * headroom