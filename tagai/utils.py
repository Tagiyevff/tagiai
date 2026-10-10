"""Shared utilities: determinism, logging, atomic IO, retry with backoff."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar

import numpy as np

T = TypeVar("T")

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_stdio() -> None:
    """Force UTF-8 on stdout/stderr.

    Azerbaijani text (ə, ı, ö, ü, ğ, ş, ç) cannot be encoded by the default
    Windows console code page, so every progress line would raise
    UnicodeEncodeError and kill a long-running download or training job.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def setup_logging(level: int = logging.INFO, log_file: str | os.PathLike[str] | None = None) -> logging.Logger:
    """Configure root logging once and return the ``tagai`` logger."""
    root = logging.getLogger()
    if not root.handlers:
        handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
        if log_file is not None:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        logging.basicConfig(level=level, format=_LOG_FORMAT, handlers=handlers, force=True)
    if log_file is not None:
        logger = logging.getLogger("tagai")
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            logger.addHandler(logging.FileHandler(log_file, encoding="utf-8"))
    return logging.getLogger("tagai")


def get_logger(name: str = "tagai") -> logging.Logger:
    return logging.getLogger(name if name.startswith("tagai") else f"tagai.{name}")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch RNGs. Torch is imported lazily."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency for training
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict[str, Any]:
    """Snapshot python/numpy/torch RNG state so a run can resume bit-exactly."""
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_write_bytes(path: str | os.PathLike[str], payload: bytes) -> Path:
    """Write ``payload`` to ``path`` atomically (temp file + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_text(path: str | os.PathLike[str], text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike[str], obj: Any, indent: int = 2) -> Path:
    def default(o: Any) -> Any:
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Object of type {type(o)!r} is not JSON serializable")

    return atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent, default=default))


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    logger: logging.Logger | None = None,
    label: str = "operation",
) -> T:
    """Call ``fn`` with bounded exponential backoff on recoverable failures."""
    log = logger or get_logger("retry")
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except exceptions as exc:  # noqa: PERF203 - retry loop
            last = exc
            if attempt == attempts:
                break
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            jitter = delay * 0.1 * random.random()
            log.warning("%s failed (attempt %d/%d): %s - retrying in %.1fs", label, attempt, attempts, exc, delay + jitter)
            time.sleep(delay + jitter)
    assert last is not None
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last}") from last


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_text_lines(path: str | os.PathLike[str], skip_empty: bool = True) -> list[str]:
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if skip_empty and not line.strip():
                continue
            out.append(line.rstrip("\n"))
    return out


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def estimate_device_memory() -> dict[str, Any]:
    """Best-effort hardware description used by the config auto-scaler."""
    import torch

    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    else:
        import platform

        info["devices"].append({"index": 0, "name": platform.processor() or platform.machine(), "total_memory_gb": None})
    return info