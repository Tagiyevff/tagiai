"""Tokenize documents and pack them into contiguous uint16/uint32 token streams.

Packing is what makes training fast: one flat array plus fixed-length windows
removes all padding waste, which matters when free GPUs are the constraint.

Documents are separated by EOS. Packing therefore never creates a training
example that spans a document boundary *as a target*: the model still sees
cross-document context (which is fine and even helpful), but each window's
loss covers only real tokens.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from ..config import PathsConfig
from ..utils import atomic_write_json, get_logger, iter_jsonl
from ..tokenizer.wrapper import TagAITokenizer

log = get_logger("data.pack")


@dataclass
class PackStats:
    documents: int = 0
    documents_empty: int = 0
    documents_truncated: int = 0
    tokens: int = 0
    tokens_truncated: int = 0
    seconds: float = 0.0
    shards: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "documents": self.documents,
            "documents_empty": self.documents_empty,
            "documents_truncated": self.documents_truncated,
            "tokens": self.tokens,
            "tokens_discarded_by_limit": self.tokens_truncated,
            "shards": len(self.shards),
            "seconds": round(self.seconds, 2),
        }


class TokenStreamWriter:
    """Append packed tokens to sharded ``.bin`` files with a metadata sidecar."""

    def __init__(self, out_dir: str | Path, prefix: str, dtype: str = "uint16", shard_tokens: int = 50_000_000) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.dtype = np.uint16 if dtype == "uint16" else np.uint32
        self.shard_tokens = shard_tokens
        self._fh = None
        self._count = 0
        self._shard_idx = 0
        self.shards: list[Path] = []
        self._open_shard()

    def _open_shard(self) -> None:
        self.close()
        self._path = self.out_dir / f"{self.prefix}_{self._shard_idx:05d}.bin"
        self._fh = open(self._path, "wb")
        self._count = 0
        self.shards.append(self._path)
        self._shard_idx += 1

    def write(self, ids: list[int]) -> None:
        if not ids:
            return
        arr = np.asarray(ids, dtype=self.dtype)
        if self._count + arr.size > self.shard_tokens and self._count > 0:
            self._open_shard()
        assert self._fh is not None
        arr.tofile(self._fh)
        self._count += arr.size

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    @property
    def total_tokens(self) -> int:
        return sum(p.stat().st_size // np.dtype(self.dtype).itemsize for p in self.shards if p.exists())

    def metadata(self) -> dict:
        return {
            "prefix": self.prefix,
            "dtype": np.dtype(self.dtype).name,
            "shard_tokens_limit": self.shard_tokens,
            "shards": [{"path": str(p), "tokens": p.stat().st_size // np.dtype(self.dtype).itemsize} for p in self.shards if p.exists()],
            "total_tokens": self.total_tokens,
        }


def read_shards(meta: dict, dtype: Optional[str] = None) -> Iterator[np.ndarray]:
    for shard in meta["shards"]:
        p = Path(shard["path"])
        if not p.exists():
            raise FileNotFoundError(f"token shard missing: {p}")
        arr = np.memmap(p, dtype=np.dtype(dtype or meta["dtype"]), mode="r")
        yield arr


def iter_windows(
    meta: dict,
    seq_len: int,
    *,
    dtype: Optional[str] = None,
    start_shard: int = 0,
    limit: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """Yield contiguous ``seq_len + 1`` token windows from the packed stream.

    Windows never straddle a shard boundary: a boundary means a new document
    region, so crossing it would fabricate tokens that were never written.
    """
    produced = 0
    for shard in meta["shards"][start_shard:]:
        p = Path(shard["path"])
        if not p.exists():
            raise FileNotFoundError(f"token shard missing: {p}")
        arr = np.memmap(p, dtype=np.dtype(dtype or meta["dtype"]), mode="r")
        usable = (len(arr) // (seq_len + 1)) * (seq_len + 1)
        for start in range(0, usable, seq_len + 1):
            yield np.asarray(arr[start : start + seq_len + 1])
            produced += 1
            if limit and produced >= limit:
                return


class PackedDataset:
    """In-memory / mmap window sampler over packed token shards."""

    def __init__(self, meta: dict, seq_len: int, dtype: Optional[str] = None, start_shard: int = 0) -> None:
        self.meta = meta
        self.seq_len = seq_len
        self.dtype = np.dtype(dtype or meta["dtype"])
        self.shards = meta["shards"][start_shard:]
        self.windows_per_shard = [int(s["tokens"] // (seq_len + 1)) for s in self.shards]
        self.total_windows = sum(self.windows_per_shard)
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.total_windows

    def _shard(self, index: int) -> np.ndarray:
        if index not in self._cache:
            if len(self._cache) > 2:  # small LRU; mmap handles are cheap but not free
                self._cache.clear()
            self._cache[index] = np.memmap(Path(self.shards[index]["path"]), dtype=self.dtype, mode="r")
        return self._cache[index]

    def get(self, window_index: int) -> np.ndarray:
        if not (0 <= window_index < self.total_windows):
            raise IndexError(window_index)
        shard_i = 0
        while window_index >= self.windows_per_shard[shard_i]:
            window_index -= self.windows_per_shard[shard_i]
            shard_i += 1
        arr = self._shard(shard_i)
        start = window_index * (self.seq_len + 1)
        return np.asarray(arr[start : start + self.seq_len + 1])


class BatchSampler:
    """Shuffled index batches over a :class:`PackedDataset`."""

    def __init__(self, dataset: PackedDataset, batch_size: int, seed: int = 1337, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def batches(self) -> Iterator[np.ndarray]:
        n = len(self.dataset)
        if n == 0:
            return
        order = self._rng.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n - self.batch_size + 1, self.batch_size):
            yield order[start : start + self.batch_size]

    def step(self) -> None:
        self.epoch += 1

    def reset(self) -> None:
        self.epoch = 0
        self._rng = np.random.default_rng(self.seed)


def pack_corpus(
    paths: PathsConfig,
    tok: TagAITokenizer,
    split: str = "train",
    *,
    max_doc_tokens: int = 2048,
    shard_tokens: int = 50_000_000,
    dtype: str = "uint16",
    append: bool = False,
    progress_every: int = 10_000,
    text_key: str = "text",
) -> tuple[dict, PackStats]:
    """Tokenize a prepared JSONL corpus into packed ``.bin`` shards."""
    from .prepare import PROCESSED_TRAIN, PROCESSED_VAL

    src = paths.processed / (PROCESSED_TRAIN if split == "train" else PROCESSED_VAL)
    if not src.exists():
        raise FileNotFoundError(f"corpus not built: {src}")

    out_dir = paths.processed / "tokens"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{split}"
    if not append:
        for old in out_dir.glob(f"{prefix}_*.bin"):
            old.unlink()
        for old in out_dir.glob(f"{prefix}_*.json"):
            old.unlink()

    started = time.time()
    stats = PackStats()
    writer = TokenStreamWriter(out_dir, prefix, dtype=dtype, shard_tokens=shard_tokens)
    batch: list[str] = []
    batch_tokens = 0
    max_batch_tokens = 2_000_000

    def encode_batch(texts: list[str]) -> None:
        nonlocal batch_tokens
        for ids in tok.encode_batch(texts, num_threads=8):
            if len(ids) > max_doc_tokens:
                stats.documents_truncated += 1
                stats.tokens_truncated += len(ids) - max_doc_tokens
                ids = ids[:max_doc_tokens]
            writer.write(ids)
            stats.tokens += len(ids)
            batch_tokens += len(ids)
            stats.documents += 1

    for row in iter_jsonl(src):
        text = row.get(text_key, "")
        if not text.strip():
            stats.documents_empty += 1
            continue
        batch.append(text)
        if len(batch) >= 1000:
            encode_batch(batch)
            batch = []
            if progress_every and stats.documents % progress_every < 1000:
                log.info("packed %d docs / %d tokens", stats.documents, stats.tokens)
    if batch:
        encode_batch(batch)

    writer.close()
    stats.shards = [str(p) for p in writer.shards]
    stats.seconds = time.time() - started

    meta = writer.metadata()
    meta.update(
        {
            "split": split,
            "seq_len_independent": True,
            "max_doc_tokens": max_doc_tokens,
            "vocab_size": tok.vocab_size,
            "eos_id": tok.eos_id,
            "bos_id": tok.bos_id,
            "pad_id": tok.pad_id,
            "stats": stats.to_dict(),
            "source_jsonl": str(src),
        }
    )
    atomic_write_json(out_dir / f"{prefix}_meta.json", meta)
    log.info("packed %s: %d documents -> %d tokens in %d shard(s)", split, stats.documents, stats.tokens, len(meta["shards"]))
    return meta, stats


def load_packed(paths: PathsConfig, split: str = "train", seq_len: Optional[int] = None) -> PackedDataset:
    meta_path = paths.processed / "tokens" / f"{split}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"packed tokens missing: {meta_path}")
    from ..utils import read_json

    meta = read_json(meta_path)
    if seq_len is None:
        raise ValueError("seq_len is required to build a PackedDataset (it defines the window size)")
    return PackedDataset(meta, seq_len)


def token_budget(meta: dict) -> int:
    return sum(s["tokens"] for s in meta["shards"])