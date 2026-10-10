"""Configuration objects for data, tokenizer, model and training stages.

Configs are plain dataclasses so they can be serialised to JSON and shipped
inside model checkpoints, which is what makes a run reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

# The 32-letter Azerbaijani Latin alphabet. These characters must survive the
# tokenizer intact: silently dropping ə or ı collapses two distinct Azerbaijani
# letters into one and makes the model unable to spell correctly.
AZERBAIJANI_ALPHABET_LOWER = "abcçdefgğhıijklmnoöpqrsştuüvyzəx"
AZERBAIJANI_ALPHABET_UPPER = "ABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVYZƏX"
AZERBAIJANI_SPECIAL_CHARS = "əıöüğşçƏİÖÜĞŞÇ"
AZERBAIJANI_ALPHABET = AZERBAIJANI_ALPHABET_LOWER + AZERBAIJANI_ALPHABET_UPPER
AZERBAIJANI_CHARSET = (
    AZERBAIJANI_ALPHABET_LOWER
    + AZERBAIJANI_ALPHABET_UPPER
    + "0123456789"
    + " .,;:!?()[]{}\"'`~@#$%^&*-+=_|<>/\\n\t"
)


@dataclass
class PathsConfig:
    """Filesystem layout. Every stage reads and writes only inside ``root``."""

    root: str = "runs/tagai"

    @property
    def root_path(self) -> Path:
        return Path(self.root)

    def _p(self, *parts: str) -> Path:
        return self.root_path.joinpath(*parts)

    # data
    @property
    def raw(self) -> Path:
        return self._p("data", "raw")

    @property
    def cache(self) -> Path:
        return self._p("data", "cache")

    @property
    def interim(self) -> Path:
        return self._p("data", "interim")

    @property
    def processed(self) -> Path:
        return self._p("data", "processed")

    # tokenizer
    @property
    def tokenizer_dir(self) -> Path:
        return self._p("tokenizer")

    # training
    @property
    def checkpoints(self) -> Path:
        return self._p("checkpoints")

    @property
    def logs(self) -> Path:
        return self._p("logs")

    @property
    def reports(self) -> Path:
        return self._p("reports")

    @property
    def samples(self) -> Path:
        return self._p("samples")

    @property
    def export(self) -> Path:
        return self._p("export")

    def ensure(self) -> "PathsConfig":
        for path in (
            self.raw,
            self.cache,
            self.interim,
            self.processed,
            self.tokenizer_dir,
            self.checkpoints,
            self.logs,
            self.reports,
            self.samples,
            self.export,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class DataConfig:
    """Corpus construction settings."""

    sources: list[str] = field(default_factory=lambda: ["azwiki"])
    language: str = "az"
    min_chars: int = 200
    max_chars: int = 50_000
    min_az_ratio: float = 0.70
    min_latin_ratio: float = 0.60
    max_repeat_ratio: float = 0.35
    dedupe_exact: bool = True
    near_dedupe: bool = True
    near_dedupe_threshold: float = 0.90
    near_dedupe_buckets: int = 64
    validation_fraction: float = 0.002
    validation_min_documents: int = 200
    shuffle_buffer: int = 20_000
    seed: int = 1337
    max_documents: int | None = None
    synthetic_ratio: float = 0.0
    # A requested source that yields nothing is almost always a missing or
    # unreadable dump; failing loudly beats silently building a smaller corpus.
    strict_sources: bool = True


@dataclass
class TokenizerConfig:
    """SentencePiece BPE settings for Azerbaijani Latin script."""

    vocab_size: int = 16_000
    model_type: str = "bpe"
    character_coverage: float = 0.9995
    input_sentence_size: int = 2_000_000
    shuffle_input_sentence: bool = True
    num_threads: int = 16
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    train_extremely_large_corpus: bool = False
    max_sentence_length: int = 4096
    # True keeps vocab_size exact, so the model's embedding matrix is known
    # before training starts. False lets SentencePiece emit a smaller vocabulary
    # when the corpus cannot support the requested size.
    hard_vocab_limit: bool = True


@dataclass
class ModelConfig:
    """Decoder-only Transformer. Every weight is randomly initialized."""

    name: str = "tagai-tiny"
    vocab_size: int = 16_000
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int | None = None  # None -> multi-head (equal to n_heads)
    d_ff: int | None = None  # None -> auto (SwiGLU: 8/3 * d_model, rounded)
    max_seq_len: int = 512
    dropout: float = 0.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    rope_theta: float = 10_000.0
    activation: str = "swiglu"  # swiglu | gelu
    norm_type: str = "rmsnorm"  # rmsnorm | layernorm
    attn_impl: str = "auto"  # auto | sdpa | eager
    initializer_range: float = 0.02
    use_bias: bool = False

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_ff is None:
            # SwiGLU keeps parameter count and quality closer to a 4x MLP at ~2/3 cost.
            self.d_ff = int(8 * self.d_model / 3)
            self.d_ff = 64 * max(1, round(self.d_ff / 64))
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.activation.lower() not in ("swiglu", "gelu"):
            raise ValueError(f"unknown activation {self.activation!r} (expected swiglu or gelu)")
        if self.norm_type.lower() not in ("rmsnorm", "layernorm", "ln"):
            raise ValueError(f"unknown norm_type {self.norm_type!r} (expected rmsnorm or layernorm)")
        if self.attn_impl.lower() not in ("auto", "sdpa", "eager"):
            raise ValueError(f"unknown attn_impl {self.attn_impl!r} (expected auto, sdpa or eager)")


@dataclass
class TrainConfig:
    """Optimisation, batching and checkpointing policy."""

    seed: int = 1337
    micro_batch_size: int = 8
    grad_accum_steps: int = 4
    max_seq_len: int = 512
    max_steps: int = 2_000
    max_tokens: int | None = None
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    betas_wd_exclude: list[str] = field(default_factory=lambda: ["bias", "norm"])
    precision: str = "auto"  # auto | bf16 | fp16 | fp32
    log_every: int = 20
    eval_every: int = 250
    eval_batches: int = 20
    save_every: int = 500
    keep_last_checkpoints: int = 3
    generate_every: int = 500
    num_generation_prompts: int = 8
    resume: str = "auto"  # auto | none | path
    out_dir: str | None = None
    benchmark_steps: int = 20
    save_cpu_optimizer_state: bool = True

    # -- storage policy -------------------------------------------------
    # Checkpoints are written to fast local disk on every save. The durable
    # copy (Google Drive on Colab, ~15 GB free) is a separate, much smaller
    # budget: only the newest checkpoint, only when validation improved.
    durable_dir: str | None = None
    durable_keep_last: int = 1
    durable_save_optimizer: bool = True
    durable_on_improvement_only: bool = True
    durable_min_steps_between: int = 250
    durable_min_free_gb: float = 3.0

    @property
    def effective_batch_tokens(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps * self.max_seq_len


# Named presets. Tiny exists purely to prove the pipeline end to end.
MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "tagai-tiny": dict(d_model=192, n_layers=4, n_heads=6, max_seq_len=256, vocab_size=8_000),
    "tagai-small": dict(d_model=512, n_layers=8, n_heads=8, max_seq_len=512, vocab_size=16_000),
    "tagai-base": dict(d_model=768, n_layers=12, n_heads=12, max_seq_len=1024, vocab_size=32_000),
    "tagai-350m": dict(d_model=1024, n_layers=24, n_heads=16, max_seq_len=2048, vocab_size=32_000),
}

TRAIN_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": dict(micro_batch_size=4, grad_accum_steps=2, max_seq_len=128, max_steps=60, warmup_steps=10, lr=3e-4, eval_every=30, save_every=30),
    "pilot": dict(micro_batch_size=16, grad_accum_steps=4, max_seq_len=512, max_steps=5_000, warmup_steps=300, lr=6e-4, eval_every=250, save_every=500),
    "main": dict(micro_batch_size=32, grad_accum_steps=8, max_seq_len=512, max_steps=60_000, warmup_steps=1_000, lr=1e-3, eval_every=500, save_every=1_000),
}


def model_from_preset(preset: str, **overrides: Any) -> ModelConfig:
    if preset not in MODEL_PRESETS:
        raise KeyError(f"unknown model preset {preset!r}; available: {sorted(MODEL_PRESETS)}")
    params = dict(MODEL_PRESETS[preset])
    params.update(overrides)
    params["name"] = params.get("name", preset)
    return ModelConfig(**params)


def train_from_preset(preset: str, **overrides: Any) -> TrainConfig:
    if preset not in TRAIN_PRESETS:
        raise KeyError(f"unknown train preset {preset!r}; available: {sorted(TRAIN_PRESETS)}")
    params = dict(TRAIN_PRESETS[preset])
    params.update(overrides)
    return TrainConfig(**params)


def config_to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj) if is_dataclass(obj) else dict(obj)


def save_config(path: str | Path, **configs: Any) -> Path:
    from .utils import atomic_write_json

    payload = {name: config_to_dict(cfg) for name, cfg in configs.items()}
    return atomic_write_json(path, payload)


def _apply_dict(obj: Any, values: dict[str, Any]) -> Any:
    valid = {f.name for f in fields(obj)}
    unknown = set(values) - valid
    if unknown:
        raise KeyError(f"unknown config keys for {type(obj).__name__}: {sorted(unknown)}")
    for key, value in values.items():
        setattr(obj, key, value)
    return obj


def load_config_file(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a config file. Comments are always stripped, so ``.json`` files
    may still carry ``//`` annotations."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    return json.loads(_strip_jsonc(text))


def _strip_jsonc(text: str) -> str:
    """Remove ``//`` and ``/* */`` comments while respecting string literals."""
    out: list[str] = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "/":
                while i < n and text[i] != "\n":
                    i += 1
                continue
            if nxt == "*":
                i += 2
                while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def build_config(raw: dict[str, Any], paths: PathsConfig | None = None) -> dict[str, Any]:
    """Turn a raw config dict into typed config objects."""
    from .data.sources import SourceRegistry  # local import to avoid cycles

    paths = paths or PathsConfig(**raw.get("paths", {}))
    out: dict[str, Any] = {"paths": paths}

    data_raw = dict(raw.get("data", {}))
    if "sources" not in data_raw:
        data_raw["sources"] = SourceRegistry.default_source_ids()
    elif "all" in data_raw["sources"]:
        data_raw["sources"] = SourceRegistry.all_source_ids()

    out["data"] = _apply_dict(DataConfig(), data_raw)
    out["tokenizer"] = _apply_dict(TokenizerConfig(), raw.get("tokenizer", {}))

    model_raw = dict(raw.get("model", {}))
    preset = model_raw.pop("preset", None)
    model_cfg = model_from_preset(preset) if preset else ModelConfig()
    out["model"] = _apply_dict(model_cfg, model_raw)

    train_raw = dict(raw.get("train", {}))
    preset = train_raw.pop("preset", None)
    train_cfg = train_from_preset(preset) if preset else TrainConfig()
    out["train"] = _apply_dict(train_cfg, train_raw)
    if out["train"].out_dir is None:
        out["train"].out_dir = str(paths.root_path / "checkpoints")
    if out["train"].max_seq_len != out["model"].max_seq_len:
        out["model"].max_seq_len = out["train"].max_seq_len
    return out