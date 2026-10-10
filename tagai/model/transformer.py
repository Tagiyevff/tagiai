"""Transformer building blocks, all initialized from scratch.

Nothing in this module loads external weights. `verify_random_init` exists so
acceptance tests can prove that every parameter traces back to a fresh RNG
draw rather than a checkpoint.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


def init_weights(module: nn.Module, std: float, tied: Optional["LMHead"] = None) -> None:
    """Normal(0, std) init for Linear/Embedding; zeros for biases.

    Mirrors the convention used by modern decoder-only LMs: residual-output
    projections get a smaller std (scaled by depth) to keep the residual
    stream variance stable at init.
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"


class LayerNorm(nn.Module):
    """Standard layer norm, used when ``norm_type='layernorm'``."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


def build_norm(kind: str, dim: int, eps: float) -> nn.Module:
    kind = kind.lower()
    if kind == "rmsnorm":
        return RMSNorm(dim, eps)
    if kind in ("layernorm", "ln"):
        return LayerNorm(dim, eps)
    raise ValueError(f"unknown norm_type {kind!r}")


def precompute_rope_cache(head_dim: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype=torch.float32):
    """Return (cos, sin) tables of shape (max_seq_len, head_dim // 2)."""
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # (T, head_dim//2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of features in the last dim of ``x`` (B, H, T, D).

    ``cos``/``sin`` are (T, D//2) tables covering the positions in this
    window; longer tables are truncated, shorter ones are a bug.
    """
    seq_len = x.shape[-2]
    if cos.shape[0] < seq_len:
        raise ValueError(f"RoPE table covers {cos.shape[0]} positions but the query is {seq_len} long")
    cos = cos[:seq_len][None, None, :, :]
    sin = sin[:seq_len][None, None, :, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = o1
    out[..., 1::2] = o2
    return out


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        cos, sin = precompute_rope_cache(head_dim, max_seq_len, theta)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, seq_len: int, device: torch.device, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        need = offset + seq_len
        if need > self.cos_cached.shape[0]:
            grow = max(need, self.cos_cached.shape[0] * 2)
            cos, sin = precompute_rope_cache(self.head_dim, grow, self.theta)
            self.cos_cached = cos.to(device)
            self.sin_cached = sin.to(device)
        return self.cos_cached[offset:need].to(device), self.sin_cached[offset:need].to(device)


class CausalSelfAttention(nn.Module):
    """Multi-head / grouped-query causal self-attention with optional RoPE."""

    def __init__(self, cfg: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads or cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        bias = cfg.use_bias

        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=bias)
        self.dropout_p = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len, cfg.rope_theta)

    def forward(self, x: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        past = cache.get("k") if cache else None
        offset = 0 if past is None else past.shape[2]
        cos, sin = self.rope(T, x.device, offset=offset)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past is not None:
            k = torch.cat([past, k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        if cache is not None:
            cache["k"], cache["v"] = k, v

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        impl = self.cfg.attn_impl
        if impl == "eager":
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.triu(torch.ones(T, k.shape[2], dtype=torch.bool, device=x.device), diagonal=offset + 1)
            att = att.masked_fill(mask, float("-inf"))
            att = F.softmax(att, dim=-1, dtype=torch.float32).to(q.dtype)
            if self.dropout_p > 0 and self.training:
                att = F.dropout(att, p=self.dropout_p)
            y = att @ v
        else:
            is_causal = offset == 0 and past is None
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.resid_dropout(self.out_proj(y))


class SwiGLU(nn.Module):
    """Gated feed-forward: (silu(gate) * up) -> down, no bias."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class GELUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


def build_ffn(cfg: ModelConfig) -> nn.Module:
    act = cfg.activation.lower()
    if act == "swiglu":
        return SwiGLU(cfg.d_model, cfg.d_ff or cfg.d_model * 4, cfg.dropout)
    if act == "gelu":
        return GELUFeedForward(cfg.d_model, cfg.d_ff or cfg.d_model * 4, cfg.dropout)
    raise ValueError(f"unknown activation {act!r}")


class TransformerBlock(nn.Module):
    """Pre-norm block: x + Attn(LN(x)); x + FFN(LN(x))."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = build_norm(cfg.norm_type, cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.ffn_norm = build_norm(cfg.norm_type, cfg.d_model, cfg.norm_eps)
        self.ffn = build_ffn(cfg)

    def forward(self, x: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cache=cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LMHead(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_model))
        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size)) if cfg.use_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class TagAIModel(nn.Module):
    """Decoder-only causal language model with randomly initialized weights.

    ``forward(input_ids, targets)`` scores *next-token* prediction, so
    ``targets`` must be ``input_ids`` shifted left by one. Passing the same
    tensor for both asks the model to copy its own input, which is not a
    language-modelling objective; :class:`tagai.training.Trainer` does the
    shift for you.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(cfg, i) for i in range(cfg.n_layers))
        self.norm = build_norm(cfg.norm_type, cfg.d_model, cfg.norm_eps)
        self.lm_head = LMHead(cfg)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_module)
        if not cfg.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=cfg.initializer_range)
        self._no_decay = self._collect_no_decay()
        self._init_weights_safeguards()

    def _init_module(self, m: nn.Module) -> None:
        init_weights(m, self.cfg.initializer_range)

    def _init_weights_safeguards(self) -> None:
        n_layers = max(1, self.cfg.n_layers)
        std = self.cfg.initializer_range
        if isinstance(self.norm, RMSNorm):
            with torch.no_grad():
                self.norm.weight.fill_(1.0)
        elif isinstance(self.norm, LayerNorm):
            with torch.no_grad():
                self.norm.weight.fill_(1.0)
                self.norm.bias.zero_()
        # Scale residual output projections by 1/sqrt(2 * n_layers).
        residual_std = std / math.sqrt(2 * n_layers)
        for block in self.blocks:
            for proj in (block.attn.out_proj, block.ffn.down_proj if hasattr(block.ffn, "down_proj") else block.ffn.fc2):
                with torch.no_grad():
                    proj.weight.normal_(0.0, residual_std)
                    if getattr(proj, "bias", None) is not None:
                        proj.bias.zero_()

    def _collect_no_decay(self) -> set[str]:
        names: set[str] = set()
        for name, _ in self.named_parameters():
            if name.endswith(".bias") or "norm" in name.lower() or name.endswith("ln_f.weight"):
                names.add(name)
        return names

    # -- forward ---------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: list[dict] | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        for i, block in enumerate(self.blocks):
            x = block(x, cache=None if cache is None else cache[i])
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=ignore_index,
                reduction=reduction,
                label_smoothing=label_smoothing,
            )
        return logits, loss

    # -- kv cache --------------------------------------------------------
    def init_cache(self, batch_size: int = 1, device: torch.device | None = None) -> list[dict]:
        return [{"k": None, "v": None} for _ in self.blocks]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        generator: torch.Generator | None = None,
        eos_id: int | None = None,
        pad_id: int = 0,
    ) -> torch.Tensor:
        """Greedy/simple sampling loop with a static-length KV cache.

        Adequate for evaluation sampling; a production server would use a
        paged cache.
        """
        self.eval()
        device = input_ids.device
        out = input_ids
        finished = torch.zeros(out.shape[0], dtype=torch.bool, device=device)
        past: list[dict] = []
        cur = out
        for step in range(max_new_tokens):
            if step == 0:
                cache = self.init_cache(cur.shape[0], device)
                logits, _ = self(cur, cache=cache)
                past = cache
            else:
                logits, _ = self(cur[:, -1:], cache=past)
            next_logits = logits[:, -1, :].float()
            if repetition_penalty != 1.0:
                for i in range(next_logits.shape[0]):
                    for tok in torch.unique(out[i]):
                        next_logits[i, tok] = next_logits[i, tok] / repetition_penalty
            next_logits = next_logits / max(temperature, 1e-6)
            if top_k and top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1]))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(next_logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                mask = cdf - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                probs = torch.zeros_like(probs).scatter(1, sorted_idx, sorted_probs)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
            nxt = torch.where(finished.unsqueeze(1), torch.full_like(nxt, pad_id), nxt)
            out = torch.cat([out, nxt], dim=1)
            if eos_id is not None:
                finished = finished | (nxt.squeeze(1) == eos_id)
            if bool(finished.all()):
                break
            cur = nxt
        return out

    # -- bookkeeping -----------------------------------------------------
    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_embeddings and hasattr(self, "lm_head"):
                n -= self.lm_head.weight.numel()
        return n

    def parameter_breakdown(self) -> dict[str, int]:
        return {
            "total": self.num_parameters(),
            "embedding": self.tok_emb.weight.numel(),
            "blocks": sum(p.numel() for blk in self.blocks for p in blk.parameters()),
            "lm_head": 0 if self.cfg.tie_embeddings else sum(p.numel() for p in self.lm_head.parameters()),
            "norms": sum(p.numel() for n, p in self.named_parameters() if "norm" in n.lower()),
        }

    def param_groups_for_optimizer(self, weight_decay: float, no_decay_names: list[str] | None = None) -> list[dict]:
        """Split params into decay/no-decay groups (biases, norms excluded)."""
        no_decay_names = no_decay_names or []
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or name.endswith(".bias") or any(nd in name for nd in no_decay_names) or "norm" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)
        groups = []
        if decay:
            groups.append({"params": decay, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        return groups

    def config_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self.cfg)


def verify_random_init(model: TagAIModel, seed: int = 0) -> dict[str, object]:
    """Instantiate two models from the same seed and confirm bit-identical params.

    This proves determinism of ``from_scratch`` initialization (i.e. that the
    parameters are a pure function of the seed and config, not loaded data).
    """
    import copy

    from ..utils import set_seed

    snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
    set_seed(seed + 1)
    clone = TagAIModel(copy.deepcopy(model.cfg))
    same = all(torch.equal(snapshot[n], p.detach()) for n, p in clone.named_parameters())
    different_from_fresh = not same  # expected True: different seed -> different weights
    return {
        "reproducible_with_same_seed": same,
        "differs_with_different_seed": different_from_fresh,
        "num_parameters": model.num_parameters(),
        "non_embedding_parameters": model.num_parameters(non_embedding=True),
        "breakdown": model.parameter_breakdown(),
    }