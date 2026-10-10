"""Model architecture for TagAI."""

from .transformer import (
    CausalSelfAttention,
    LMHead,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    TagAIModel,
    TransformerBlock,
    build_ffn,
    build_norm,
    precompute_rope_cache,
    verify_random_init,
)

__all__ = [
    "TagAIModel",
    "TransformerBlock",
    "CausalSelfAttention",
    "RotaryEmbedding",
    "RMSNorm",
    "SwiGLU",
    "LMHead",
    "build_ffn",
    "build_norm",
    "precompute_rope_cache",
    "verify_random_init",
]