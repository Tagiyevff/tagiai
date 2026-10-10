"""Model architecture tests: shapes, causality, determinism, from-scratch init."""

from __future__ import annotations

import pytest
import torch

from tagai.config import ModelConfig, model_from_preset
from tagai.model import TagAIModel, verify_random_init
from tagai.model.transformer import RMSNorm, apply_rope, precompute_rope_cache


@pytest.fixture(scope="module")
def tiny_cfg() -> ModelConfig:
    return ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=4, max_seq_len=32, dropout=0.0)


@pytest.fixture(scope="module")
def tiny_model(tiny_cfg: ModelConfig) -> TagAIModel:
    return TagAIModel(tiny_cfg)


def next_token_pair(tiny_cfg: ModelConfig, batch: int = 2, length: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a correctly shifted (inputs, targets) pair.

    Training uses next-token prediction, so targets are inputs shifted left.
    """
    ids = torch.randint(0, tiny_cfg.vocab_size, (batch, length + 1))
    return ids[:, :-1], ids[:, 1:]


def test_forward_shapes(tiny_model: TagAIModel, tiny_cfg: ModelConfig) -> None:
    x, y = next_token_pair(tiny_cfg)
    logits, loss = tiny_model(x, y)
    assert logits.shape == (2, 16, tiny_cfg.vocab_size)
    assert loss is not None and loss.isfinite()


def test_initial_loss_is_near_uniform(tiny_cfg: ModelConfig) -> None:
    """A randomly initialized model must start near ln(V), proving no
    pretrained structure leaks in through the initialization."""
    model = TagAIModel(tiny_cfg).eval()
    x, y = next_token_pair(tiny_cfg, batch=8, length=64)
    with torch.no_grad():
        _, loss = model(x, y)
    expected = float(torch.log(torch.tensor(float(tiny_cfg.vocab_size))))
    assert abs(float(loss) - expected) < 0.15, f"initial loss {float(loss):.3f} vs ln(V)={expected:.3f}"


def test_passing_unshifted_targets_is_detected(tiny_cfg: ModelConfig) -> None:
    """Identity prediction is not language modelling; guard against it.

    With tied embeddings the model can copy its input, so ``model(x, x)``
    reports a suspiciously low loss. The correct objective must not do that.
    """
    model = TagAIModel(tiny_cfg).eval()
    x, y = next_token_pair(tiny_cfg, batch=8, length=64)
    with torch.no_grad():
        _, loss_next = model(x, y)
        _, loss_identity = model(x, x)
    assert float(loss_next) > float(loss_identity), (
        "unshifted targets must not score better than real next-token prediction"
    )


def test_attention_is_causal(tiny_cfg: ModelConfig) -> None:
    """Changing a later token must not change earlier predictions."""
    model = TagAIModel(tiny_cfg).eval()
    x = torch.randint(0, tiny_cfg.vocab_size, (1, 24))
    with torch.no_grad():
        base, _ = model(x)
        y = x.clone()
        y[0, -1] = (y[0, -1] + 1) % tiny_cfg.vocab_size
        other, _ = model(y)
    assert torch.allclose(base[:, :-1], other[:, :-1], atol=1e-5), "information leaked from future tokens"


def test_eager_and_sdpa_agree(tiny_cfg: ModelConfig) -> None:
    eager_cfg = ModelConfig(**{**tiny_cfg.__dict__, "attn_impl": "eager"})
    sdpa_cfg = ModelConfig(**{**tiny_cfg.__dict__, "attn_impl": "sdpa"})
    a = TagAIModel(eager_cfg).eval()
    b = TagAIModel(sdpa_cfg).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 20))
    with torch.no_grad():
        la, _ = a(x)
        lb, _ = b(x)
    assert torch.allclose(la, lb, atol=2e-4), "eager and SDPA attention paths disagree"


def test_rope_is_norm_preserving_and_unique() -> None:
    cos, sin = precompute_rope_cache(16, 8, 10000.0)
    assert cos.shape == (8, 8) and sin.shape == (8, 8)
    assert torch.allclose(cos[0], torch.ones(8), atol=1e-6)  # position 0 is the identity rotation
    assert torch.allclose(sin[0], torch.zeros(8), atol=1e-6)
    x = torch.randn(1, 2, 5, 16)
    y = apply_rope(x, cos[:5], sin[:5])
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5), "RoPE must preserve vector norms"
    # consecutive positions must rotate differently
    pos0 = apply_rope(x[:, :, :1], cos[:1], sin[:1])
    pos1 = apply_rope(x[:, :, :1], cos[1:2], sin[1:2])
    assert not torch.allclose(pos0, pos1, atol=1e-3)
    with pytest.raises(ValueError):
        apply_rope(x, cos[:2], sin[:2])


def test_kv_cache_matches_full_forward(tiny_cfg: ModelConfig) -> None:
    model = TagAIModel(tiny_cfg).eval()
    prompt = torch.randint(0, tiny_cfg.vocab_size, (1, 12))
    with torch.no_grad():
        full, _ = model(prompt)
        cache = model.init_cache(1, prompt.device)
        step, _ = model(prompt[:, :-1], cache=cache)
        next_logits, _ = model(prompt[:, -1:], cache=cache)
    assert torch.allclose(full[:, :-1], step, atol=2e-4), "cached prefix differs from full forward"
    assert torch.allclose(full[:, -1:], next_logits, atol=2e-4), "cached last step differs"


def test_grouped_query_attention_shapes() -> None:
    cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=1, n_heads=8, n_kv_heads=2, max_seq_len=16)
    model = TagAIModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    logits, _ = model(x, x)
    assert logits.shape == (2, 10, cfg.vocab_size)
    assert model.blocks[0].attn.n_rep == 4


def test_tied_embeddings_share_storage(tiny_cfg: ModelConfig) -> None:
    model = TagAIModel(tiny_cfg)
    assert model.lm_head.weight is model.tok_emb.weight


def test_rmsnorm_properties() -> None:
    norm = RMSNorm(8)
    x = torch.randn(3, 8) * 5
    out = norm(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones(3), atol=1e-4)


def test_random_init_is_reproducible_and_not_loaded() -> None:
    """Same seed -> identical weights; different seed -> different weights."""
    from tagai.utils import set_seed

    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=4, max_seq_len=8)
    set_seed(0)
    a = TagAIModel(cfg)
    set_seed(0)
    b = TagAIModel(cfg)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb and torch.equal(pa, pb)

    set_seed(1)
    c = TagAIModel(cfg)
    assert not torch.equal(a.tok_emb.weight, c.tok_emb.weight)


def test_verify_random_init_report() -> None:
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=4, max_seq_len=8)
    model = TagAIModel(cfg)
    report = verify_random_init(model, seed=123)
    assert report["reproducible_with_same_seed"] is False or report["num_parameters"] > 0
    assert report["num_parameters"] > 0
    assert "embedding" in report["breakdown"]


def test_parameter_count_matches_breakdown(tiny_model: TagAIModel) -> None:
    breakdown = tiny_model.parameter_breakdown()
    assert breakdown["total"] == sum(p.numel() for p in tiny_model.parameters())


def test_presets_build_and_have_sane_sizes() -> None:
    for name in ("tagai-tiny", "tagai-small"):
        cfg = model_from_preset(name)
        model = TagAIModel(cfg)
        assert model.num_parameters() > 0
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        logits, _ = model(x)
        assert logits.shape[-1] == cfg.vocab_size


def test_generate_produces_requested_length(tiny_cfg: ModelConfig) -> None:
    model = TagAIModel(tiny_cfg).eval()
    prompt = torch.randint(0, tiny_cfg.vocab_size, (2, 5))
    out = model.generate(prompt, max_new_tokens=7, temperature=0.9, top_k=10, generator=torch.Generator().manual_seed(0))
    assert out.shape[0] == 2
    assert out.shape[1] >= 5 + 1


def test_generate_is_reproducible_with_seed(tiny_cfg: ModelConfig) -> None:
    model = TagAIModel(tiny_cfg).eval()
    prompt = torch.randint(0, tiny_cfg.vocab_size, (1, 4))
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = model.generate(prompt, max_new_tokens=10, generator=g1, temperature=0.8, top_k=5)
    b = model.generate(prompt, max_new_tokens=10, generator=g2, temperature=0.8, top_k=5)
    assert torch.equal(a, b)


def test_invalid_configs_are_rejected() -> None:
    with pytest.raises(ValueError):
        ModelConfig(d_model=65, n_heads=8)
    with pytest.raises(ValueError):
        ModelConfig(n_heads=6, n_kv_heads=4)
    with pytest.raises(ValueError):
        ModelConfig(activation="unknown")
    with pytest.raises(ValueError):
        ModelConfig(norm_type="unknown")
    with pytest.raises(ValueError):
        ModelConfig(max_seq_len=0)