"""Training tests: optimizer schedule, checkpoint recovery, end-to-end training."""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest
import torch

from tagai.config import ModelConfig, TrainConfig
from tagai.data.pack import BatchSampler, PackedDataset, TokenStreamWriter
from tagai.model import TagAIModel
from tagai.training.checkpoint import (
    compatibility_check,
    find_latest_valid,
    list_checkpoints,
    list_checkpoints_in,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
    write_manifest,
)
from tagai.training.optimizer import build_optimizer, build_param_groups, check_gradients, lr_at
from tagai.training.trainer import Trainer, load_model_from_checkpoint
from tagai.utils import set_seed


# ------------------------------------------------------------------ schedule
def test_lr_schedule_warms_up_then_decays() -> None:
    cfg = TrainConfig(lr=1e-3, warmup_steps=100, max_steps=1000, min_lr_ratio=0.1)
    assert lr_at(0, cfg) == pytest.approx(1e-5)
    assert lr_at(99, cfg) == pytest.approx(1e-3)
    assert lr_at(100, cfg) == pytest.approx(1e-3)
    assert lr_at(550, cfg) < cfg.lr
    assert lr_at(1000, cfg) == pytest.approx(1e-4, rel=1e-3)
    # Monotonically decreasing after warmup.
    values = [lr_at(s, cfg) for s in range(100, 1000, 50)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_lr_schedule_is_deterministic_across_resume() -> None:
    """The schedule is a closed-form function of step, so resuming matches."""
    cfg = TrainConfig(lr=1e-3, warmup_steps=100, max_steps=1000)
    saved = [lr_at(s, cfg) for s in range(0, 1000, 37)]
    rebuilt = TrainConfig(lr=1e-3, warmup_steps=100, max_steps=1000)
    assert saved == [lr_at(s, rebuilt) for s in range(0, 1000, 37)]


def test_param_groups_exclude_norms_and_biases() -> None:
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)
    model = TagAIModel(cfg)
    groups = build_param_groups(model, TrainConfig())
    decay = next(g for g in groups if g["weight_decay"] > 0)
    no_decay = next(g for g in groups if g["weight_decay"] == 0)
    assert all(p.ndim >= 2 for p in decay["params"])
    assert all(p.ndim < 2 for p in no_decay["params"])


def test_optimizer_counts_tied_weights_once() -> None:
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=4, max_seq_len=16)
    model = TagAIModel(cfg)
    groups = build_param_groups(model, TrainConfig())
    total = sum(len(g["params"]) for g in groups)
    unique = len({id(p) for g in groups for p in g["params"]})
    assert total == unique
    assert total == len(list(model.parameters()))


def test_check_gradients_clips_and_reports() -> None:
    model = torch.nn.Linear(4, 4)
    model.weight.grad = torch.full((4, 4), 10.0)
    diag = check_gradients(model, clip=1.0)
    assert diag.finite
    assert diag.grad_norm > 1.0
    assert model.weight.grad.norm().item() <= 1.0 + 1e-4


def test_check_gradients_sanitizes_nonfinite() -> None:
    model = torch.nn.Linear(4, 4)
    model.weight.grad = torch.full((4, 4), float("nan"))
    diag = check_gradients(model, clip=1.0)
    assert not diag.finite
    assert diag.nonfinite_params == 1
    assert torch.isfinite(model.weight.grad).all(), "non-finite grads must be sanitised, not propagated"


def test_build_optimizer_runs() -> None:
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=4, max_seq_len=16)
    opt = build_optimizer(TagAIModel(cfg), TrainConfig(lr=1e-3))
    assert isinstance(opt, torch.optim.AdamW)


# ------------------------------------------------------------------ checkpoints
@pytest.fixture
def tiny_run(tmp_path):
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)
    set_seed(0)
    model = TagAIModel(cfg)
    opt = build_optimizer(model, TrainConfig(lr=1e-3))
    return tmp_path, model, opt, cfg


def test_save_and_load_roundtrip(tiny_run) -> None:
    root, model, opt, cfg = tiny_run
    path = save_checkpoint(root, 10, model, opt, config=cfg.__dict__)
    assert path.exists()

    set_seed(99)
    fresh = TagAIModel(cfg)
    fresh_opt = build_optimizer(fresh, TrainConfig(lr=1e-3))
    meta = load_checkpoint(path, fresh, fresh_opt)
    assert meta["step"] == 10
    for (n1, p1), (n2, p2) in zip(model.state_dict().items(), fresh.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2)


def test_checkpoint_is_atomic(tiny_run) -> None:
    root, model, opt, cfg = tiny_run
    path = save_checkpoint(root, 5, model, opt, config=cfg.__dict__)
    leftovers = list((root / "checkpoints").glob("*.tmp"))
    assert leftovers == [], f"temporary files left behind: {leftovers}"
    assert (root / "checkpoints" / "latest.pt").exists()


def test_corrupted_checkpoint_is_detected(tiny_run) -> None:
    root, model, opt, cfg = tiny_run
    path = save_checkpoint(root, 7, model, opt, config=cfg.__dict__)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    ok, why = validate_checkpoint(path)
    assert not ok, "corruption must be detected"
    assert why


def test_truncated_checkpoint_is_detected(tiny_run) -> None:
    root, model, opt, cfg = tiny_run
    path = save_checkpoint(root, 9, model, opt, config=cfg.__dict__)
    path.write_bytes(path.read_bytes()[:100])
    ok, _ = validate_checkpoint(path)
    assert not ok


def test_find_latest_valid_skips_corrupted(tiny_run) -> None:
    root, model, opt, cfg = tiny_run
    for step in (1, 2, 3):
        save_checkpoint(root, step, model, opt, config=cfg.__dict__)
    newest = root / "checkpoints" / "step000003.pt"
    newest.write_bytes(b"not a torch file")
    found = find_latest_valid(root)
    assert found is not None
    assert found.name == "step000002.pt", f"expected fallback to the previous checkpoint, got {found}"


def test_retention_never_deletes_the_newest(tiny_run) -> None:
    root, model, opt, cfg = tiny_run
    for step in range(1, 11):
        save_checkpoint(root, step, model, opt, config=cfg.__dict__, keep_last=2)
    entries = list_checkpoints(root)
    assert entries[-1].step == 10, "the newest checkpoint was deleted"
    assert len(entries) <= 2


def test_free_space_handles_a_path_that_does_not_exist_yet(tmp_path) -> None:
    """A Drive folder that has never been written to has no local path.

    shutil.disk_usage raises FileNotFoundError on it, which is exactly what
    broke the first Colab run.
    """
    import shutil

    from tagai.training.checkpoint import free_space_gb

    missing = tmp_path / "does" / "not" / "exist" / "yet"
    assert not missing.exists()
    with pytest.raises((FileNotFoundError, OSError)):
        shutil.disk_usage(str(missing))

    free = free_space_gb(missing)
    assert free > 0.0, "free_space_gb must walk up to the nearest existing parent"


def test_free_space_matches_shutil_for_an_existing_path(tmp_path) -> None:
    from tagai.training.checkpoint import free_space_gb

    expected = shutil.disk_usage(str(tmp_path)).free / 1024**3
    assert abs(free_space_gb(tmp_path) - expected) < 0.01


def test_durable_directory_keeps_only_the_last(tmp_path, tiny_run) -> None:
    """A 15 GB Drive account cannot hold three full checkpoints."""
    root, model, opt, cfg = tiny_run
    durable = tmp_path / "drive"
    for step in range(1, 8):
        save_checkpoint(root, step, model, opt, config=cfg.__dict__, keep_last=3)
        save_checkpoint(root, step, model, opt, config=cfg.__dict__, keep_last=1, dest_dir=durable)
    kept = list_checkpoints_in(durable)
    assert len(kept) == 1, f"durable directory should hold one checkpoint, found {len(kept)}"
    assert kept[0].step == 7, "the durable copy must be the newest checkpoint"


def test_durable_directory_does_not_pollute_the_run(tmp_path, tiny_run) -> None:
    root, model, opt, cfg = tiny_run
    durable = tmp_path / "drive"
    save_checkpoint(root, 1, model, opt, config=cfg.__dict__)
    save_checkpoint(root, 2, model, opt, config=cfg.__dict__, dest_dir=durable)
    assert len(list_checkpoints(root)) == 1, "the durable save must not create a run-local checkpoint"
    assert (durable / "latest.pt").exists()


def test_durable_save_is_skipped_without_free_space(tmp_path, tiny_run) -> None:
    from tagai.training.trainer import Trainer

    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=4, max_seq_len=8)
    train_data = _make_dataset(tmp_path / "d")
    tcfg = TrainConfig(
        max_steps=1, micro_batch_size=2, grad_accum_steps=1, warmup_steps=1,
        eval_every=0, save_every=1, max_seq_len=8,
        durable_dir=str(tmp_path / "drive"),
        durable_min_free_gb=10_000.0,  # unreachable on purpose
    )
    trainer = Trainer(TagAIModel(cfg), train_data, None, tcfg, cfg, tmp_path / "run")
    trainer.state.best_val_loss = 1.0
    trainer.train()
    assert not (tmp_path / "drive").exists(), "durable save should have been refused for lack of space"
    assert trainer.storage_report()["local_checkpoints"] >= 1, "the local checkpoint must still exist"


def test_durable_save_skipped_when_validation_did_not_improve(tmp_path, tiny_run) -> None:
    from tagai.training.trainer import Trainer

    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=4, max_seq_len=8)
    train_data = _make_dataset(tmp_path / "d2")
    durable = tmp_path / "drive"
    tcfg = TrainConfig(
        max_steps=4, micro_batch_size=2, grad_accum_steps=1, warmup_steps=1,
        eval_every=0, save_every=1, max_seq_len=8,
        durable_dir=str(durable), durable_on_improvement_only=True, durable_min_steps_between=0,
    )
    trainer = Trainer(TagAIModel(cfg), train_data, None, tcfg, cfg, tmp_path / "run")
    trainer.state.best_val_loss = 5.0
    trainer.save(1)  # first durable write is allowed and records best_val = 5.0
    assert (durable / "step000001.pt").exists()
    trainer.save(2)  # best_val unchanged -> no new durable copy
    assert len(list_checkpoints_in(durable)) == 1


def test_resume_falls_back_to_the_durable_directory(tmp_path, tiny_run) -> None:
    """A fresh Colab session has no local disk; the Drive copy must be found."""
    from tagai.training.trainer import Trainer

    root, model, opt, cfg = tiny_run
    durable = tmp_path / "drive"
    save_checkpoint(root, 42, model, opt, config=cfg.__dict__, dest_dir=durable)

    tcfg = TrainConfig(max_steps=1, micro_batch_size=2, grad_accum_steps=1, warmup_steps=1,
                       eval_every=0, save_every=0, max_seq_len=8, resume="auto",
                       durable_dir=str(durable))
    trainer = Trainer(TagAIModel(cfg), _make_dataset(tmp_path / "d3", seq_len=8), None, tcfg, cfg, root)
    assert trainer.resolve_resume_path() is not None
    assert trainer.resolve_resume_path().parent == durable


def test_storage_report_reports_both_volumes(tmp_path, tiny_run) -> None:
    from tagai.training.trainer import Trainer

    root, model, opt, cfg = tiny_run
    durable = tmp_path / "drive"
    save_checkpoint(root, 1, model, opt, config=cfg.__dict__)
    save_checkpoint(root, 2, model, opt, config=cfg.__dict__, dest_dir=durable)
    tcfg = TrainConfig(max_steps=1, durable_dir=str(durable), max_seq_len=8)
    trainer = Trainer(TagAIModel(cfg), _make_dataset(tmp_path / "d4", seq_len=8), None, tcfg, cfg, root)
    report = trainer.storage_report()
    assert report["local_checkpoints"] == 1
    assert report["durable_checkpoints"] == 1
    assert report["local_gb"] >= 0.0 and report["durable_free_gb"] >= 0.0


def test_manifest_records_checkpoints(tiny_run) -> None:
    root, model, opt, cfg = tiny_run
    save_checkpoint(root, 4, model, opt, config=cfg.__dict__)
    p = write_manifest(root)
    assert p.exists()
    import json

    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["latest_step"] == 4
    assert data["checkpoint_count"] >= 1


def test_compatibility_check_rejects_drift() -> None:
    current = {"vocab_size": 100, "d_model": 64, "n_layers": 2, "n_heads": 4, "max_seq_len": 128, "n_kv_heads": 4, "d_ff": 128}
    same = dict(current)
    assert compatibility_check({"model_config": same}, current, 100, None)[0]

    changed = dict(current, d_model=128)
    ok, problems = compatibility_check({"model_config": same}, changed, 100, None)
    assert not ok and any("d_model" in p for p in problems)

    ok2, problems2 = compatibility_check({"model_config": same}, current, 200, None)
    assert not ok2 and any("tokenizer" in p for p in problems2)

    ok3, problems3 = compatibility_check({"model_config": same, "data_fingerprint": "aaa"}, current, 100, "bbb")
    assert not ok3 and any("packed data" in p for p in problems3)


def test_load_model_from_checkpoint_rebuilds_config(tiny_run) -> None:
    root, model, _, cfg = tiny_run
    path = save_checkpoint(root, 3, model, None, config=cfg.__dict__)
    rebuilt, rebuilt_cfg = load_model_from_checkpoint(path)
    assert rebuilt_cfg.d_model == cfg.d_model
    assert rebuilt_cfg.n_layers == cfg.n_layers
    assert torch.equal(rebuilt.tok_emb.weight, model.tok_emb.weight)


def test_exported_manifest_counts_tied_weights_once(tmp_path, az_corpus) -> None:
    """Tied embeddings appear twice in a state dict; the manifest must not
    report a parameter count larger than the model actually has."""
    from tagai.config import TokenizerConfig
    from tagai.export import export_bundle, load_bundle
    from tagai.inference import GenerateConfig
    from tagai.tokenizer.trainer import train_tokenizer

    tok_dir = tmp_path / "tok"
    tok_stats = train_tokenizer(
        az_corpus * 20, tok_dir,
        TokenizerConfig(vocab_size=400, num_threads=2, input_sentence_size=0),
        probe_texts=az_corpus,
    )

    cfg = ModelConfig(vocab_size=tok_stats.vocab_size, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)
    set_seed(0)
    model = TagAIModel(cfg)
    assert cfg.tie_embeddings
    path = save_checkpoint(tmp_path, 4, model, None, config=cfg.__dict__)

    manifest = export_bundle(path, tok_dir, tmp_path / "export")
    reported = manifest["model"]["parameters"]
    assert reported == model.num_parameters(), (
        f"manifest reports {reported} parameters but the model has {model.num_parameters()}"
    )
    # The bundle must still load and generate despite the omitted tied alias.
    bundle = load_bundle(tmp_path / "export", device="cpu")
    assert bundle.model.num_parameters() == model.num_parameters()
    text = bundle.generate("Azərbaycan", cfg=GenerateConfig(max_new_tokens=4, seed=0))
    assert isinstance(text, str) and "Azərbaycan" in text


# ------------------------------------------------------------------ packing
def test_packed_dataset_windows_are_correct(tmp_path) -> None:
    writer = TokenStreamWriter(tmp_path, "train", dtype="uint16", shard_tokens=1000)
    writer.write(list(range(1, 100)))
    writer.close()
    meta = writer.metadata()
    assert meta["total_tokens"] == 99

    ds = PackedDataset(meta, seq_len=8)
    assert len(ds) == 99 // 9
    w = ds.get(0)
    assert len(w) == 9
    assert list(w) == list(range(1, 10))
    assert list(ds.get(1)) == list(range(10, 19))


def test_windows_do_not_cross_shard_boundaries(tmp_path) -> None:
    writer = TokenStreamWriter(tmp_path, "train", dtype="uint16", shard_tokens=30)
    writer.write(list(range(1, 26)))
    writer.write(list(range(100, 125)))
    writer.close()
    ds = PackedDataset(writer.metadata(), seq_len=8)
    for i in range(len(ds)):
        w = list(ds.get(i))
        # a valid window is strictly increasing within a shard's own numbering
        assert w == sorted(w)


def test_batch_sampler_yields_batches(tmp_path) -> None:
    writer = TokenStreamWriter(tmp_path, "train", dtype="uint16", shard_tokens=10_000)
    writer.write(list(range(1, 500)))
    writer.close()
    ds = PackedDataset(writer.metadata(), seq_len=8)
    sampler = BatchSampler(ds, batch_size=4, seed=0)
    batches = list(sampler.batches())
    assert all(len(b) == 4 for b in batches)
    assert sum(len(b) for b in batches) <= len(ds)
    # different epochs shuffle differently
    sampler.step()
    assert not np.array_equal(np.array(batches[0]), np.array(list(sampler.batches())[0]))


# ------------------------------------------------------------------ end-to-end
def _markov_tokens(n_tokens: int, vocab: int, seed: int = 0) -> np.ndarray:
    """First-order Markov chain: token t+1 depends on token t, so there is real
    structure to learn. Uniform noise would let a model score no better than
    chance and would make the loss-decrease test meaningless."""
    rng = np.random.default_rng(seed)
    transitions = rng.random((vocab, vocab))
    transitions /= transitions.sum(axis=1, keepdims=True)
    seq = np.zeros(n_tokens, dtype=np.int64)
    seq[0] = rng.integers(0, vocab)
    # Vectorised sampling keeps 40k tokens fast.
    for i in range(1, n_tokens):
        seq[i] = rng.choice(vocab, p=transitions[seq[i - 1]])
    return seq


def _make_dataset(tmp_path, n_tokens=20_000, seq_len=16, vocab=64, pattern: bool = False, seed: int = 0):
    writer = TokenStreamWriter(tmp_path, "train", dtype="uint16", shard_tokens=1_000_000)
    rng = np.random.default_rng(seed)
    seq = _markov_tokens(n_tokens, vocab, seed) if pattern else rng.integers(0, vocab, size=n_tokens)
    writer.write(list(np.asarray(seq, dtype=np.uint16)))
    writer.close()
    ds = PackedDataset(writer.metadata(), seq_len)
    return BatchSampler(ds, batch_size=4, seed=0)


def test_tiny_training_run_completes_and_logs(tmp_path) -> None:
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)
    tcfg = TrainConfig(max_steps=12, micro_batch_size=4, grad_accum_steps=2, warmup_steps=3, log_every=2, eval_every=6, save_every=6, eval_batches=2)
    train_data = _make_dataset(tmp_path / "data")
    val_data = _make_dataset(tmp_path / "val")

    set_seed(0)
    model = TagAIModel(cfg)
    trainer = Trainer(model, train_data, val_data, tcfg, cfg, tmp_path / "run")
    state = trainer.train()

    assert state.step == 12
    assert state.tokens_seen > 0
    log = (tmp_path / "run" / "train_log.csv").read_text(encoding="utf-8")
    assert "step,loss" in log
    assert len([ln for ln in log.splitlines() if ln.strip()]) > 1


def test_training_loss_decreases(tmp_path) -> None:
    """The model must learn a real next-token distribution, not noise."""
    cfg = ModelConfig(vocab_size=16, d_model=32, n_layers=1, n_heads=4, max_seq_len=8)
    tcfg = TrainConfig(
        max_steps=120, micro_batch_size=16, grad_accum_steps=1, warmup_steps=10,
        lr=3e-3, log_every=1000, eval_every=0, save_every=0, max_seq_len=8,
    )
    train_data = _make_dataset(tmp_path / "d2", n_tokens=40_000, seq_len=8, vocab=16, pattern=True, seed=0)
    val_data = _make_dataset(tmp_path / "v2", n_tokens=8_000, seq_len=8, vocab=16, pattern=True, seed=0)
    set_seed(0)
    trainer = Trainer(TagAIModel(cfg), train_data, val_data, tcfg, cfg, tmp_path / "run2")
    before = trainer.evaluate(batches=8)
    trainer.train()
    after = trainer.evaluate(batches=8)
    assert np.isfinite(before) and np.isfinite(after), f"non-finite evaluation: {before} -> {after}"
    assert before > 0.5 * 3, f"the evaluation set must contain signal, got {before}"
    assert after < before, f"loss did not decrease: {before:.4f} -> {after:.4f}"


def test_evaluation_is_nan_without_validation_data(tmp_path) -> None:
    """Missing validation data is reported, not silently scored as 0."""
    cfg = ModelConfig(vocab_size=16, d_model=32, n_layers=1, n_heads=4, max_seq_len=8)
    tcfg = TrainConfig(max_steps=1, micro_batch_size=2, grad_accum_steps=1, warmup_steps=1, max_seq_len=8)
    trainer = Trainer(TagAIModel(cfg), _make_dataset(tmp_path / "dnv", seq_len=8, vocab=16), None, tcfg, cfg, tmp_path / "rnv")
    assert np.isnan(trainer.evaluate(batches=2))


def test_training_resumes_from_checkpoint(tmp_path) -> None:
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)
    train_data = _make_dataset(tmp_path / "data3")

    t1 = TrainConfig(max_steps=8, micro_batch_size=4, grad_accum_steps=1, warmup_steps=2, log_every=100, save_every=1000, eval_every=0)
    set_seed(0)
    trainer1 = Trainer(TagAIModel(cfg), train_data, None, t1, cfg, tmp_path / "run3")
    trainer1.train()
    trainer1.save()
    tokens_at_pause = trainer1.state.tokens_seen
    weight_at_pause = trainer1.model.tok_emb.weight.detach().clone()

    t2 = TrainConfig(max_steps=16, micro_batch_size=4, grad_accum_steps=1, warmup_steps=2, log_every=100, save_every=1000, eval_every=0, resume="auto")
    trainer2 = Trainer(TagAIModel(cfg), train_data, None, t2, cfg, tmp_path / "run3")
    resume_path = trainer2.resolve_resume_path()
    assert resume_path is not None
    step = trainer2.resume(resume_path)
    assert step == 8
    assert trainer2.state.tokens_seen == tokens_at_pause
    assert torch.equal(trainer2.model.tok_emb.weight, weight_at_pause), "resume did not restore the saved weights"
    trainer2.train()
    assert trainer2.state.step == 16
    assert trainer2.state.tokens_seen > tokens_at_pause


def test_resume_refuses_incompatible_checkpoint(tmp_path) -> None:
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)
    train_data = _make_dataset(tmp_path / "d4")
    tcfg = TrainConfig(max_steps=2, micro_batch_size=4, grad_accum_steps=1, warmup_steps=1, eval_every=0, save_every=1)
    set_seed(0)
    trainer = Trainer(TagAIModel(cfg), train_data, None, tcfg, cfg, tmp_path / "run4")
    trainer.train()
    trainer.save()

    other_cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, max_seq_len=16)
    tcfg2 = TrainConfig(max_steps=4, micro_batch_size=4, grad_accum_steps=1, warmup_steps=1, eval_every=0)
    trainer2 = Trainer(TagAIModel(other_cfg), train_data, None, tcfg2, other_cfg, tmp_path / "run4")
    with pytest.raises(RuntimeError, match="refusing to resume"):
        trainer2.resume(trainer2.resolve_resume_path())


def test_resume_none_starts_from_scratch(tmp_path) -> None:
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)
    tcfg = TrainConfig(max_steps=1, micro_batch_size=2, grad_accum_steps=1, warmup_steps=1, eval_every=0, save_every=0, resume="none")
    trainer = Trainer(TagAIModel(cfg), _make_dataset(tmp_path / "d5"), None, tcfg, cfg, tmp_path / "run5")
    assert trainer.resolve_resume_path() is None


def test_token_budget_stops_training(tmp_path) -> None:
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=4, max_seq_len=8)
    tcfg = TrainConfig(max_steps=10_000, max_tokens=512, micro_batch_size=4, grad_accum_steps=1, warmup_steps=2, eval_every=0, save_every=0)
    trainer = Trainer(TagAIModel(cfg), _make_dataset(tmp_path / "d6", seq_len=8), None, tcfg, cfg, tmp_path / "run6")
    state = trainer.train()
    assert state.tokens_seen < 10_000 * 8 * 4
    assert state.step < 10_000