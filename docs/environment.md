# Environment

Versions this repository was **actually verified against**. Cloud notebooks
ship different images; the table below records what was tested, and
`requirements.txt` carries lower bounds rather than exact pins so a fresh
image resolves something compatible.

## Verified locally

| Component | Version |
|---|---|
| OS | Windows 11 (10.0.26200), AMD64 |
| Python | 3.13.13 |
| PyTorch | 2.8.0+cpu |
| CUDA available | **no** (CPU-only torch build) |
| NumPy | 2.5.1 |
| SentencePiece | 0.2.2 |
| safetensors | 0.8.0 |
| langdetect | installed |
| pytest | 8.4.1 |

Python requirement: `>=3.9`. The code uses `X | None` annotations under
`from __future__ import annotations`, so it runs on 3.9+.

## Not yet verified

The following were **not** exercised and are stated as untested:

- Any CUDA / GPU execution. This machine has no GPU and torch is the CPU build.
  bf16 and fp16 autocast paths, the CUDA GradScaler, fused AdamW, and
  `torch.cuda` RNG capture are written but have not been run.
- Python 3.9, 3.10 and 3.11.
- Colab and Kaggle notebook images.

The first GPU run will exercise code that has never executed. Checkpoints are
written before anything is asserted, so a failure there is recoverable.

## Installing

```bash
pip install -r requirements.txt
```

Core: `torch`, `sentencepiece`, `numpy`, `tqdm`, `langdetect`, `safetensors`.
Optional: `matplotlib` (loss curves), `pytest` (tests).

`sentencepiece` and `torch` ship binary wheels for the common platforms; no
compiler is needed.

## Memory notes

`torch` is imported lazily inside functions in `tagai/utils.py` so that
`--help` and config resolution work without it. Everything else imports it at
module level and genuinely requires it.

The Windows console default code page cannot encode Azerbaijani letters.
`tagai.utils.configure_stdio()` forces UTF-8 on stdout/stderr, and
`tagai.cli.main` calls it before anything else. Without it, any run printing
Azerbaijani text raises `UnicodeEncodeError` — this is a real failure mode on
Windows, not a hypothetical one.

## Free-GPU notes

Measured on a free Colab T4 or Kaggle P100 (16 GB), `configs/tagai_main.json`
uses `micro_batch_size=32`, `grad_accum_steps=8`, `seq_len=512` for
`tagai-base`, an effective batch of 131 072 tokens per optimizer step.

Those numbers are arithmetic, not measurement. Before committing GPU hours,
run notebook cell 15 ("Training configuration selection"), which prints the
batch size chosen for the detected GPU and the implied epoch count.

If a run runs out of memory, halve `train.micro_batch_size` first and raise
`train.grad_accum_steps` to keep the effective batch unchanged.