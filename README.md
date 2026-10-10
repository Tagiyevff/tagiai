# TagAI

**An Azerbaijani (Azərbaycan dili) language model trained from randomly initialized weights.**

Project owner and developer: **Tagiyevff**
Primary development agent: **OpenCode**

TagAI learns Azerbaijani vocabulary, grammar, spelling, sentence structure and
idiom from raw text. Every trainable parameter is randomly initialised at the
start of every run. No pretrained weights, no LoRA, no fine-tuning of an
existing model.

---

## Status: what has actually been executed and verified

This section is deliberately precise about what ran on real hardware and what
did not. Nothing below is aspirational.

| Item | Status | Evidence |
|---|---|---|
| Source code (data, tokenizer, model, training, evaluation, export) | **Done** | `tagai/` |
| Automated test suite | **Done — 180 passing** | `python -m pytest` |
| End-to-end pipeline test (XML dump -> export) | **Done** | `tests/test_pipeline_e2e.py` |
| One-command real pipeline | **Done — 90 s on CPU** | `scripts/run_smoke_pipeline.py` |
| Real Azerbaijani Wikipedia download + parse | **Done** | 2 400 real articles fetched from the azwiki multistream dump |
| Real corpus: clean, language-ID, dedupe, split | **Done** | 2 311 articles in -> 590 documents out |
| Tokenizer trained on real Azerbaijani | **Done** | vocab 8 000, 64/64 alphabet letters, 100 % round-trip |
| Training from random weights on real text | **Done (CPU smoke)** | loss 8.92 -> 6.66 over 200 steps |
| Checkpoint save / load / resume | **Done** | verified in tests and on the real run |
| Text generation | **Done** | samples produced; see [docs/training_report.md](docs/training_report.md) |
| Evaluation + export bundle | **Done** | `export.zip`, 12.63 MB of weights |
| **Large-scale GPU pretraining** | **NOT DONE — requires your authorisation** | see [Run it on Colab](#run-it-on-colab) |

The last row is the important one. This machine has **no GPU**
(`nvidia-smi` not found; torch is CPU-only). The full training run must happen
on a GPU session that only you can start. Everything needed for that is built,
tested and waiting.

---

## Quick start (local, CPU, ~90 seconds)

One command runs the entire pipeline on a real sample of Azerbaijani Wikipedia:

```bash
pip install -r requirements.txt
python scripts/run_smoke_pipeline.py --root runs/demo
```

It fetches 2 400 real azwiki articles, cleans and deduplicates them, trains a
SentencePiece tokenizer, packs the tokens, trains from random weights,
generates text, evaluates, and exports a bundle. On a CPU this takes about 90
seconds and ends with a `runs/demo/export.zip` you can load.

The individual stages, if you prefer to run them yourself:

```bash
python -m tagai.cli --config configs/tagai_smoke.json --root runs/demo info     # hardware + resolved config
python -m tagai.cli sources                                                     # corpus licences
python -m tagai.cli --config configs/tagai_smoke.json --root runs/demo sample --streams 25
python -m tagai.cli --config configs/tagai_smoke.json --root runs/demo clean --min-chars 300
python -m tagai.cli --config configs/tagai_smoke.json --root runs/demo tokenize --vocab-size 8000
python -m tagai.cli --config configs/tagai_smoke.json --root runs/demo pack
python -m tagai.cli --config configs/tagai_smoke.json --root runs/demo train
python -m tagai.cli --config configs/tagai_smoke.json --root runs/demo generate
python -m tagai.cli --config configs/tagai_smoke.json --root runs/demo evaluate
python -m tagai.cli --config configs/tagai_smoke.json --root runs/demo export
```

---

## Storage budget (important on free tiers)

Google Drive's free tier is **15 GB for the whole account**. A single
`tagai-base` checkpoint is ~1.6 GB (weights + AdamW state), so naively keeping
three of them plus token shards fills it immediately. TagAI therefore uses two
separate stores:

| Store | What | Policy |
|---|---|---|
| Fast local disk (`/content` on Colab) | every `save_every` steps | keeps the last 3 |
| Durable (Google Drive) | **one** checkpoint | only when validation improves, never more often than `durable_min_steps_between`, and skipped if < `durable_min_free_gb` is free |

Verified behaviour on a real run: local 2 checkpoints (74 MB), durable exactly
1 (37 MB). Deleting the local disk and resuming picks the Drive copy back up
automatically.

| Setting | Default | Effect |
|---|---|---|
| `train.durable_dir` | `null` | set to a Drive path to enable, `null` to stay session-local |
| `train.durable_keep_last` | `1` | checkpoints held on the durable volume |
| `train.durable_on_improvement_only` | `true` | write only on a new best validation loss |
| `train.durable_min_steps_between` | `250` | minimum gap between durable writes |
| `train.durable_save_optimizer` | `true` | `false` cuts the file ~65 % (resume restarts AdamW) |
| `train.durable_min_free_gb` | `3.0` | refuse to write if the volume is nearly full |

The notebook's packaging step deliberately **excludes token shards** — they are
multi-GB, already live on the local disk, and can be regenerated. Only the
model bundle, tokenizer and reports go to Drive.

```bash
python -m tagai.cli --config configs/tagai_main.json --root runs/main \
    train --durable-dir /content/drive/MyDrive/tagai/checkpoints
```

---

## Run it on Colab (the real training run)

**You must do three things yourself.** The agent cannot and does not have
access to your Google account, a browser, or a GPU session.

1. **Push this repository to GitHub** under your account (`Tagiyevff/tagai`),
   or edit the `USER` variable in notebook cell 3.
2. **Open the notebook** — `notebooks/tagai_from_scratch_colab.ipynb`.
3. **Enable a GPU:** `Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU`,
   then `Runtime -> Run all`.

The notebook performs, in order: environment and GPU detection, dependency
installation, Drive mounting, source/licence inspection, resumable dataset
download, cleaning, deduplication, tokenizer training, packing, model creation
with random weights, parameter verification, smoke test, configuration
selection, pretraining with periodic checkpoints, validation, generation,
evaluation, export and reporting.

**If you prefer Kaggle**, use `notebooks/tagai_from_scratch_kaggle.ipynb`
instead. It drives the identical `tagai` package; the only differences are the
storage path and the GPU-detection code.

**Colab limits.** A free session is not an infinite machine. That is why
checkpoints are written atomically every `save_every` steps and the run
resumes automatically from the newest valid checkpoint — kill the session and
start a new one, and training continues.

---

## Repository layout

```
tagai/
  config.py              Paths/Data/Tokenizer/Model/Train configs + presets
  utils.py               seeding, atomic IO, retry with backoff, hardware probe
  cli.py                 every stage as a subcommand
  export.py              self-contained model bundle (+ safetensors + zip)
  model/                 decoder-only Transformer, all weights random
  tokenizer/             SentencePiece BPE training, encode/decode, Unicode checks
  data/                  sources, download, wiki parsing, clean, dedupe, langid,
                         synthetic generation, quality report, packing
  training/              AdamW + schedule, loop, atomic checkpoints, resume
  evaluation/            perplexity, Azerbaijani prompt suite, quality metrics
  inference/             sampling and text generation
configs/                 smoke / pilot / main run configurations
notebooks/               generated Colab and Kaggle notebooks
scripts/                 build_notebooks.py, run_smoke_pipeline.py,
                         build_training_report.py
tests/                   180 tests
docs/                    architecture, dataset provenance, licensing, reports
```

---

## Model

A decoder-only Transformer in plain PyTorch (`tagai/model/transformer.py`):

- RoPE positional encoding
- RMSNorm pre-norm blocks, optional LayerNorm
- SwiGLU feed-forward (optional GELU)
- Grouped-query attention (optional)
- Tied embeddings
- KV-cache generation with top-k / top-p / temperature / repetition penalty
- `scaled_dot_product_attention` with an eager fallback, verified to agree

### Configured sizes

| preset | d_model | layers | heads | context | vocab | parameters |
|---|---|---|---|---|---|---|
| `tagai-tiny` | 192 | 4 | 6 | 256 | 8 000 | ~3.3 M |
| `tagai-small` | 512 | 8 | 8 | 512 | 16 000 | ~45 M |
| `tagai-base` | 768 | 12 | 12 | 1024 | 32 000 | ~135 M |
| `tagai-350m` | 1024 | 24 | 16 | 2048 | 32 000 | ~340 M |

`configs/tagai_main.json` targets a 16 GB GPU (Colab T4 / Kaggle P100).
Larger presets exist but a bigger model is not automatically better: with a
free GPU, a model you can actually finish several epochs on is worth more than
one that barely moves.

---

## Proof that nothing pretrained is used

1. `TagAIModel.__init__` initialises every parameter from scratch. No code path
   in `tagai/` loads external weights; the only checkpoint reader is
   `tagai/training/checkpoint.py`, which loads TagAI's own checkpoints.
2. A freshly initialised model scores `ln(V)` on random next-token pairs
   (`test_initial_loss_is_near_uniform`). A pretrained model does not.
3. `test_passing_unshifted_targets_is_detected` proves the model cannot cheat
   by copying its input.
4. `tests/test_notebook_api.py` asserts the notebooks never reference
   `transformers`, `AutoModelForCausalLM` or any remote model id.
5. `export_bundle` writes `"pretrained_weights_used": false` into every
   exported manifest.

---

## Dataset and licensing

Sources, licences and inclusion decisions live in
[`tagai/data/sources.py`](tagai/data/sources.py) and are written into every run
manifest. **Included by default:** Azerbaijani Wikipedia, Wiktionary,
Wikisource, Wikiquote and Wikinews (all CC BY-SA 4.0 or public domain).

**Excluded pending licence review:** OSCAR, mC4, Common Crawl and the
Wikidata labels. They are not in the corpus and `tagai` will not add them
without an explicit review. See
[docs/dataset_provenance.md](docs/dataset_provenance.md).

Synthetic Azerbaijani is generated locally by template composition — no paid or
hosted model is called — and is capped by `data.synthetic_ratio` so it can
never dominate the corpus.

---

## Documentation

- [docs/architecture.md](docs/architecture.md) — model, tokenizer, data flow
- [docs/dataset_provenance.md](docs/dataset_provenance.md) — sources, licences, limitations
- [docs/licensing.md](docs/licensing.md) — licence obligations and exclusions
- [docs/training_report.md](docs/training_report.md) — the verified run, with real numbers
- [docs/not_covered.md](docs/not_covered.md) — what is explicitly *not* done yet
- [docs/environment.md](docs/environment.md) — verified software versions

---

## Tests

```bash
python -m pytest              # 180 tests
```

The suite covers the model (causality, RoPE, KV-cache equivalence, tied
embeddings, reproducibility), the tokenizer (64-letter alphabet coverage,
Unicode round-trip, fragmentation), the data pipeline (cleaning, Azerbaijani
language ID, deduplication, licence gating), training (schedule, checkpoint
corruption, resume compatibility), export (parameter counting with tied
embeddings), the notebooks (syntax, imports, no pretrained weights), and a
full end-to-end pipeline run.

Reproducible artefacts:

```bash
python scripts/build_notebooks.py                       # regenerate the notebooks
python scripts/build_training_report.py --root runs/real # regenerate the report from a run
```

---

## Licence

Code: MIT. Data: each source keeps its own licence — see
[docs/licensing.md](docs/licensing.md). Models trained on CC BY-SA corpora
inherit share-alike obligations for derivative datasets.