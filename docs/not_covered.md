# Not covered / not completed

This file exists so nobody has to guess. Everything below is genuinely missing
from this repository as it stands.

## 1. Large-scale pretraining has not been run

**This is the single biggest gap.** The verified training run used the
`tagai-tiny` model (3.3 M parameters) on a 429 k-token sample for 200 steps on
a **CPU**. That proves the pipeline works. It does not produce a usable
language model.

Blocked on: a GPU session, which only the project owner can start.
Required: push this repo to GitHub, open
`notebooks/tagai_from_scratch_colab.ipynb`, enable a T4 GPU, run all.

Not attempted: any paid cloud instance, any subscription, any API. None are
available and none were requested.

## 2. No pretrained weights — this is intentional, not a gap

The specification requires from-scratch pretraining, so the following are
absent by design and were **not** used:

- no pretrained checkpoints of any kind
- no LoRA / QLoRA / adapters
- no fine-tuning of an existing model
- no distillation from an existing model
- no tokenizer borrowed from another model

`TagAIModel.__init__` initialises every parameter from scratch and there is no
code path in `tagai/` that loads external weights.

## 3. Sources not yet collected

Only `azwiki` has been exercised against live data. Registered, licence-cleared
but never downloaded or parsed in a verified run:

- `azwiktionary`, `azwikisource`, `azwikiquote`, `azwikinews`

Their collectors share the same code path as `azwiki` and the registry marks
them eligible, but claiming they work would be unverified. Run
`python -m tagai.cli fetch --source azwiktionary` to exercise one.

Excluded pending licence review (see `docs/licensing.md`): OSCAR, mC4, Common
Crawl, Wikidata labels, the mawdoo mirror.

## 4. Corpus gaps

- **No conversational or colloquial Azerbaijani.** The corpus is
  encyclopedic. Everyday speech, slang and informal writing are missing.
- **No instruction or QA data.** A licensed Azerbaijani instruction dataset has
  not been reviewed or incorporated.
- **No Cyrillic-script Azerbaijani.** Rejected by a >2 % Cyrillic filter.
- **No code-switching dataset.** Some code-switched text survives cleaning, but
  it is not curated.
- **No PII redaction.** See `docs/licensing.md`.

## 5. Synthetic generation is deliberately limited

Current synthetic data is template composition. It is reproducible, free and
licence-clean, but it is **stiff and repetitive by construction**. Measured
acceptance rate 17.6 % — most candidates are rejected as repetitive.

Not done: any model-generated synthetic corpus. Not claimed: native-speaker
verification of any synthetic text.

## 6. Evaluation limitations

- The 14-prompt evaluation suite is hand-written and small. It measures
  repetition, degeneracy and perplexity, **not** fluency.
- No native-speaker human evaluation has been performed.
- No downstream task benchmarks (HellaSwag-style cloze, MMLU-style questions)
  exist for Azerbaijani yet, so no such numbers are reported.
- Perplexity is not comparable across tokenizers with different vocabularies;
  bits-per-token and bits-per-character are reported alongside for that reason.

## 7. Engineering gaps

- **Generation throughput.** `TagAIModel.generate` uses a KV cache but is a
  reference implementation; no batching, no paged attention. Fine for
  evaluation, not for serving.
- **No ONNX/TensorRT export.**
- **Single-process only.** No DDP/FSDP, so training is limited to one GPU.
- **No resume test across a real process kill.** Resume is verified by loading
  a checkpoint in a new `Trainer`, not by SIGKILL-ing a live run.
- **`scripts/build_notebooks.py` embeds the notebook cells.** The notebooks
  call the real package rather than inlining the model, so they cannot drift
  in logic, but they are generated rather than hand-edited.

## 8. Resource figures not established

No measurement exists yet for:

- real azwiki throughput (articles/second) on the full ~563 000-article dump
- peak GPU memory for `tagai-base` at `micro_batch_size=32`
- tokens/second for each model preset on a T4 or P100
- how many epochs fit in a free Colab session

`configs/tagai_main.json` is sized by arithmetic and by the standard behaviour
of this architecture, not by a measured run. The notebook cell "Training
configuration selection" prints the computed batch size and epoch estimate so
you can sanity-check it before committing GPU hours.

## 9. Honest summary

**Done and verified:** the entire pipeline, end to end, on real Azerbaijani
Wikipedia data, from random weights, with tests.

**Not done:** the training run that would make TagAI actually useful. That
requires a GPU you control.