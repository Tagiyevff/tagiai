# TagAI architecture

This document describes what the code actually does. Every claim here is
backed by a test in `tests/`.

## 1. Data flow

```
sources.py     registry: what exists, under which licence, included or not
download.py    resumable HTTP, multistream index parsing, byte-range fetching
wikitext.py    streaming MediaWiki XML reader (namespace-agnostic)
clean.py       NFKC normalisation, markup removal, Azerbaijani-aware filters
langid.py      az / tr / ru / en discrimination
dedupe.py      exact blake2b keys + MinHash/LSH near-duplicate detection
synthetic.py   offline template-based Azerbaijani generation
prepare.py     orchestration -> train/validation JSONL + manifest
quality.py     measured dataset quality report
pack.py        tokenize -> uint16 shards -> fixed-length windows
```

Memory is flat regardless of corpus size. `prepare.iter_clean_documents` is a
generator chain and `write_splits` streams to disk; nothing holds the corpus in
RAM.

### Cleaning decisions and why

Azerbaijani Wikipedia wikitext is mostly not prose. What follows is what the
cleaner removes and the reason for each rule — most of these were found by
running the pipeline against real dump data and reading the samples.

| Removed | Reason |
|---|---|
| `{{...}}` templates, including nested | Infobox parameters (`\| url = ...`) are presentation, not language. Nesting is handled by a brace-balanced scanner because a regex cannot: `{{Infobox\|child={{T}}}}` breaks any non-greedy pattern. |
| `{\| ... \|}` tables, `\| key = value` rows | Table markup. |
| `[[File:...]]`, `[[Category:...]]` links | Their targets are page names, so unlinking them injects navigation junk (`Kateqoriya:İlin günləri`) into the training text. |
| `[[target\|label]]` | Keeps the target (a real article title), drops the label. |
| `'''bold'''`, `**bold**` | Markup, not punctuation. Leaving them teaches the model to emit stray stars. |
| URLs and e-mail addresses | Non-linguistic noise; long URLs also defeat the `too_short` filter. |
| HTML tags and comments | Residual markup. |
| Mojibake | Repaired via cp1254 → cp1252 → latin-1 in that order. cp1254 matters most: it is the Windows code page for Azerbaijani, so pages damaged by it carry `ə` and `ş`, and no other single-byte codec round-trips them. |

Azerbaijani namespace prefixes are filtered in their Azerbaijani spellings
(`Fayl:`, `Şablon:`, `Kateqoriya:`, `Vikipediya:`, `Danışma:`, `Qovluq:`, …).
Filtering only the English spellings silently admits every File and Category
page — a defect that was only visible after running against the real dump.

### Language identification

`langdetect` is used when installed, but the final decision is overridden by
character evidence: `ə` (U+0259) does not exist in modern Turkish orthography
at all, so its presence is decisive. Cyrillic is detected by script ratio
before any statistical guess. If `langdetect` is missing the character
classifier still works, so the pipeline runs offline.

### Deduplication

Exact duplicates use a normalised blake2b key: lowercase, digits and
punctuation removed, whitespace collapsed. Normalising away digits means
`Bakı, 15 — 2024!` and `Bakı 15 2024` collapse. Azerbaijani `ı` is *not*
folded to `u` — they are different letters, and conflating them would merge
genuinely different documents.

Near duplicates use character 5-gram MinHash with LSH banding (64
permutations, 32 bands at a configurable threshold). Character shingles need no
tokenizer and work on any language.

## 2. Tokenizer

SentencePiece BPE, trained on the cleaned training split only — never on
validation, so evaluation stays honest.

The alphabet guarantee is the important part. After training, every one of the
64 Azerbaijani letters is probed with `x<letter>x`. If any would encode to
`<unk>`, training is retried at `character_coverage=1.0`; if letters are still
missing, the run **raises**. A corpus with gaps in the alphabet produces a
model that cannot spell, and that must not be a silent failure discovered
months later.

`hard_vocab_limit=True` by default keeps `vocab_size` exact so the model's
embedding matrix is known before training starts. If the corpus cannot support
the requested size, SentencePiece's opaque error is translated into an
actionable message naming the maximum achievable size.

Measured on the real run: 8 000 pieces, 64/64 letters, 100 % round-trip,
2.93 characters per token.

## 3. Model

`tagai/model/transformer.py`, decoder-only:

```
tokens -> Embedding
  -> N x [ RMSNorm -> CausalSelfAttention(RoPE, GQA) -> + ]
          [ RMSNorm -> SwiGLU -> + ]
  -> RMSNorm -> LMHead (tied to the embedding)
```

- **RoPE** applied per head, cache grows on demand. The cache regrowth uses the
  configured `rope_theta`, not a hard-coded default.
- **Residual output projections** (`attn.out_proj`, `ffn.down_proj`) are
  initialised with `std / sqrt(2 * n_layers)` to keep residual-stream variance
  stable at initialisation.
- **Attention causality** is enforced by `is_causal` on the first forward pass
  and by full visibility over the cached prefix afterwards;
  `test_attention_is_causal` proves no future token can influence an earlier
  prediction.
- **`forward(input_ids, targets)` scores next-token prediction**, so `targets`
  must be `input_ids` shifted by one. `test_passing_unshifted_targets_is_detected`
  exists because `model(x, x)` looks natural, asks the model to copy its input,
  and reports a suspiciously low loss.

### Known limitation

`generate()` is a reference implementation: correct, with a KV cache, but it
re-prefills the full context per call in the cached path and is not tuned for
throughput. It is suitable for evaluation sampling, not for serving.

## 4. Training

- Objective: next-token cross-entropy on packed windows.
- Optimiser: AdamW, `betas=(0.9, 0.95)`, decoupled weight decay on 2-D weight
  matrices only. Tied parameters are counted once.
- Schedule: linear warmup then cosine decay to `min_lr_ratio * lr`, defined in
  closed form as a function of step. Because it is closed form, resuming
  reproduces the identical schedule without storing scheduler state.
- Gradient accumulation, gradient clipping, non-finite gradient detection
  (which aborts and checkpoints rather than writing a poisoned run).
- Precision: `auto` selects bf16 when supported, else fp16 with a GradScaler,
  else fp32. CPU always runs fp32.

### Packing

Documents are tokenized, terminated with EOS, and concatenated into `uint16`
shards. Training windows are `seq_len + 1` tokens long and never straddle a
shard boundary — crossing it would fabricate tokens that were never written.

### Evaluation honesty

`Trainer.evaluate` returns `NaN`, not `0.0`, when the validation set cannot
produce even one batch. Reporting `0.0` would be indistinguishable from a
perfect model; this was a real defect found when a smoke run produced a
validation split of two documents.

## 5. Checkpointing and recovery

Two separate stores with different budgets, because a free Google Drive account
holds 15 GB for the entire account and one `tagai-base` checkpoint is ~1.6 GB.

| Store | Trigger | Retention |
|---|---|---|
| Local (fast disk) | every `save_every` steps | `keep_last_checkpoints` (3) |
| Durable (Drive) | new best validation loss only | `durable_keep_last` (1) |

The durable copy additionally refuses to run more often than
`durable_min_steps_between` and skips entirely if less than
`durable_min_free_gb` remains on the volume. `Trainer.storage_report()` prints
the current footprint of both so the budget is never invisible.

`resolve_resume_path()` checks the local store first and falls back to the
durable directory, so a fresh session whose local disk was wiped continues from
the Drive copy. Token shards are never written to Drive: they are large, they
already live on the fast local disk, and they are regenerable.

Both stores share the same guarantees:

- Written to a `.tmp` file then `os.replace`d, so a crash never leaves a
  half-written checkpoint where a valid one was.
- A SHA-256 sidecar detects corruption on load.
- `latest.pt` is a **copy**, not a symlink: Colab and Drive mounts frequently
  break symlinks across filesystems.
- `find_latest_valid` walks backwards and skips anything that fails to load.
- Retention pruning never deletes the newest checkpoint.
- Resume performs a compatibility check **before touching any weights**, so a
  mismatched run fails with "refusing to resume: model.d_model differs…" rather
  than a wall of PyTorch shape errors.

## 6. Reproducibility

`set_seed` seeds python, numpy and torch (CUDA included) and enables
deterministic cuDNN. Checkpoints store RNG state for all three, so a resumed
run continues the same random stream. The train/validation split is a hash of
document content, not a position, so it is stable under corpus growth and
identical on every machine.