# Dataset provenance and licensing

Everything here is written from measured output of the pipeline, not from
estimates. Re-running `python -m tagai.cli report` regenerates
`runs/<root>/reports/dataset_quality.json` and `dataset_quality.md`.

## Sources included by default

| id | Source | Script | Licence | Included |
|---|---|---|---|---|
| `azwiki` | Azerbaijani Wikipedia | Latin | CC BY-SA 4.0 | yes |
| `azwiktionary` | Azerbaijani Wiktionary | Latin | CC BY-SA 4.0 | yes |
| `azwikisource` | Azerbaijani Wikisource | Latin | CC BY-SA 4.0 / public domain | yes |
| `azwikiquote` | Azerbaijani Wikiquote | Latin | CC BY-SA 4.0 | yes |
| `azwikinews` | Azerbaijani Wikinews | Latin | CC BY-SA 4.0 | yes |
| `az_synthetic_templates` | TagAI synthetic corpus | Latin | CC0 1.0 (generated locally) | yes, ratio-capped |

All Wikimedia content is also available under the GNU Free Documentation
License; CC BY-SA 4.0 is used as the stated term.

## Sources excluded pending licence review

These are registered so the decision is visible and reversible, but they are
**not** in the default corpus:

| id | Source | Why excluded |
|---|---|---|
| `hf_oscar_az` | OSCAR (az) | Aggregated web crawl. Per-document licences are heterogeneous and cannot be filtered reliably. |
| `mc4_az` | mC4 (az) | Web-derived; per-document provenance is not preserved through the filtering pipeline. |
| `common_crawl_az` | Common Crawl | Per-subset licence varies by crawl; requires the full index and substantial disk. |
| `mawdoo` | Azerbaijan Wikipedia mirror | Site scraping mirrors Wikimedia content and may conflict with the host's terms. The official dump is used instead. |
| `azwikidata` | Wikidata (az labels) | Labels are CC0, but the dump is ~100 GB and would need a dedicated extraction step. |

`SourceRegistry.select()` refuses these unless `allow_review_required=True` is
passed explicitly, and logs the exclusion. The decision and the reason are
copied into every run manifest under `licenses`.

## Measured results — real Azerbaijani Wikipedia

Produced by `python -m tagai.cli --config configs/tagai_smoke.json --root runs/real ...`.

### Corpus construction

| Measurement | Value |
|---|---|
| Articles fetched from the azwiki multistream dump | 2 400 |
| Articles reaching the cleaner | 2 311 |
| Articles retained from `azwiki` | 590 |
| Synthetic documents added | 17 (cap: 3 % of collected) |
| Total documents in corpus | 607 |
| Retention rate | 26.3 % |
| Cleaned characters | 2 406 894 |
| Language distribution | `az`: 591 (100 % of language-checked documents) |
| Mean Azerbaijani language-ID confidence | 0.9997 (min 0.90) |
| Train / validation split | 602 / 5 |

The language distribution covers the 591 collected documents. Synthetic
documents do not pass through `detect_language` — they are template-generated
and gated separately — so they are counted under their own source.

Drop reasons, measured over 2 311 articles:

| Reason | Documents |
|---|---|
| `too_short` | 1 642 |
| `too_long` | 49 |
| `cyrillic_heavy` | 17 |
| `not_latin` | 9 |
| `duplicate_lines` | 2 |
| `dup_near` | 1 |
| `boilerplate` | 1 |

A 71 % "too short" rate is expected for Wikipedia: most Azerbaijani pages are
stubs. Raising `data.min_chars` trades recall for cleaner prose; the default
is 250.

### Alphabet coverage

64/64 Azerbaijani letters appear in the corpus. Rarest letters, by share of
documents containing them:

| Letter | Documents |
|---|---|
| Ğ | 0.66 % |
| Ü | 32.6 % |
| J | 38.0 % |
| Z | 44.5 % |
| U | 55.0 % |
| Ç | 55.3 % |

This is a useful quality signal: `ğ` and `ü` are genuinely rare in Azerbaijani
prose, and a model that produces them at the frequency of English `th` would
be wrong.

### Tokenizer

| Measurement | Value |
|---|---|
| Vocabulary | 8 000 SentencePiece BPE pieces |
| Training documents | 602 |
| Training text | 2.71 MB |
| Azerbaijani alphabet coverage | 64/64 (1.0) |
| Unicode round-trip rate | 1.0 |
| Characters per token | 3.02 |

### Packed training data

| Split | Documents | Tokens |
|---|---|---|
| train | 602 | 429 500 |
| validation | 5 | 3 116 |

The validation split is small because the *whole* sample corpus is small. The
pipeline warns when it falls below `data.validation_min_documents` (200) rather
than pretending the held-out set is adequate. A full-dump run produces a
proportionally larger split.

## Known limitations of this corpus

These are real and are not fixed by this pipeline:

1. **Encyclopedic bias.** Wikipedia is reference text: dense, factual,
   declarative. Conversational Azerbaijani, slang, and informal writing are
   under-represented. The synthetic corpus partly offsets this but is
   template-based and therefore stiff.
2. **Limited dialectal and regional variety.** azwiki is predominantly written
   in the Baku standard register.
3. **No Cyrillic-script Azerbaijani.** The cleaner rejects documents with more
   than 2 % Cyrillic. Azerbaijani has a real Cyrillic-script history; including
   it is a deliberate future decision, not an oversight.
4. **No conversational or instruction data** from a licensed source has been
   reviewed yet.
5. **The web is absent.** The largest source of everyday Azerbaijani is the
   open web, but every accessible crawl corpus (OSCAR, mC4, Common Crawl) has
   unresolved per-document licensing and is therefore excluded.
6. **Statistics are computed on the sample, not the full dump.** The numbers
   above describe 2 400 articles, not azwiki's ~563 000.

## Synthetic data

`az_synthetic_templates` is generated locally by template composition in
`tagai/data/synthetic.py`. No external model is called, so the output is
reproducible from a seed, carries no licence encumbrance, and costs nothing.

Categories: conversation, grammar, vocabulary, spelling, reasoning, summary.

Quality gates applied to every generated document: Azerbaijani language ID,
5-gram repetition detection, duplicate rejection, minimum length. Measured
acceptance rate on a 500-document request: **17.6 %** — the gates reject
roughly five of every six candidates, mostly as repetitive. In the run above,
17 of 200 available synthetic documents were admitted under a 3 % cap.

**These are not native-speaker verified.** They have not been reviewed by a
native Azerbaijani speaker, and this repository makes no such claim.