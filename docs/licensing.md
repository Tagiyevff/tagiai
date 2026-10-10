# Licensing

This document states what TagAI may use, why, and what obligations come with
each source. It is intentionally conservative: **text that is publicly
accessible is not automatically licensed for reuse.**

## Code

The TagAI source code is MIT licensed. That covers `tagai/`, `configs/`,
`scripts/`, `tests/` and the notebooks. It does **not** cover the data.

## Data — included

| Source | Licence | Obligations |
|---|---|---|
| Azerbaijani Wikipedia (azwiki) | CC BY-SA 4.0 | Attribution + share-alike on derivatives |
| Azerbaijani Wiktionary | CC BY-SA 4.0 | same |
| Azerbaijani Wikisource | CC BY-SA 4.0 / public domain | same |
| Azerbaijani Wikiquote | CC BY-SA 4.0 | same |
| Azerbaijani Wikinews | CC BY-SA 4.0 | same |
| TagAI synthetic corpus | CC0 1.0 | none |

Wikimedia content is dual-licensed CC BY-SA 4.0 / GFDL; CC BY-SA 4.0 is used
here. **A model trained on CC BY-SA data is a derivative work.** Anyone
redistributing TagAI weights or a TagAI-trained corpus inherits the share-alike
requirement on that material. This is a real obligation, not a formality, and it
is the main reason the open-web corpora below stay excluded.

## Data — excluded, and why

These are registered in `tagai/data/sources.py` so the decision is auditable,
but `include_by_default=False` and `review_required=True`. The pipeline refuses
them and logs the exclusion into every run manifest.

| Source | Why excluded |
|---|---|
| OSCAR (az) | Aggregated Common Crawl. Per-document licences are heterogeneous; the Azerbaijani slice has no reliable filtering mechanism, and ODC-By covers the *dataset*, not each scraped page's copyright. |
| mC4 (az) | Same web-derivation problem as OSCAR. |
| Common Crawl | Licence varies per crawl; requires the full index and a large disk budget. |
| mawdoo (Wikipedia mirror) | Scraping a mirror can conflict with the host's terms. The official Wikimedia dump is used instead. |
| Wikidata labels | Labels are CC0, but the dump is ~100 GB and needs a dedicated extraction step before it can be reviewed. |

To include any of these you must first perform and document a licence review,
then set `include_by_default=True` and clear `review_required` for that entry.

## No scraping, no paywall bypass, no access-control circumvention

TagAI downloads only official, unauthenticated dump endpoints. It does not
bypass paywalls, logins, rate limits or access controls, and it sends an
identifying User-Agent. The multistream-index reader used by
`tagai.cli sample` issues ordinary HTTP Range requests against public files.

## Synthetic data

Generated locally by template composition in `tagai/data/synthetic.py`. No
hosted or paid model is called, so there is no third-party output licence to
inherit and no per-request cost. The output is released as CC0.

If model-generated Azerbaijani is ever added, the following must be documented
before it enters the corpus: which model produced it, under what licence the
model is available, whether its terms restrict the generated text, and what
quality-control process was applied. None of that is currently satisfied by a
model-generated source, which is why none is used.

## Personal information

Wikipedia and Wiktionary content is community-authored and can contain names
and contact details that were public. TagAI does not perform PII redaction
today. If the corpus is used for a published release rather than research, a
redaction pass over the prepared JSONL should be added first.

## What a user must do

If you release TagAI weights:

1. State which sources were used (the manifest in your run records this).
2. Apply CC BY-SA 4.0 to the dataset-derived material, or obtain permission.
3. Preserve attribution for Wikimedia content.

TagAI writes `sources_used`, `sources_excluded_pending_review` and the full
licence manifest into every run's `dataset_manifest.json` so this information
travels with the artefacts.