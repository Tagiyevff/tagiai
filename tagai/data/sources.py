"""Registry of candidate Azerbaijani corpora with explicit licence metadata.

Every entry records where the data comes from, what licence it carries and
whether TagAI may include it in the default training mix. Anything with an
unclear or non-redistributable licence is marked ``include_by_default=False``
and stays out of the corpus until a human reviews the terms.

Adding a source here is the only edit required to pull it into the pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..utils import get_logger

log = get_logger("data.sources")


@dataclass
class Source:
    """One candidate corpus."""

    id: str
    name: str
    language: str
    script: str
    url: str
    license: str
    license_url: str
    access: str  # direct_download | api | hf_dataset | manual
    include_by_default: bool
    notes: str = ""
    expected_documents: int | None = None
    expected_bytes: int | None = None
    enabled: bool = True
    tags: list[str] = field(default_factory=list)
    review_required: bool = False
    review_notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


SOURCES: list[Source] = [
    Source(
        id="azwiki",
        name="Azerbaijani Wikipedia",
        language="az",
        script="Latin",
        url="https://dumps.wikimedia.org/azwiki/latest/",
        license="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        access="direct_download",
        include_by_default=True,
        notes=(
            "Wikimedia XML dump. Article prose is CC BY-SA 4.0; user-interface chrome and "
            "short/low-content stubs are filtered out during cleaning."
        ),
        expected_documents=1_100_000,
        expected_bytes=900_000_000,
        tags=["encyclopedic", "general", "reference"],
    ),
    Source(
        id="azwiktionary",
        name="Azerbaijani Wiktionary",
        language="az",
        script="Latin",
        url="https://dumps.wikimedia.org/azwiktionary/latest/",
        license="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        access="direct_download",
        include_by_default=True,
        notes="Definitions, morphology and example sentences. Valuable for vocabulary coverage.",
        expected_documents=180_000,
        expected_bytes=250_000_000,
        tags=["lexical", "definitions"],
    ),
    Source(
        id="azwikisource",
        name="Azerbaijani Wikisource",
        language="az",
        script="Latin",
        url="https://dumps.wikimedia.org/azwikisource/latest/",
        license="CC BY-SA 4.0 / PD",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        access="direct_download",
        include_by_default=True,
        notes="Public-domain literary and historical Azerbaijani texts.",
        expected_documents=30_000,
        expected_bytes=90_000_000,
        tags=["literature", "public_domain"],
    ),
    Source(
        id="azwikiquote",
        name="Azerbaijani Wikiquote",
        language="az",
        script="Latin",
        url="https://dumps.wikimedia.org/azwikiquote/latest/",
        license="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        access="direct_download",
        include_by_default=True,
        notes="Quotations and proverbs; small but high-quality idiomatic material.",
        expected_documents=12_000,
        expected_bytes=15_000_000,
        tags=["idioms", "proverbs"],
    ),
    Source(
        id="azwikinews",
        name="Azerbaijani Wikinews",
        language="az",
        script="Latin",
        url="https://dumps.wikimedia.org/azwikinews/latest/",
        license="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        access="direct_download",
        include_by_default=True,
        notes="News-style journalistic Azerbaijani.",
        expected_documents=60_000,
        expected_bytes=120_000_000,
        tags=["news", "journalistic"],
    ),
    Source(
        id="azwikidata",
        name="Wikidata (Azerbaijani labels + descriptions)",
        language="az",
        script="Latin",
        url="https://www.wikidata.org/wiki/Wikidata:Database_download",
        license="CC0 1.0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        access="manual",
        include_by_default=False,
        review_required=True,
        review_notes="Wikidata labels are CC0, but the dump is ~100GB; extract the az labels before use.",
        tags=["entities", "lexical"],
    ),
    Source(
        id="mawdoo",
        name="Azərbaycan Wikipedia mirror (mawdoo)",
        language="az",
        script="Latin",
        url="https://dumps.wikimedia.org/azwiki/latest/",
        license="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        access="direct_download",
        include_by_default=False,
        review_required=True,
        review_notes="Site scraping mirrors Wikimedia content; prefer the official dump to avoid ToS issues.",
        tags=["encyclopedic"],
    ),
    Source(
        id="hf_oscar_az",
        name="OSCAR (az subset) via Hugging Face",
        language="az",
        script="Latin",
        url="https://huggingface.co/datasets/oscar-corpus/OSCAR-2301",
        license="ODC-By 1.0 (source-website licences vary)",
        license_url="https://opendatacommons.org/licenses/odc-by/1-0/",
        access="hf_dataset",
        include_by_default=False,
        review_required=True,
        review_notes=(
            "Aggregated web crawl; per-document licences are heterogeneous and the Azerbaijani "
            "slice is not small. Excluded until a licence review is completed."
        ),
        tags=["web", "noisy"],
    ),
    Source(
        id="common_crawl_az",
        name="Common Crawl (Azerbaijani filter)",
        language="az",
        script="Latin",
        url="https://commoncrawl.org/",
        license="per-subset, varies by crawl",
        license_url="https://commoncrawl.org/terms",
        access="manual",
        include_by_default=False,
        review_required=True,
        review_notes="Requires the CC index + substantial disk and a language model; excluded by default.",
        tags=["web", "noisy", "large"],
    ),
    Source(
        id="mc4_az",
        name="mC4 (az subset) via Hugging Face",
        language="az",
        script="Latin",
        url="https://huggingface.co/datasets/allenai/c4",
        license="ODC-By 1.0",
        license_url="https://opendatacommons.org/licenses/odc-by/1-0/",
        access="hf_dataset",
        include_by_default=False,
        review_required=True,
        review_notes="Cleaner than OSCAR but still web-derived; needs a licence review before inclusion.",
        tags=["web", "noisy"],
    ),
    Source(
        id="az_synthetic_templates",
        name="TagAI synthetic template corpus (generated locally)",
        language="az",
        script="Latin",
        url="local://tagai/data/synthetic",
        license="CC0 1.0 (generated by TagAI)",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        access="local",
        include_by_default=True,
        notes=(
            "Deterministic template-based Azerbaijani generation. No external model is queried, "
            "so it is reproducible and licence-clean. Kept in a separate stream and capped by "
            "data.synthetic_ratio."
        ),
        tags=["synthetic", "conversational", "grammar"],
    ),
]


class SourceRegistry:
    @staticmethod
    def all_sources() -> list[Source]:
        return list(SOURCES)

    @staticmethod
    def default_sources() -> list[Source]:
        return [s for s in SOURCES if s.include_by_default and s.enabled]

    @staticmethod
    def default_source_ids() -> list[str]:
        return [s.id for s in SourceRegistry.default_sources()]

    @staticmethod
    def all_source_ids() -> list[str]:
        return [s.id for s in SOURCES if s.enabled]

    @staticmethod
    def get(source_id: str) -> Source:
        for s in SOURCES:
            if s.id == source_id:
                return s
        raise KeyError(f"unknown source id {source_id!r}; known: {[s.id for s in SOURCES]}")

    @staticmethod
    def select(ids: list[str], allow_review_required: bool = False) -> list[Source]:
        out: list[Source] = []
        for sid in ids:
            src = SourceRegistry.get(sid)
            if not src.enabled:
                log.warning("source %s is disabled; skipping", sid)
                continue
            if src.review_required and not allow_review_required:
                log.warning(
                    "source %s requires licence review (enabled=%s); excluded from the default mix. "
                    "Pass allow_review_required=True to override.",
                    sid,
                    src.license,
                )
                continue
            out.append(src)
        return out

    @staticmethod
    def license_manifest() -> list[dict]:
        return [s.to_dict() for s in SOURCES]

    @staticmethod
    def eligible_manifest(ids: list[str] | None = None) -> dict:
        ids = ids or SourceRegistry.all_source_ids()
        chosen = SourceRegistry.select(ids)
        excluded = [SourceRegistry.get(i).to_dict() for i in ids if SourceRegistry.get(i) not in chosen]
        return {"eligible": [s.to_dict() for s in chosen], "excluded_pending_review": excluded}