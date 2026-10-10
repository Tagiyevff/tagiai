"""Dataset discovery, collection, cleaning, deduplication and packaging."""

from .clean import CleanStats, char_stats, clean_document, normalize_unicode
from .dedupe import Deduplicator, exact_key, minhash, shingles
from .langid import detect_language, is_azerbaijani
from .pack import PackedDataset, BatchSampler, pack_corpus, read_shards, iter_windows
from .prepare import build_corpus, iter_corpus
from .quality import analyse as dataset_quality_report
from .sources import Source, SourceRegistry
from .synthetic import generate as generate_synthetic

__all__ = [
    "Source",
    "SourceRegistry",
    "CleanStats",
    "clean_document",
    "normalize_unicode",
    "char_stats",
    "Deduplicator",
    "exact_key",
    "minhash",
    "shingles",
    "detect_language",
    "is_azerbaijani",
    "build_corpus",
    "iter_corpus",
    "pack_corpus",
    "PackedDataset",
    "BatchSampler",
    "read_shards",
    "iter_windows",
    "dataset_quality_report",
    "generate_synthetic",
]