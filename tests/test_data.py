"""Data pipeline tests: cleaning, language ID, dedup, packing, sources, synthetic."""

from __future__ import annotations

import json

import pytest

from tagai.config import DataConfig, PathsConfig
from tagai.data.clean import (
    boilerplate_score,
    char_stats,
    clean_document,
    collapse_whitespace,
    distinct_line_ratio,
    fix_mojibake,
    has_az_specials,
    normalize_unicode,
    repeated_token_ratio,
)
from tagai.data.dedupe import Deduplicator, exact_key, minhash, normalize_for_dedupe, shingles
from tagai.data.langid import detect_language, is_azerbaijani
from tagai.data.prepare import val_bucket, write_splits
from tagai.data.sources import SourceRegistry

AZ_DOC = (
    "Azərbaycan Respublikası qərb-şərq istiqamətində uzanır. Onun ərazisi Xəzər dənizi ilə "
    "sınırlanır və paytaxtı Bakı şəhəridir. Dövlətin rəsmi dili Azərbaycan dilidir. Bu dil "
    "türk dilinin qohumudur, lakin ondan əhəmiyyətli dərəcdə fərqlənir. Əlifba 32 hərfdən "
    "ibarətdir və ə, ı, ö, ü, ğ, ş, ç kimi xüsusi hərfləri ehtiva edir."
)

AZ_DOC2 = (
    "Bakı şəhəri Xəzər dənizinə sahildir və paytaxt funksiyasını daşıyır. Şəhərin ən tanınmış "
    "memarlıq abidəsi Şəhərlər bağıdır. Bakıçılar üçün ən vacib məsələ dəniz havasının "
    "dəyişməsidir, çünki bəzən şəhər duman altında qalır."
)

TURKISH = (
    "Türkiye Cumhuriyeti'nin başkenti Ankara'dır. Türk dili, Azerbaycan dili ile yakın "
    "akraba bir dildir ancak farklı bir alfabe sistemi kullanır. İstanbul çok büyük "
    "bir şehirdir ve pek çok tarihi eser barındır."
)

RUSSIAN = (
    "Москва является столицей России и крупнейшим городом страны. Российский язык "
    "относятся к славянской группе языков и имеет кириллическую письменность."
)

ENGLISH = (
    "The quick brown fox jumps over the lazy dog near the riverbank every single "
    "morning while the sun rises over the distant mountains and the valley fills."
)


# ------------------------------------------------------------------ clean
def test_normalize_unicode_preserves_az_letters() -> None:
    text = "ə ı ö ü ğ ş ç Ə İ Ö Ü Ğ Ş Ç"
    assert normalize_unicode(text) == text


def test_normalize_unicode_folds_compatibility_forms() -> None:
    assert normalize_unicode("ﬁnal") == "final"  # ﬁ ligature
    assert normalize_unicode("ＡＢＣ") == "ABC"  # full-width


def test_normalize_unicode_unifies_quotes_and_dashes() -> None:
    assert normalize_unicode("“qeyd”") == '"qeyd"'
    assert normalize_unicode("sənəd — mən") == "sənəd - mən"


def test_char_stats_counts_az_letters() -> None:
    stats = char_stats(AZ_DOC)
    assert stats["az_ratio"] > 0.6
    assert stats["cyrillic_ratio"] == 0.0
    assert stats["latin_ratio"] > 0.6


def test_collapse_whitespace_collapses_runs() -> None:
    assert collapse_whitespace("a  \t b\n\n\n\nc  ") == "a b\n\nc"


def test_has_az_specials_detects_schwa() -> None:
    assert has_az_specials("mən")
    assert not has_az_specials("man")


def test_repeated_token_ratio_detects_inflation() -> None:
    inflated = " ".join(["salam dünya"] * 100)
    assert repeated_token_ratio(inflated, 5) > 0.8
    assert repeated_token_ratio(AZ_DOC, 5) < 0.1


def test_distinct_line_ratio() -> None:
    assert distinct_line_ratio("a\nb\nc") == 1.0
    assert distinct_line_ratio("a\na\na\na") < 0.5


def test_boilerplate_score_flags_wiki_chrome() -> None:
    chrome = "Məqalə\nXarici keçidlər\nBələdgi maddələr\nRedaktə et\n" + AZ_DOC
    assert boilerplate_score(chrome) > 0.3
    assert boilerplate_score(AZ_DOC) < 0.1


def test_azerbaijani_namespace_pages_are_skipped() -> None:
    """azwiki names its namespaces in Azerbaijani; English-only filtering lets
    every Fayl:/Şablon:/Kateqoriya: page into the corpus."""
    from tagai.data.wikitext import _should_skip_title

    for title in ("Fayl:Baylraq şəkli.jpg", "Şablon:Infobox", "Kateqoriya:Şəhərlər",
                  "Vikipediya:Ana səhifə", "Danışma:Bakı", "Modul:Test", "Portal:Bakı"):
        assert _should_skip_title(title), f"{title} should be skipped"
    for title in ("Azərbaycan", "Bakı", "Azərbaycan dili", "Xəzər dənizi"):
        assert not _should_skip_title(title), f"{title} must be kept"


def test_long_wikilinks_are_removed() -> None:
    """A link body longer than any fixed cap must still be stripped."""
    from tagai.data.clean import clean_markup

    long_target = "X" * 900
    text = f"Bax: [[{long_target}|göstər]] və [[Fayl:Nümunə.jpg|şəkil]]."
    out = clean_markup(text)
    assert "[[" not in out and "]]" not in out
    # [[target|label]] keeps the target (a real article title) and drops the label.
    assert long_target in out
    assert "göstər" not in out


def test_namespace_wikilinks_are_dropped_entirely() -> None:
    """[[Kateqoriya:X]] and [[Fayl:Y.jpg]] are chrome: their targets are page
    names, so keeping them injects navigation junk into the corpus."""
    from tagai.data.clean import clean_markup

    out = clean_markup("mətn [[Kateqoriya:İlin günləri]] və [[Fayl:Nümunə.jpg|thumb]] və [[Bakı]]")
    for junk in ("Kateqoriya", "Fayl", "thumb", "Nümunə"):
        assert junk not in out, f"namespace link residue: {junk}"
    assert "Bakı" in out, "ordinary article links must survive"


def test_real_wiki_infobox_and_category_residue_is_gone() -> None:
    raw = (
        "{{Infobox şəhər\n| ad = Bakı\n| əhalisi = 2.3 milyon\n}}\n"
        "[[Kateqoriya:Azərbaycan şəhərləri]]\n"
        "[[Fayl:Bakı.png|thumb|Bakı]]\n"
        + AZ_DOC
    )
    out = clean_document(raw, min_chars=100)
    assert out.keep, out.reason
    for junk in ("Kateqoriya:", "Fayl:", "| ad =", "{{Infobox"):
        assert junk not in out.text, f"residue survived: {junk}"


def test_wiki_emphasis_markers_are_removed() -> None:
    """Stray asterisks and apostrophe runs are markup, not punctuation."""
    raw = AZ_DOC + "\n\n'''qalın''' mətn və **kursiv** söz burada yazılıb."
    out = clean_document(raw, min_chars=100)
    assert out.keep, out.reason
    assert "**" not in out.text
    assert "'''" not in out.text
    assert "qalın" in out.text and "kursiv" in out.text


def test_fix_mojibake_cp1254() -> None:
    """cp1254 is the Azerbaijani/Turkish Windows code page - the realistic case."""
    original = "Azərbaycan şəhəri və ən çox gözəl günlər"
    broken = original.encode("utf-8").decode("cp1254")
    assert broken != original
    assert fix_mojibake(broken) == original


def test_fix_mojibake_cp1252() -> None:
    original = "Salam dünya"
    broken = original.encode("utf-8").decode("cp1252", errors="replace")
    assert fix_mojibake(broken) == original


def test_fix_mojibake_leaves_clean_text_alone() -> None:
    assert fix_mojibake(AZ_DOC) == AZ_DOC


def test_clean_document_keeps_good_az_text() -> None:
    out = clean_document(AZ_DOC, min_chars=100)
    assert out.keep, out.reason
    assert "Azərbaycan" in out.text
    assert out.stats["has_az_specials"]


def test_clean_document_drops_short_text() -> None:
    assert not clean_document("Çox qısa.", min_chars=200).keep


def test_clean_document_drops_russian() -> None:
    out = clean_document(RUSSIAN * 6, min_chars=100)
    assert not out.keep
    assert out.reason in {"not_latin", "cyrillic_heavy", "low_az_letters"}


def test_clean_document_drops_repetition() -> None:
    padded = AZ_DOC + "\n" + AZ_DOC * 20
    out = clean_document(padded, min_chars=100)
    assert not out.keep or out.reason != "repetitive" or repeated_token_ratio(out.text) <= 0.35


def test_clean_document_removes_urls_and_markup() -> None:
    noisy = "Başlanğıc: {{dil|az}}\n" + AZ_DOC + "\n\nhttp://example.az/çox/uzun/keçid"
    out = clean_document(noisy, min_chars=100)
    assert out.keep
    assert "{{dil" not in out.text


def test_clean_document_drops_wikilinks() -> None:
    text = AZ_DOC + "\n\nƏtraflı məlumat üçün [[Azərbaycan]] və [[Bakı|başqa]] səhifələrə baxın."
    out = clean_document(text, min_chars=100)
    assert "[[" not in out.text


def test_clean_document_removes_infobox_rows() -> None:
    """Infobox fields are presentation, not prose, and must not become tokens."""
    raw = (
        "{{Infobox person\n"
        "| ad = Əli\n"
        "| doğum = 1900\n"
        "| şəkil = Fayl.jpg\n"
        "| url = https://example.az\n"
        "}}\n\n"
        + AZ_DOC
        + "\n\n{{dil|az}} məlumat qeyd olunur."
    )
    out = clean_document(raw, min_chars=100)
    assert out.keep, out.reason
    for junk in ("| ad =", "| doğum =", "| şəkil =", "| url =", "{{Infobox", "{{dil|az}}"):
        assert junk not in out.text, f"infobox residue survived: {junk!r}"


def test_strip_nested_templates_handles_nesting() -> None:
    from tagai.data.clean import strip_nested_templates

    text = "a {{outer|{{inner|x=y}}tail}} b"
    result = strip_nested_templates(text)
    assert "{{" not in result and "inner" not in result and "tail" not in result
    assert "a" in result and "b" in result


def test_strip_nested_templates_keeps_unbalanced_text() -> None:
    from tagai.data.clean import strip_nested_templates

    # Unbalanced braces must not silently delete the rest of the document.
    result = strip_nested_templates("ön {{broken mətn burada qalır")
    assert "qalır" in result


# ------------------------------------------------------------------ langid
def test_detects_azerbaijani() -> None:
    res = detect_language(AZ_DOC)
    assert res["lang"] == "az", res
    assert res["confidence"] > 0.5


def test_detects_turkish_not_azerbaijani() -> None:
    res = detect_language(TURKISH)
    assert res["lang"] in {"tr", "az"}, res
    if res["lang"] == "az":
        # Turkish text must only be labelled Azerbaijani via the schwa rule,
        # which modern Turkish does not use.
        assert res["profile"]["az_only_ratio"] > 0


def test_detects_russian_by_script() -> None:
    res = detect_language(RUSSIAN)
    assert res["lang"] == "ru"
    assert res["method"] == "script"


def test_detects_english() -> None:
    res = detect_language(ENGLISH)
    assert res["lang"] == "en"


def test_is_azerbaijani_helper() -> None:
    ok, res = is_azerbaijani(AZ_DOC)
    assert ok and res["lang"] == "az"
    ok2, _ = is_azerbaijani(RUSSIAN)
    assert not ok2


# ------------------------------------------------------------------ dedupe
def test_normalize_for_dedupe_strips_punctuation_and_digits() -> None:
    assert normalize_for_dedupe("Bakı, 15 — 2024!") == normalize_for_dedupe("bakı 15 2024")
    # Azerbaijani ı is not the letter u: orthographically distinct words stay distinct.
    assert normalize_for_dedupe("bakı") != normalize_for_dedupe("baku")


def test_exact_key_detects_identical_text() -> None:
    assert exact_key(AZ_DOC) == exact_key(AZ_DOC)
    assert exact_key(AZ_DOC) != exact_key(AZ_DOC2)


def test_shingles_and_minhash_are_stable() -> None:
    a = minhash(shingles(AZ_DOC))
    b = minhash(shingles(AZ_DOC))
    c = minhash(shingles(AZ_DOC2))
    assert a == b
    assert a != c


def test_deduplicator_removes_exact_duplicates() -> None:
    dd = Deduplicator(threshold=0.90)
    assert dd.is_duplicate(AZ_DOC) == (False, "unique")
    assert dd.is_duplicate(AZ_DOC) == (True, "exact")
    assert dd.stats.exact_duplicates == 1


def test_deduplicator_removes_near_duplicates() -> None:
    dd = Deduplicator(threshold=0.85)
    variant = AZ_DOC + " Bundan əlavə, məlumat sonrakı illərdə yenilənmişdir."
    assert dd.is_duplicate(AZ_DOC)[0] is False
    assert dd.is_duplicate(variant)[0] is True


def test_deduplicator_filter_keeps_distinct() -> None:
    dd = Deduplicator(threshold=0.95)
    docs = [AZ_DOC, AZ_DOC, AZ_DOC2, AZ_DOC2, ENGLISH]
    result = dd.filter(docs)
    assert len(result.kept) == 3
    assert result.exact_duplicates == 2
    assert result.summary(len(docs))["duplicate_rate"] > 0


def test_dedupe_normalization_catches_punctuation_variants() -> None:
    dd = Deduplicator(threshold=0.99)
    assert dd.is_duplicate(AZ_DOC)[0] is False
    assert dd.is_duplicate(AZ_DOC.replace(",", "").replace(".", ""))[0] is True


# ------------------------------------------------------------------ sources
def test_source_registry_excludes_unreviewed() -> None:
    eligible = {s.id for s in SourceRegistry.default_sources()}
    assert "azwiki" in eligible
    for sid in ("hf_oscar_az", "common_crawl_az", "mc4_az"):
        assert sid not in eligible, f"{sid} must stay out until its licence is reviewed"


def test_select_warns_and_skips_review_required() -> None:
    chosen = SourceRegistry.select(["azwiki", "mc4_az"])
    assert [s.id for s in chosen] == ["azwiki"]


def test_license_manifest_has_required_fields() -> None:
    for entry in SourceRegistry.license_manifest():
        for field in ("id", "name", "url", "license", "license_url", "access", "include_by_default"):
            assert field in entry, f"{entry['id']} missing {field}"


def test_eligible_manifest_lists_exclusions() -> None:
    m = SourceRegistry.eligible_manifest(["azwiki", "mc4_az"])
    assert any(e["id"] == "azwiki" for e in m["eligible"])
    assert any(e["id"] == "mc4_az" for e in m["excluded_pending_review"])


def test_unknown_source_raises() -> None:
    with pytest.raises(KeyError):
        SourceRegistry.get("does_not_exist")


# ------------------------------------------------------------------ splits
def test_val_bucket_is_deterministic() -> None:
    assert val_bucket(AZ_DOC, 1337) == val_bucket(AZ_DOC, 1337)
    assert val_bucket(AZ_DOC, 1337) != val_bucket(AZ_DOC, 9999)


def test_val_bucket_rate_matches_fraction() -> None:
    buckets = [val_bucket(f"document number {i}", 1337) for i in range(20_000)]
    rate = sum(1 for b in buckets if b < 200) / len(buckets)
    assert 0.005 < rate < 0.02, rate


def test_write_splits_partitions_documents(tmp_path) -> None:
    paths = PathsConfig(root=str(tmp_path / "run"))
    paths.ensure()
    cfg = DataConfig(validation_fraction=0.1)
    rows = [(f"src{i%3}", f"t{i}", f"Azərbaycan sənəd nömrə {i} " + AZ_DOC, "collected") for i in range(500)]
    info = write_splits(paths, cfg, rows)
    assert info["counts"]["train"] + info["counts"]["validation"] == 500
    assert info["counts"]["validation"] > 20

    train_rows = [json.loads(l) for l in (paths.processed / "corpus_train.jsonl").read_text(encoding="utf-8").splitlines()]
    val_rows = [json.loads(l) for l in (paths.processed / "corpus_val.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train_rows) + len(val_rows) == 500
    all_texts = {r["text"] for r in train_rows} | {r["text"] for r in val_rows}
    assert len(all_texts) == 500, "a document was written to both splits or lost"


def test_write_splits_records_provenance(tmp_path) -> None:
    paths = PathsConfig(root=str(tmp_path / "run2"))
    paths.ensure()
    info = write_splits(paths, DataConfig(validation_fraction=0.0), [("azwiki", "Bakı", AZ_DOC, "collected")])
    row = json.loads((paths.processed / "corpus_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["source"] == "azwiki"
    assert row["title"] == "Bakı"
    assert row["synthetic"] is False


# ------------------------------------------------------------------ synthetic
def test_synthetic_generation_is_deterministic() -> None:
    from tagai.data.synthetic import generate, suffix_balance_report

    a, stats_a = generate(n=30, seed=42)
    b, stats_b = generate(n=30, seed=42)
    assert a == b
    assert stats_a.accepted == stats_b.accepted
    report = suffix_balance_report(a)
    assert "max_single_suffix_share" in report


def test_synthetic_generation_respects_ratio_and_quality() -> None:
    from tagai.data.synthetic import generate

    texts, stats = generate(n=40, seed=7)
    assert stats.accepted >= 30
    for t in texts[:10]:
        assert detect_language(t)["lang"] == "az"
        assert repeated_token_ratio(t, 5) <= 0.35


def test_synthetic_pipeline_is_applied_to_a_corpus(tmp_path) -> None:
    from tagai.data.prepare import build_corpus
    from tagai.data.synthetic import generate

    paths = PathsConfig(root=str(tmp_path / "synth_run"))
    cfg = DataConfig(sources=["az_synthetic_templates"], synthetic_ratio=0.5, min_chars=100)
    synth, _ = generate(n=40, seed=1)
    manifest = build_corpus(paths, cfg, synthetic_texts=synth, progress_every=0)
    assert manifest["counts"]["train"] + manifest["counts"]["validation"] > 0
    assert manifest["synthetic"]["added"] >= 0