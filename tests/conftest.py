"""Shared fixtures.

`az_corpus` exists because training a SentencePiece tokenizer on text that
does not contain all 64 Azerbaijani letters fails by design (see
`tagai.tokenizer.trainer.train_tokenizer`). Writing that text by hand in every
test is error-prone, so it lives here once.
"""

from __future__ import annotations

import pytest

# Sentences chosen to cover the whole Azerbaijani Latin alphabet, the digits,
# and the punctuation the tokenizer must round-trip losslessly.
AZ_ALPHABET_SENTENCES = [
    "Azərbaycan əlifbası abcçdefgğhıijklmnoöpqrsştuüvyzə sıralaması ilə yazılır.",
    "Böyük hərflər ABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVYZƏ şəklində yazılır və otuz iki hərf var.",
    "Xəzər dənizi, Zaqatala, Quba, Qusar, İsmayıllı, Xaçmaz, Cəbrayıl, Ağdam, Tovuz və Füzuli rayonları.",
    "Jurnalistika, informasiya texnologiyaları, qərərar, təqdimat, memorandum və təqdimat sözleri.",
    "Azərbaycan Respublikası müstəqil dövlətdir və paytaxtı Bakı şəhəridir.",
    "Bakı şəhəri qərb-şərq istiqamətində uzanır, Xəzər dənizinə sahildir.",
    "Nizami Gəncəvi, Səməd Vurğun, Nəsimi və Fuzuli Azərbaycan ədəbiyyatının əsas sənətkarıdır.",
    "Qızıldaş yataqları Zaqatala, Quba və Tovuz rayonlarında yerləşir və relyef şəraiti ilə məşhurdur.",
    "12,75% (2024-cü il) — \"qeyd\" [1]; 3/4 = 0,75; 15 + 27 = 42! Necə? Yaxşı! Diqqət edin.",
    "Fayl adı: sənəd.pdf; e-poçt: numune@ornek.az; hər ikisi də vacibdir.",
    "İnsan özü üçün ən vacib məqsəd olan işi görəndə xoşbəxt olur, çünki əmək insana hörmət qazandırır.",
    "Müasir texnologiyalar süni intellekt sahəsində inqilab yaratdı və həyatımızı dəyişdi.",
]


@pytest.fixture(scope="session")
def az_corpus() -> list[str]:
    """Azerbaijani text covering the full alphabet, digits and punctuation."""
    return list(AZ_ALPHABET_SENTENCES)


@pytest.fixture(scope="session")
def az_training_corpus(az_corpus: list[str]) -> list[str]:
    """A repeated corpus large enough for a small SentencePiece vocabulary."""
    return az_corpus * 20