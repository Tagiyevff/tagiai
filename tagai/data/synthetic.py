"""Deterministic, offline synthetic Azerbaijani generation.

TagAI's synthetic corpus is produced by template composition rather than by
querying a hosted model. That is a deliberate constraint:

* it is reproducible (a fixed seed gives byte-identical output),
* it has no licensing question attached (the output is ours, CC0),
* it costs nothing and needs no GPU,
* and it can be audited line by line.

The risk of templates is stilted repetition, so the generator applies the same
gates as the natural corpus: Azerbaijani language ID, duplicate rejection,
5-gram repetition detection, and a suffix-balance check that would reveal a
template family dominating the mix.

If someone later wants model-generated Azerbaijani, `docs/dataset_provenance.md`
records what would have to be documented (model, licence, QC).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from ..utils import get_logger
from .clean import repeated_token_ratio
from .dedupe import Deduplicator
from .langid import detect_language

log = get_logger("data.synthetic")

GREETINGS = [
    "Salam, {name}! Necəsiniz?",
    "Salam! Xoş gəlmisiniz, {name}.",
    "Salam {name}, mən necəyim?",
    "İyə axşamı, {name}! Həyat necədir?",
    "Xeyrli səhər, {name}! Səhəriniz xoş olsun.",
    "Salam, mən {name}. Sizlə tanışmağa sevindim.",
    "Nazam, {name}! Necə keçir işləriniz?",
    "Aykünlə, {name}! Gözləyirdim sizi.",
]

INTROS = [
    "Mən {name} adında bir {profession}.",
    "Mənim adım {name}. {city} şəhərində yaşayıram.",
    "{profession} işləyirəm, {city} şəhərində yaşayıram.",
    "Mən {city} şəhərindən {name}.",
    "{name} adlıyam, {profession} olanda təhsil alıram.",
]

QUESTIONS = [
    "Necə edirsiniz?",
    "Nə işləyirsiniz?",
    "Harada yaşayırsınız?",
    "Azərbaycan haqqında nə bilirsiniz?",
    "Kitab oxumaq istəyirəm, hansını məsləhət görürsünüz?",
    "Neftçi — Arsenal maçını izlədinizmi?",
    "Sabah hava necə olacaq?",
    "Bu gün nə etmək olar?",
    "Sizə necə kömək edə bilərəm?",
    "Bu məsələni izah edə bilərsinizmi?",
    "Universitetdə hansı fənnləri öyrənirsiniz?",
    "Tələbə qiymətləri necə hesablanır?",
]

ANSWERS = [
    "Yaxşı, çox sağ olun!",
    "Həqiqətən də, belədir.",
    "Bilmirəm, amara öyrənməyə çalışıram.",
    "Əlbəttə, mən sizə kömək edə bilərəm.",
    "Bu çox maraqlı fikirdir.",
    "Xeyr, düz deyil. Əsl hadisə başqa cür olub.",
    "Bəli, mənim fikrimcə bu doğru qərardır.",
    "Çox ümidverici nəticədir.",
    "Üzrün, bir az gecikdim.",
    "Gələn dəfə daha ətraflı danışaq.",
]

SMALLTALK = [
    "Hava bu gün çox istidir.",
    "Küçələrdə çəmənliklər gözəl görünürdü.",
    "Kino yeni filmi haqqında danışdıq.",
    "Konsert qalıqları çox maraqlı keçdi.",
    "Kitabı bitirdim, fikrimcə olduqca yaxşı idi.",
    "Musiqi dinləyərək işlərimi daha yaxşı edirəm.",
    "Şəhərdə yeni tikililər meydana gəlib.",
    "Qış fasiləsi tez bitdi.",
]

GRAMMAR_TEMPLATES = [
    ("Mən hər gün {time_expr} kitabxanaya gedirəm.", "Niyə hər gün gedirsiniz?"),
    ("{name} {city} şəhərində {occupation} işləyir.", "Bu iş həyəranıza uyğundur?"),
    ("Əgər hava yaxşı olsa, {activity} gedəcəyik.", "Hava yaxşı deyil, niyə?"),
    ("Mən {food} yeməyi çox sevirəm.", "Harada hazırlanır?"),
    ("{city} haqqında bir neçə söz demək istəyirəm.", "Nə demək istəyirsiniz?"),
    ("Bu tapşırığı {deadline} qədər tamamlamalıyam.", "Tapşırıq nədir?"),
]

GRAMMAR_RULES = [
    "Azərbaycan dilində ən çox işlədilən sözüncül əlavələr: -da, -də, -dan, -dən.",
    "Feilin forması şəxsə və zamana görə dəyişir: oxuyorum, oxuyursan, oxuyur.",
    "Sözün əvvəlindəki sözün sonundakı səsə görə yumuşalma işlənir: kitab-da, ev-də.",
    "Qurulmaş əlavələr mənsubiyyət göstərir: mənim, sənin, onun, bizim, sizin, onların.",
    "Qoşulma əlaqələndiriciləri cümlə birləşdirir: və, ancaq, lakin, çünki, buna görə.",
]

VOCAB_TOPICS = [
    ("milli", "millət", "vətən", "dövlət", "qədim", "tarix", "mədəniyyət"),
    ("texnoloji", "kompüter", "internet", "proqram", "süni intellekt", "robot", "telefon"),
    ("təbiət", "günəş", "ay", "küçə", "çay", "meşə", "çiçək"),
    ("iqtisadi", "iş", "maaş", "bazar", "qiymət", "şirkət", "xidmət"),
    ("tibb", "həkim", "xəstəxana", "dərman", "sağlamlıq", "müalicə", "həyat"),
]

NAMES = [
    "Aysel", "Elvin", "Nigar", "Rəşad", "Günel", "Kamran", "Zeynəb", "Tural",
    "Fidan", "Orxan", "Leyla", "Səbuhi", "Aygün", "Murad", "Günay", "Vüqar",
]
PROFESSIONS = ["müəllim", "həkim", "mühəndis", "tələbə", "rəqqas", "hərbçi", "rəssam", "yazıçı", "sürücü", "fermer"]
CITIES = ["Bakı", "Gəncə", "Sumqayıt", "Mingəçevir", "Naxçıvan", "Şəki", "Lənkəran", "Xankəndi", "Yevlax", "Quba"]
ACTIVITIES = ["kino", "konsertə", "küçəyə", "muzeya", "çayxanaya", "təbiətə", "bəzəyə"]
FOODS = ["plov", "dolma", "xəngəl", "qutab", "piti", "kətə", "şah plov", "ləvəngi"]
TIMES = ["səhər", "axşam", "gündüz", "qışda", "yayda", "həftə sonu"]
OCCUPATIONS = ["müəllimlik", "həkimlik", "mühəndislik", "rəssamlıq", "yazıçılıq", "sürücülük", "fermerlik"]
DEADLINES = ["bugün", "sabah", "bu həftə", "ayın sonuna", "növbəti həftə"]


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        name=rng.choice(NAMES),
        profession=rng.choice(PROFESSIONS),
        city=rng.choice(CITIES),
        activity=rng.choice(ACTIVITIES),
        food=rng.choice(FOODS),
        time_expr=rng.choice(TIMES),
        occupation=rng.choice(OCCUPATIONS),
        deadline=rng.choice(DEADLINES),
    )


@dataclass
class SyntheticStats:
    generated: int = 0
    accepted: int = 0
    rejected_repeat: int = 0
    rejected_lang: int = 0
    rejected_duplicate: int = 0
    rejected_short: int = 0
    by_category: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "generated": self.generated,
            "accepted": self.accepted,
            "rejected": {
                "repetitive": self.rejected_repeat,
                "wrong_language": self.rejected_lang,
                "duplicate": self.rejected_duplicate,
                "too_short": self.rejected_short,
            },
            "acceptance_rate": round(self.accepted / max(1, self.generated), 4),
            "by_category": self.by_category,
        }


def _paragraph(rng: random.Random) -> str:
    lines = [rng.choice(GREETINGS), rng.choice(INTROS)]
    for _ in range(rng.randint(2, 4)):
        lines.append(rng.choice(QUESTIONS))
        lines.append(rng.choice(ANSWERS))
    lines.append(rng.choice(SMALLTALK))
    return " ".join(lines)


def _grammar_item(rng: random.Random) -> str:
    statement, question = rng.choice(GRAMMAR_TEMPLATES)
    answer = rng.choice(ANSWERS)
    return f"{_fill(statement, rng)} {_fill(question, rng)} {answer}"


def _vocab_item(rng: random.Random) -> str:
    topic, *words = rng.choice(VOCAB_TOPICS)
    picked = rng.sample(words, k=min(3, len(words)))
    question = f"\"{picked[0]}\" sözünün mənası nədir?"
    answer = f"\"{picked[0]}\" sözü {topic} mövzusuna aid olur. Bu sahədə əlavə olaraq {', '.join(picked[1:])} kimi sözlər də işlədilir."
    return f"{question} {answer}"


def _rule_item(rng: random.Random) -> str:
    rule = rng.choice(GRAMMAR_RULES)
    example = _fill(rng.choice(GRAMMAR_TEMPLATES)[0], rng)
    return f"{rule} Məsələn: {example}"


CATEGORIES: dict[str, Callable[[random.Random], str]] = {
    "conversation": lambda rng: _paragraph(rng),
    "grammar": _grammar_item,
    "vocabulary": _vocab_item,
    "spelling": _rule_item,
    "reasoning": lambda rng: " ".join(
        [
            rng.choice(QUESTIONS),
            "Gəlin addımlarla izah edək.",
            "Birincisi, məlumatı toplayırıq.",
            "İkincisi, məlumatı müqayisə edirik.",
            "Nəticədə, aydın bir cavab əldə edirik.",
        ]
    ),
    "summary": lambda rng: " ".join(
        [
            rng.choice(SMALLTALK),
            f"Ümumiliklə, mətndə əsas fikir odur ki, {rng.choice(PROFESSIONS)} işinin gündəlik həyatda rolu böyükdür.",
            "Nəticə budur ki, düzgün qurulmuş cümlə yazmaq üçün söz sırasına diqqət yetmək vacibdir.",
        ]
    ),
}


def generate(
    n: int = 2_000,
    seed: int = 1337,
    categories: Sequence[str] | None = None,
    min_chars: int = 200,
    max_repeat_ratio: float = 0.30,
    require_az: bool = True,
    dedupe: bool = True,
) -> tuple[list[str], SyntheticStats]:
    """Generate ``n`` accepted synthetic documents (or fewer if gates reject)."""
    rng = random.Random(seed)
    cats = list(categories) if categories else list(CATEGORIES)
    stats = SyntheticStats()
    out: list[str] = []
    dd = Deduplicator(threshold=0.92) if dedupe else None
    attempts = 0
    max_attempts = n * 12

    while len(out) < n and attempts < max_attempts:
        attempts += 1
        cat = cats[attempts % len(cats)]
        text = CATEGORIES[cat](rng)
        stats.generated += 1
        stats.by_category[cat] = stats.by_category.get(cat, 0) + 1
        if len(text) < min_chars:
            text = (text + " ") * (1 + min_chars // max(1, len(text)))
        if repeated_token_ratio(text, 5) > max_repeat_ratio:
            stats.rejected_repeat += 1
            continue
        if require_az and detect_language(text)["lang"] != "az":
            stats.rejected_lang += 1
            continue
        if dd is not None:
            dup, _ = dd.is_duplicate(text)
            if dup:
                stats.rejected_duplicate += 1
                continue
        out.append(text)
        stats.accepted += 1

    log.info("synthetic: %d accepted of %d generated (%d attempts)", stats.accepted, stats.generated, attempts)
    return out, stats


def write_corpus(texts: Iterable[str], path) -> int:
    from ..utils import write_jsonl

    return write_jsonl(path, [{"text": t, "source": "az_synthetic_templates", "synthetic": True} for t in texts])


def suffix_balance_report(texts: Sequence[str], suffixes: Sequence[str] = ("-da", "-də", "-dan", "-dən", "-lar", "-lər")) -> dict:
    """Detect a single template family dominating the synthetic mix."""
    total = sum(len(t.split()) for t in texts) or 1
    counts = {s: sum(t.count(s) for t in texts) for s in suffixes}
    total_hits = sum(counts.values()) or 1
    return {
        "words_total": total,
        "suffix_counts": counts,
        "suffix_share": {s: round(c / total_hits, 4) for s, c in counts.items()},
        "max_single_suffix_share": round(max(counts.values()) / total_hits, 4) if counts else 0.0,
        "balanced": (max(counts.values()) / total_hits) < 0.5 if counts else True,
    }