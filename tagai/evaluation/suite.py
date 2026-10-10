"""Azerbaijani evaluation suite (held out from all gradient updates).

The prompts are hand-written Azerbaijani covering several registers. They live
in this module rather than in the training corpus precisely so they can never
be trained on.
"""

from __future__ import annotations

from typing import Sequence

# Register tags: conv = conversational, formal = formal writing, gram = grammar,
# edu = educational, tech = technical, idiom = idiomatic.
EVAL_PROMPTS: list[dict] = [
    {"id": "conv_greeting", "register": "conv", "prompt": "Salam, mənə kömək edə bilərsinizmi?", "expects": ["salat", "qarşı", "buyur", "çox sağ"]},
    {"id": "conv_selfintro", "register": "conv", "prompt": "Mən Əli adında tələbəyəm. Siz kimsiniz?", "expects": ["adım", "mən", "tanışmaq", "siz"]},
    {"id": "conv_smalltalk", "register": "conv", "prompt": "Bu gün hava çox gözəl idi.", "expects": ["hava", "gün", "gözəl", "bəli"]},
    {"id": "formal_letter", "register": "formal", "prompt": "Hörmətli qonaq, əvvəlcə dəyərli məktubunuz üçün təşəkkür edirəm.", "expects": ["hörmət", "məktub", "təşəkkür", "nizam"]},
    {"id": "formal_request", "register": "formal", "prompt": "Zəhmət olmasa məlumatı göndərə bilərsinizmi?", "expects": ["zəhmət", "məlumat", "göndər", "bilər"]},
    {"id": "gram_suffix", "register": "gram", "prompt": "Kitabı kitabxanadan götürdüm.", "expects": ["kitabxanada", "götür", "oxu", "kitab"]},
    {"id": "gram_question", "register": "gram", "prompt": "Azərbaycanın baş payıtaxtı nədir?", "expects": ["bakı", "paytaxt", "azərbaycan"]},
    {"id": "edu_math", "register": "edu", "prompt": "Əlavə etmə problemi: 15 + 27 neçədir?", "expects": ["42", "əlavə", "cəm"]},
    {"id": "edu_science", "register": "edu", "prompt": "Günəş nədir?", "expects": ["ulduz", "günəş", "işıq", "enerji"]},
    {"id": "tech_explain", "register": "tech", "prompt": "Kompüter proqramı necə işləyir?", "expects": ["proqram", "kompüter", "əmrlər", "məlumat"]},
    {"id": "idiom_ proverb", "register": "idiom", "prompt": "Yaxşılıq eləyən, yaxşılıq görür.", "expects": ["yaxşılıq", "görür", "qarşılığı"]},
    {"id": "geo_azerbaijan", "register": "edu", "prompt": "Azərbaycan Respublikası haqqında məlumat ver.", "expects": ["azərbaycan", "respublika", "baku", "baku", "xalq"]},
    {"id": "reasoning_step", "register": "edu", "prompt": "Problem: Gələn ay 5 kitab oxudum, bu ay 8 kitab oxudum. Neçə kitab oxudum?", "expects": ["13", "cəmi", "topla"]},
    {"id": "story_open", "register": "conv", "prompt": "Bir gün Bakıda yağış yağdı və", "expects": ["bakı", "gün", "yağış"]},
]

SPELLING_PAIRS: list[tuple[str, str]] = [
    ("Mən məktəbdə oxuyuram.", "Mən məktəbdə oxuyuram."),
    ("Beləliklə bu çox vacibdir.", "Beləliklə bu çox vacibdir."),
    ("Onun əlində böyük kitab var idi.", "Onun əlində böyük kitab var idi."),
    ("Səhər yığışması saat 9-da başlayır.", "Səhər yığışması saat 9-da başlayır."),
    ("Mən öz qardaşım ilə gedirəm.", "Mən öz qardaşım ilə gedirəm."),
]


class EvalSuite:
    """Scores a generator on prompt completion, perplexity and repetition."""

    def __init__(self, prompts: Sequence[dict] | None = None, max_new_tokens: int = 96, seed: int = 1337) -> None:
        self.prompts = list(prompts or EVAL_PROMPTS)
        self.max_new_tokens = max_new_tokens
        self.seed = seed

    def run(self, generator, *, max_new_tokens: int | None = None, verbose: bool = False) -> dict:
        from .metrics import repetition_ratio, token_statistics

        cfg_kwargs = dict(max_new_tokens=max_new_tokens or self.max_new_tokens, seed=self.seed)
        results = []
        for item in self.prompts:
            cfg = _make_cfg(generator, **cfg_kwargs)
            completion = generator.generate(item["prompt"], cfg)
            nll_stats = generator.score(item["prompt"] + " " + completion) if hasattr(generator, "score") else {}
            results.append(
                {
                    **item,
                    "completion": completion,
                    "generated_chars": len(completion),
                    "nll": nll_stats.get("nll"),
                    "repetition_ratio": repetition_ratio(completion),
                    "token_stats": token_statistics(completion),
                }
            )
            if verbose:
                print(f"[{item['id']}] {item['prompt']}\n  -> {completion[:300]}\n")

        mean_nll = _mean([r["nll"] for r in results if r["nll"] is not None])
        mean_rep = _mean([r["repetition_ratio"] for r in results])
        empty = sum(1 for r in results if not r["completion"].strip())
        return {
            "num_prompts": len(results),
            "mean_nll": round(mean_nll, 4) if mean_nll is not None else None,
            "mean_perplexity": round(float(__import__("math").exp(mean_nll)), 3) if mean_nll is not None else None,
            "mean_repetition_ratio": round(mean_rep, 4),
            "empty_completions": empty,
            "results": results,
        }


def _make_cfg(generator, **kwargs):
    from ..inference.generate import GenerateConfig

    return GenerateConfig(**kwargs)


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and v == v]
    return sum(vals) / len(vals) if vals else None