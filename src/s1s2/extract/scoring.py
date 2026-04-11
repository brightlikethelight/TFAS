"""Behavioral scoring of model generations.

Every generation is classified into one of four categories:

- ``correct``: answer_pattern regex matches somewhere in the answer.
- ``lure``: lure_pattern regex matches AND the item is a conflict item.
- ``refusal``: the model refused via a canonical refusal phrase.
- ``other_wrong``: none of the above.

Precedence: we check refusal first (so a refusal phrase wins over a stray
number that might coincidentally match ``answer_pattern``), then correct,
then lure, then other_wrong. This is a deliberate choice -- it gives us the
most generous behavioral counts for "correct" while still treating refusals
as a distinct outcome category.

The item is expected to provide ``answer_pattern`` and ``lure_pattern`` as
raw regex strings (see ``data/benchmark/SCHEMA.md``). We compile them lazily
and cache per-item.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from s1s2.utils.types import ResponseCategory

# Canonical refusal phrases. Case-insensitive substring match.
_REFUSAL_PHRASES: tuple[str, ...] = (
    "i cannot",
    "i can't help",
    "i can not help",
    "i refuse",
    "i will not",
    "i won't",
    "as an ai",
    "as a language model",
    "i'm sorry, but i",
    "i am sorry, but i",
    "i'm not able to",
    "i am not able to",
)


class BenchmarkItemProto(Protocol):
    """Structural protocol for the subset of BenchmarkItem fields we need.

    Keeping this as a protocol lets us accept the real loader's dataclass and
    a plain dict or SimpleNamespace in tests, without importing the benchmark
    module (which may not exist yet when extraction runs in isolation).
    """

    @property
    def answer_pattern(self) -> str: ...
    @property
    def lure_pattern(self) -> str: ...
    @property
    def conflict(self) -> bool: ...
    @property
    def correct_answer(self) -> str: ...
    @property
    def lure_answer(self) -> str: ...


@dataclass(frozen=True)
class ScoringResult:
    category: ResponseCategory
    predicted_answer: str          # What we extracted as the model's answer
    matched_correct: bool
    matched_lure: bool
    refused: bool


def _compile(pat: str) -> re.Pattern[str] | None:
    if not pat:
        return None
    try:
        return re.compile(pat, flags=re.IGNORECASE)
    except re.error:
        # Fall back to a literal match if the item's regex is malformed.
        return re.compile(re.escape(pat), flags=re.IGNORECASE)


def _is_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _REFUSAL_PHRASES)


def score_response(
    generated_text: str,
    item: BenchmarkItemProto,
) -> tuple[ResponseCategory, str]:
    """Classify a generation and return (category, extracted_answer).

    The ``extracted_answer`` is the first regex match (correct or lure,
    whichever fired); for other_wrong we return the final ~64 chars of the
    stripped generation as a best-effort hint for downstream inspection.
    """
    result = score_response_detailed(generated_text, item)
    return result.category, result.predicted_answer


def score_response_detailed(
    generated_text: str,
    item: BenchmarkItemProto,
) -> ScoringResult:
    """Full scoring result. Use this when you need the booleans separately."""
    if generated_text is None:
        generated_text = ""

    # Refusal check has priority: a refusal that happens to contain a digit
    # matching answer_pattern should still be classified as refusal.
    refused = _is_refusal(generated_text)
    if refused:
        return ScoringResult(
            category="refusal",
            predicted_answer="[refusal]",
            matched_correct=False,
            matched_lure=False,
            refused=True,
        )

    ans_re = _compile(item.answer_pattern)
    lure_re = _compile(item.lure_pattern) if item.lure_pattern else None

    ans_match = ans_re.search(generated_text) if ans_re is not None else None
    lure_match = lure_re.search(generated_text) if lure_re is not None else None

    matched_correct = ans_match is not None
    matched_lure = (lure_match is not None) and bool(item.conflict)

    if matched_correct:
        # If both patterns match, prefer whichever occurs later in the text:
        # the model's "final answer" is typically at the end, and reasoning
        # models often mention the lure mid-trace then correct themselves.
        if matched_lure and lure_match is not None and lure_match.start() > ans_match.start():
            return ScoringResult(
                category="lure",
                predicted_answer=lure_match.group(0),
                matched_correct=True,
                matched_lure=True,
                refused=False,
            )
        return ScoringResult(
            category="correct",
            predicted_answer=ans_match.group(0),
            matched_correct=True,
            matched_lure=matched_lure,
            refused=False,
        )

    if matched_lure and lure_match is not None:
        return ScoringResult(
            category="lure",
            predicted_answer=lure_match.group(0),
            matched_correct=False,
            matched_lure=True,
            refused=False,
        )

    # other_wrong: return a short suffix of the generation as the "predicted
    # answer" for inspection. Downstream consumers treat this as opaque.
    stripped = generated_text.strip()
    tail = stripped[-64:] if len(stripped) > 64 else stripped
    return ScoringResult(
        category="other_wrong",
        predicted_answer=tail,
        matched_correct=False,
        matched_lure=False,
        refused=False,
    )
