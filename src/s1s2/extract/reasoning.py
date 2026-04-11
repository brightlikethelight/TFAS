"""Reasoning-trace handling: ``<think>...</think>`` parsing and behavioral scoring.

The thinking-span parser lives in :mod:`s1s2.extract.parsing`; this module
re-exports the canonical API and adds the behavioral parsing layer
(predicted-answer extraction + correct/lure/refusal classification). Keeping
these next to each other makes the extractor's post-generation logic
one-import instead of two.

Why behavioral parsing lives here and not in ``benchmark/``: it consumes a
*generated* response and an item schema simultaneously, so it straddles the
two concerns. The benchmark module stays pure (loading + validation), and the
extraction pipeline owns response -> label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from beartype import beartype

from s1s2.extract.parsing import (
    THINK_CLOSE,
    THINK_OPEN,
    PositionInfo,
    ThinkingSpan,
    build_token_char_spans,
    compute_positions,
    find_thinking_span,
    split_thinking_answer,
)

__all__ = [
    "THINK_CLOSE",
    "THINK_OPEN",
    "BehavioralResult",
    "PositionInfo",
    "ThinkingSpan",
    "build_token_char_spans",
    "compute_positions",
    "find_thinking_span",
    "parse_behavioral_response",
    "split_thinking_answer",
]

ResponseCategoryStr = Literal["correct", "lure", "other_wrong", "refusal"]

# Phrases a model may use to refuse to engage. We keep the list short to avoid
# false positives on genuine answers that happen to contain these words.
_REFUSAL_PHRASES: tuple[str, ...] = (
    "i cannot answer",
    "i can't answer",
    "i cannot help",
    "i can't help",
    "i'm unable to",
    "i am unable to",
    "as an ai language model",
    "i do not have the ability",
    "i refuse to",
)


@dataclass(frozen=True)
class BehavioralResult:
    """Output of :func:`parse_behavioral_response`.

    Attributes
    ----------
    predicted_answer:
        Best-effort short string representing the model's final answer. For
        numeric tasks this is the last number in the response; for multiple
        choice it's a single uppercase letter; for free-text, the last
        sentence truncated to 128 chars.
    correct:
        True iff the predicted answer matches the ground-truth correct answer.
        Checked via the item's ``answer_pattern`` regex first, then by a more
        lenient substring match against the canonical string.
    matches_lure:
        True iff the response matches the S1 lure pattern. Only meaningful
        for conflict items; for controls it is always False.
    response_category:
        Categorical label for downstream grouping.
    """

    predicted_answer: str
    correct: bool
    matches_lure: bool
    response_category: ResponseCategoryStr


def _last_number(text: str) -> str | None:
    """Return the last standalone number in ``text``, or None."""
    # Capture signed decimals with optional thousands separators. We tolerate
    # trailing punctuation/units outside the match so "$5." and "5 cents" both
    # return "5".
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def _last_sentence(text: str) -> str:
    """Return the last non-empty sentence. Very small heuristic."""
    # Split on sentence terminators but keep them simple — the fancier
    # alternative (e.g. spaCy) isn't worth the dependency weight here.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    for seg in reversed(parts):
        if seg.strip():
            return seg.strip()[:128]
    return ""


def _single_letter_choice(text: str) -> str | None:
    """Return the last standalone uppercase A-E choice letter, or None."""
    # Require a word boundary and optional surrounding punctuation. The
    # regex is greedy from the right: we use findall and pick the last.
    matches = re.findall(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", text)
    if not matches:
        return None
    return matches[-1]


@beartype
def parse_behavioral_response(
    response_text: str,
    *,
    correct_answer: str,
    lure_answer: str,
    answer_pattern: str,
    lure_pattern: str,
) -> BehavioralResult:
    """Parse a model response and classify the outcome.

    The response may be the FULL generation including the thinking trace;
    this function strips ``<think>...</think>`` first so the classification
    only sees the answer segment.

    Parameters
    ----------
    response_text:
        Raw model generation (prompt stripped).
    correct_answer, lure_answer:
        Canonical strings from the benchmark item. ``lure_answer`` may be
        empty for non-conflict items.
    answer_pattern, lure_pattern:
        Python regexes from the benchmark item. The extractor uses these to
        make matching robust to whitespace/formatting noise. ``lure_pattern``
        may be empty for non-conflict items.
    """
    # Strip thinking trace before scoring. Reasoning models often rehearse
    # both the lure and the correct answer inside <think>, so scoring on the
    # full text would produce false positives.
    _thinking, answer_segment, _span = split_thinking_answer(response_text)
    stripped = answer_segment.strip()

    if not stripped:
        return BehavioralResult(
            predicted_answer="",
            correct=False,
            matches_lure=False,
            response_category="refusal",
        )

    low = stripped.lower()
    for phrase in _REFUSAL_PHRASES:
        if phrase in low:
            return BehavioralResult(
                predicted_answer=stripped[:128],
                correct=False,
                matches_lure=False,
                response_category="refusal",
            )

    # Pattern-first matching: the benchmark schema guarantees the regex is
    # a tight match for the canonical answer, so this is the primary signal.
    correct_hit = False
    if answer_pattern:
        try:
            correct_hit = re.search(answer_pattern, stripped, flags=re.IGNORECASE) is not None
        except re.error:
            correct_hit = False
    lure_hit = False
    if lure_pattern:
        try:
            lure_hit = re.search(lure_pattern, stripped, flags=re.IGNORECASE) is not None
        except re.error:
            lure_hit = False

    # Fall back to canonical substring match when the regex is absent or
    # silently fails.
    if not correct_hit and correct_answer:
        correct_hit = correct_answer.lower() in low
    if not lure_hit and lure_answer:
        lure_hit = lure_answer.lower() in low

    # Short predicted_answer extraction. We try the three shapes in order of
    # specificity; the best match wins for downstream display.
    letter = _single_letter_choice(stripped)
    number = _last_number(stripped)
    if letter is not None and (
        re.fullmatch(r"[A-E]", correct_answer.strip())
        or re.fullmatch(r"[A-E]", lure_answer.strip())
    ):
        predicted = letter
    elif number is not None and (
        _last_number(correct_answer) is not None or _last_number(lure_answer) is not None
    ):
        predicted = number
    else:
        predicted = _last_sentence(stripped)

    # Classification rule: correct wins over lure if both match (a response
    # that contains both is usually the model giving the correct answer and
    # also mentioning the lure). Conflict framing still reports matches_lure
    # so downstream code can flag mixed responses.
    if correct_hit:
        category: ResponseCategoryStr = "correct"
    elif lure_hit:
        category = "lure"
    else:
        category = "other_wrong"

    return BehavioralResult(
        predicted_answer=predicted[:128],
        correct=correct_hit,
        matches_lure=lure_hit,
        response_category=category,
    )
