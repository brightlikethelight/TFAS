"""Programmatic generators for CRT-style benchmark templates.

The module exposes:

* :func:`bat_ball_isomorph` -- generate a novel bat-and-ball ratio
  problem with user-supplied cover and numbers.
* :func:`widgets_machines_isomorph` -- work-rate isomorph.
* :func:`lily_pad_isomorph` -- exponential-doubling isomorph.
* :func:`expand_paraphrases` -- turn one primary :class:`BenchmarkItem`
  into a flat list of paraphrase records with suffixed ids. The caller
  is responsible for writing the expanded list to disk; the benchmark
  file keeps paraphrases inside the primary record only.

Every generator returns a tuple ``(conflict_item, control_item)`` where
the two items share a ``matched_pair_id`` and identical structural
difficulty. Generators raise :class:`ValueError` on inconsistent input
(e.g. numeric parameters that don't sum correctly) -- this is how we
keep the pipeline honest against a human-authoring bug.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from beartype import beartype

from s1s2.benchmark.loader import BenchmarkItem

# ---- low-level helpers ------------------------------------------------


def _bounded_regex(literal: str) -> str:
    """Build an ``\\b<literal>\\b`` regex around an already-escaped literal.

    Word boundaries don't trigger around punctuation, so for answers
    like "0.05" we fall back to a lookaround.
    """
    esc = re.escape(literal)
    if re.match(r"^\w", literal) and re.search(r"\w$", literal):
        return rf"\b{esc}\b"
    return rf"(?<![\w.]){esc}(?![\w.])"


@beartype
def _build_pair(
    *,
    pair_id: str,
    category: str,
    subcategory: str,
    difficulty: int,
    conflict_prompt: str,
    control_prompt: str,
    correct_answer: str,
    lure_answer: str,
    source: str,
    provenance_note: str,
    paraphrases_conflict: list[str],
    paraphrases_control: list[str] | None = None,
) -> tuple[BenchmarkItem, BenchmarkItem]:
    """Construct a matched (conflict, control) pair of :class:`BenchmarkItem`."""
    conflict = BenchmarkItem(
        id=f"{pair_id}_conflict",
        category=category,  # type: ignore[arg-type]
        subcategory=subcategory,
        conflict=True,
        difficulty=difficulty,
        prompt=conflict_prompt,
        system_prompt=None,
        correct_answer=correct_answer,
        lure_answer=lure_answer,
        answer_pattern=_bounded_regex(correct_answer),
        lure_pattern=_bounded_regex(lure_answer),
        matched_pair_id=pair_id,
        source=source,
        provenance_note=provenance_note,
        paraphrases=tuple(paraphrases_conflict),
    )
    control = BenchmarkItem(
        id=f"{pair_id}_control",
        category=category,  # type: ignore[arg-type]
        subcategory=subcategory,
        conflict=False,
        difficulty=difficulty,
        prompt=control_prompt,
        system_prompt=None,
        correct_answer=correct_answer,
        lure_answer="",
        answer_pattern=_bounded_regex(correct_answer),
        lure_pattern="",
        matched_pair_id=pair_id,
        source=source,
        provenance_note=provenance_note + " (structural control).",
        paraphrases=tuple(paraphrases_control or []),
    )
    return conflict, control


# ---- CRT ratio (bat-and-ball-style) -----------------------------------


@beartype
def bat_ball_isomorph(
    *,
    pair_id: str,
    object_a: str,
    object_b: str,
    total_cents: int,
    diff_cents: int,
    difficulty: int = 2,
    cover: str = "cost",
) -> tuple[BenchmarkItem, BenchmarkItem]:
    """Build a novel bat-and-ball isomorph.

    The classic problem is: ``A bat and a ball cost $1.10 in total. The
    bat costs $1.00 more than the ball. How much does the ball cost?``
    The intuitive (wrong) answer is 10 cents; the correct answer is 5
    cents. Here we generalise the structure with arbitrary objects and
    numbers; the function asserts that ``diff_cents > 0``, ``total_cents
    > diff_cents``, and that the solution ``ball = (total - diff) / 2``
    is an integer (in cents).

    The returned pair shares a difficulty and a matched_pair_id and
    differs only in the lure: the control poses the same question in a
    form with no S1 lure (we give the explicit price of object_a so
    the remainder is trivial subtraction).
    """
    if diff_cents <= 0:
        raise ValueError("diff_cents must be positive")
    if total_cents <= diff_cents:
        raise ValueError("total_cents must exceed diff_cents")
    remainder = total_cents - diff_cents
    if remainder % 2 != 0:
        raise ValueError(
            "total_cents - diff_cents must be even so that ball is an integer"
        )
    b_cents = remainder // 2
    a_cents = b_cents + diff_cents
    if b_cents <= 0:
        raise ValueError("derived ball price is non-positive")

    lure_cents = total_cents - diff_cents  # the intuitive wrong answer
    if lure_cents == b_cents:
        raise ValueError(
            "lure collapses onto the correct answer; pick different numbers"
        )

    total_dollars = f"${total_cents / 100:.2f}"
    diff_dollars = f"${diff_cents / 100:.2f}"
    a_dollars = f"${a_cents / 100:.2f}"

    conflict_prompt = (
        f"A {object_a} and a {object_b} {cover} {total_dollars} in total. "
        f"The {object_a} {cover}s {diff_dollars} more than the {object_b}. "
        f"How much does the {object_b} {cover}, in cents? "
        "Answer with the number of cents only."
    )
    # Control: give the explicit price of object_a so only subtraction is
    # required; the correct answer is still b_cents.
    control_prompt = (
        f"A {object_a} and a {object_b} {cover} {total_dollars} in total. "
        f"The {object_a} alone {cover}s {a_dollars}. "
        f"How much does the {object_b} {cover}, in cents? "
        "Answer with the number of cents only."
    )

    paraphrases_conflict = [
        (
            f"Together, a {object_a} and a {object_b} {cover} {total_dollars}. "
            f"The {object_a} is {diff_dollars} more expensive than the {object_b}. "
            f"In cents, what is the price of the {object_b}?"
        ),
        (
            f"Imagine two items: a {object_a} and a {object_b}. Their combined "
            f"price is {total_dollars}, and the {object_a} {cover}s "
            f"{diff_dollars} more than the {object_b}. "
            f"Give the price of the {object_b} in cents."
        ),
        (
            f"Two products are sold together for {total_dollars}: a "
            f"{object_a} and a {object_b}. The {object_a} {cover}s "
            f"{diff_dollars} more than the {object_b}. "
            f"How many cents does the {object_b} {cover}?"
        ),
        (
            f"A shopkeeper sells a {object_a} and a {object_b}. The total is "
            f"{total_dollars}. The {object_a} {cover}s {diff_dollars} more "
            f"than the {object_b}. Report the {object_b}'s price in cents."
        ),
    ]
    paraphrases_control = [
        (
            f"A {object_a} and a {object_b} together {cover} {total_dollars}. "
            f"The {object_a} by itself {cover}s {a_dollars}. In cents, what is "
            f"the price of the {object_b}?"
        ),
    ]

    return _build_pair(
        pair_id=pair_id,
        category="crt",
        subcategory="ratio",
        difficulty=difficulty,
        conflict_prompt=conflict_prompt,
        control_prompt=control_prompt,
        correct_answer=str(b_cents),
        lure_answer=str(lure_cents),
        source="novel",
        provenance_note=(
            "Novel bat-and-ball structural isomorph. "
            f"Cover: {object_a}/{object_b}; total={total_dollars}, "
            f"diff={diff_dollars}."
        ),
        paraphrases_conflict=paraphrases_conflict,
        paraphrases_control=paraphrases_control,
    )


# ---- CRT work-rate (widgets-machines) ---------------------------------


@beartype
def widgets_machines_isomorph(
    *,
    pair_id: str,
    worker_label: str,
    output_label: str,
    base_rate: int,
    scale: int,
    difficulty: int = 3,
) -> tuple[BenchmarkItem, BenchmarkItem]:
    """Work-rate isomorph of the widgets-machines CRT problem.

    Classic: ``5 machines make 5 widgets in 5 minutes. How long for 100
    machines to make 100 widgets?``. Intuitive answer: 100; correct: 5.

    Here ``base_rate`` plays the role of 5 and ``scale`` the role of
    100. For the structure to work we need ``base_rate < scale`` and
    positive integers.
    """
    if base_rate <= 0 or scale <= 0:
        raise ValueError("base_rate and scale must be positive")
    if base_rate == scale:
        raise ValueError("base_rate equals scale; no lure exists")

    conflict_prompt = (
        f"{base_rate} {worker_label}s together produce {base_rate} "
        f"{output_label}s in {base_rate} minutes. "
        f"Working at the same per-unit rate, how many minutes would "
        f"{scale} {worker_label}s take to produce {scale} {output_label}s? "
        "Answer with the number of minutes only."
    )
    control_prompt = (
        f"Each {worker_label} produces one {output_label} every {base_rate} "
        f"minutes. With {scale} {worker_label}s working in parallel, how "
        f"many minutes does it take to produce {scale} {output_label}s? "
        "Answer with the number of minutes only."
    )

    paraphrases_conflict = [
        (
            f"{base_rate} {worker_label}s turn out {base_rate} "
            f"{output_label}s over {base_rate} minutes. At that rate, how "
            f"long does it take {scale} {worker_label}s to turn out "
            f"{scale} {output_label}s? Report your answer in minutes."
        ),
        (
            f"A factory runs {base_rate} {worker_label}s; together they "
            f"complete {base_rate} {output_label}s in {base_rate} minutes. "
            f"If the factory instead runs {scale} {worker_label}s at the "
            f"same per-unit rate, how many minutes are needed to complete "
            f"{scale} {output_label}s?"
        ),
        (
            f"It takes {base_rate} {worker_label}s exactly {base_rate} "
            f"minutes to build {base_rate} {output_label}s, one per "
            f"{worker_label}. Scaling the line up to {scale} "
            f"{worker_label}s and {scale} {output_label}s, how many "
            f"minutes is the new job?"
        ),
        (
            f"When {base_rate} parallel {worker_label}s each complete one "
            f"{output_label} in {base_rate} minutes, how many minutes would "
            f"{scale} parallel {worker_label}s each take to complete "
            f"{scale} {output_label}s? Answer in minutes."
        ),
    ]
    paraphrases_control = [
        (
            f"One {worker_label} makes one {output_label} every "
            f"{base_rate} minutes. Running {scale} {worker_label}s in "
            f"parallel, how many minutes does it take to produce "
            f"{scale} {output_label}s total?"
        )
    ]

    return _build_pair(
        pair_id=pair_id,
        category="crt",
        subcategory="work_rate",
        difficulty=difficulty,
        conflict_prompt=conflict_prompt,
        control_prompt=control_prompt,
        correct_answer=str(base_rate),
        lure_answer=str(scale),
        source="novel",
        provenance_note=(
            "Novel widgets-machines structural isomorph "
            f"({worker_label}/{output_label}, base={base_rate}, scale={scale})."
        ),
        paraphrases_conflict=paraphrases_conflict,
        paraphrases_control=paraphrases_control,
    )


# ---- CRT exponential doubling (lily-pad) ------------------------------


@beartype
def lily_pad_isomorph(
    *,
    pair_id: str,
    entity: str,
    habitat: str,
    days_to_full: int,
    difficulty: int = 3,
) -> tuple[BenchmarkItem, BenchmarkItem]:
    """Exponential-doubling isomorph.

    Classic: ``A patch of lily pads doubles in size every day. If it
    takes 48 days to cover the lake, how long does it take to cover
    half the lake?`` Correct: 47, intuitive lure: 24.
    """
    if days_to_full < 2:
        raise ValueError("days_to_full must be at least 2 for the structure to hold")
    correct = days_to_full - 1
    lure = days_to_full // 2
    if lure == correct:
        raise ValueError("lure collapses onto correct answer")

    conflict_prompt = (
        f"A colony of {entity} doubles in size every day. It takes "
        f"{days_to_full} days for the colony to fill the entire {habitat}. "
        f"How many days does it take to fill exactly half of the {habitat}? "
        "Answer with the number of days."
    )
    control_prompt = (
        f"A colony of {entity} grows by the same fixed number every day. "
        f"After {days_to_full} days the colony fills exactly the whole "
        f"{habitat}, and after {correct} days it fills the (full_size - "
        f"daily_growth) amount. How many days does it take to fill that "
        "(full_size - daily_growth) amount? Answer with the number of days."
    )

    paraphrases_conflict = [
        (
            f"In a pond, a population of {entity} doubles every 24 hours. "
            f"It takes {days_to_full} days for the population to cover the "
            f"entire {habitat}. On which day is half of the {habitat} "
            f"covered? Answer with the day number."
        ),
        (
            f"Day by day the {entity} colony doubles. It needs "
            f"{days_to_full} days to cover the {habitat}. When was the "
            f"{habitat} half-covered? Give the day as a number."
        ),
        (
            f"{entity} in the {habitat} exhibit perfect daily doubling. "
            f"They reach full coverage on day {days_to_full}. On what day "
            f"number did they reach 50 percent coverage?"
        ),
        (
            f"Assuming perfect doubling every day, the {entity} fill the "
            f"{habitat} in {days_to_full} days. Report the day on which "
            f"they filled half the {habitat}."
        ),
    ]
    paraphrases_control = [
        (
            f"A colony of {entity} grows by a constant amount per day. "
            f"It fills the {habitat} on day {days_to_full}. On which day "
            f"does the colony's size equal "
            f"(full_size - one_day_of_growth)? Answer with the day number."
        )
    ]

    return _build_pair(
        pair_id=pair_id,
        category="crt",
        subcategory="exponential_growth",
        difficulty=difficulty,
        conflict_prompt=conflict_prompt,
        control_prompt=control_prompt,
        correct_answer=str(correct),
        lure_answer=str(lure),
        source="novel",
        provenance_note=(
            "Novel lily-pad (exponential doubling) structural isomorph "
            f"({entity}/{habitat}, full={days_to_full}d)."
        ),
        paraphrases_conflict=paraphrases_conflict,
        paraphrases_control=paraphrases_control,
    )


# ---- paraphrase expansion --------------------------------------------


@beartype
def expand_paraphrases(items: list[BenchmarkItem]) -> list[BenchmarkItem]:
    """Expand ``paraphrases`` into sibling :class:`BenchmarkItem` records.

    Each paraphrase becomes its own benchmark line with id
    ``<base_id>__p<N>``, inheriting everything except ``prompt`` and
    ``paraphrases`` (which is reset to an empty tuple on paraphrase
    siblings). The original primary remains in place. Callers typically
    pass the full benchmark through this function once before writing
    to disk if they want one record per stimulus.

    For the on-disk benchmark we prefer to KEEP paraphrases nested
    inside the primary record (so the JSONL stays compact and each
    item's provenance is centralised), and run ``expand_paraphrases``
    lazily at extraction time.
    """
    out: list[BenchmarkItem] = []
    for it in items:
        out.append(it)
        for i, p in enumerate(it.paraphrases, start=1):
            sibling = replace(
                it,
                id=f"{it.id}__p{i}",
                prompt=p,
                paraphrases=(),
                provenance_note=it.provenance_note + f" [paraphrase {i}]",
            )
            out.append(sibling)
    return out


# ---- bulk helpers ----------------------------------------------------


@beartype
def make_many(
    maker: Callable[..., tuple[BenchmarkItem, BenchmarkItem]],
    specs: list[dict[str, Any]],
) -> list[BenchmarkItem]:
    """Apply a generator across a list of parameter dicts.

    Raises the underlying generator's error on the first spec that
    fails, which gives a clean line number when debugging benchmark
    construction.
    """
    out: list[BenchmarkItem] = []
    for spec in specs:
        try:
            c, x = maker(**spec)
        except Exception as e:
            raise ValueError(f"failed building item with spec {spec}: {e}") from e
        out.append(c)
        out.append(x)
    return out
