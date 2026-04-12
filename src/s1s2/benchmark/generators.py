"""Generators for the six non-CRT benchmark categories.

Each public function returns a ``(conflict, control)`` tuple of
:class:`~s1s2.benchmark.loader.BenchmarkItem`. The pair shares a
``matched_pair_id`` and difficulty so downstream matched-subset
analyses stay honest. The conflict item carries a System-1 lure that
"feels" correct but is wrong; the control is a structural isomorph
where intuition and deliberation agree. Generators are pure -- they
raise :class:`ValueError` on inconsistent input and otherwise produce
deterministic output for the same arguments.

Why six files and not one big file? The CRT generators live in
:mod:`s1s2.benchmark.templates` because they came first and the rest
of the codebase already imports them from that module. Adding the
heterogeneous non-CRT generators there would crowd an already-long
file, so this module is its own home.

Design invariants (enforced per-generator):

* Conflict and control share ``category``, ``matched_pair_id``, and
  ``difficulty``.
* Every answer and lure is a short literal string matched by a
  bounded regex (reused from :mod:`templates`).
* Each generator supplies 2-3 paraphrases on the conflict item and 1
  on the control so the paraphrase expander has something to work
  with uniformly.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from beartype import beartype

from s1s2.benchmark.loader import BenchmarkItem
from s1s2.benchmark.templates import _build_pair  # re-used pair builder

# --------------------------------------------------------------------- #
# shared helpers                                                        #
# --------------------------------------------------------------------- #


def _bounded_regex(literal: str) -> str:
    """Mirror of :func:`templates._bounded_regex` to avoid circular imports.

    Kept as a thin local copy so this module does not depend on a
    private symbol in :mod:`templates`. Both copies must stay in sync.
    """
    esc = re.escape(literal)
    if re.match(r"^\w", literal) and re.search(r"\w$", literal):
        return rf"\b{esc}\b"
    return rf"(?<![\w.]){esc}(?![\w.])"


def _pct(x: float) -> str:
    """Render a probability 0.0..1.0 as a whole-percent string."""
    return f"{round(x * 100)}%"


# --------------------------------------------------------------------- #
# base-rate neglect                                                     #
# --------------------------------------------------------------------- #


@beartype
def base_rate_isomorph(
    *,
    pair_id: str,
    description: str,
    rare_group: str,
    common_group: str,
    rare_rate: float,
    common_rate: float,
    subcategory: str = "representativeness",
    difficulty: int = 3,
) -> tuple[BenchmarkItem, BenchmarkItem]:
    """Base-rate-neglect pair in the representativeness-heuristic family.

    ``rare_group`` is a low-base-rate category that the
    ``description`` stereotypically matches; ``common_group`` is the
    high-base-rate majority. The System-1 lure is to say the person
    belongs to ``rare_group`` because the description "fits the
    stereotype"; the normatively correct answer is ``common_group``
    (or equivalently "more likely ``common_group``") because the base
    rates dominate any weak evidence from a one-line description.

    The control poses a structurally identical question without the
    stereotype cue: the description is neutral so deliberation and
    intuition both point at ``common_group``.

    ``rare_rate`` and ``common_rate`` are the marginal base rates; we
    require ``common_rate >> rare_rate`` for the lure to exist.
    """
    if not (0.0 <= rare_rate <= 1.0 and 0.0 <= common_rate <= 1.0):
        raise ValueError("rates must be in [0,1]")
    if common_rate <= rare_rate * 3:
        raise ValueError(
            "common_rate must be clearly larger than rare_rate (>=3x) "
            "so the correct answer is unambiguous"
        )
    if rare_group.strip().lower() == common_group.strip().lower():
        raise ValueError("rare_group and common_group cannot be identical")

    rare_pct = _pct(rare_rate)
    common_pct = _pct(common_rate)

    conflict_prompt = (
        f"In a large survey, {common_pct} of respondents are "
        f"{common_group}s and only {rare_pct} are {rare_group}s. "
        f"One respondent is described as follows: {description} "
        f"Given only this information, is this respondent more likely "
        f"to be a {common_group} or a {rare_group}? "
        f"Answer with the single word {common_group!s} or "
        f"{rare_group!s}."
    )
    control_prompt = (
        f"In a large survey, {common_pct} of respondents are "
        f"{common_group}s and only {rare_pct} are {rare_group}s. "
        f"A respondent is picked uniformly at random with no "
        f"additional information. Is this respondent more likely to "
        f"be a {common_group} or a {rare_group}? Answer with the "
        f"single word {common_group!s} or {rare_group!s}."
    )

    paraphrases_conflict = [
        (
            f"Suppose {common_pct} of a population are {common_group}s "
            f"and {rare_pct} are {rare_group}s. Here is a short "
            f"description of one person drawn from this population: "
            f"{description} Without further evidence, which group is "
            f"this person more likely to belong to, {common_group} or "
            f"{rare_group}? Give the group name only."
        ),
        (
            f"Base rates in the population: {common_group} = "
            f"{common_pct}, {rare_group} = {rare_pct}. We observe one "
            f"person with this one-sentence profile: {description} "
            f"Ignoring stereotypes, is the profile more probably a "
            f"{common_group} or a {rare_group}? Answer: "
            f"{common_group} / {rare_group}."
        ),
        (
            f"A random sample from a population where {common_pct} "
            f"are {common_group}s and {rare_pct} are {rare_group}s "
            f"yields one individual. The individual's brief profile "
            f"reads: {description} Which label is more probable, "
            f"{common_group} or {rare_group}? Respond with one word."
        ),
    ]
    paraphrases_control = [
        (
            f"A population has {common_pct} {common_group}s and "
            f"{rare_pct} {rare_group}s. A random person is selected "
            f"with no further information. Which group is more "
            f"likely, {common_group} or {rare_group}? One word."
        )
    ]

    return _build_pair(
        pair_id=pair_id,
        category="base_rate",
        subcategory=subcategory,
        difficulty=difficulty,
        conflict_prompt=conflict_prompt,
        control_prompt=control_prompt,
        correct_answer=common_group,
        lure_answer=rare_group,
        source="novel",
        provenance_note=(
            "Base-rate-neglect isomorph in the Kahneman-Tversky "
            "representativeness tradition. Conflict has a stereotype "
            f"cue ({description[:40]}...) that pulls toward the rare "
            f"group ({rare_group}, base rate {rare_pct}); correct "
            f"answer is the dominant group ({common_group}, "
            f"base rate {common_pct})."
        ),
        paraphrases_conflict=paraphrases_conflict,
        paraphrases_control=paraphrases_control,
    )


# --------------------------------------------------------------------- #
# belief-bias syllogism                                                 #
# --------------------------------------------------------------------- #


@beartype
def belief_bias_syllogism(
    *,
    pair_id: str,
    conflict_major: str,
    conflict_minor: str,
    conflict_conclusion: str,
    conflict_is_valid: bool,
    is_believable: bool,
    control_major: str,
    control_minor: str,
    control_conclusion: str | None = None,
    subcategory: str | None = None,
    difficulty: int = 3,
) -> tuple[BenchmarkItem, BenchmarkItem]:
    """Belief-bias syllogism in the Evans (1983) 2x2 paradigm.

    The four cells of the 2x2 are:

    ============  =============  =========================
    validity      believability  pattern
    ============  =============  =========================
    valid         believable     **control** (agree)
    valid         unbelievable   **conflict** (belief lure)
    invalid       believable     **conflict** (belief lure)
    invalid       unbelievable   **control** (agree)
    ============  =============  =========================

    The conflict cells are where the system-1 response disagrees with
    the logical answer; the control cells are where both agree.

    API contract: the caller supplies BOTH cells of the matched pair
    explicitly. The conflict cell is identified by ``conflict_*``
    arguments and ``conflict_is_valid``; the matched control is the
    OPPOSITE-validity cell on the same believability axis. The
    caller must construct the control's premises so they are
    logically opposite-validity to the conflict's, and the control's
    conclusion must be believable iff the conflict's is. The
    ``is_believable`` flag is recorded in the provenance for both
    items but does not change the items' fields.

    Why explicit control premises? Auto-flipping a syllogism's major
    premise (e.g. swapping subject and predicate) does not reliably
    convert valid forms into invalid ones across all four moods, so
    we trust the human author of the spec instead.
    """
    if subcategory is None:
        v = "valid" if conflict_is_valid else "invalid"
        b = "believable" if is_believable else "unbelievable"
        subcategory = f"belief_bias_{v}_{b}"

    if not (conflict_is_valid ^ is_believable):
        raise ValueError(
            "belief_bias_syllogism expects (conflict_is_valid, is_believable) "
            "to identify the CONFLICT cell, which must be mismatched "
            "(valid XOR believable)."
        )

    # If the caller doesn't provide a separate control conclusion, we
    # reuse the conflict conclusion. This is the most common case --
    # the conclusion is the believable / unbelievable string and we
    # want both items to be evaluated against the same surface text.
    if control_conclusion is None:
        control_conclusion = conflict_conclusion

    correct_conflict = "valid" if conflict_is_valid else "invalid"
    lure_conflict = "invalid" if conflict_is_valid else "valid"
    correct_control = "invalid" if conflict_is_valid else "valid"

    conflict_prompt = (
        "Assume the premises are true. Is the conclusion logically "
        "entailed by the premises?\n\n"
        f"Premise 1: {conflict_major}\n"
        f"Premise 2: {conflict_minor}\n"
        f"Conclusion: {conflict_conclusion}\n\n"
        "A conclusion is 'valid' if it must be true whenever the "
        "premises are true, and 'invalid' otherwise. Answer 'valid' "
        "or 'invalid'."
    )
    control_prompt = (
        "Assume the premises are true. Is the conclusion logically "
        "entailed by the premises?\n\n"
        f"Premise 1: {control_major}\n"
        f"Premise 2: {control_minor}\n"
        f"Conclusion: {control_conclusion}\n\n"
        "A conclusion is 'valid' if it must be true whenever the "
        "premises are true, and 'invalid' otherwise. Answer 'valid' "
        "or 'invalid'."
    )

    paraphrases_conflict = [
        (
            f"Premise 1: {conflict_major}\n"
            f"Premise 2: {conflict_minor}\n"
            f"Conclusion: {conflict_conclusion}\n"
            "Taking the premises as true, does the conclusion "
            "logically follow? Answer 'valid' or 'invalid'."
        ),
        (
            "Consider the following argument:\n"
            f"  (1) {conflict_major}\n"
            f"  (2) {conflict_minor}\n"
            f"  Therefore: {conflict_conclusion}\n"
            "Is this argument logically valid? Respond with 'valid' "
            "or 'invalid' only."
        ),
        (
            f"If we assume '{conflict_major}' and '{conflict_minor}', "
            f"can we conclude that '{conflict_conclusion}'? Ignore "
            "real-world truth of the premises; only assess logical "
            "entailment. Say 'valid' or 'invalid'."
        ),
    ]
    paraphrases_control = [
        (
            "Consider the following argument:\n"
            f"  (1) {control_major}\n"
            f"  (2) {control_minor}\n"
            f"  Therefore: {control_conclusion}\n"
            "Is this argument logically valid? Respond with 'valid' "
            "or 'invalid' only."
        )
    ]

    # Belief-bias pairs need DIFFERENT correct_answers on the two
    # sides (valid vs invalid), so we can't use `_build_pair` which
    # assumes a shared correct_answer. Build both items explicitly.
    provenance = (
        "Belief-bias syllogism in the Evans (1983) 2x2 paradigm. "
        f"Conflict cell: valid={conflict_is_valid}, "
        f"believable={is_believable}; matched control flips validity "
        "while keeping the conclusion's believability."
    )
    conflict_item = BenchmarkItem(
        id=f"{pair_id}_conflict",
        category="syllogism",  # type: ignore[arg-type]
        subcategory=subcategory,
        conflict=True,
        difficulty=difficulty,
        prompt=conflict_prompt,
        system_prompt=None,
        correct_answer=correct_conflict,
        lure_answer=lure_conflict,
        answer_pattern=_bounded_regex(correct_conflict),
        lure_pattern=_bounded_regex(lure_conflict),
        matched_pair_id=pair_id,
        source="novel",
        provenance_note=provenance,
        paraphrases=tuple(paraphrases_conflict),
    )
    control_item = _build_control_override(
        pair_id=pair_id,
        category="syllogism",
        subcategory=subcategory,
        difficulty=difficulty,
        prompt=control_prompt,
        correct_answer=correct_control,
        source="novel",
        provenance_note=provenance,
        paraphrases_control=paraphrases_control,
    )
    return conflict_item, control_item


# The syllogism generator's control item has a different
# `correct_answer` from its conflict item (valid vs invalid), so the
# shared `_build_pair` helper -- which copies the conflict's
# correct_answer onto the control -- can't be used. The override
# below builds the control explicitly while keeping every other
# field consistent with the conflict.


def _build_control_override(
    *,
    pair_id: str,
    category: str,
    subcategory: str,
    difficulty: int,
    prompt: str,
    correct_answer: str,
    source: str,
    provenance_note: str,
    paraphrases_control: list[str],
) -> BenchmarkItem:
    """Build a control item whose correct_answer differs from conflict's.

    The matched control in a belief-bias pair has a different
    correct_answer (the 'other' validity cell) from the conflict.
    :func:`_build_pair` hard-codes one correct answer for both items,
    so we sidestep it for the control side of the syllogism pair.
    """
    return BenchmarkItem(
        id=f"{pair_id}_control",
        category=category,  # type: ignore[arg-type]
        subcategory=subcategory,
        conflict=False,
        difficulty=difficulty,
        prompt=prompt,
        system_prompt=None,
        correct_answer=correct_answer,
        lure_answer="",
        answer_pattern=_bounded_regex(correct_answer),
        lure_pattern="",
        matched_pair_id=pair_id,
        source=source,
        provenance_note=provenance_note + " (structural control).",
        paraphrases=tuple(paraphrases_control),
    )


# --------------------------------------------------------------------- #
# anchoring                                                             #
# --------------------------------------------------------------------- #


@beartype
def anchoring_isomorph(
    *,
    pair_id: str,
    question: str,
    true_value: int,
    high_anchor: int,
    low_anchor: int,
    units: str = "",
    difficulty: int = 2,
) -> tuple[BenchmarkItem, BenchmarkItem]:
    """Anchoring bias on a numeric estimate question.

    The conflict item first primes the model with a misleading
    numeric anchor (``high_anchor``), asking whether the true value
    is above or below that number; THEN it asks for a point estimate.
    The intuition literature (Tversky & Kahneman 1974) finds point
    estimates are pulled towards the anchor even when the anchor is
    known to be arbitrary.

    The control uses a neutral anchor (``low_anchor``) that is still
    above the true value by roughly the same margin so the
    comparison-question word length is matched, but phrased as a
    baseline reference that does not bias the estimate. In this
    benchmark the correct_answer is always the ``true_value``
    regardless of anchor. The lure is the ``high_anchor`` itself --
    if a model answers with the anchor verbatim it's obviously
    anchoring.

    All inputs are integers; the question is free-form text that
    should end with a period and NOT contain the true value.
    """
    if high_anchor <= true_value:
        raise ValueError("high_anchor must be above true_value for the lure")
    if low_anchor >= true_value:
        raise ValueError("low_anchor must be below true_value")
    if high_anchor == low_anchor:
        raise ValueError("high_anchor and low_anchor must differ")
    if str(true_value) == str(high_anchor):
        raise ValueError("lure collapses onto correct answer")
    if not question.strip():
        raise ValueError("question must be non-empty")

    u_anchor = f" {units}" if units else ""
    u_ans = f" Express your answer in {units}." if units else ""
    conflict_prompt = (
        f"Consider this question: {question}\n\n"
        f"Is the true answer greater or less than {high_anchor}{u_anchor}? "
        "After deciding, give your best single-number point estimate "
        f"of the true answer.{u_ans} Answer with the number only."
    )
    control_prompt = (
        f"Consider this question: {question}\n\n"
        "Give your best single-number point estimate of the true "
        f"answer.{u_ans} Answer with the number only."
    )

    paraphrases_conflict = [
        (
            f"{question}\n"
            f"First decide: is the true answer above or below "
            f"{high_anchor}{u_anchor}? Then give your best numeric "
            f"estimate of the true answer. Respond with a single "
            "integer."
        ),
        (
            f"You are asked: {question} Before answering, note that "
            f"the number {high_anchor}{u_anchor} has been proposed by "
            "another respondent as a ballpark. Is the truth higher "
            f"or lower than {high_anchor}? What is your own best "
            "single-number estimate? Give the number only."
        ),
        (
            f"Question: {question} Treat {high_anchor}{u_anchor} as a "
            "rough starting guess. Adjust upward or downward as you "
            "see fit, then report a single integer point estimate."
        ),
    ]
    paraphrases_control = [
        (
            f"Question: {question} Provide your best single-number "
            f"estimate. Respond with one integer."
        )
    ]

    return _build_pair(
        pair_id=pair_id,
        category="anchoring",
        subcategory="numeric_estimate",
        difficulty=difficulty,
        conflict_prompt=conflict_prompt,
        control_prompt=control_prompt,
        correct_answer=str(true_value),
        lure_answer=str(high_anchor),
        source="novel",
        provenance_note=(
            "Numeric-anchoring isomorph in the Tversky-Kahneman "
            f"tradition. Conflict anchor={high_anchor}, "
            f"true_value={true_value}, control is unanchored."
        ),
        paraphrases_conflict=paraphrases_conflict,
        paraphrases_control=paraphrases_control,
    )


# --------------------------------------------------------------------- #
# framing (Asian-disease style)                                         #
# --------------------------------------------------------------------- #


@beartype
def framing_isomorph(
    *,
    pair_id: str,
    scenario: str,
    n_total: int,
    n_certain_save: int,
    prob_save_all: float,
    prefer_sure: bool,
    subcategory: str = "asian_disease",
    difficulty: int = 2,
) -> tuple[BenchmarkItem, BenchmarkItem]:
    """Asian-disease framing pair.

    Classic Tversky & Kahneman (1981) structure: a population of
    ``n_total`` faces a threat. Option A is CERTAIN: saves
    ``n_certain_save`` of them (or, equivalently, lets ``n_total -
    n_certain_save`` die). Option B is PROBABILISTIC: saves all
    ``n_total`` with probability ``prob_save_all``, or none with
    probability ``1 - prob_save_all``.

    Expected values match by construction: the caller picks a
    ``n_certain_save`` such that ``n_certain_save == prob_save_all *
    n_total`` (the generator validates this). With EVs equal, the
    normatively correct answer depends on the caller's
    ``prefer_sure`` flag -- but in the framing paradigm what actually
    matters is that the SAME option A is framed positively (gain) in
    the conflict cell and negatively (loss) in the other cell. The
    lure is the "flip" induced by the frame.

    To match with a single correct answer we adopt the convention:
    both items share the same underlying numbers; the conflict item
    uses a gain frame ("200 will be saved") and the control uses an
    explicit neutral frame ("the outcome distribution is ... the
    correct answer is the EV-maximising choice"). The correct answer
    is the SURE option when ``prefer_sure=True`` and the GAMBLE
    otherwise, chosen by the caller so multi-item data sets can test
    both cells.

    The generator raises if the EVs are not exactly equal. This is
    stricter than the literature's convention but keeps the
    "correct_answer" well-defined.
    """
    if n_total <= 0 or n_certain_save < 0 or n_certain_save > n_total:
        raise ValueError("invalid populations")
    if not 0.0 < prob_save_all < 1.0:
        raise ValueError("prob_save_all must be strictly between 0 and 1")
    # EV equality: n_certain_save == prob_save_all * n_total.
    ev_gamble = prob_save_all * n_total
    if abs(ev_gamble - n_certain_save) > 1e-9:
        raise ValueError(
            f"EV mismatch: gamble={ev_gamble}, sure={n_certain_save}. "
            "Pick numbers so ev_gamble == n_certain_save exactly."
        )

    n_certain_die = n_total - n_certain_save
    prob_die_all = 1.0 - prob_save_all
    p_save_pct = _pct(prob_save_all)
    p_die_pct = _pct(prob_die_all)

    sure_label = "A"
    gamble_label = "B"
    correct_label = sure_label if prefer_sure else gamble_label
    lure_label = gamble_label if prefer_sure else sure_label

    # Gain frame: conflict item phrases the sure option as LIVES SAVED.
    conflict_prompt = (
        f"{scenario} The threat will kill {n_total} people if "
        "nothing is done. Two rescue plans are proposed:\n\n"
        f"Plan {sure_label}: {n_certain_save} people will be saved "
        "for certain.\n"
        f"Plan {gamble_label}: with probability {p_save_pct} all "
        f"{n_total} people will be saved; with probability "
        f"{p_die_pct} nobody will be saved.\n\n"
        "Assume the expected number of lives saved is the sole "
        "criterion. Which plan should be chosen? Answer 'A' or 'B'."
    )
    # Control: neutral frame that states BOTH lives-saved and
    # lives-lost, and explicitly instructs the model to pick on EV.
    control_prompt = (
        f"{scenario} The threat puts {n_total} people at risk. Two "
        "rescue plans are proposed, with outcomes fully specified:\n\n"
        f"Plan {sure_label}: exactly {n_certain_save} saved and "
        f"exactly {n_certain_die} lost.\n"
        f"Plan {gamble_label}: with probability {p_save_pct} all "
        f"{n_total} saved; with probability {p_die_pct} all "
        f"{n_total} lost.\n\n"
        "Choose whichever plan maximises the expected number of "
        "lives saved. If the expected values are tied, pick the "
        f"plan the instructions favour ({'A' if prefer_sure else 'B'} "
        "by default). Answer 'A' or 'B'."
    )

    paraphrases_conflict = [
        (
            f"{scenario} Without intervention {n_total} people will "
            "die. Choose between:\n"
            f"  A) {n_certain_save} people saved for certain.\n"
            f"  B) {p_save_pct} chance of saving all {n_total}, "
            f"{p_die_pct} chance of saving none.\n"
            "Which option should be picked on expected-value grounds? "
            "'A' or 'B'."
        ),
        (
            f"{scenario} There are two response plans for a threat to "
            f"{n_total} people. Plan A: {n_certain_save} lives saved "
            f"with certainty. Plan B: a {p_save_pct} chance of saving "
            "everyone, otherwise no one. Pick the plan that maximises "
            "expected lives saved. Reply with one letter."
        ),
    ]
    paraphrases_control = [
        (
            f"{scenario} Two plans are on the table for a crisis "
            f"affecting {n_total} people. Plan A yields exactly "
            f"{n_certain_save} saved and {n_certain_die} lost. Plan B "
            f"yields a {p_save_pct}/{p_die_pct} split over "
            f"(all saved, none saved). Assume EV-maximising choice. "
            "Answer 'A' or 'B'."
        )
    ]

    return _build_pair(
        pair_id=pair_id,
        category="framing",
        subcategory=subcategory,
        difficulty=difficulty,
        conflict_prompt=conflict_prompt,
        control_prompt=control_prompt,
        correct_answer=correct_label,
        lure_answer=lure_label,
        source="novel",
        provenance_note=(
            "Framing isomorph in the Tversky-Kahneman (1981) "
            f"Asian-disease family. n_total={n_total}, "
            f"sure={n_certain_save}, p_save={prob_save_all}. "
            f"Conflict uses a gain frame; control is explicit/neutral. "
            f"Correct = {'sure' if prefer_sure else 'gamble'}."
        ),
        paraphrases_conflict=paraphrases_conflict,
        paraphrases_control=paraphrases_control,
    )


# --------------------------------------------------------------------- #
# conjunction fallacy (Linda)                                           #
# --------------------------------------------------------------------- #


@beartype
def conjunction_fallacy_isomorph(
    *,
    pair_id: str,
    person_name: str,
    person_description: str,
    feature_a: str,
    feature_b: str,
    subcategory: str = "linda_isomorph",
    difficulty: int = 2,
) -> tuple[BenchmarkItem, BenchmarkItem]:
    """Linda-style conjunction-fallacy pair.

    The conflict item describes a person in a way that matches
    ``feature_b`` (the "narrow" stereotype). The model must rank two
    statements:

      (A) "``person_name`` is a ``feature_a``."
      (B) "``person_name`` is a ``feature_a`` AND a ``feature_b``."

    P(A) >= P(A and B) for any B, so A is always at least as
    probable. The System-1 lure is to rank the conjunction (B) as
    more probable because the description matches the conjunction's
    stereotype.

    The control uses a description that is NEUTRAL -- no stereotype
    push -- so deliberation and intuition both prefer A.

    Answers are returned as 'A' or 'B' so extraction is simple.
    """
    if feature_a.strip().lower() in feature_b.strip().lower():
        raise ValueError(
            "feature_a appears inside feature_b -- statements collapse"
        )
    if not person_description.strip():
        raise ValueError("person_description must be non-empty")

    option_a = f"{person_name} is a {feature_a}."
    option_b = f"{person_name} is a {feature_a} and a {feature_b}."

    conflict_prompt = (
        f"{person_description}\n\n"
        "Which of the following statements is more probable?\n"
        f"A) {option_a}\n"
        f"B) {option_b}\n\n"
        "Answer with the single letter A or B."
    )
    control_prompt = (
        f"{person_name} is a randomly selected adult; no further "
        "personal details are known.\n\n"
        "Which of the following statements is more probable?\n"
        f"A) {option_a}\n"
        f"B) {option_b}\n\n"
        "Answer with the single letter A or B."
    )

    paraphrases_conflict = [
        (
            f"Consider this description of a person. {person_description} "
            f"Which is strictly more probable under ordinary "
            f"probability axioms?\n"
            f"(A) {option_a}\n"
            f"(B) {option_b}\n"
            "Respond with one letter."
        ),
        (
            f"{person_description} Rank the following by probability:\n"
            f"(A) {option_a}\n"
            f"(B) {option_b}\n"
            "Which is MORE probable, A or B? One letter only."
        ),
        (
            f"A description of {person_name}: {person_description} "
            f"Given this, which statement has higher probability, "
            f"A ({option_a}) or B ({option_b})? Answer with A or B."
        ),
    ]
    paraphrases_control = [
        (
            f"{person_name} is drawn uniformly at random from the adult "
            f"population. Which statement has higher probability?\n"
            f"A) {option_a}\n"
            f"B) {option_b}\n"
            "Answer with one letter."
        )
    ]

    return _build_pair(
        pair_id=pair_id,
        category="conjunction",
        subcategory=subcategory,
        difficulty=difficulty,
        conflict_prompt=conflict_prompt,
        control_prompt=control_prompt,
        correct_answer="A",
        lure_answer="B",
        source="novel",
        provenance_note=(
            "Conjunction-fallacy isomorph in the Tversky-Kahneman "
            f"(1983) Linda tradition. feature_a={feature_a!r}, "
            f"feature_b={feature_b!r}. Conflict has a stereotype cue "
            "matching the conjunction; control is neutral."
        ),
        paraphrases_conflict=paraphrases_conflict,
        paraphrases_control=paraphrases_control,
    )


# --------------------------------------------------------------------- #
# base-rate neglect — natural frequency framing (Gigerenzer)            #
# --------------------------------------------------------------------- #


@beartype
def base_rate_natural_freq_isomorph(
    *,
    pair_id: str,
    description: str,
    rare_group: str,
    common_group: str,
    n_total: int,
    n_rare: int,
    n_common: int,
    subcategory: str = "natural_frequency",
    difficulty: int = 3,
) -> tuple[BenchmarkItem, BenchmarkItem]:
    """Base-rate pair using natural frequency format (Gigerenzer, 1995).

    Gigerenzer argues that base rate neglect is an artefact of
    probability framing and largely disappears when the same information
    is presented as natural frequencies ("10 out of 1000" instead of
    "1%"). If the model still falls for the lure under natural frequency
    framing, the vulnerability is robust to Gigerenzer's critique.

    Same logic as :func:`base_rate_isomorph` but the prompt says
    "Out of N people, X are rare_group and Y are common_group"
    instead of "X% are rare_group."
    """
    if n_rare >= n_common:
        raise ValueError("n_rare must be less than n_common")
    if n_rare + n_common > n_total:
        raise ValueError("n_rare + n_common cannot exceed n_total")
    if n_common < n_rare * 3:
        raise ValueError("n_common must dominate n_rare (>=3x)")

    conflict_prompt = (
        f"Imagine a room of {n_total} people. Of these {n_total} "
        f"people, {n_common} are {common_group}s and {n_rare} are "
        f"{rare_group}s. One person is drawn at random and described "
        f"as follows: {description} "
        f"Out of the {n_total} people in the room, is this person "
        f"more likely to be one of the {n_common} {common_group}s "
        f"or one of the {n_rare} {rare_group}s? "
        f"Answer with the single word {common_group!s} or "
        f"{rare_group!s}."
    )
    control_prompt = (
        f"Imagine a room of {n_total} people. Of these {n_total} "
        f"people, {n_common} are {common_group}s and {n_rare} are "
        f"{rare_group}s. One person is drawn at random with no "
        f"further information. Out of the {n_total} people, is this "
        f"person more likely to be one of the {n_common} "
        f"{common_group}s or one of the {n_rare} {rare_group}s? "
        f"Answer with the single word {common_group!s} or "
        f"{rare_group!s}."
    )

    paraphrases_conflict = [
        (
            f"There are {n_total} people. {n_common} of them are "
            f"{common_group}s. {n_rare} of them are {rare_group}s. "
            f"One person has this profile: {description} "
            f"Which group does this person more likely belong to — "
            f"the {n_common} {common_group}s or the {n_rare} "
            f"{rare_group}s? One word."
        ),
        (
            f"In a group of {n_total}, there are {n_common} "
            f"{common_group}s and {n_rare} {rare_group}s. A randomly "
            f"selected individual is described: {description} "
            f"Is this individual more likely a {common_group} or a "
            f"{rare_group}? Respond with one word."
        ),
    ]
    paraphrases_control = [
        (
            f"In a group of {n_total}, there are {n_common} "
            f"{common_group}s and {n_rare} {rare_group}s. One "
            f"person is selected at random with no description. "
            f"More likely a {common_group} or {rare_group}? One word."
        ),
    ]

    return _build_pair(
        pair_id=pair_id,
        category="base_rate",
        subcategory=subcategory,
        difficulty=difficulty,
        conflict_prompt=conflict_prompt,
        control_prompt=control_prompt,
        correct_answer=common_group,
        lure_answer=rare_group,
        source="novel",
        provenance_note=(
            "Base-rate neglect isomorph in Gigerenzer's (1995) natural "
            f"frequency format. n_total={n_total}, n_rare={n_rare}, "
            f"n_common={n_common}. Tests whether vulnerability is "
            "robust to framing format."
        ),
        paraphrases_conflict=paraphrases_conflict,
        paraphrases_control=paraphrases_control,
    )


# --------------------------------------------------------------------- #
# sunk cost fallacy                                                     #
# --------------------------------------------------------------------- #


@beartype
def sunk_cost_isomorph(
    *,
    pair_id: str,
    scenario: str,
    investment_description: str,
    amount_spent: str,
    negative_signal: str,
    alternative: str,
    subcategory: str = "loss_aversion",
    difficulty: int = 3,
) -> tuple[BenchmarkItem, BenchmarkItem]:
    """Sunk cost fallacy pair in the loss-aversion heuristic family.

    The conflict item describes a project where substantial resources
    have been invested, but new information makes it clear the rational
    choice is to stop and switch to an alternative. The System-1 lure
    is to continue because of the sunk investment ("we've already put
    so much in..."). The normatively correct answer is to stop/switch
    because sunk costs are irrelevant to future-looking decisions.

    The control poses a structurally identical scenario but with a
    POSITIVE signal: the investment IS on track and continuing is
    genuinely rational, so both S1 and S2 agree on "continue."

    This category tests a DIFFERENT heuristic family (loss aversion)
    than base_rate/conjunction/syllogism (representativeness), which
    broadens the domain-specificity claim.
    """
    if not scenario.strip() or not investment_description.strip():
        raise ValueError("scenario and investment_description must be non-empty")
    if not negative_signal.strip() or not alternative.strip():
        raise ValueError("negative_signal and alternative must be non-empty")

    conflict_prompt = (
        f"{scenario} {investment_description} So far, {amount_spent} has "
        f"been spent on this project. However, {negative_signal} "
        f"Meanwhile, {alternative}\n\n"
        "Based purely on expected future outcomes (ignoring what has "
        "already been spent), should you continue the current project "
        "or switch to the alternative?\n\n"
        "Answer with one word: Continue or Switch."
    )
    control_prompt = (
        f"{scenario} {investment_description} So far, {amount_spent} has "
        "been spent on this project. Recent evaluations confirm the "
        "project is on track, with projected returns exceeding remaining "
        "costs by a wide margin. No better alternative is currently "
        "available.\n\n"
        "Based purely on expected future outcomes (ignoring what has "
        "already been spent), should you continue the current project "
        "or abandon it?\n\n"
        "Answer with one word: Continue or Abandon."
    )

    paraphrases_conflict = [
        (
            f"Consider this scenario. {scenario} {investment_description} "
            f"Already spent: {amount_spent}. New information: "
            f"{negative_signal} An alternative exists: {alternative} "
            "Ignoring sunk costs entirely, which option has higher "
            "expected value going forward: continuing or switching? "
            "One word: Continue or Switch."
        ),
        (
            f"{scenario} {investment_description} Investment to date: "
            f"{amount_spent}. {negative_signal} {alternative} "
            "A rational decision-maker who ignores past expenditures "
            "would choose to: Continue or Switch? Answer in one word."
        ),
    ]
    paraphrases_control = [
        (
            f"{scenario} {investment_description} Investment to date: "
            f"{amount_spent}. The project is proceeding well with "
            "strong projected returns. No superior alternative exists. "
            "Should you continue or abandon? One word."
        ),
    ]

    return _build_pair(
        pair_id=pair_id,
        category="sunk_cost",
        subcategory=subcategory,
        difficulty=difficulty,
        conflict_prompt=conflict_prompt,
        control_prompt=control_prompt,
        correct_answer="Switch",
        lure_answer="Continue",
        source="novel",
        provenance_note=(
            "Sunk cost fallacy isomorph in the loss-aversion "
            "tradition (Arkes & Blumer, 1985; Thaler, 1980). "
            f"amount_spent={amount_spent!r}. Conflict item has a "
            "negative signal making continuation irrational; "
            "control item has the project on track."
        ),
        paraphrases_conflict=paraphrases_conflict,
        paraphrases_control=paraphrases_control,
    )


# --------------------------------------------------------------------- #
# arithmetic trap                                                       #
# --------------------------------------------------------------------- #


@beartype
def arithmetic_trap_isomorph(
    *,
    pair_id: str,
    start: int,
    steps: list[tuple[str, int]],
    trap_step: int,
    difficulty: int = 2,
    scenario: str | None = None,
) -> tuple[BenchmarkItem, BenchmarkItem]:
    """Multi-step arithmetic with a tempting-wrong-intermediate lure.

    ``steps`` is a sequence of ``(op, value)`` pairs applied in order
    from ``start``. Supported operators are ``+ - * /``. Division must
    be exact at every intermediate.

    The conflict prompt wraps the computation in a short narrative
    that tempts a System-1 reader to apply ``trap_step`` in the WRONG
    order or with the WRONG sign; the control prompt lays out the
    same arithmetic as an explicit numbered list so there is no
    misinterpretation temptation.

    ``trap_step`` is the index into ``steps`` of the "tempting wrong
    step." The lure answer is computed by MIS-APPLYING that step --
    specifically, by using the "obvious" surface interpretation
    (addition instead of subtraction when the narrative contains "X
    less", etc.). The lure is computed automatically as
    ``correct ± 2 * |value|`` for the trap step; if the two collapse,
    the generator raises.
    """
    if not steps:
        raise ValueError("steps must be non-empty")
    if not (0 <= trap_step < len(steps)):
        raise ValueError("trap_step is out of range")

    # Correct answer: apply steps exactly.
    running = start
    for op, v in steps:
        running = _apply(running, op, v)
    correct = running

    # Lure answer: flip the sign / operation of the trap step and
    # recompute. This is a structural operationalisation of "the
    # tempting wrong intermediate." For trap_step we map + -> -,
    # - -> +, * -> /, / -> *.
    lure_steps: list[tuple[str, int]] = list(steps)
    op, v = lure_steps[trap_step]
    lure_steps[trap_step] = (_flip_op(op), v)
    lure = start
    for op2, v2 in lure_steps:
        lure = _apply(lure, op2, v2)
    if lure == correct:
        raise ValueError(
            "lure arithmetic equals correct arithmetic; pick different steps"
        )

    scenario = scenario or (
        "A shopkeeper tracks the contents of a display box through "
        "the day."
    )
    # Build a narrative prompt that contains the operations in words.
    narrative_parts: list[str] = [
        f"{scenario} The box starts with {start} items."
    ]
    for i, (op, v) in enumerate(steps):
        narrative_parts.append(_narrate_step(i, op, v, trap_index=trap_step))
    conflict_prompt = (
        " ".join(narrative_parts)
        + " How many items are in the box at the end of the day? "
        "Answer with a single integer."
    )

    # Control: numbered arithmetic sequence, no narrative distractor.
    control_lines = [f"Start with the number {start}."]
    for i, (op, v) in enumerate(steps):
        control_lines.append(f"Step {i+1}: apply {op} {v}.")
    control_lines.append(
        "Apply the steps in the order listed. What is the final "
        "number? Answer with a single integer."
    )
    control_prompt = "\n".join(control_lines)

    paraphrases_conflict = [
        (
            " ".join(narrative_parts)
            + " What is the final item count in the box? Respond "
            "with one integer."
        ),
        (
            " ".join(narrative_parts)
            + " Report the final count at day's end. Integer only."
        ),
    ]
    paraphrases_control = [
        (
            f"Begin at {start}. "
            + " ".join(
                f"Then apply {op} {v}." for op, v in steps
            )
            + " What is the resulting number?"
        )
    ]

    return _build_pair(
        pair_id=pair_id,
        category="arithmetic",
        subcategory="multistep_trap",
        difficulty=difficulty,
        conflict_prompt=conflict_prompt,
        control_prompt=control_prompt,
        correct_answer=str(correct),
        lure_answer=str(lure),
        source="novel",
        provenance_note=(
            f"Multi-step arithmetic isomorph, start={start}, "
            f"steps={steps}, trap_step={trap_step}. Conflict item "
            "wraps the arithmetic in a narrative that tempts the "
            "wrong op on the trap step; control lists the steps "
            "explicitly."
        ),
        paraphrases_conflict=paraphrases_conflict,
        paraphrases_control=paraphrases_control,
    )


def _apply(x: int, op: str, v: int) -> int:
    """Apply one arithmetic step; raise on non-exact divisions."""
    if op == "+":
        return x + v
    if op == "-":
        return x - v
    if op == "*":
        return x * v
    if op == "/":
        if v == 0 or x % v != 0:
            raise ValueError(f"non-exact division {x}/{v}")
        return x // v
    raise ValueError(f"unsupported operator {op!r}")


def _flip_op(op: str) -> str:
    return {"+": "-", "-": "+", "*": "/", "/": "*"}[op]


def _narrate_step(i: int, op: str, v: int, *, trap_index: int) -> str:
    """Render a step as a one-sentence English clause.

    The trap step uses ambiguous or surface-misleading language
    (e.g. 'X less than the starting amount' instead of 'subtract X'),
    while other steps are clearly phrased.
    """
    if i == trap_index:
        if op == "+":
            return (
                f"A regular customer returns {v} items which end up "
                "back in the box."
            )
        if op == "-":
            return (
                f"A sign on the box reads 'today there are {v} fewer "
                "than at opening' -- remove that many."
            )
        if op == "*":
            return (
                f"Mid-morning, a bulk shipment multiplies the current "
                f"count by {v}."
            )
        if op == "/":
            return (
                f"Near close of day, staff move items so that only "
                f"one-in-{v} remain in the display box."
            )
    # Non-trap steps: clear imperative language.
    if op == "+":
        return f"Then {v} items are added."
    if op == "-":
        return f"Then {v} items are removed."
    if op == "*":
        return f"The count is then multiplied by {v}."
    if op == "/":
        return f"The count is then divided by {v}."
    raise ValueError(f"unsupported operator {op!r}")


# --------------------------------------------------------------------- #
# loss aversion                                                         #
# --------------------------------------------------------------------- #


@beartype
def loss_aversion_isomorph(
    *,
    pair_id: str,
    scenario: str,
    win_prob: float,
    win_amount: int | float,
    lose_prob: float,
    lose_amount: int | float,
    subcategory: str = "positive_ev_gamble",
    difficulty: int = 3,
) -> tuple[BenchmarkItem, BenchmarkItem]:
    """Loss aversion pair: reject a positive-EV gamble because of loss framing.

    The conflict item presents a gamble with clearly positive expected
    value (win_prob * win_amount - lose_prob * lose_amount > 0) but
    frames it with an explicit loss component. The System-1 lure is
    to Reject the gamble due to loss aversion even though the EV is
    positive. The correct answer is Accept.

    The control presents the same EV without any loss framing (as a
    guaranteed expected gain), so both S1 and S2 agree on Accept.

    The EV ratio (EV / lose_amount) must be > 1.2 so the correct
    answer is unambiguous.
    """
    if not (0.0 < win_prob < 1.0 and 0.0 < lose_prob < 1.0):
        raise ValueError("probabilities must be strictly between 0 and 1")
    if abs(win_prob + lose_prob - 1.0) > 1e-9:
        raise ValueError("win_prob + lose_prob must equal 1.0")
    if win_amount <= 0 or lose_amount <= 0:
        raise ValueError("amounts must be positive")

    ev = win_prob * win_amount - lose_prob * lose_amount
    if ev <= 0:
        raise ValueError(f"EV must be positive, got {ev:.2f}")
    if ev / lose_amount < 1.2:
        raise ValueError(
            f"EV/lose_amount ratio must be > 1.2 for unambiguous answer, "
            f"got {ev / lose_amount:.2f}"
        )

    ev_str = f"${ev:,.0f}" if ev >= 1 else f"${ev:.2f}"
    win_pct = _pct(win_prob)
    lose_pct = _pct(lose_prob)

    conflict_prompt = (
        f"{scenario} You are offered the following gamble: "
        f"{win_pct} chance of winning ${win_amount:,.0f}, "
        f"{lose_pct} chance of losing ${lose_amount:,.0f}. "
        f"The expected value is {ev_str}.\n\n"
        "Should you accept or reject this gamble based on expected "
        "value alone?\n\n"
        "Answer with one word: Accept or Reject."
    )
    control_prompt = (
        f"{scenario} You are offered a payment with an expected "
        f"value of {ev_str}. There is no possibility of loss. "
        "Should you accept or reject this offer based on expected "
        "value alone?\n\n"
        "Answer with one word: Accept or Reject."
    )

    paraphrases_conflict = [
        (
            f"Consider this gamble. {scenario} "
            f"Win ${win_amount:,.0f} with probability {win_pct}; "
            f"lose ${lose_amount:,.0f} with probability {lose_pct}. "
            f"EV = {ev_str}. On expected value grounds, should you "
            "accept or reject? One word: Accept or Reject."
        ),
        (
            f"{scenario} A gamble offers {win_pct} chance of "
            f"+${win_amount:,.0f} and {lose_pct} chance of "
            f"-${lose_amount:,.0f} (EV = {ev_str}). "
            "Judging purely by expected value, do you Accept or "
            "Reject? Answer in one word."
        ),
    ]
    paraphrases_control = [
        (
            f"{scenario} You can receive an expected payment of "
            f"{ev_str} with no downside risk. Accept or Reject? "
            "One word."
        ),
    ]

    return _build_pair(
        pair_id=pair_id,
        category="loss_aversion",
        subcategory=subcategory,
        difficulty=difficulty,
        conflict_prompt=conflict_prompt,
        control_prompt=control_prompt,
        correct_answer="Accept",
        lure_answer="Reject",
        source="novel",
        provenance_note=(
            "Loss aversion isomorph in the Kahneman-Tversky (1979) "
            "prospect theory tradition. Conflict presents a positive-EV "
            f"gamble (EV={ev_str}) with explicit loss framing; "
            "control removes the loss frame."
        ),
        paraphrases_conflict=paraphrases_conflict,
        paraphrases_control=paraphrases_control,
    )


# --------------------------------------------------------------------- #
# certainty effect                                                       #
# --------------------------------------------------------------------- #


@beartype
def certainty_effect_isomorph(
    *,
    pair_id: str,
    scenario: str,
    certain_amount: int | float,
    gamble_prob: float,
    gamble_amount: int | float,
    control_a_prob: float,
    control_a_amount: int | float,
    control_b_prob: float,
    control_b_amount: int | float,
    subcategory: str = "allais_isomorph",
    difficulty: int = 3,
) -> tuple[BenchmarkItem, BenchmarkItem]:
    """Certainty effect pair: prefer certain option despite lower EV.

    The conflict item offers A (certain_amount for sure) vs B
    (gamble_prob chance of gamble_amount). B has higher EV, but the
    certainty of A acts as a System-1 lure. Correct answer is B.

    The control offers two probabilistic options (no certainty anchor)
    where B still has higher EV. Both S1 and S2 agree on B.

    Answer format: A or B. Correct is always B.
    """
    if not (0.0 < gamble_prob <= 1.0):
        raise ValueError("gamble_prob must be in (0, 1]")
    if certain_amount <= 0 or gamble_amount <= 0:
        raise ValueError("amounts must be positive")

    ev_certain = certain_amount
    ev_gamble = gamble_prob * gamble_amount
    if ev_gamble <= ev_certain:
        raise ValueError(
            f"gamble EV ({ev_gamble:.0f}) must exceed certain amount "
            f"({ev_certain:.0f}) for B to be correct"
        )

    if not (0.0 < control_a_prob <= 1.0 and 0.0 < control_b_prob <= 1.0):
        raise ValueError("control probabilities must be in (0, 1]")
    ev_ctrl_a = control_a_prob * control_a_amount
    ev_ctrl_b = control_b_prob * control_b_amount
    if ev_ctrl_b <= ev_ctrl_a:
        raise ValueError(
            f"control B EV ({ev_ctrl_b:.0f}) must exceed control A EV "
            f"({ev_ctrl_a:.0f}) for B to be correct"
        )

    gamble_pct = _pct(gamble_prob)
    ctrl_a_pct = _pct(control_a_prob)
    ctrl_b_pct = _pct(control_b_prob)

    conflict_prompt = (
        f"{scenario}\n\n"
        f"Option A: ${certain_amount:,.0f} for certain.\n"
        f"Option B: {gamble_pct} chance of ${gamble_amount:,.0f} "
        f"(expected value = ${ev_gamble:,.0f}).\n\n"
        "Which option has higher expected value? Answer A or B."
    )
    control_prompt = (
        f"{scenario}\n\n"
        f"Option A: {ctrl_a_pct} chance of ${control_a_amount:,.0f} "
        f"(expected value = ${ev_ctrl_a:,.0f}).\n"
        f"Option B: {ctrl_b_pct} chance of ${control_b_amount:,.0f} "
        f"(expected value = ${ev_ctrl_b:,.0f}).\n\n"
        "Which option has higher expected value? Answer A or B."
    )

    paraphrases_conflict = [
        (
            f"{scenario} Choose between: "
            f"A) a guaranteed ${certain_amount:,.0f}; "
            f"B) a {gamble_pct} chance of ${gamble_amount:,.0f} "
            f"(EV=${ev_gamble:,.0f}). "
            "Which has higher expected value, A or B? One letter."
        ),
        (
            f"{scenario} "
            f"A: ${certain_amount:,.0f} certain (EV=${ev_certain:,.0f}). "
            f"B: {gamble_pct} of ${gamble_amount:,.0f} "
            f"(EV=${ev_gamble:,.0f}). "
            "Comparing expected values only, which is better? "
            "Reply A or B."
        ),
    ]
    paraphrases_control = [
        (
            f"{scenario} "
            f"A: {ctrl_a_pct} of ${control_a_amount:,.0f} "
            f"(EV=${ev_ctrl_a:,.0f}). "
            f"B: {ctrl_b_pct} of ${control_b_amount:,.0f} "
            f"(EV=${ev_ctrl_b:,.0f}). "
            "Which option has higher expected value? A or B."
        ),
    ]

    return _build_pair(
        pair_id=pair_id,
        category="certainty_effect",
        subcategory=subcategory,
        difficulty=difficulty,
        conflict_prompt=conflict_prompt,
        control_prompt=control_prompt,
        correct_answer="B",
        lure_answer="A",
        source="novel",
        provenance_note=(
            "Certainty effect isomorph in the Allais (1953) / "
            "Kahneman-Tversky (1979) tradition. Conflict offers a "
            f"certain ${certain_amount:,.0f} vs {gamble_pct} of "
            f"${gamble_amount:,.0f} (EV=${ev_gamble:,.0f}); "
            "control removes the certainty anchor."
        ),
        paraphrases_conflict=paraphrases_conflict,
        paraphrases_control=paraphrases_control,
    )


# --------------------------------------------------------------------- #
# availability heuristic                                                 #
# --------------------------------------------------------------------- #


@beartype
def availability_isomorph(
    *,
    pair_id: str,
    vivid_event: str,
    mundane_event: str,
    vivid_deaths: int,
    mundane_deaths: int,
    control_event_a: str,
    control_event_b: str,
    control_deaths_a: int,
    control_deaths_b: int,
    subcategory: str = "frequency_judgment",
    difficulty: int = 3,
) -> tuple[BenchmarkItem, BenchmarkItem]:
    """Availability heuristic pair: vivid/media-salient vs mundane frequency.

    The conflict item asks which of two causes kills more people per
    year: a vivid, media-prominent but rarer event vs a mundane but
    more frequent one. The System-1 lure is to name the vivid event
    (availability bias). The correct answer is the mundane event.

    The control uses two events with similar salience where the more
    frequent one is also more salient, so S1 and S2 agree.

    The vivid event must be at least 3x rarer than the mundane one.
    """
    if vivid_deaths <= 0 or mundane_deaths <= 0:
        raise ValueError("death counts must be positive")
    if mundane_deaths < vivid_deaths * 3:
        raise ValueError(
            f"mundane_deaths ({mundane_deaths}) must be at least 3x "
            f"vivid_deaths ({vivid_deaths}) for the lure to exist"
        )
    if vivid_event.strip().lower() == mundane_event.strip().lower():
        raise ValueError("events must be different")

    if control_deaths_a <= 0 or control_deaths_b <= 0:
        raise ValueError("control death counts must be positive")
    if control_deaths_b <= control_deaths_a:
        raise ValueError(
            "control_deaths_b must exceed control_deaths_a so B is correct"
        )

    conflict_prompt = (
        "Which of the following causes more deaths per year in the "
        "United States?\n\n"
        f"A) {vivid_event}\n"
        f"B) {mundane_event}\n\n"
        "Answer with one letter: A or B."
    )
    control_prompt = (
        "Which of the following causes more deaths per year in the "
        "United States?\n\n"
        f"A) {control_event_a}\n"
        f"B) {control_event_b}\n\n"
        "Answer with one letter: A or B."
    )

    paraphrases_conflict = [
        (
            f"In the US, which kills more people annually: "
            f"{vivid_event.lower()} or {mundane_event.lower()}? "
            "Answer A or B."
        ),
        (
            "Consider two causes of death in the United States. "
            f"A: {vivid_event}. B: {mundane_event}. "
            "Which has a higher annual death toll? Reply with one "
            "letter."
        ),
    ]
    paraphrases_control = [
        (
            "In the US, which kills more people annually: "
            f"A) {control_event_a}, or B) {control_event_b}? "
            "Answer A or B."
        ),
    ]

    return _build_pair(
        pair_id=pair_id,
        category="availability",
        subcategory=subcategory,
        difficulty=difficulty,
        conflict_prompt=conflict_prompt,
        control_prompt=control_prompt,
        correct_answer="B",
        lure_answer="A",
        source="novel",
        provenance_note=(
            "Availability heuristic isomorph in the Tversky-Kahneman "
            "(1973) tradition. Conflict: vivid/media-salient "
            f"{vivid_event!r} (~{vivid_deaths}/yr) vs mundane "
            f"{mundane_event!r} (~{mundane_deaths}/yr). "
            "Control uses events with aligned salience and frequency."
        ),
        paraphrases_conflict=paraphrases_conflict,
        paraphrases_control=paraphrases_control,
    )


# --------------------------------------------------------------------- #
# bulk helper                                                           #
# --------------------------------------------------------------------- #


@beartype
def make_many(
    maker: Callable[..., tuple[BenchmarkItem, BenchmarkItem]],
    specs: list[dict[str, Any]],
) -> list[BenchmarkItem]:
    """Apply a generator across a list of parameter dicts.

    Duplicates the helper in :mod:`templates` so callers of this
    module don't need to import both files.
    """
    out: list[BenchmarkItem] = []
    for spec in specs:
        try:
            c, x = maker(**spec)
        except Exception as e:
            raise ValueError(
                f"failed building item with spec {spec}: {e}"
            ) from e
        out.append(c)
        out.append(x)
    return out
