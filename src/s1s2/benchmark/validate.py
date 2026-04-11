"""Structural validator for the cognitive bias benchmark.

Run directly::

    python -m s1s2.benchmark.validate data/benchmark/benchmark.jsonl

Exit code is 0 when every check passes and 1 when any fails. The checks
are intentionally strict — downstream analyses assume the benchmark is
perfectly clean (e.g. every conflict item has exactly one sibling
control with the same difficulty), so catching issues here avoids
silently biased statistics later.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

from beartype import beartype

from s1s2.benchmark.loader import BenchmarkItem, load_benchmark
from s1s2.utils.types import ALL_CATEGORIES, TaskCategory

# Target counts for the PRIMARY benchmark items (paraphrases do not count
# towards these targets). These numbers come from the project brief.
TARGET_COUNTS: dict[TaskCategory, dict[str, int]] = {
    "crt": {"conflict": 30, "control": 30},
    "base_rate": {"conflict": 20, "control": 20},
    "syllogism": {"conflict": 25, "control": 25},
    "anchoring": {"conflict": 15, "control": 15},
    "framing": {"conflict": 15, "control": 15},
    "conjunction": {"conflict": 12, "control": 12},
    "arithmetic": {"conflict": 25, "control": 25},
}

# Token-count heuristic bounds for the prompt. Benchmark items should be
# self-contained mini-problems, not short prompts and not essays.
MIN_PROMPT_TOKENS = 10
MAX_PROMPT_TOKENS = 500

# Prompt-length guard uses a cheap whitespace tokeniser for the validator
# (real tokenisation lives in the extract workstream).
_WHITESPACE_SPLIT = re.compile(r"\S+")


def _approx_tokens(s: str) -> int:
    return len(_WHITESPACE_SPLIT.findall(s))


@dataclass
class ValidationReport:
    """Collected issues from a validation run."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


# ---- individual checks -------------------------------------------------


def _is_primary(item: BenchmarkItem) -> bool:
    """A primary item is one used to satisfy target counts.

    Paraphrases carry ``__p<N>`` in their id and are sibling records.
    """
    return "__p" not in item.id


@beartype
def _check_unique_ids(items: list[BenchmarkItem], r: ValidationReport) -> None:
    c = Counter(it.id for it in items)
    dups = [i for i, n in c.items() if n > 1]
    if dups:
        r.add_error(f"duplicate ids: {dups[:5]} (total {len(dups)})")


@beartype
def _check_answers_distinct(items: list[BenchmarkItem], r: ValidationReport) -> None:
    for it in items:
        if it.conflict:
            if it.lure_answer == "":
                r.add_error(
                    f"{it.id}: conflict=True but lure_answer is empty"
                )
                continue
            if it.correct_answer.strip() == it.lure_answer.strip():
                r.add_error(
                    f"{it.id}: correct_answer equals lure_answer ({it.correct_answer!r})"
                )
        else:
            if it.lure_answer != "":
                r.add_error(
                    f"{it.id}: conflict=False but lure_answer is non-empty ({it.lure_answer!r})"
                )


@beartype
def _check_regex_roundtrip(items: list[BenchmarkItem], r: ValidationReport) -> None:
    for it in items:
        try:
            ap = re.compile(it.answer_pattern, re.IGNORECASE)
        except re.error as e:
            r.add_error(f"{it.id}: answer_pattern is not valid regex: {e}")
            continue
        if not ap.search(it.correct_answer):
            r.add_error(
                f"{it.id}: answer_pattern {it.answer_pattern!r} does not match "
                f"correct_answer {it.correct_answer!r}"
            )

        if it.conflict:
            try:
                lp = re.compile(it.lure_pattern, re.IGNORECASE)
            except re.error as e:
                r.add_error(f"{it.id}: lure_pattern is not valid regex: {e}")
                continue
            if not lp.search(it.lure_answer):
                r.add_error(
                    f"{it.id}: lure_pattern {it.lure_pattern!r} does not match "
                    f"lure_answer {it.lure_answer!r}"
                )
        else:
            if it.lure_pattern != "":
                r.add_error(
                    f"{it.id}: conflict=False but lure_pattern is non-empty"
                )


@beartype
def _check_pair_matching(items: list[BenchmarkItem], r: ValidationReport) -> None:
    by_pair: dict[str, dict[str, list[BenchmarkItem]]] = defaultdict(
        lambda: {"conflict": [], "control": []}
    )
    for it in items:
        key = "conflict" if it.conflict else "control"
        by_pair[it.matched_pair_id][key].append(it)

    for pair_id, group in by_pair.items():
        conflicts = group["conflict"]
        controls = group["control"]

        primary_conflicts = [c for c in conflicts if _is_primary(c)]
        primary_controls = [c for c in controls if _is_primary(c)]

        if primary_conflicts and not primary_controls:
            r.add_error(
                f"matched_pair_id {pair_id!r}: conflict primary present "
                f"({primary_conflicts[0].id}) but no control primary"
            )
        if primary_controls and not primary_conflicts:
            r.add_error(
                f"matched_pair_id {pair_id!r}: control primary present "
                f"({primary_controls[0].id}) but no conflict primary"
            )

        if len(primary_conflicts) > 1:
            r.add_error(
                f"matched_pair_id {pair_id!r}: {len(primary_conflicts)} conflict "
                f"primaries ({[c.id for c in primary_conflicts]})"
            )
        if len(primary_controls) > 1:
            r.add_error(
                f"matched_pair_id {pair_id!r}: {len(primary_controls)} control "
                f"primaries ({[c.id for c in primary_controls]})"
            )

        if primary_conflicts and primary_controls:
            pc = primary_conflicts[0]
            pcon = primary_controls[0]
            if pc.category != pcon.category:
                r.add_error(
                    f"matched_pair_id {pair_id!r}: category mismatch "
                    f"({pc.category} vs {pcon.category})"
                )
            if pc.difficulty != pcon.difficulty:
                r.add_warning(
                    f"matched_pair_id {pair_id!r}: difficulty mismatch "
                    f"({pc.difficulty} vs {pcon.difficulty})"
                )


@beartype
def _check_category_counts(items: list[BenchmarkItem], r: ValidationReport) -> None:
    primaries = [it for it in items if _is_primary(it)]
    cat_counts: dict[TaskCategory, Counter[str]] = {
        c: Counter() for c in ALL_CATEGORIES
    }
    for it in primaries:
        cat_counts[it.category]["conflict" if it.conflict else "control"] += 1

    for cat, targets in TARGET_COUNTS.items():
        actual = cat_counts[cat]
        for flavour in ("conflict", "control"):
            got = actual[flavour]
            want = targets[flavour]
            if got < want:
                r.add_error(
                    f"category {cat}/{flavour}: have {got} primary items, need {want}"
                )
            elif got > want:
                r.add_warning(
                    f"category {cat}/{flavour}: have {got} primary items, target is {want} (over)"
                )


@beartype
def _check_prompt_length(items: list[BenchmarkItem], r: ValidationReport) -> None:
    for it in items:
        n = _approx_tokens(it.prompt)
        if n < MIN_PROMPT_TOKENS:
            r.add_error(
                f"{it.id}: prompt is {n} tokens, below floor {MIN_PROMPT_TOKENS}"
            )
        elif n > MAX_PROMPT_TOKENS:
            r.add_error(
                f"{it.id}: prompt is {n} tokens, above ceiling {MAX_PROMPT_TOKENS}"
            )


@beartype
def _check_difficulty_balance(items: list[BenchmarkItem], r: ValidationReport) -> None:
    primaries = [it for it in items if _is_primary(it)]
    if not primaries:
        return
    diffs = Counter(it.difficulty for it in primaries)
    total = sum(diffs.values())
    # With difficulty 1-5 a perfectly balanced distribution would have
    # 20% per bin. We warn when any bin is below 5% -- this keeps
    # analyses stratifiable by difficulty.
    for lvl in range(1, 6):
        share = diffs.get(lvl, 0) / total
        if share < 0.05:
            r.add_warning(
                f"difficulty {lvl}: only {diffs.get(lvl, 0)}/{total} primaries "
                f"({share:.1%}) -- stratified analyses may be underpowered"
            )


@beartype
def _collect_stats(items: list[BenchmarkItem], r: ValidationReport) -> None:
    primaries = [it for it in items if _is_primary(it)]

    per_cat: dict[str, dict[str, int]] = {}
    for cat in ALL_CATEGORIES:
        cat_prim = [it for it in primaries if it.category == cat]
        per_cat[cat] = {
            "primary_total": len(cat_prim),
            "conflict": sum(1 for it in cat_prim if it.conflict),
            "control": sum(1 for it in cat_prim if not it.conflict),
            "paraphrases": sum(
                len(it.paraphrases) for it in items if it.category == cat
            ),
        }

    diffs = [it.difficulty for it in primaries]
    r.stats = {
        "n_lines": len(items),
        "n_primary": len(primaries),
        "n_paraphrase_records": len(items) - len(primaries),
        "total_paraphrase_strings": sum(len(it.paraphrases) for it in items),
        "conflict_ratio_primary": (
            sum(1 for it in primaries if it.conflict) / max(len(primaries), 1)
        ),
        "mean_difficulty": mean(diffs) if diffs else 0.0,
        "median_difficulty": median(diffs) if diffs else 0.0,
        "per_category": per_cat,
    }


# ---- top-level -------------------------------------------------------


@beartype
def validate_benchmark(path: str | Path) -> ValidationReport:
    """Run the full validation suite on a benchmark file."""
    items = load_benchmark(path)
    r = ValidationReport()

    _check_unique_ids(items, r)
    _check_answers_distinct(items, r)
    _check_regex_roundtrip(items, r)
    _check_pair_matching(items, r)
    _check_category_counts(items, r)
    _check_prompt_length(items, r)
    _check_difficulty_balance(items, r)
    _collect_stats(items, r)
    return r


def _print_report(r: ValidationReport, path: Path) -> None:
    line = "=" * 78
    print(line)
    print(f"Cognitive bias benchmark validation: {path}")
    print(line)

    s = r.stats
    print(f"Total JSONL records : {s.get('n_lines')}")
    print(f"Primary items       : {s.get('n_primary')}")
    print(
        f"Paraphrase strings  : {s.get('total_paraphrase_strings')} "
        "(expand via templates.expand_paraphrases())"
    )
    print(f"Conflict ratio      : {s.get('conflict_ratio_primary'):.2%}")
    print(
        f"Difficulty mean/med : "
        f"{s.get('mean_difficulty'):.2f} / {s.get('median_difficulty'):.1f}"
    )
    print()

    print("Per-category primary counts:")
    print(f"  {'category':<14}{'conflict':>10}{'control':>10}{'paraphrases':>14}")
    print(f"  {'-'*14}{'-'*10}{'-'*10}{'-'*14}")
    per_cat = s.get("per_category", {})
    assert isinstance(per_cat, dict)
    for cat in ALL_CATEGORIES:
        row = per_cat.get(cat, {})
        print(
            f"  {cat:<14}"
            f"{row.get('conflict', 0):>10}"
            f"{row.get('control', 0):>10}"
            f"{row.get('paraphrases', 0):>14}"
        )

    print()
    if r.warnings:
        print(f"WARNINGS ({len(r.warnings)}):")
        for w in r.warnings[:30]:
            print(f"  - {w}")
        if len(r.warnings) > 30:
            print(f"  ... and {len(r.warnings) - 30} more")
        print()

    if r.errors:
        print(f"ERRORS ({len(r.errors)}):")
        for e in r.errors[:30]:
            print(f"  - {e}")
        if len(r.errors) > 30:
            print(f"  ... and {len(r.errors) - 30} more")
    else:
        print("No errors.")
    print(line)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(
            "usage: python -m s1s2.benchmark.validate path/to/benchmark.jsonl",
            file=sys.stderr,
        )
        return 2

    path = Path(argv[0])
    report = validate_benchmark(path)
    _print_report(report, path)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
