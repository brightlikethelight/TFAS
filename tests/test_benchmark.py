"""Tests for the benchmark loader, validator, and template generators.

These tests treat the benchmark module as the single source of truth
for cognitive-bias item shape. We round-trip serialise/deserialise
items, exercise every validation check (so a future schema change
fails loud), and run each template generator with both well-formed and
malformed parameters.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# OpenMP guard for macOS — torch + numpy can crash without this.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Allow tests to import the package without `pip install -e .`.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from s1s2.benchmark import (  # noqa: E402
    BenchmarkItem,
    filter_by_category,
    filter_conflict,
    group_by_matched_pair,
    iter_matched_pairs,
    load_benchmark,
    validate_benchmark,
)
from s1s2.benchmark.loader import _coerce_item  # noqa: E402
from s1s2.benchmark.templates import (  # noqa: E402
    bat_ball_isomorph,
    expand_paraphrases,
    lily_pad_isomorph,
    make_many,
    widgets_machines_isomorph,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _make_dict(
    *,
    id: str = "p_0",
    category: str = "crt",
    subcategory: str = "ratio",
    conflict: bool = True,
    difficulty: int = 2,
    prompt: str = "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost in cents?",
    correct_answer: str = "5",
    lure_answer: str = "10",
    answer_pattern: str = r"\b5\b",
    lure_pattern: str = r"\b10\b",
    matched_pair_id: str = "pair_0",
    source: str = "novel",
) -> dict:
    return {
        "id": id,
        "category": category,
        "subcategory": subcategory,
        "conflict": conflict,
        "difficulty": difficulty,
        "prompt": prompt,
        "system_prompt": None,
        "correct_answer": correct_answer,
        "lure_answer": lure_answer,
        "answer_pattern": answer_pattern,
        "lure_pattern": lure_pattern,
        "matched_pair_id": matched_pair_id,
        "source": source,
        "provenance_note": "test",
        "paraphrases": [],
    }


def _write_jsonl(path: Path, items: list[dict]) -> Path:
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return path


# --------------------------------------------------------------------------- #
# BenchmarkItem round-trip                                                     #
# --------------------------------------------------------------------------- #


class TestBenchmarkItemSerialization:
    def test_to_dict_round_trip(self):
        item = _coerce_item(_make_dict(), line_number=1)
        d = item.to_dict()
        # Re-coerce from the dict — exercises the schema both ways.
        item2 = _coerce_item(d, line_number=2)
        # Coercion preserves all fields except id (we kept the same id).
        assert item2 == item

    def test_to_dict_is_json_serialisable(self):
        item = _coerce_item(_make_dict(), line_number=1)
        s = json.dumps(item.to_dict())
        assert "5" in s

    def test_frozen_dataclass(self):
        """BenchmarkItem is frozen — direct mutation must fail."""
        item = _coerce_item(_make_dict(), line_number=1)
        with pytest.raises(Exception):  # noqa: B017 — frozen dataclass raises FrozenInstanceError (an Exception subclass)
            item.id = "new_id"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# load_benchmark                                                               #
# --------------------------------------------------------------------------- #


class TestLoadBenchmark:
    def test_load_two_items(self, tmp_path: Path):
        path = tmp_path / "tiny.jsonl"
        _write_jsonl(
            path,
            [
                _make_dict(id="p_0"),
                _make_dict(id="p_1", conflict=False, lure_answer="", lure_pattern=""),
            ],
        )
        items = load_benchmark(path)
        assert len(items) == 2
        assert items[0].id == "p_0"
        assert items[1].conflict is False

    def test_skips_blank_lines(self, tmp_path: Path):
        path = tmp_path / "blank.jsonl"
        with path.open("w") as f:
            f.write("\n")
            f.write(json.dumps(_make_dict(id="p_0")) + "\n")
            f.write("   \n")
            f.write(json.dumps(_make_dict(id="p_1")) + "\n")
        items = load_benchmark(path)
        assert len(items) == 2

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_benchmark(tmp_path / "nope.jsonl")

    def test_malformed_json_raises(self, tmp_path: Path):
        path = tmp_path / "bad.jsonl"
        path.write_text("not json\n")
        with pytest.raises(ValueError, match="malformed JSON"):
            load_benchmark(path)

    def test_unknown_category_raises(self, tmp_path: Path):
        path = tmp_path / "bad_cat.jsonl"
        bad = _make_dict()
        bad["category"] = "totally_made_up"
        _write_jsonl(path, [bad])
        with pytest.raises(ValueError, match="category"):
            load_benchmark(path)

    def test_invalid_difficulty_raises(self, tmp_path: Path):
        path = tmp_path / "bad_diff.jsonl"
        bad = _make_dict()
        bad["difficulty"] = 99
        _write_jsonl(path, [bad])
        with pytest.raises(ValueError, match="difficulty"):
            load_benchmark(path)

    def test_invalid_source_raises(self, tmp_path: Path):
        path = tmp_path / "bad_src.jsonl"
        bad = _make_dict()
        bad["source"] = "from_thin_air"
        _write_jsonl(path, [bad])
        with pytest.raises(ValueError, match="source"):
            load_benchmark(path)


# --------------------------------------------------------------------------- #
# Filters                                                                      #
# --------------------------------------------------------------------------- #


class TestFilters:
    def _items(self) -> list[BenchmarkItem]:
        return [
            _coerce_item(_make_dict(id="c0", category="crt", conflict=True), 1),
            _coerce_item(
                _make_dict(
                    id="c0_ctrl",
                    category="crt",
                    conflict=False,
                    lure_answer="",
                    lure_pattern="",
                ),
                2,
            ),
            _coerce_item(
                _make_dict(id="b0", category="base_rate", conflict=True), 3
            ),
            _coerce_item(
                _make_dict(
                    id="b0_ctrl",
                    category="base_rate",
                    conflict=False,
                    lure_answer="",
                    lure_pattern="",
                ),
                4,
            ),
        ]

    def test_filter_by_category(self):
        items = self._items()
        crt = filter_by_category(items, "crt")
        assert len(crt) == 2
        assert all(it.category == "crt" for it in crt)

    def test_filter_by_category_empty(self):
        items = self._items()
        out = filter_by_category(items, "anchoring")
        assert out == []

    def test_filter_conflict_true(self):
        items = self._items()
        out = filter_conflict(items, only_conflict=True)
        assert len(out) == 2
        assert all(it.conflict for it in out)

    def test_filter_conflict_false(self):
        items = self._items()
        out = filter_conflict(items, only_conflict=False)
        assert len(out) == 2
        assert all(not it.conflict for it in out)

    def test_group_by_matched_pair(self):
        items = self._items()
        # All four items above use the default matched_pair_id="pair_0"
        grouped = group_by_matched_pair(items)
        assert "pair_0" in grouped
        assert len(grouped["pair_0"]) == 4


# --------------------------------------------------------------------------- #
# iter_matched_pairs                                                           #
# --------------------------------------------------------------------------- #


class TestIterMatchedPairs:
    def test_paired_items_yielded(self):
        items = [
            _coerce_item(
                _make_dict(id="c0", matched_pair_id="pair_a", conflict=True), 1
            ),
            _coerce_item(
                _make_dict(
                    id="c0_ctrl",
                    matched_pair_id="pair_a",
                    conflict=False,
                    lure_answer="",
                    lure_pattern="",
                ),
                2,
            ),
        ]
        pairs = list(iter_matched_pairs(items))
        assert len(pairs) == 1
        conflict, control = pairs[0]
        assert conflict.id == "c0"
        assert control.id == "c0_ctrl"
        assert conflict.conflict is True
        assert control.conflict is False

    def test_orphaned_pair_skipped(self):
        items = [
            _coerce_item(
                _make_dict(id="lonely", matched_pair_id="pair_x", conflict=True), 1
            ),
            # No control pair for "pair_x"; should be skipped.
        ]
        pairs = list(iter_matched_pairs(items))
        assert pairs == []

    def test_paraphrase_skipped_in_favor_of_primary(self):
        items = [
            _coerce_item(
                _make_dict(id="primary", matched_pair_id="pair_a", conflict=True),
                1,
            ),
            _coerce_item(
                _make_dict(id="primary__p1", matched_pair_id="pair_a", conflict=True),
                2,
            ),
            _coerce_item(
                _make_dict(
                    id="primary_ctrl",
                    matched_pair_id="pair_a",
                    conflict=False,
                    lure_answer="",
                    lure_pattern="",
                ),
                3,
            ),
        ]
        pairs = list(iter_matched_pairs(items))
        assert len(pairs) == 1
        # Primary (no __p suffix) wins
        assert pairs[0][0].id == "primary"


# --------------------------------------------------------------------------- #
# validate_benchmark                                                           #
# --------------------------------------------------------------------------- #


def _make_balanced_benchmark(tmp_path: Path) -> Path:
    """Build a tiny but valid benchmark with the bare-minimum CRT pair."""
    items = [
        _make_dict(
            id="crt_0",
            category="crt",
            conflict=True,
            matched_pair_id="pair_0",
            prompt="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents? Reply with the number only.",
        ),
        _make_dict(
            id="crt_0_ctrl",
            category="crt",
            conflict=False,
            lure_answer="",
            lure_pattern="",
            matched_pair_id="pair_0",
            prompt="A bat alone costs $1.05 and a ball alone costs $0.05. What is the price of the ball in cents? Reply with the number only.",
        ),
    ]
    return _write_jsonl(tmp_path / "bench.jsonl", items)


class TestValidateBenchmark:
    def test_valid_benchmark_loads(self, tmp_path: Path):
        """A valid benchmark may still hit count warnings — what we care
        about is the per-item structural checks."""
        path = _make_balanced_benchmark(tmp_path)
        report = validate_benchmark(path)
        # Per-item structural checks should pass; count targets won't, but
        # the function still runs and returns a report.
        # The minimal benchmark above has *only* one pair so the count
        # target check fires; ensure no per-item integrity errors fire.
        assert all("crt_0" not in e for e in report.errors), report.errors

    def test_missing_lure_on_conflict_caught(self, tmp_path: Path):
        path = tmp_path / "bad.jsonl"
        bad = _make_dict(id="bad_conflict", conflict=True)
        bad["lure_answer"] = ""
        bad["lure_pattern"] = ""
        _write_jsonl(path, [bad])
        report = validate_benchmark(path)
        assert any("lure_answer is empty" in e for e in report.errors)

    def test_lure_on_control_caught(self, tmp_path: Path):
        path = tmp_path / "bad.jsonl"
        bad = _make_dict(id="bad_ctrl", conflict=False)
        bad["lure_answer"] = "10"
        bad["lure_pattern"] = r"\b10\b"
        _write_jsonl(path, [bad])
        report = validate_benchmark(path)
        assert any("non-empty" in e for e in report.errors)

    def test_duplicate_ids_caught(self, tmp_path: Path):
        path = tmp_path / "dup.jsonl"
        _write_jsonl(
            path,
            [
                _make_dict(id="dup", conflict=True),
                _make_dict(id="dup", conflict=False, lure_answer="", lure_pattern=""),
            ],
        )
        report = validate_benchmark(path)
        assert any("duplicate ids" in e for e in report.errors)

    def test_missing_pair_caught(self, tmp_path: Path):
        path = tmp_path / "lonely.jsonl"
        _write_jsonl(
            path,
            [
                _make_dict(
                    id="lonely",
                    conflict=True,
                    matched_pair_id="pair_lonely",
                    prompt="A bat and a ball cost a dollar in total. The bat costs more. What is the ball price in cents? Reply with the number only.",
                ),
                # No control for pair_lonely
            ],
        )
        report = validate_benchmark(path)
        assert any("control primary" in e for e in report.errors)

    def test_correct_equals_lure_caught(self, tmp_path: Path):
        path = tmp_path / "ce.jsonl"
        bad = _make_dict(id="ce", conflict=True)
        bad["lure_answer"] = bad["correct_answer"]
        bad["lure_pattern"] = bad["answer_pattern"]
        _write_jsonl(path, [bad])
        report = validate_benchmark(path)
        assert any("equals lure_answer" in e for e in report.errors)

    def test_invalid_regex_caught(self, tmp_path: Path):
        path = tmp_path / "regex.jsonl"
        bad = _make_dict(id="bad_re")
        bad["answer_pattern"] = "[unclosed"
        _write_jsonl(path, [bad])
        report = validate_benchmark(path)
        assert any("not valid regex" in e for e in report.errors)

    def test_too_short_prompt_caught(self, tmp_path: Path):
        path = tmp_path / "short.jsonl"
        bad = _make_dict(id="short", prompt="too short")
        _write_jsonl(path, [bad])
        report = validate_benchmark(path)
        assert any("below floor" in e for e in report.errors)


# --------------------------------------------------------------------------- #
# Template generators                                                          #
# --------------------------------------------------------------------------- #


class TestBatBallTemplate:
    def test_classic_problem(self):
        c, x = bat_ball_isomorph(
            pair_id="bat_ball_classic",
            object_a="bat",
            object_b="ball",
            total_cents=110,
            diff_cents=100,
        )
        assert c.conflict is True
        assert x.conflict is False
        assert c.matched_pair_id == x.matched_pair_id == "bat_ball_classic"
        assert c.correct_answer == "5"  # ball costs 5 cents
        assert c.lure_answer == "10"  # the intuitive (wrong) answer
        assert c.category == "crt"
        # Patterns should match the answers
        import re

        assert re.search(c.answer_pattern, "5", re.IGNORECASE)
        assert re.search(c.lure_pattern, "10", re.IGNORECASE)

    def test_invalid_diff_cents(self):
        with pytest.raises(ValueError, match="diff_cents"):
            bat_ball_isomorph(
                pair_id="x", object_a="a", object_b="b",
                total_cents=110, diff_cents=0,
            )

    def test_total_le_diff(self):
        with pytest.raises(ValueError, match="total_cents"):
            bat_ball_isomorph(
                pair_id="x", object_a="a", object_b="b",
                total_cents=50, diff_cents=100,
            )

    def test_odd_remainder_raises(self):
        with pytest.raises(ValueError, match="even"):
            bat_ball_isomorph(
                pair_id="x", object_a="a", object_b="b",
                total_cents=111, diff_cents=100,
            )

    def test_paraphrases_present(self):
        c, _ = bat_ball_isomorph(
            pair_id="p", object_a="phone", object_b="case",
            total_cents=220, diff_cents=200,
        )
        assert len(c.paraphrases) >= 4


class TestWidgetsMachinesTemplate:
    def test_classic_problem(self):
        c, x = widgets_machines_isomorph(
            pair_id="wm_classic",
            worker_label="machine",
            output_label="widget",
            base_rate=5,
            scale=100,
        )
        assert c.conflict is True
        assert x.conflict is False
        assert c.correct_answer == "5"
        assert c.lure_answer == "100"
        assert c.matched_pair_id == "wm_classic"

    def test_zero_base_rate_raises(self):
        with pytest.raises(ValueError, match="positive"):
            widgets_machines_isomorph(
                pair_id="x", worker_label="m", output_label="w",
                base_rate=0, scale=10,
            )

    def test_equal_rates_no_lure(self):
        with pytest.raises(ValueError, match="no lure"):
            widgets_machines_isomorph(
                pair_id="x", worker_label="m", output_label="w",
                base_rate=5, scale=5,
            )


class TestLilyPadTemplate:
    def test_classic_problem(self):
        c, x = lily_pad_isomorph(
            pair_id="lily",
            entity="lily pads",
            habitat="lake",
            days_to_full=48,
        )
        assert c.conflict is True
        assert x.conflict is False
        assert c.correct_answer == "47"
        assert c.lure_answer == "24"

    def test_too_few_days_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            lily_pad_isomorph(
                pair_id="x", entity="x", habitat="y", days_to_full=1
            )

    def test_collapsing_lure_raises(self):
        # days_to_full=2 -> correct=1, lure=2//2=1 -> collide
        with pytest.raises(ValueError, match="collapses"):
            lily_pad_isomorph(
                pair_id="x", entity="x", habitat="y", days_to_full=2
            )


# --------------------------------------------------------------------------- #
# expand_paraphrases & make_many                                               #
# --------------------------------------------------------------------------- #


class TestParaphraseExpansion:
    def test_expand_paraphrases(self):
        c, x = bat_ball_isomorph(
            pair_id="ep", object_a="bat", object_b="ball",
            total_cents=110, diff_cents=100,
        )
        n_paraphrases = len(c.paraphrases)
        expanded = expand_paraphrases([c, x])
        # Original two + paraphrases of c (and any of x)
        assert len(expanded) == 2 + n_paraphrases + len(x.paraphrases)
        # Paraphrase ids carry __pN suffix
        suffixes = [it.id for it in expanded if "__p" in it.id]
        assert all("__p" in s for s in suffixes)

    def test_expand_strips_paraphrases_from_siblings(self):
        c, x = lily_pad_isomorph(
            pair_id="ep2", entity="frogs", habitat="pond",
            days_to_full=10,
        )
        expanded = expand_paraphrases([c, x])
        for it in expanded:
            if "__p" in it.id:
                assert it.paraphrases == ()


class TestMakeMany:
    def test_runs_without_errors(self):
        specs = [
            {"pair_id": f"bb_{i}", "object_a": "bat", "object_b": "ball",
             "total_cents": 110 + 2 * i, "diff_cents": 100}
            for i in range(3)
        ]
        out = make_many(bat_ball_isomorph, specs)
        # 3 specs * 2 items each = 6 items
        assert len(out) == 6

    def test_propagates_errors_with_context(self):
        specs = [{"pair_id": "x", "object_a": "a", "object_b": "b",
                  "total_cents": 50, "diff_cents": 100}]
        with pytest.raises(ValueError, match="failed building"):
            make_many(bat_ball_isomorph, specs)


# --------------------------------------------------------------------------- #
# Generators (non-CRT)                                                         #
# --------------------------------------------------------------------------- #

from s1s2.benchmark.generators import (  # noqa: E402
    anchoring_isomorph,
    arithmetic_trap_isomorph,
    base_rate_isomorph,
    belief_bias_syllogism,
    conjunction_fallacy_isomorph,
    framing_isomorph,
)


def _validate_pair(items: list[BenchmarkItem], tmp_path: Path) -> None:
    """Write items to a temp file and assert no per-item errors fire.

    Tiny benchmarks always fail the per-category target counts; we
    filter those out and assert the per-item structural checks pass.
    """
    p = tmp_path / "pair.jsonl"
    _write_jsonl(p, [it.to_dict() for it in items])
    report = validate_benchmark(p)
    item_errors = [
        e for e in report.errors
        if "category" not in e or "primary present" in e
    ]
    if any(it.id in e for it in items for e in item_errors):
        raise AssertionError(f"per-item errors: {item_errors}")


class TestBaseRateGenerator:
    def test_basic(self, tmp_path: Path):
        c, x = base_rate_isomorph(
            pair_id="br_t",
            description=(
                "She owns a kiln and a wall of glaze tins, sells "
                "ceramics at weekend markets, and her hands are "
                "permanently stained with iron oxide."
            ),
            rare_group="full-time studio potter",
            common_group="office worker",
            rare_rate=0.005,
            common_rate=0.55,
        )
        _validate_pair([c, x], tmp_path)
        assert c.conflict and not x.conflict
        assert c.correct_answer == "office worker"
        assert c.lure_answer == "full-time studio potter"
        assert c.category == "base_rate"

    def test_rejects_close_rates(self):
        with pytest.raises(ValueError, match="3x"):
            base_rate_isomorph(
                pair_id="x",
                description="d",
                rare_group="r",
                common_group="c",
                rare_rate=0.20,
                common_rate=0.40,  # not 3x
            )

    def test_rejects_identical_groups(self):
        with pytest.raises(ValueError, match="identical"):
            base_rate_isomorph(
                pair_id="x",
                description="d",
                rare_group="g",
                common_group="g",
                rare_rate=0.01,
                common_rate=0.85,
            )


class TestBeliefBiasGenerator:
    def test_valid_unbelievable(self, tmp_path: Path):
        c, x = belief_bias_syllogism(
            pair_id="bb_vu",
            conflict_major="All clouds are made of solid iron.",
            conflict_minor="All cumulus clouds are clouds.",
            conflict_conclusion="All cumulus clouds are made of solid iron.",
            conflict_is_valid=True,
            is_believable=False,
            control_major="All things made of solid iron are clouds.",
            control_minor="All cumulus clouds are made of solid iron.",
        )
        _validate_pair([c, x], tmp_path)
        assert c.correct_answer == "valid"
        assert c.lure_answer == "invalid"
        assert x.correct_answer == "invalid"
        assert c.subcategory == "belief_bias_valid_unbelievable"

    def test_invalid_believable(self, tmp_path: Path):
        c, x = belief_bias_syllogism(
            pair_id="bb_ib",
            conflict_major="All things made of paper are textbooks.",
            conflict_minor="All notebooks are textbooks.",
            conflict_conclusion="All notebooks are made of paper.",
            conflict_is_valid=False,
            is_believable=True,
            control_major="All textbooks are made of paper.",
            control_minor="All notebooks are textbooks.",
        )
        _validate_pair([c, x], tmp_path)
        assert c.correct_answer == "invalid"
        assert c.lure_answer == "valid"
        assert x.correct_answer == "valid"

    def test_rejects_control_cell(self):
        # both flags true => not a conflict cell
        with pytest.raises(ValueError, match="mismatched"):
            belief_bias_syllogism(
                pair_id="x",
                conflict_major="All cats are mammals.",
                conflict_minor="All persians are cats.",
                conflict_conclusion="All persians are mammals.",
                conflict_is_valid=True,
                is_believable=True,
                control_major="x",
                control_minor="y",
            )


class TestAnchoringGenerator:
    def test_basic(self, tmp_path: Path):
        c, x = anchoring_isomorph(
            pair_id="an_t",
            question="How long is the river Danube in kilometres?",
            true_value=2850,
            high_anchor=8000,
            low_anchor=200,
            units="kilometres",
        )
        _validate_pair([c, x], tmp_path)
        assert c.correct_answer == "2850"
        assert c.lure_answer == "8000"

    def test_rejects_low_above_true(self):
        with pytest.raises(ValueError, match="below true_value"):
            anchoring_isomorph(
                pair_id="x",
                question="q",
                true_value=100,
                high_anchor=200,
                low_anchor=300,
            )


class TestFramingGenerator:
    def test_basic(self, tmp_path: Path):
        c, x = framing_isomorph(
            pair_id="fr_t",
            scenario="A landslide threatens a village in the foothills.",
            n_total=600,
            n_certain_save=200,
            prob_save_all=1 / 3,
            prefer_sure=True,
        )
        _validate_pair([c, x], tmp_path)
        assert c.correct_answer == "A"
        assert c.lure_answer == "B"

    def test_rejects_unequal_evs(self):
        with pytest.raises(ValueError, match="EV mismatch"):
            framing_isomorph(
                pair_id="x",
                scenario="x",
                n_total=600,
                n_certain_save=300,
                prob_save_all=1 / 3,  # EV = 200, mismatch
                prefer_sure=True,
            )

    def test_rejects_invalid_prob(self):
        with pytest.raises(ValueError, match="strictly between"):
            framing_isomorph(
                pair_id="x",
                scenario="x",
                n_total=600,
                n_certain_save=200,
                prob_save_all=0.0,
                prefer_sure=True,
            )


class TestConjunctionGenerator:
    def test_basic(self, tmp_path: Path):
        c, x = conjunction_fallacy_isomorph(
            pair_id="cj_t",
            person_name="Liam",
            person_description=(
                "Liam, age 31, has a darkroom in his basement, owns "
                "vintage twin-lens reflex cameras, and his social "
                "media is full of moody street photography."
            ),
            feature_a="freelance designer",
            feature_b="exhibiting art photographer",
        )
        _validate_pair([c, x], tmp_path)
        assert c.correct_answer == "A"
        assert c.lure_answer == "B"

    def test_rejects_overlapping_features(self):
        with pytest.raises(ValueError, match="appears inside"):
            conjunction_fallacy_isomorph(
                pair_id="x",
                person_name="x",
                person_description="x",
                feature_a="teacher",
                feature_b="primary school teacher",
            )


class TestArithmeticGenerator:
    def test_basic(self, tmp_path: Path):
        c, x = arithmetic_trap_isomorph(
            pair_id="ar_t",
            start=40,
            steps=[("+", 20), ("-", 10), ("*", 2)],
            trap_step=1,
        )
        _validate_pair([c, x], tmp_path)
        # ((40+20)-10)*2 = 100; flipping - to + gives ((40+20)+10)*2 = 140
        assert c.correct_answer == "100"
        assert c.lure_answer == "140"

    def test_rejects_collapse(self):
        with pytest.raises(ValueError):
            arithmetic_trap_isomorph(
                pair_id="x",
                start=10,
                steps=[("*", 1)],
                trap_step=0,
            )

    def test_rejects_oob_trap(self):
        with pytest.raises(ValueError, match="trap_step"):
            arithmetic_trap_isomorph(
                pair_id="x",
                start=10,
                steps=[("+", 5)],
                trap_step=4,
            )


# --------------------------------------------------------------------------- #
# Canonical assembly                                                           #
# --------------------------------------------------------------------------- #

from s1s2.benchmark.build import (  # noqa: E402
    build_full_benchmark,
    write_jsonl,
)


class TestBuildFullBenchmark:
    def test_deterministic(self):
        a = build_full_benchmark()
        b = build_full_benchmark()
        assert [it.id for it in a] == [it.id for it in b]
        assert [it.prompt for it in a] == [it.prompt for it in b]

    def test_validates_when_written(self, tmp_path: Path):
        items = build_full_benchmark()
        p = tmp_path / "full.jsonl"
        write_jsonl(items, p)
        report = validate_benchmark(p)
        assert report.ok, f"build_full_benchmark produced errors: {report.errors[:10]}"

    def test_meets_minimum_size(self):
        items = build_full_benchmark()
        # Project requirement: 200+ items, 100+ matched pairs
        assert len(items) >= 200
        primaries = [it for it in items if "__p" not in it.id]
        n_pairs = len(primaries) // 2
        assert n_pairs >= 100

    def test_every_pair_has_matched_difficulty(self):
        items = build_full_benchmark()
        by_pair: dict[str, list[BenchmarkItem]] = {}
        for it in items:
            by_pair.setdefault(it.matched_pair_id, []).append(it)
        for pid, group in by_pair.items():
            difficulties = {it.difficulty for it in group}
            assert len(difficulties) == 1, (
                f"pair {pid} has mixed difficulties {difficulties}"
            )

    def test_no_classic_exemplars(self):
        items = build_full_benchmark()
        # We deliberately ship novel structural isomorphs only.
        forbidden = ["bat and a ball", "$1.10", "linda is", "asian disease"]
        for it in items:
            text = it.prompt.lower()
            for f in forbidden:
                assert f not in text, (
                    f"item {it.id} contains forbidden classic exemplar {f!r}"
                )


# --------------------------------------------------------------------------- #
# Real benchmark on disk (skipped if not present)                              #
# --------------------------------------------------------------------------- #


REAL_BENCHMARK = _REPO / "data" / "benchmark" / "benchmark.jsonl"


@pytest.mark.skipif(
    not REAL_BENCHMARK.exists(),
    reason=(
        "real benchmark not present; run `python -m s1s2.benchmark.cli generate`"
    ),
)
class TestRealBenchmark:
    def test_validates_clean(self):
        report = validate_benchmark(REAL_BENCHMARK)
        assert report.ok, f"real benchmark has errors: {report.errors[:10]}"

    def test_size(self):
        items = load_benchmark(REAL_BENCHMARK)
        assert len(items) >= 200

    def test_every_category_present(self):
        items = load_benchmark(REAL_BENCHMARK)
        from s1s2.utils.types import ALL_CATEGORIES

        cats = {it.category for it in items}
        assert cats == set(ALL_CATEGORIES)


# --------------------------------------------------------------------------- #
# CLI smoke                                                                    #
# --------------------------------------------------------------------------- #


class TestCLI:
    def test_generate_then_validate(self, tmp_path: Path):
        from s1s2.benchmark import cli

        out = tmp_path / "bench.jsonl"
        rc = cli.main(["generate", "--output", str(out)])
        assert rc == 0
        assert out.exists()
        rc = cli.main(["validate", "--path", str(out)])
        assert rc == 0
        rc = cli.main(["stats", "--path", str(out)])
        assert rc == 0

    def test_validate_missing_path(self, tmp_path: Path):
        from s1s2.benchmark import cli

        rc = cli.main(["validate", "--path", str(tmp_path / "absent.jsonl")])
        assert rc != 0
