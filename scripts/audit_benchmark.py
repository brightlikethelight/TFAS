#!/usr/bin/env python3
"""Programmatic audit of the s1s2 cognitive-bias benchmark.

Checks:
  1. Schema / structural integrity
  2. Regex pattern matching (answer_pattern matches correct_answer, etc.)
  3. Matched-pair integrity (same category, difficulty, one conflict + one control)
  4. CRT mathematical correctness (bat-ball, widgets, lily-pad invariants)
  5. Framing EV equality
  6. Arithmetic correctness (recompute every multi-step item)
  7. Syllogism validity (AAA-1 / AAA-2 structural checks)
  8. Base-rate consistency (common_rate > rare_rate from prompt text)
  9. Conjunction fallacy (correct=A always, lure=B always)
  10. Anchoring (true_value < high_anchor from prompt)

Run:
    python scripts/audit_benchmark.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Issue:
    item_id: str
    issue_type: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    description: str


BENCHMARK_PATH = Path(__file__).resolve().parent.parent / "data" / "benchmark" / "benchmark.jsonl"


def load_items(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            raw["_lineno"] = lineno
            items.append(raw)
    return items


# ---- helper: apply arithmetic ops ----

def _apply(x: int, op: str, v: int) -> int:
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
    raise ValueError(f"unsupported op {op!r}")


def _flip_op(op: str) -> str:
    return {"+": "-", "-": "+", "*": "/", "/": "*"}[op]


# ---- checks ----

def check_schema(items: list[dict[str, Any]]) -> list[Issue]:
    """Verify every required field is present and has correct types."""
    issues = []
    required = [
        "id", "category", "subcategory", "conflict", "difficulty",
        "prompt", "system_prompt", "correct_answer", "lure_answer",
        "answer_pattern", "lure_pattern", "matched_pair_id",
        "source", "provenance_note", "paraphrases",
    ]
    valid_categories = {
        "crt", "base_rate", "syllogism", "anchoring",
        "framing", "conjunction", "arithmetic",
    }
    valid_sources = {"novel", "hagendorff_2023", "template", "adapted"}

    for it in items:
        iid = it.get("id", f"line_{it.get('_lineno', '?')}")
        for field in required:
            if field not in it:
                issues.append(Issue(iid, "missing_field", "CRITICAL",
                                    f"Missing required field: {field}"))
        if it.get("category") not in valid_categories:
            issues.append(Issue(iid, "invalid_category", "CRITICAL",
                                f"Invalid category: {it.get('category')}"))
        if it.get("source") not in valid_sources:
            issues.append(Issue(iid, "invalid_source", "HIGH",
                                f"Invalid source: {it.get('source')}"))
        if not isinstance(it.get("difficulty"), int) or not (1 <= it.get("difficulty", 0) <= 5):
            issues.append(Issue(iid, "invalid_difficulty", "HIGH",
                                f"Invalid difficulty: {it.get('difficulty')}"))
        if not isinstance(it.get("paraphrases"), list):
            issues.append(Issue(iid, "invalid_paraphrases", "MEDIUM",
                                f"paraphrases is not a list"))
        if not isinstance(it.get("conflict"), bool):
            issues.append(Issue(iid, "invalid_conflict", "CRITICAL",
                                f"conflict is not bool: {it.get('conflict')}"))
    return issues


def check_regex_patterns(items: list[dict[str, Any]]) -> list[Issue]:
    """Verify answer_pattern matches correct_answer; lure_pattern matches lure_answer."""
    issues = []
    for it in items:
        iid = it["id"]
        correct = it["correct_answer"]
        pattern = it["answer_pattern"]
        lure = it["lure_answer"]
        lure_pat = it["lure_pattern"]

        # answer_pattern must match correct_answer
        if pattern:
            try:
                if not re.search(pattern, correct):
                    issues.append(Issue(
                        iid, "answer_pattern_mismatch", "CRITICAL",
                        f"answer_pattern '{pattern}' does not match "
                        f"correct_answer '{correct}'"
                    ))
            except re.error as e:
                issues.append(Issue(
                    iid, "answer_pattern_invalid", "CRITICAL",
                    f"answer_pattern is invalid regex: {e}"
                ))
        else:
            issues.append(Issue(
                iid, "answer_pattern_empty", "CRITICAL",
                "answer_pattern is empty"
            ))

        # lure_pattern must match lure_answer (only for conflict items)
        if it["conflict"]:
            if not lure:
                issues.append(Issue(
                    iid, "missing_lure_answer", "CRITICAL",
                    "Conflict item has empty lure_answer"
                ))
            if lure_pat:
                try:
                    if lure and not re.search(lure_pat, lure):
                        issues.append(Issue(
                            iid, "lure_pattern_mismatch", "CRITICAL",
                            f"lure_pattern '{lure_pat}' does not match "
                            f"lure_answer '{lure}'"
                        ))
                except re.error as e:
                    issues.append(Issue(
                        iid, "lure_pattern_invalid", "CRITICAL",
                        f"lure_pattern is invalid regex: {e}"
                    ))
            else:
                issues.append(Issue(
                    iid, "missing_lure_pattern", "HIGH",
                    "Conflict item has empty lure_pattern"
                ))
        else:
            # Control items should have empty lure
            if lure:
                issues.append(Issue(
                    iid, "control_has_lure", "MEDIUM",
                    f"Control item has non-empty lure_answer: '{lure}'"
                ))

        # Cross-check: answer_pattern should NOT match lure_answer
        # (otherwise the pattern is too broad)
        if it["conflict"] and lure and pattern:
            try:
                if re.search(pattern, lure):
                    issues.append(Issue(
                        iid, "pattern_too_broad", "HIGH",
                        f"answer_pattern '{pattern}' also matches "
                        f"lure_answer '{lure}'"
                    ))
            except re.error:
                pass

        # lure_pattern should NOT match correct_answer
        if it["conflict"] and lure_pat and correct:
            try:
                if re.search(lure_pat, correct):
                    issues.append(Issue(
                        iid, "lure_pattern_too_broad", "HIGH",
                        f"lure_pattern '{lure_pat}' also matches "
                        f"correct_answer '{correct}'"
                    ))
            except re.error:
                pass

    return issues


def check_matched_pairs(items: list[dict[str, Any]]) -> list[Issue]:
    """Verify matched pair integrity."""
    issues = []
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        by_pair[it["matched_pair_id"]].append(it)

    for pid, group in by_pair.items():
        conflicts = [it for it in group if it["conflict"]]
        controls = [it for it in group if not it["conflict"]]

        if len(conflicts) == 0:
            issues.append(Issue(
                pid, "missing_conflict", "CRITICAL",
                f"Matched pair '{pid}' has no conflict item"
            ))
        if len(controls) == 0:
            issues.append(Issue(
                pid, "missing_control", "CRITICAL",
                f"Matched pair '{pid}' has no control item"
            ))
        if len(conflicts) > 1:
            issues.append(Issue(
                pid, "multiple_conflicts", "HIGH",
                f"Matched pair '{pid}' has {len(conflicts)} conflict items"
            ))
        if len(controls) > 1:
            issues.append(Issue(
                pid, "multiple_controls", "HIGH",
                f"Matched pair '{pid}' has {len(controls)} control items"
            ))

        if conflicts and controls:
            c = conflicts[0]
            ctrl = controls[0]
            if c["category"] != ctrl["category"]:
                issues.append(Issue(
                    pid, "category_mismatch", "CRITICAL",
                    f"Pair category mismatch: {c['category']} vs {ctrl['category']}"
                ))
            if c["difficulty"] != ctrl["difficulty"]:
                issues.append(Issue(
                    pid, "difficulty_mismatch", "CRITICAL",
                    f"Pair difficulty mismatch: {c['difficulty']} vs {ctrl['difficulty']}"
                ))
            # Subcategory should match
            if c["subcategory"] != ctrl["subcategory"]:
                issues.append(Issue(
                    pid, "subcategory_mismatch", "MEDIUM",
                    f"Pair subcategory mismatch: {c['subcategory']} vs {ctrl['subcategory']}"
                ))

    # Check for duplicate IDs
    ids = [it["id"] for it in items]
    id_counts = Counter(ids)
    for iid, count in id_counts.items():
        if count > 1:
            issues.append(Issue(
                iid, "duplicate_id", "CRITICAL",
                f"Duplicate item ID (appears {count} times)"
            ))

    return issues


def check_crt_math(items: list[dict[str, Any]]) -> list[Issue]:
    """Verify CRT items' mathematical correctness."""
    issues = []
    crt_items = [it for it in items if it["category"] == "crt" and it["conflict"]]

    for it in crt_items:
        iid = it["id"]
        sub = it["subcategory"]
        prompt = it["prompt"]
        correct = it["correct_answer"]
        lure = it["lure_answer"]

        if sub == "ratio":
            # bat-and-ball: extract total and diff from prompt
            # Format: "cost $X.XX in total" and "costs $Y.YY more"
            total_match = re.search(r'\$(\d+\.\d{2}) in total', prompt)
            diff_match = re.search(r'\$(\d+\.\d{2}) more', prompt)
            if total_match and diff_match:
                total_cents = round(float(total_match.group(1)) * 100)
                diff_cents = round(float(diff_match.group(1)) * 100)
                expected_correct = (total_cents - diff_cents) // 2
                expected_lure = total_cents - diff_cents
                if str(expected_correct) != correct:
                    issues.append(Issue(
                        iid, "crt_ratio_wrong_answer", "CRITICAL",
                        f"bat-ball: total={total_cents}, diff={diff_cents}, "
                        f"expected correct={(total_cents-diff_cents)//2}, "
                        f"got '{correct}'"
                    ))
                if str(expected_lure) != lure:
                    issues.append(Issue(
                        iid, "crt_ratio_wrong_lure", "CRITICAL",
                        f"bat-ball: expected lure={expected_lure}, got '{lure}'"
                    ))
                # Verify invariant: ball = (total - diff) / 2
                # and that lure != correct
                if expected_lure == expected_correct:
                    issues.append(Issue(
                        iid, "crt_ratio_degenerate", "CRITICAL",
                        "Lure equals correct answer -- degenerate problem"
                    ))
            else:
                issues.append(Issue(
                    iid, "crt_ratio_parse_fail", "MEDIUM",
                    "Could not parse total/diff from prompt"
                ))

        elif sub == "work_rate":
            # widgets-machines: "N workers make N items in N minutes"
            # correct = base_rate, lure = scale
            m = re.search(
                r'(\d+) [\w\s]+ together produce (\d+) [\w\s]+ in (\d+) minutes',
                prompt
            )
            if m:
                n_workers = int(m.group(1))
                n_output = int(m.group(2))
                base_rate = int(m.group(3))
                # The triple should be equal: N workers make N items in N minutes
                if not (n_workers == n_output == base_rate):
                    issues.append(Issue(
                        iid, "crt_workrate_unequal_triple", "HIGH",
                        f"Work-rate triple not equal: workers={n_workers}, "
                        f"output={n_output}, time={base_rate}"
                    ))
                # Correct answer is the base_rate (each worker makes 1 per base_rate min)
                if str(base_rate) != correct:
                    issues.append(Issue(
                        iid, "crt_workrate_wrong_answer", "CRITICAL",
                        f"Work-rate: expected correct={base_rate}, got '{correct}'"
                    ))
                # Lure is the scale number
                scale_match = re.search(
                    r'how many minutes would (\d+) [\w\s]+ take to produce (\d+)',
                    prompt
                )
                if scale_match:
                    scale = int(scale_match.group(1))
                    if str(scale) != lure:
                        issues.append(Issue(
                            iid, "crt_workrate_wrong_lure", "CRITICAL",
                            f"Work-rate: expected lure={scale}, got '{lure}'"
                        ))
            else:
                issues.append(Issue(
                    iid, "crt_workrate_parse_fail", "MEDIUM",
                    "Could not parse work-rate triple from prompt"
                ))

        elif sub == "exponential_growth":
            # lily-pad: "takes N days to fill entire habitat"
            # correct = N-1, lure = N//2
            m = re.search(r'(\d+) days for', prompt)
            if m:
                days_to_full = int(m.group(1))
                expected_correct = days_to_full - 1
                expected_lure = days_to_full // 2
                if str(expected_correct) != correct:
                    issues.append(Issue(
                        iid, "crt_exp_wrong_answer", "CRITICAL",
                        f"Exp growth: days_to_full={days_to_full}, "
                        f"expected correct={expected_correct}, got '{correct}'"
                    ))
                if str(expected_lure) != lure:
                    issues.append(Issue(
                        iid, "crt_exp_wrong_lure", "CRITICAL",
                        f"Exp growth: expected lure={expected_lure}, got '{lure}'"
                    ))
                if expected_lure == expected_correct:
                    issues.append(Issue(
                        iid, "crt_exp_degenerate", "CRITICAL",
                        "Lure equals correct answer -- degenerate"
                    ))
            else:
                issues.append(Issue(
                    iid, "crt_exp_parse_fail", "MEDIUM",
                    "Could not parse days_to_full from prompt"
                ))

    return issues


def check_arithmetic(items: list[dict[str, Any]]) -> list[Issue]:
    """Recompute every arithmetic item's correct and lure answers."""
    issues = []
    arith_items = [it for it in items if it["category"] == "arithmetic"]

    for it in arith_items:
        iid = it["id"]
        # Parse provenance_note for the step spec
        prov = it["provenance_note"]
        # Extract start, steps, trap_step from provenance
        start_m = re.search(r'start=(\d+)', prov)
        steps_m = re.search(r'steps=\[(.+?)\]', prov)
        trap_m = re.search(r'trap_step=(\d+)', prov)

        if not (start_m and steps_m and trap_m):
            if it["conflict"]:
                issues.append(Issue(
                    iid, "arith_parse_fail", "MEDIUM",
                    "Could not parse arithmetic spec from provenance"
                ))
            continue

        start = int(start_m.group(1))
        trap_step = int(trap_m.group(1))

        # Parse steps like ('*', 2), ('-', 30), ('+', 15)
        steps_str = steps_m.group(1)
        steps: list[tuple[str, int]] = []
        for step_match in re.finditer(r"\('([+\-*/])',\s*(\d+)\)", steps_str):
            op = step_match.group(1)
            val = int(step_match.group(2))
            steps.append((op, val))

        if not steps:
            issues.append(Issue(
                iid, "arith_no_steps", "MEDIUM",
                "Could not parse steps from provenance"
            ))
            continue

        # Compute correct answer
        try:
            running = start
            for op, v in steps:
                running = _apply(running, op, v)
            expected_correct = running
        except ValueError as e:
            issues.append(Issue(
                iid, "arith_compute_error", "CRITICAL",
                f"Error computing correct answer: {e}"
            ))
            continue

        if it["conflict"]:
            if str(expected_correct) != it["correct_answer"]:
                issues.append(Issue(
                    iid, "arith_wrong_correct", "CRITICAL",
                    f"Arithmetic: expected {expected_correct}, "
                    f"got '{it['correct_answer']}'"
                ))

            # Compute lure answer (flip the trap step's operation)
            try:
                lure_steps = list(steps)
                op_orig, v_orig = lure_steps[trap_step]
                lure_steps[trap_step] = (_flip_op(op_orig), v_orig)
                lure_running = start
                for op, v in lure_steps:
                    lure_running = _apply(lure_running, op, v)
                expected_lure = lure_running
                if str(expected_lure) != it["lure_answer"]:
                    issues.append(Issue(
                        iid, "arith_wrong_lure", "CRITICAL",
                        f"Arithmetic lure: expected {expected_lure}, "
                        f"got '{it['lure_answer']}'"
                    ))
                if expected_lure == expected_correct:
                    issues.append(Issue(
                        iid, "arith_degenerate", "CRITICAL",
                        "Lure equals correct answer"
                    ))
            except (ValueError, ZeroDivisionError) as e:
                issues.append(Issue(
                    iid, "arith_lure_compute_error", "HIGH",
                    f"Error computing lure: {e}"
                ))
        else:
            # Control item should have same correct answer
            if str(expected_correct) != it["correct_answer"]:
                issues.append(Issue(
                    iid, "arith_control_wrong_correct", "CRITICAL",
                    f"Arithmetic control: expected {expected_correct}, "
                    f"got '{it['correct_answer']}'"
                ))

    return issues


def check_framing_ev(items: list[dict[str, Any]]) -> list[Issue]:
    """Verify framing items have matching expected values.

    The build spec uses exact fractions (e.g. prob_save_all=1/3) so the
    underlying EVs DO match. However, the prompt renders 1/3 as "33%"
    (via ``round(x * 100)``), which introduces a ~1% rounding gap. We
    flag this as a MEDIUM quality issue (prompt-level ambiguity), not
    CRITICAL, since the generator enforces exact EV equality at
    construction time.
    """
    issues = []
    framing_items = [it for it in items if it["category"] == "framing" and it["conflict"]]

    for it in framing_items:
        iid = it["id"]
        prompt = it["prompt"]

        # Extract n_total, n_certain_save, prob_save_all from prompt
        total_m = re.search(r'kill (\d+) people', prompt)
        save_m = re.search(r'(\d+) people will be saved', prompt)
        prob_m = re.search(r'probability (\d+)% all', prompt)

        if total_m and save_m and prob_m:
            n_total = int(total_m.group(1))
            n_certain_save = int(save_m.group(1))
            prob_pct = int(prob_m.group(1))
            prob_save_all = prob_pct / 100.0

            ev_sure = n_certain_save
            ev_gamble = prob_save_all * n_total

            # Check with the exact fraction (33% = 1/3, 25% = 1/4, etc.)
            # Common fractions used in the specs:
            fraction_map = {20: 1/5, 25: 1/4, 33: 1/3, 50: 1/2, 67: 2/3}
            exact_prob = fraction_map.get(prob_pct, prob_save_all)
            ev_gamble_exact = exact_prob * n_total

            if abs(ev_sure - ev_gamble_exact) > 1e-6:
                issues.append(Issue(
                    iid, "framing_ev_mismatch", "CRITICAL",
                    f"EV mismatch even with exact fraction: "
                    f"sure={ev_sure}, gamble={ev_gamble_exact:.4f} "
                    f"(n_total={n_total}, n_save={n_certain_save}, "
                    f"p_exact={exact_prob:.6f})"
                ))

            # Flag rounding ambiguity at prompt level
            if abs(ev_sure - ev_gamble) > 0.5:
                issues.append(Issue(
                    iid, "framing_prompt_rounding", "MEDIUM",
                    f"Prompt says {prob_pct}% but actual prob is "
                    f"{exact_prob:.6f}. Prompt-level EV: "
                    f"sure={ev_sure}, gamble(prompt)={ev_gamble:.1f}. "
                    f"A literal reader may see unequal EVs."
                ))
        else:
            issues.append(Issue(
                iid, "framing_parse_fail", "MEDIUM",
                "Could not parse framing parameters from prompt"
            ))

    return issues


def check_syllogism_structure(items: list[dict[str, Any]]) -> list[Issue]:
    """Check syllogism structural validity.

    Key checks:
    - Answer values are "valid" or "invalid"
    - Conflict items: the syllogistic form matches the claimed validity
    - Control items: conclusion must NOT be identical to a premise
      (otherwise it's trivially entailed regardless of syllogistic form)
    """
    issues = []
    syll_items = [it for it in items if it["category"] == "syllogism"]

    for it in syll_items:
        iid = it["id"]
        prompt = it["prompt"]
        correct = it["correct_answer"]
        lure = it["lure_answer"]

        # Check answer is "valid" or "invalid"
        if it["conflict"]:
            if correct not in ("valid", "invalid"):
                issues.append(Issue(
                    iid, "syll_bad_correct", "CRITICAL",
                    f"Syllogism correct_answer is '{correct}', expected 'valid' or 'invalid'"
                ))
            if lure not in ("valid", "invalid"):
                issues.append(Issue(
                    iid, "syll_bad_lure", "CRITICAL",
                    f"Syllogism lure_answer is '{lure}', expected 'valid' or 'invalid'"
                ))
            if correct == lure:
                issues.append(Issue(
                    iid, "syll_degenerate", "CRITICAL",
                    "Syllogism correct == lure"
                ))
        else:
            if correct not in ("valid", "invalid"):
                issues.append(Issue(
                    iid, "syll_control_bad_correct", "CRITICAL",
                    f"Syllogism control correct_answer is '{correct}', "
                    "expected 'valid' or 'invalid'"
                ))

        # Extract premises and check structure
        p1_m = re.search(r'Premise 1: (.+)', prompt)
        p2_m = re.search(r'Premise 2: (.+)', prompt)
        conc_m = re.search(r'Conclusion: (.+)', prompt)

        if p1_m and p2_m and conc_m:
            p1 = p1_m.group(1).strip()
            p2 = p2_m.group(1).strip()
            conc = conc_m.group(1).strip()

            # All premises and conclusion should start with "All"
            for label, text in [("P1", p1), ("P2", p2), ("Conclusion", conc)]:
                if not text.startswith("All "):
                    issues.append(Issue(
                        iid, "syll_not_universal", "MEDIUM",
                        f"{label} does not start with 'All': '{text[:60]}'"
                    ))

            # CRITICAL: conclusion identical to a premise means the
            # argument is trivially valid regardless of form.
            # If marked "invalid" this is a wrong answer.
            if conc == p2 or conc == p1:
                which_prem = "P2" if conc == p2 else "P1"
                if correct == "invalid":
                    issues.append(Issue(
                        iid, "syll_trivially_valid", "CRITICAL",
                        f"Conclusion is identical to {which_prem}, making the "
                        f"argument trivially valid, but correct_answer='invalid'. "
                        f"The control_conclusion defaulted to the conflict's "
                        f"conclusion, which happens to equal the control's "
                        f"{which_prem}. Fix: supply an explicit control_conclusion "
                        f"in the build spec that is the AAA-2 conclusion "
                        f"(e.g. 'All S are P' from the AAA-2 form)."
                    ))

    return issues


def check_base_rate(items: list[dict[str, Any]]) -> list[Issue]:
    """Check base-rate items for validity."""
    issues = []
    br_items = [it for it in items if it["category"] == "base_rate" and it["conflict"]]

    for it in br_items:
        iid = it["id"]
        prompt = it["prompt"]
        correct = it["correct_answer"]
        lure = it["lure_answer"]

        # The correct answer should be the common group (high base rate).
        # Prompt format: "X% of respondents are <group>s and only Y% are <group>s."
        # The group names are PLURALIZED in the prompt. We parse up to
        # " and only" for the common group, and up to "." for the rare group,
        # then strip the trailing "s" to recover the singular form used in
        # correct_answer / lure_answer.
        rate_m = re.search(
            r'(\d+)% of respondents are (.+?) and only (\d+)% are (.+?)\.',
            prompt
        )
        if rate_m:
            common_pct = int(rate_m.group(1))
            common_group_plural = rate_m.group(2).strip()
            rare_pct = int(rate_m.group(3))
            rare_group_plural = rate_m.group(4).strip()

            # Recover singular by stripping trailing 's'
            common_group = common_group_plural.rstrip("s")
            rare_group = rare_group_plural.rstrip("s")

            if common_pct <= rare_pct:
                issues.append(Issue(
                    iid, "br_rates_not_dominated", "CRITICAL",
                    f"Common rate ({common_pct}%) not > rare rate ({rare_pct}%)"
                ))

            # Correct answer should be the common group
            if correct.lower() != common_group.lower():
                issues.append(Issue(
                    iid, "br_wrong_correct", "CRITICAL",
                    f"Correct answer '{correct}' != common group '{common_group}'"
                ))

            # Lure should be the rare group
            if lure.lower() != rare_group.lower():
                issues.append(Issue(
                    iid, "br_wrong_lure", "CRITICAL",
                    f"Lure answer '{lure}' != rare group '{rare_group}'"
                ))

            # Check the ratio is compelling (at least 3x)
            if common_pct < rare_pct * 3:
                issues.append(Issue(
                    iid, "br_weak_ratio", "HIGH",
                    f"Common/rare ratio too small: {common_pct}/{rare_pct} < 3"
                ))
        else:
            issues.append(Issue(
                iid, "br_parse_fail", "MEDIUM",
                "Could not parse base rates from prompt"
            ))

    return issues


def check_conjunction(items: list[dict[str, Any]]) -> list[Issue]:
    """Check conjunction fallacy items."""
    issues = []
    conj_items = [it for it in items if it["category"] == "conjunction"]

    for it in conj_items:
        iid = it["id"]
        prompt = it["prompt"]
        correct = it["correct_answer"]
        lure = it["lure_answer"]

        if it["conflict"]:
            # Correct must be A (the single feature), lure must be B (the conjunction)
            if correct != "A":
                issues.append(Issue(
                    iid, "conj_wrong_correct", "CRITICAL",
                    f"Conjunction correct should be 'A', got '{correct}'"
                ))
            if lure != "B":
                issues.append(Issue(
                    iid, "conj_wrong_lure", "CRITICAL",
                    f"Conjunction lure should be 'B', got '{lure}'"
                ))

            # Check that option B is a conjunction of A
            # B should contain "and" joining feature_a and feature_b
            a_m = re.search(r'A\) .+ is a (.+?)\.', prompt)
            b_m = re.search(r'B\) .+ is a (.+?)\.', prompt)
            if a_m and b_m:
                feature_a_text = a_m.group(1).strip()
                feature_b_text = b_m.group(1).strip()
                # B should contain feature_a as a substring
                if feature_a_text.lower() not in feature_b_text.lower():
                    issues.append(Issue(
                        iid, "conj_b_not_superset", "HIGH",
                        f"B ('{feature_b_text}') does not contain A ('{feature_a_text}')"
                    ))
        else:
            if correct != "A":
                issues.append(Issue(
                    iid, "conj_control_wrong", "CRITICAL",
                    f"Conjunction control correct should be 'A', got '{correct}'"
                ))

    return issues


def check_anchoring(items: list[dict[str, Any]]) -> list[Issue]:
    """Check anchoring items for validity."""
    issues = []
    anchor_items = [it for it in items if it["category"] == "anchoring"]

    for it in anchor_items:
        iid = it["id"]
        correct = it["correct_answer"]
        lure = it["lure_answer"]

        if it["conflict"]:
            # Correct is true_value, lure is high_anchor
            try:
                true_val = int(correct)
                high_anchor = int(lure)
                if high_anchor <= true_val:
                    issues.append(Issue(
                        iid, "anchor_not_above", "CRITICAL",
                        f"High anchor ({high_anchor}) not above true value ({true_val})"
                    ))
                # Check the anchor is at least 1.5x the true value
                if true_val > 0 and high_anchor / true_val < 1.3:
                    issues.append(Issue(
                        iid, "anchor_too_close", "HIGH",
                        f"Anchor ({high_anchor}) only {high_anchor/true_val:.2f}x "
                        f"true value ({true_val}) -- may be too close"
                    ))
            except ValueError:
                issues.append(Issue(
                    iid, "anchor_non_numeric", "CRITICAL",
                    f"Non-numeric answers: correct='{correct}', lure='{lure}'"
                ))

    return issues


def check_paraphrases(items: list[dict[str, Any]]) -> list[Issue]:
    """Check that conflict items have at least 2 paraphrases and controls have at least 1."""
    issues = []
    for it in items:
        iid = it["id"]
        paraphrases = it.get("paraphrases", [])
        if it["conflict"]:
            if len(paraphrases) < 2:
                issues.append(Issue(
                    iid, "few_paraphrases", "LOW",
                    f"Conflict item has only {len(paraphrases)} paraphrase(s), expected >= 2"
                ))
        else:
            if len(paraphrases) < 1:
                issues.append(Issue(
                    iid, "no_paraphrases_control", "LOW",
                    "Control item has no paraphrases"
                ))

    return issues


def check_prompt_quality(items: list[dict[str, Any]]) -> list[Issue]:
    """Basic quality checks on prompts."""
    issues = []
    for it in items:
        iid = it["id"]
        prompt = it["prompt"]

        # Empty prompt
        if not prompt.strip():
            issues.append(Issue(iid, "empty_prompt", "CRITICAL", "Prompt is empty"))
            continue

        # Prompt too short (< 50 chars)
        if len(prompt) < 50:
            issues.append(Issue(
                iid, "short_prompt", "MEDIUM",
                f"Prompt is very short ({len(prompt)} chars)"
            ))

        # Check for response format instruction
        has_format = any(phrase in prompt.lower() for phrase in [
            "answer with", "respond with", "reply with", "give the",
            "report", "one letter", "one word", "integer only",
            "number only", "number of", "answer 'a' or 'b'",
            "answer 'valid' or 'invalid'", "single letter",
            "single integer", "single word",
        ])
        if not has_format:
            issues.append(Issue(
                iid, "no_format_instruction", "MEDIUM",
                "Prompt may lack an explicit answer format instruction"
            ))

    return issues


def verify_factual_anchoring(items: list[dict[str, Any]]) -> list[Issue]:
    """Verify known factual anchoring true values."""
    issues = []
    known_facts = {
        "anchor_un_members": (193, "UN member states as of 2023"),
        "anchor_periodic_elements": (118, "Elements on periodic table"),
        "anchor_country_capital_year": (1913, "Canberra officially founded"),
        "anchor_amazon_river_length": (6400, "Amazon River length in km (approx)"),
        "anchor_human_chromosomes": (46, "Human somatic cell chromosomes"),
        "anchor_sun_radius": (109, "Sun/Earth radius ratio"),
        "anchor_hist_olympics": (1896, "First modern Olympics in Athens"),
        "anchor_keys_piano": (88, "Standard grand piano keys"),
        "anchor_great_wall": (21000, "Great Wall total length in km"),
        "anchor_marathon_distance": (42195, "Olympic marathon in metres"),
        "anchor_mount_fuji": (3776, "Mount Fuji height in metres"),
        "anchor_dna_pairs": (3000, "Human genome base pairs in millions"),
        "anchor_us_states": (50, "US states"),
        "anchor_speed_light": (300, "Speed of light in thousand km/s"),
        "anchor_everest_meters": (8849, "Mount Everest height in metres"),
    }

    for it in items:
        if it["category"] != "anchoring" or not it["conflict"]:
            continue

        pair_id = it["matched_pair_id"]
        if pair_id in known_facts:
            expected_val, fact_desc = known_facts[pair_id]
            try:
                actual = int(it["correct_answer"])
                if actual != expected_val:
                    issues.append(Issue(
                        it["id"], "factual_error", "CRITICAL",
                        f"Factual error for {fact_desc}: "
                        f"expected {expected_val}, got {actual}"
                    ))
            except ValueError:
                pass

    return issues


def main() -> None:
    path = BENCHMARK_PATH
    if not path.exists():
        print(f"ERROR: Benchmark file not found: {path}")
        sys.exit(1)

    print(f"Loading benchmark from: {path}")
    items = load_items(path)
    print(f"Loaded {len(items)} items")
    print()

    all_issues: list[Issue] = []
    checks = [
        ("Schema & structure", check_schema),
        ("Regex patterns", check_regex_patterns),
        ("Matched pairs", check_matched_pairs),
        ("CRT mathematics", check_crt_math),
        ("Arithmetic recomputation", check_arithmetic),
        ("Framing EV equality", check_framing_ev),
        ("Syllogism structure", check_syllogism_structure),
        ("Base-rate validity", check_base_rate),
        ("Conjunction fallacy", check_conjunction),
        ("Anchoring validity", check_anchoring),
        ("Factual anchoring values", verify_factual_anchoring),
        ("Paraphrase coverage", check_paraphrases),
        ("Prompt quality", check_prompt_quality),
    ]

    for name, fn in checks:
        issues = fn(items)
        all_issues.extend(issues)
        count_by_sev = Counter(i.severity for i in issues)
        status = "PASS" if not issues else "FAIL"
        sev_str = ", ".join(f"{s}: {c}" for s, c in sorted(count_by_sev.items()))
        print(f"[{status}] {name}: {len(issues)} issues" +
              (f" ({sev_str})" if issues else ""))

    print()
    print("=" * 70)
    print(f"TOTAL ISSUES: {len(all_issues)}")
    by_severity = Counter(i.severity for i in all_issues)
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        print(f"  {sev}: {by_severity.get(sev, 0)}")
    print("=" * 70)

    if all_issues:
        print()
        print("DETAILED ISSUES:")
        print("-" * 70)
        for issue in sorted(all_issues, key=lambda x: (
            {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[x.severity],
            x.item_id
        )):
            print(f"[{issue.severity:8s}] {issue.item_id}")
            print(f"           Type: {issue.issue_type}")
            print(f"           {issue.description}")
            print()

    # Summary statistics
    print()
    print("SUMMARY STATISTICS:")
    print(f"  Total items: {len(items)}")
    cats = Counter(it["category"] for it in items)
    for c, n in sorted(cats.items()):
        print(f"    {c}: {n}")
    print(f"  Conflict items: {sum(1 for it in items if it['conflict'])}")
    print(f"  Control items: {sum(1 for it in items if not it['conflict'])}")
    print(f"  Matched pairs: {len(set(it['matched_pair_id'] for it in items))}")

    # Exit code
    n_critical = by_severity.get("CRITICAL", 0)
    n_high = by_severity.get("HIGH", 0)
    if n_critical > 0:
        print(f"\nEXIT: {n_critical} CRITICAL issues found!")
        sys.exit(2)
    elif n_high > 0:
        print(f"\nEXIT: {n_high} HIGH issues found (no CRITICAL)")
        sys.exit(1)
    else:
        print("\nEXIT: No CRITICAL or HIGH issues. Benchmark passes.")
        sys.exit(0)


if __name__ == "__main__":
    main()
