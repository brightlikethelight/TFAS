#!/usr/bin/env python3
"""Compute full behavior reporting table with correct/lure/other rates.

Produces per-category and overall:
  - Conflict items: n_conflict, n_correct, n_lure, n_other, conflict_accuracy, lure_rate, other_rate
  - Control items: n_control, control_accuracy
  - Gap: control_accuracy - conflict_accuracy
  - Wilson 95% CIs on all rates
  - McNemar test on matched conflict vs control pairs

Output:
  - LaTeX table for the paper
  - JSON summary at results/summary/full_behavior_table.json
  - Human-readable stdout
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------

def wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion. Returns (lower, upper) in [0, 1]."""
    if n_total == 0:
        return (0.0, 0.0)
    p_hat = n_success / n_total
    denom = 1 + z**2 / n_total
    centre = (p_hat + z**2 / (2 * n_total)) / denom
    margin = (z / denom) * math.sqrt(
        p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# McNemar's test (exact, mid-p variant)
# ---------------------------------------------------------------------------

def mcnemar_test(b: int, c: int) -> dict[str, Any]:
    """McNemar test on 2x2 discordant pair counts.

    b = pairs where conflict correct, control wrong
    c = pairs where conflict wrong, control correct

    Returns dict with chi2, p_asymptotic, p_exact (binomial mid-p).
    """
    from math import comb, factorial  # noqa: F811

    result: dict[str, Any] = {"b": b, "c": c, "n_discordant": b + c}

    # Asymptotic (chi-squared with continuity correction)
    if b + c == 0:
        result["chi2"] = 0.0
        result["p_asymptotic"] = 1.0
        result["p_exact_midp"] = 1.0
        return result

    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    # p from chi2 with 1 df -- use survival function approximation
    # For simplicity, compute exact binomial p-value (more appropriate for small n)
    result["chi2"] = chi2

    # Exact binomial mid-p test: H0: P(success) = 0.5 on n = b+c trials
    n = b + c
    k = min(b, c)
    # Two-sided p-value: 2 * P(X <= k) under Binom(n, 0.5)
    # Mid-p: subtract 0.5 * P(X = k) from each tail
    p_tail = 0.0
    for i in range(k + 1):
        p_tail += comb(n, i) / (2**n)
    p_exact = 2 * p_tail
    # mid-p correction
    p_at_k = comb(n, k) / (2**n)
    p_midp = p_exact - p_at_k  # subtract one copy of P(X=k) since we double-counted
    p_midp = min(1.0, max(0.0, p_midp))

    result["p_asymptotic"] = _chi2_sf(chi2, 1)
    result["p_exact_midp"] = p_midp
    return result


def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function for chi-squared distribution (1 df only).

    Uses the complementary error function relation:
    P(chi2 > x | df=1) = erfc(sqrt(x/2))
    """
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent.parent
BEHAVIORAL_DIRS = [
    BASE / "results" / "behavioral",
    BASE / "results_pod" / "behavioral",
]
OUTPUT_DIR = BASE / "results" / "summary"

# Canonical model order and display names
MODEL_SPECS: list[dict[str, Any]] = [
    {
        "key": "llama",
        "display": "Llama-3.1-8B",
        "display_latex": "Llama-3.1-8B",
        "files": ["llama31_8b_ALL.json"],
    },
    {
        "key": "r1_distill",
        "display": "R1-Distill-8B",
        "display_latex": "R1-Distill-8B",
        "files": ["r1_distill_llama_ALL.json"],
    },
    {
        "key": "qwen_nothink",
        "display": "Qwen3-8B (no think)",
        "display_latex": r"Qwen3-8B {\small(no think)}",
        "files": ["qwen3_8b_no_think.json"],
    },
    {
        "key": "qwen_think",
        "display": "Qwen3-8B (think)",
        "display_latex": r"Qwen3-8B {\small(think)}",
        "files": ["qwen3_8b_think.json"],
    },
    {
        "key": "olmo_instruct",
        "display": "OLMo-3-7B-Instruct",
        "display_latex": "OLMo-3-7B-Instruct",
        "files": ["olmo3_instruct_ALL.json"],
    },
    {
        "key": "olmo_think",
        "display": "OLMo-3-7B-Think",
        "display_latex": "OLMo-3-7B-Think",
        "files": ["olmo3_think_ALL.json"],
    },
    {
        "key": "olmo32b_instruct",
        "display": "OLMo-2-32B-Instruct",
        "display_latex": "OLMo-2-32B-Instruct",
        "files": ["olmo32b_instruct_ALL.json"],
    },
    {
        "key": "olmo32b_think",
        "display": "OLMo-2-32B-Think",
        "display_latex": "OLMo-2-32B-Think",
        "files": ["olmo32b_think_ALL.json"],
    },
]

ORIGINAL_CATEGORIES = [
    "crt",
    "base_rate",
    "conjunction",
    "syllogism",
    "arithmetic",
    "framing",
    "anchoring",
]

NEW_CATEGORIES = [
    "sunk_cost",
    "availability",
    "certainty_effect",
    "loss_aversion",
]

ALL_CATEGORIES = ORIGINAL_CATEGORIES + NEW_CATEGORIES

CATEGORY_DISPLAY: dict[str, str] = {
    "crt": "CRT",
    "base_rate": "Base Rate",
    "conjunction": "Conjunction",
    "syllogism": "Syllogism",
    "arithmetic": "Arithmetic",
    "framing": "Framing",
    "anchoring": "Anchoring",
    "sunk_cost": "Sunk Cost",
    "availability": "Availability",
    "certainty_effect": "Certainty Eff.",
    "loss_aversion": "Loss Aversion",
}


def load_results(filenames: list[str]) -> list[dict[str, Any]]:
    """Load results from JSON files, searching across behavioral directories.

    Deduplicates by item ID; first found wins.
    """
    seen_ids: set[str] = set()
    merged: list[dict[str, Any]] = []
    for fname in filenames:
        found = False
        for bdir in BEHAVIORAL_DIRS:
            path = bdir / fname
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                for item in data["results"]:
                    item_id = item["id"]
                    if item_id not in seen_ids:
                        seen_ids.add(item_id)
                        merged.append(item)
                found = True
                break  # use first directory that has the file
        if not found:
            print(f"WARNING: {fname} not found in any behavioral dir", file=sys.stderr)
    return merged


# ---------------------------------------------------------------------------
# Per-category statistics -- FULL (correct/lure/other + control)
# ---------------------------------------------------------------------------

def compute_category_stats(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute per-category stats on both conflict and control items.

    Returns dict[category] -> {
        n_conflict, n_correct, n_lure, n_other,
        conflict_accuracy, lure_rate, other_rate,
        conflict_accuracy_ci, lure_rate_ci, other_rate_ci,
        n_control, n_control_correct, control_accuracy, control_accuracy_ci,
        gap, gap_pp
    }
    """
    # Accumulate counts
    conflict_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "lure": 0, "other": 0}
    )
    control_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )

    for item in results:
        cat = item["category"]
        if item.get("conflict", False):
            conflict_counts[cat]["total"] += 1
            v = item["verdict"]
            if v == "correct":
                conflict_counts[cat]["correct"] += 1
            elif v == "lure":
                conflict_counts[cat]["lure"] += 1
            else:
                conflict_counts[cat]["other"] += 1
        else:
            control_counts[cat]["total"] += 1
            if item["verdict"] == "correct":
                control_counts[cat]["correct"] += 1

    all_cats = set(conflict_counts.keys()) | set(control_counts.keys())
    stats: dict[str, dict[str, Any]] = {}
    for cat in all_cats:
        cc = conflict_counts[cat]
        nc = cc["total"]
        n_correct = cc["correct"]
        n_lure = cc["lure"]
        n_other = cc["other"]

        conflict_acc = n_correct / nc if nc > 0 else 0.0
        lure_rate = n_lure / nc if nc > 0 else 0.0
        other_rate = n_other / nc if nc > 0 else 0.0

        ctrl = control_counts[cat]
        n_ctrl = ctrl["total"]
        n_ctrl_correct = ctrl["correct"]
        ctrl_acc = n_ctrl_correct / n_ctrl if n_ctrl > 0 else 0.0

        gap = ctrl_acc - conflict_acc

        stats[cat] = {
            "n_conflict": nc,
            "n_correct": n_correct,
            "n_lure": n_lure,
            "n_other": n_other,
            "conflict_accuracy": conflict_acc,
            "conflict_accuracy_ci": wilson_ci(n_correct, nc),
            "lure_rate": lure_rate,
            "lure_rate_ci": wilson_ci(n_lure, nc),
            "other_rate": other_rate,
            "other_rate_ci": wilson_ci(n_other, nc),
            "n_control": n_ctrl,
            "n_control_correct": n_ctrl_correct,
            "control_accuracy": ctrl_acc,
            "control_accuracy_ci": wilson_ci(n_ctrl_correct, n_ctrl),
            "gap": gap,
            "gap_pp": gap * 100,
        }
    return stats


def compute_overall(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute overall stats across all categories."""
    nc = n_correct = n_lure = n_other = 0
    n_ctrl = n_ctrl_correct = 0

    for item in results:
        if item.get("conflict", False):
            nc += 1
            v = item["verdict"]
            if v == "correct":
                n_correct += 1
            elif v == "lure":
                n_lure += 1
            else:
                n_other += 1
        else:
            n_ctrl += 1
            if item["verdict"] == "correct":
                n_ctrl_correct += 1

    conflict_acc = n_correct / nc if nc > 0 else 0.0
    lure_rate = n_lure / nc if nc > 0 else 0.0
    other_rate = n_other / nc if nc > 0 else 0.0
    ctrl_acc = n_ctrl_correct / n_ctrl if n_ctrl > 0 else 0.0

    return {
        "n_conflict": nc,
        "n_correct": n_correct,
        "n_lure": n_lure,
        "n_other": n_other,
        "conflict_accuracy": conflict_acc,
        "conflict_accuracy_ci": wilson_ci(n_correct, nc),
        "lure_rate": lure_rate,
        "lure_rate_ci": wilson_ci(n_lure, nc),
        "other_rate": other_rate,
        "other_rate_ci": wilson_ci(n_other, nc),
        "n_control": n_ctrl,
        "n_control_correct": n_ctrl_correct,
        "control_accuracy": ctrl_acc,
        "control_accuracy_ci": wilson_ci(n_ctrl_correct, n_ctrl),
        "gap": ctrl_acc - conflict_acc,
        "gap_pp": (ctrl_acc - conflict_acc) * 100,
    }


# ---------------------------------------------------------------------------
# McNemar on matched pairs
# ---------------------------------------------------------------------------

def compute_mcnemar(results: list[dict[str, Any]]) -> dict[str, Any]:
    """McNemar test on matched conflict/control pairs.

    Pairs are matched by ID stem: {stem}_conflict and {stem}_control.
    Counts discordant pairs where one is correct and the other isn't.
    """
    # Index results by ID
    by_id: dict[str, dict[str, Any]] = {r["id"]: r for r in results}

    # Find matched pairs
    conflict_items = [r for r in results if r.get("conflict", False)]
    b = 0  # conflict correct, control wrong
    c = 0  # conflict wrong, control correct
    n_pairs = 0

    for item in conflict_items:
        stem = item["id"].replace("_conflict", "")
        control_id = stem + "_control"
        if control_id not in by_id:
            continue
        ctrl = by_id[control_id]
        n_pairs += 1
        conflict_correct = item["verdict"] == "correct"
        control_correct = ctrl["verdict"] == "correct"

        if conflict_correct and not control_correct:
            b += 1
        elif not conflict_correct and control_correct:
            c += 1

    test_result = mcnemar_test(b, c)
    test_result["n_matched_pairs"] = n_pairs
    return test_result


# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------

def _fmt_rate(rate: float, ci: tuple[float, float]) -> str:
    """Format rate with CI: '45.2 [40, 50]'"""
    return f"{rate * 100:.1f}" + r"\%" + r"\,{\tiny [" + f"{ci[0]*100:.0f},{ci[1]*100:.0f}" + "]}"


def _fmt_pct(rate: float) -> str:
    """Format rate as percentage without CI."""
    return f"{rate * 100:.1f}\\%"


def generate_latex_table(
    all_stats: dict[str, dict[str, dict[str, Any]]],
    all_overall: dict[str, dict[str, Any]],
    all_mcnemar: dict[str, dict[str, Any]],
) -> str:
    """Full LaTeX table: models x categories, showing correct/lure/other + control + gap."""
    # Determine which categories have data
    cats_with_data: list[str] = []
    for cat in ALL_CATEGORIES:
        has_data = any(
            cat in all_stats[spec["key"]] for spec in MODEL_SPECS
            if spec["key"] in all_stats
        )
        if has_data:
            cats_with_data.append(cat)

    n_cats = len(cats_with_data)
    # Table: Model | Overall columns | per-category columns
    # For each category block: Acc_conflict | Lure | Other | Acc_control | Gap
    # That's too wide. Use a more compact layout:
    # Model | Metric | Cat1 | Cat2 | ... | Overall

    lines = [
        r"% Full behavioral results table -- correct/lure/other rates",
        f"% Generated: {datetime.now(UTC).isoformat()}",
        r"% Script: scripts/compute_full_behavior_table.py",
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Complete behavioral results across models and categories. "
        r"Conflict accuracy (Acc), lure rate (Lure), and other-response rate (Other) on conflict items; "
        r"control accuracy (Ctrl); and accuracy gap (Gap = Ctrl $-$ Acc). "
        r"Wilson 95\% CIs in brackets. "
        r"McNemar mid-$p$ tests conflict vs.\ matched control accuracy.}",
        r"\label{tab:behavioral-full}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
    ]

    col_spec = "l l " + " ".join(["r"] * n_cats) + " r"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header
    cat_headers = " & ".join(CATEGORY_DISPLAY[c] for c in cats_with_data)
    lines.append(f"Model & Metric & {cat_headers} & Overall \\\\")
    lines.append(r"\midrule")

    for spec in MODEL_SPECS:
        key = spec["key"]
        if key not in all_stats:
            continue
        display = spec["display_latex"]
        cat_stats = all_stats[key]
        overall = all_overall[key]
        mcn = all_mcnemar[key]

        # Row 1: Conflict accuracy
        cells = []
        for cat in cats_with_data:
            s = cat_stats.get(cat)
            if s is None:
                cells.append("---")
            else:
                cells.append(_fmt_rate(s["conflict_accuracy"], s["conflict_accuracy_ci"]))
        cells.append(_fmt_rate(overall["conflict_accuracy"], overall["conflict_accuracy_ci"]))
        lines.append(f"{display} & Acc & " + " & ".join(cells) + r" \\")

        # Row 2: Lure rate
        cells = []
        for cat in cats_with_data:
            s = cat_stats.get(cat)
            if s is None:
                cells.append("---")
            else:
                val = _fmt_rate(s["lure_rate"], s["lure_rate_ci"])
                if s["lure_rate"] > 0.10:
                    val = r"\textbf{" + val + "}"
                cells.append(val)
        ov_lure = _fmt_rate(overall["lure_rate"], overall["lure_rate_ci"])
        if overall["lure_rate"] > 0.10:
            ov_lure = r"\textbf{" + ov_lure + "}"
        cells.append(ov_lure)
        lines.append(f" & Lure & " + " & ".join(cells) + r" \\")

        # Row 3: Other rate
        cells = []
        for cat in cats_with_data:
            s = cat_stats.get(cat)
            if s is None:
                cells.append("---")
            else:
                cells.append(_fmt_pct(s["other_rate"]))
        cells.append(_fmt_pct(overall["other_rate"]))
        lines.append(f" & Other & " + " & ".join(cells) + r" \\")

        # Row 4: Control accuracy
        cells = []
        for cat in cats_with_data:
            s = cat_stats.get(cat)
            if s is None:
                cells.append("---")
            else:
                cells.append(_fmt_rate(s["control_accuracy"], s["control_accuracy_ci"]))
        cells.append(_fmt_rate(overall["control_accuracy"], overall["control_accuracy_ci"]))
        lines.append(f" & Ctrl & " + " & ".join(cells) + r" \\")

        # Row 5: Gap + McNemar p-value
        cells = []
        for cat in cats_with_data:
            s = cat_stats.get(cat)
            if s is None:
                cells.append("---")
            else:
                g = s["gap_pp"]
                sign = "+" if g >= 0 else ""
                cells.append(f"{sign}{g:.1f}pp")
        g_ov = overall["gap_pp"]
        sign = "+" if g_ov >= 0 else ""
        p_str = _fmt_p(mcn["p_exact_midp"])
        cells.append(f"{sign}{g_ov:.1f}pp {p_str}")
        lines.append(f" & Gap & " + " & ".join(cells) + r" \\")

        lines.append(r"\midrule")

    # Remove last midrule, replace with bottomrule
    lines[-1] = r"\bottomrule"

    lines.extend([
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def _fmt_p(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return "($p < .001$)"
    elif p < 0.01:
        return f"($p = {p:.3f}$)"
    elif p < 0.05:
        return f"($p = {p:.3f}$)"
    else:
        return f"($p = {p:.2f}$)"


# ---------------------------------------------------------------------------
# Human-readable stdout
# ---------------------------------------------------------------------------

def print_full_summary(
    all_stats: dict[str, dict[str, dict[str, Any]]],
    all_overall: dict[str, dict[str, Any]],
    all_mcnemar: dict[str, dict[str, Any]],
) -> None:
    """Print clear ASCII summary with all rates."""
    print("=" * 110)
    print("FULL BEHAVIORAL RESULTS -- Correct / Lure / Other on Conflict + Control Accuracy + Gap")
    print("=" * 110)

    for spec in MODEL_SPECS:
        key = spec["key"]
        if key not in all_stats:
            continue
        display = spec["display"]
        overall = all_overall[key]
        cat_stats = all_stats[key]
        mcn = all_mcnemar[key]

        print(f"\n{'='*80}")
        print(f"  {display}")
        print(f"{'='*80}")

        # Overall
        ov = overall
        print(f"  OVERALL (n_conflict={ov['n_conflict']}, n_control={ov['n_control']}):")
        print(f"    Conflict accuracy: {ov['n_correct']}/{ov['n_conflict']} = "
              f"{ov['conflict_accuracy']*100:.1f}% "
              f"[{ov['conflict_accuracy_ci'][0]*100:.0f}, {ov['conflict_accuracy_ci'][1]*100:.0f}]")
        print(f"    Lure rate:         {ov['n_lure']}/{ov['n_conflict']} = "
              f"{ov['lure_rate']*100:.1f}% "
              f"[{ov['lure_rate_ci'][0]*100:.0f}, {ov['lure_rate_ci'][1]*100:.0f}]")
        print(f"    Other rate:        {ov['n_other']}/{ov['n_conflict']} = "
              f"{ov['other_rate']*100:.1f}% "
              f"[{ov['other_rate_ci'][0]*100:.0f}, {ov['other_rate_ci'][1]*100:.0f}]")
        print(f"    Control accuracy:  {ov['n_control_correct']}/{ov['n_control']} = "
              f"{ov['control_accuracy']*100:.1f}% "
              f"[{ov['control_accuracy_ci'][0]*100:.0f}, {ov['control_accuracy_ci'][1]*100:.0f}]")
        print(f"    Gap (ctrl - acc):  {ov['gap_pp']:+.1f} pp")
        print(f"    McNemar: b={mcn['b']}, c={mcn['c']}, "
              f"n_discordant={mcn['n_discordant']}, "
              f"n_pairs={mcn['n_matched_pairs']}, "
              f"p_midp={mcn['p_exact_midp']:.4f}")

        # Per category
        print()
        hdr = (f"  {'Category':>16s}  {'n':>4s}  "
               f"{'Acc%':>6s} {'[CI]':>10s}  "
               f"{'Lure%':>6s} {'[CI]':>10s}  "
               f"{'Oth%':>5s}  "
               f"{'Ctrl%':>6s} {'[CI]':>10s}  "
               f"{'Gap':>7s}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for cat in ALL_CATEGORIES:
            s = cat_stats.get(cat)
            if s is None:
                continue
            acc_ci = s["conflict_accuracy_ci"]
            lure_ci = s["lure_rate_ci"]
            ctrl_ci = s["control_accuracy_ci"]
            marker = " ***" if s["lure_rate"] > 0.10 else ""
            print(
                f"  {CATEGORY_DISPLAY[cat]:>16s}  {s['n_conflict']:>4d}  "
                f"{s['conflict_accuracy']*100:5.1f}% [{acc_ci[0]*100:4.0f},{acc_ci[1]*100:4.0f}]  "
                f"{s['lure_rate']*100:5.1f}% [{lure_ci[0]*100:4.0f},{lure_ci[1]*100:4.0f}]  "
                f"{s['other_rate']*100:4.1f}%  "
                f"{s['control_accuracy']*100:5.1f}% [{ctrl_ci[0]*100:4.0f},{ctrl_ci[1]*100:4.0f}]  "
                f"{s['gap_pp']:+6.1f}pp{marker}"
            )

    print("\n" + "=" * 110)


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------

def build_json_summary(
    all_stats: dict[str, dict[str, dict[str, Any]]],
    all_overall: dict[str, dict[str, Any]],
    all_mcnemar: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build JSON-serializable summary."""

    def _ser_ci(ci: tuple[float, float]) -> list[float]:
        return [round(ci[0], 4), round(ci[1], 4)]

    def _ser_stats(s: dict[str, Any]) -> dict[str, Any]:
        return {
            "n_conflict": s["n_conflict"],
            "n_correct": s["n_correct"],
            "n_lure": s["n_lure"],
            "n_other": s["n_other"],
            "conflict_accuracy": round(s["conflict_accuracy"], 4),
            "conflict_accuracy_ci": _ser_ci(s["conflict_accuracy_ci"]),
            "lure_rate": round(s["lure_rate"], 4),
            "lure_rate_ci": _ser_ci(s["lure_rate_ci"]),
            "other_rate": round(s["other_rate"], 4),
            "other_rate_ci": _ser_ci(s["other_rate_ci"]),
            "n_control": s["n_control"],
            "n_control_correct": s["n_control_correct"],
            "control_accuracy": round(s["control_accuracy"], 4),
            "control_accuracy_ci": _ser_ci(s["control_accuracy_ci"]),
            "gap_pp": round(s["gap_pp"], 2),
        }

    models = {}
    for spec in MODEL_SPECS:
        key = spec["key"]
        if key not in all_stats:
            continue
        overall = all_overall[key]
        cat_stats = all_stats[key]
        mcn = all_mcnemar[key]

        m: dict[str, Any] = {
            "display_name": spec["display"],
            "overall": _ser_stats(overall),
            "mcnemar": {
                "n_matched_pairs": mcn["n_matched_pairs"],
                "b_conflict_correct_control_wrong": mcn["b"],
                "c_conflict_wrong_control_correct": mcn["c"],
                "n_discordant": mcn["n_discordant"],
                "chi2_continuity": round(mcn["chi2"], 4),
                "p_asymptotic": round(mcn["p_asymptotic"], 6),
                "p_exact_midp": round(mcn["p_exact_midp"], 6),
            },
            "categories": {},
        }
        for cat in ALL_CATEGORIES:
            s = cat_stats.get(cat)
            if s is None:
                m["categories"][cat] = None
            else:
                m["categories"][cat] = _ser_stats(s)
        models[key] = m

    return {
        "generated": datetime.now(UTC).isoformat(),
        "script": "scripts/compute_full_behavior_table.py",
        "description": "Full behavioral results: correct/lure/other rates on conflict items, "
                       "control accuracy, gap, and McNemar tests.",
        "categories_original": ORIGINAL_CATEGORIES,
        "categories_new": NEW_CATEGORIES,
        "models": models,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_stats: dict[str, dict[str, dict[str, Any]]] = {}
    all_overall: dict[str, dict[str, Any]] = {}
    all_mcnemar: dict[str, dict[str, Any]] = {}

    for spec in MODEL_SPECS:
        key = spec["key"]
        results = load_results(spec["files"])
        if not results:
            print(f"WARNING: No results loaded for {spec['display']}, skipping",
                  file=sys.stderr)
            continue

        cat_stats = compute_category_stats(results)
        overall = compute_overall(results)
        mcn = compute_mcnemar(results)

        all_stats[key] = cat_stats
        all_overall[key] = overall
        all_mcnemar[key] = mcn

    # Print human-readable summary
    print_full_summary(all_stats, all_overall, all_mcnemar)

    # Generate LaTeX
    latex = generate_latex_table(all_stats, all_overall, all_mcnemar)

    # Write outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tex_path = OUTPUT_DIR / "full_behavior_tables.tex"
    with open(tex_path, "w") as f:
        f.write(latex + "\n")
    print(f"\nLaTeX table written to: {tex_path}")

    json_path = OUTPUT_DIR / "full_behavior_table.json"
    summary = build_json_summary(all_stats, all_overall, all_mcnemar)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary written to: {json_path}")

    # Print LaTeX to stdout
    print("\n" + "=" * 110)
    print("LATEX TABLE")
    print("=" * 110)
    print(latex)


if __name__ == "__main__":
    main()
