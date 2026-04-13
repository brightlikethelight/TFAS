#!/usr/bin/env python3
"""Generate comprehensive behavioral results tables for the s1s2 paper.

Produces:
  1. Full LaTeX table (6 models x 11 categories) for supplementary materials
  2. Compact LaTeX table (6 models x 7 original categories + overall) for main paper
  3. JSON summary at results/summary/behavioral_complete.json
  4. Human-readable stdout summary

Wilson score interval for binomial 95% CI on all cells.
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
    """Wilson score interval for binomial proportion.

    Returns (lower, upper) as proportions in [0, 1].
    """
    if n_total == 0:
        return (0.0, 0.0)
    p_hat = n_success / n_total
    denom = 1 + z**2 / n_total
    centre = (p_hat + z**2 / (2 * n_total)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent.parent
BEHAVIORAL_DIR = BASE / "results_pod" / "behavioral"
OUTPUT_DIR = BASE / "results" / "summary"

# Canonical model order and display names
MODEL_SPECS: list[dict[str, Any]] = [
    {
        "key": "llama",
        "display": "Llama-3.1-8B",
        "files": ["llama31_8b_ALL.json", "new_items_llama_31_8b_instruct.json"],
    },
    {
        "key": "r1_distill",
        "display": "R1-Distill-8B",
        "files": ["r1_distill_llama_ALL.json", "new_items_r1_distill_llama_8b.json"],
    },
    {
        "key": "qwen_nothink",
        "display": r"Qwen3-8B {\small(no think)}",
        "files": ["qwen3_8b_no_think.json"],
    },
    {
        "key": "qwen_think",
        "display": r"Qwen3-8B {\small(think)}",
        "files": ["qwen3_8b_think.json"],
    },
    {
        "key": "olmo_instruct",
        "display": "OLMo-3-7B-Instruct",
        "files": ["olmo3_instruct_ALL.json"],
    },
    {
        "key": "olmo_think",
        "display": "OLMo-3-7B-Think",
        "files": ["olmo3_think_ALL.json"],
    },
]

# All 11 categories in display order.
# First 7 are the original benchmark; last 4 are new categories.
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
    """Load and merge results from one or more JSON files, deduplicating by item ID."""
    seen_ids: set[str] = set()
    merged: list[dict[str, Any]] = []
    for fname in filenames:
        path = BEHAVIORAL_DIR / fname
        if not path.exists():
            print(f"WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        with open(path) as f:
            data = json.load(f)
        for item in data["results"]:
            item_id = item["id"]
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                merged.append(item)
    return merged


# ---------------------------------------------------------------------------
# Per-category statistics
# ---------------------------------------------------------------------------

def compute_category_stats(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute per-category lure statistics on CONFLICT items only.

    Returns dict[category] -> {n_conflict, n_lure, lure_rate, ci_lo, ci_hi}
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"conflict": 0, "lure": 0})
    for item in results:
        if not item.get("conflict", False):
            continue
        cat = item["category"]
        counts[cat]["conflict"] += 1
        if item["verdict"] == "lure":
            counts[cat]["lure"] += 1

    stats: dict[str, dict[str, Any]] = {}
    for cat, c in counts.items():
        n = c["conflict"]
        k = c["lure"]
        rate = k / n if n > 0 else 0.0
        lo, hi = wilson_ci(k, n)
        stats[cat] = {
            "n_conflict": n,
            "n_lure": k,
            "lure_rate": rate,
            "ci_lo": lo,
            "ci_hi": hi,
        }
    return stats


def compute_overall(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute overall lure rate across ALL conflict items."""
    n = sum(1 for r in results if r.get("conflict", False))
    k = sum(1 for r in results if r.get("conflict", False) and r["verdict"] == "lure")
    rate = k / n if n > 0 else 0.0
    lo, hi = wilson_ci(k, n)
    return {"n_conflict": n, "n_lure": k, "lure_rate": rate, "ci_lo": lo, "ci_hi": hi}


# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------

def _fmt_cell(stats: dict[str, Any] | None, bold_threshold: float = 0.10) -> str:
    """Format a single table cell: rate [CI], bolded if above threshold."""
    if stats is None:
        return "---"
    rate = stats["lure_rate"]
    lo = stats["ci_lo"]
    hi = stats["ci_hi"]
    cell = f"{rate * 100:.1f}" + r"\%" + r"\,{\tiny " + f"[{lo * 100:.0f},{hi * 100:.0f}]" + "}"
    if rate > bold_threshold:
        cell = r"\textbf{" + cell + "}"
    return cell


def _fmt_cell_compact(stats: dict[str, Any] | None, bold_threshold: float = 0.10) -> str:
    """Compact cell: just the percentage, bolded if above threshold."""
    if stats is None:
        return "---"
    rate = stats["lure_rate"]
    cell = f"{rate * 100:.1f}\\%"
    if rate > bold_threshold:
        cell = r"\textbf{" + cell + "}"
    return cell


def generate_full_table(
    all_stats: dict[str, dict[str, dict[str, Any]]],
    all_overall: dict[str, dict[str, Any]],
) -> str:
    """Full supplementary table: 6 models x 11 categories + overall, with CIs."""
    n_cats = len(ALL_CATEGORIES)
    col_spec = "l c " + " ".join(["c"] * n_cats)

    lines = [
        r"% Full behavioral results table — all models x all categories",
        f"% Generated: {datetime.now(UTC).isoformat()}",
        r"% Script: scripts/make_behavioral_table.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Lure rates (\%) on conflict items by model and cognitive-bias category, "
        r"with Wilson 95\% CIs. "
        r"\textbf{Bold} indicates lure rate $>$ 10\%. "
        r"``---'' indicates the category was not evaluated for that model.}",
        r"\label{tab:behavioral-full}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Header row
    header_cats = " & ".join(CATEGORY_DISPLAY[c] for c in ALL_CATEGORIES)
    lines.append(f"Model & Overall & {header_cats} \\\\")

    # Midrule after original 7, before new 4
    # Use cmidrule to visually separate
    lines.append(r"\midrule")

    for spec in MODEL_SPECS:
        key = spec["key"]
        display = spec["display"]
        overall = all_overall[key]
        cat_stats = all_stats[key]

        overall_cell = _fmt_cell(overall)
        cat_cells = []
        for cat in ALL_CATEGORIES:
            cat_cells.append(_fmt_cell(cat_stats.get(cat)))

        row = f"{display} & {overall_cell} & " + " & ".join(cat_cells) + r" \\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def generate_compact_table(
    all_orig_stats: dict[str, dict[str, dict[str, Any]]],
    all_orig_overall: dict[str, dict[str, Any]],
) -> str:
    """Compact main-paper table: 6 models x 7 original categories + overall.

    Uses original-only stats (excludes natural frequency / new-category items)
    so that per-category cells and overall are self-consistent.
    """
    cats = ORIGINAL_CATEGORIES
    n_cats = len(cats)
    col_spec = "l c " + " ".join(["c"] * n_cats)

    lines = [
        r"% Compact behavioral results table — main paper",
        f"% Generated: {datetime.now(UTC).isoformat()}",
        r"% Script: scripts/make_behavioral_table.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Lure rates (\%) on conflict items across six models and seven cognitive-bias categories. "
        r"\textbf{Bold} indicates lure rate $>$ 10\%.}",
        r"\label{tab:behavioral}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    header_cats = " & ".join(CATEGORY_DISPLAY[c] for c in cats)
    lines.append(f"Model & Overall & {header_cats} \\\\")
    lines.append(r"\midrule")

    for spec in MODEL_SPECS:
        key = spec["key"]
        display = spec["display"]
        overall = all_orig_overall[key]
        cat_stats = all_orig_stats[key]

        overall_cell = _fmt_cell_compact(overall)
        cat_cells = [_fmt_cell_compact(cat_stats.get(cat)) for cat in cats]
        row = f"{display} & {overall_cell} & " + " & ".join(cat_cells) + r" \\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

def print_summary(
    all_stats: dict[str, dict[str, dict[str, Any]]],
    all_overall: dict[str, dict[str, Any]],
) -> None:
    """Print a clear ASCII summary to stdout."""
    print("=" * 90)
    print("BEHAVIORAL RESULTS — LURE RATES ON CONFLICT ITEMS (Wilson 95% CI)")
    print("=" * 90)

    for spec in MODEL_SPECS:
        key = spec["key"]
        display = spec["display"].replace(r"{\small(", "(").replace(")}", ")")
        overall = all_overall[key]
        cat_stats = all_stats[key]

        print(f"\n{display}")
        print(f"  Overall: {overall['n_lure']}/{overall['n_conflict']} = "
              f"{overall['lure_rate']*100:.1f}% "
              f"[{overall['ci_lo']*100:.0f}, {overall['ci_hi']*100:.0f}]")

        for cat in ALL_CATEGORIES:
            s = cat_stats.get(cat)
            if s is None:
                print(f"  {CATEGORY_DISPLAY[cat]:>16s}: ---")
            else:
                marker = " ***" if s["lure_rate"] > 0.10 else ""
                print(f"  {CATEGORY_DISPLAY[cat]:>16s}: "
                      f"{s['n_lure']:>3d}/{s['n_conflict']:>3d} = "
                      f"{s['lure_rate']*100:5.1f}% "
                      f"[{s['ci_lo']*100:4.0f}, {s['ci_hi']*100:4.0f}]{marker}")

    print("\n" + "=" * 90)


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------

def build_json_summary(
    all_stats: dict[str, dict[str, dict[str, Any]]],
    all_overall: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a serializable summary dict."""
    models = {}
    for spec in MODEL_SPECS:
        key = spec["key"]
        display = spec["display"].replace(r"{\small(", "(").replace(")}", ")")
        overall = all_overall[key]
        cat_stats = all_stats[key]

        m: dict[str, Any] = {
            "display_name": display,
            "overall": {
                "n_conflict": overall["n_conflict"],
                "n_lure": overall["n_lure"],
                "lure_rate": round(overall["lure_rate"], 4),
                "ci_95": [round(overall["ci_lo"], 4), round(overall["ci_hi"], 4)],
            },
            "categories": {},
        }
        for cat in ALL_CATEGORIES:
            s = cat_stats.get(cat)
            if s is None:
                m["categories"][cat] = None
            else:
                m["categories"][cat] = {
                    "n_conflict": s["n_conflict"],
                    "n_lure": s["n_lure"],
                    "lure_rate": round(s["lure_rate"], 4),
                    "ci_95": [round(s["ci_lo"], 4), round(s["ci_hi"], 4)],
                }
        models[key] = m

    return {
        "generated": datetime.now(UTC).isoformat(),
        "script": "scripts/make_behavioral_table.py",
        "categories_original": ORIGINAL_CATEGORIES,
        "categories_new": NEW_CATEGORIES,
        "models": models,
    }


# ---------------------------------------------------------------------------
# Original-item filtering for compact table
# ---------------------------------------------------------------------------

# Canonical original-330 item IDs from llama31_8b_ALL.json (the reference file).
# All models in the original benchmark share the same 330 items.
_ORIGINAL_330_IDS: set[str] | None = None


def _get_original_330_ids() -> set[str]:
    """Lazily load the canonical set of 330 item IDs from the Llama reference file."""
    global _ORIGINAL_330_IDS
    if _ORIGINAL_330_IDS is None:
        ref_path = BEHAVIORAL_DIR / "llama31_8b_ALL.json"
        with open(ref_path) as f:
            data = json.load(f)
        _ORIGINAL_330_IDS = {r["id"] for r in data["results"]}
    return _ORIGINAL_330_IDS


def _filter_original_items(
    spec: dict[str, Any], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Filter results to only the original 330-item benchmark.

    Uses the canonical Llama 330-item ID set as reference. This excludes
    natural frequency base_rate items (brnf_*), sunk cost, availability,
    certainty effect, and loss aversion items — even for OLMo which has
    them all in a single file.
    """
    orig_ids = _get_original_330_ids()
    return [r for r in results if r["id"] in orig_ids]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_stats: dict[str, dict[str, dict[str, Any]]] = {}
    all_overall: dict[str, dict[str, Any]] = {}
    # Original-only stats for the compact table (excludes new-category / natural-freq items)
    all_orig_stats: dict[str, dict[str, dict[str, Any]]] = {}
    all_orig_overall: dict[str, dict[str, Any]] = {}

    for spec in MODEL_SPECS:
        key = spec["key"]
        results = load_results(spec["files"])

        # Full stats (all categories merged)
        cat_stats = compute_category_stats(results)
        overall = compute_overall(results)
        all_stats[key] = cat_stats
        all_overall[key] = overall

        # Original-330-item-only stats: restrict to items from the original files
        # (i.e., categories in ORIGINAL_CATEGORIES, and only IDs from the *_ALL.json files)
        orig_results = _filter_original_items(spec, results)
        orig_cat_stats = compute_category_stats(orig_results)
        orig_overall = compute_overall(orig_results)
        all_orig_stats[key] = orig_cat_stats
        all_orig_overall[key] = orig_overall

    # Print human-readable summary
    print_summary(all_stats, all_overall)

    # Generate LaTeX
    full_tex = generate_full_table(all_stats, all_overall)
    compact_tex = generate_compact_table(all_orig_stats, all_orig_overall)

    # Write LaTeX
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = OUTPUT_DIR / "behavioral_tables.tex"
    with open(tex_path, "w") as f:
        f.write("% " + "=" * 70 + "\n")
        f.write("% Comprehensive behavioral results tables for s1s2 paper\n")
        f.write(f"% Generated: {datetime.now(UTC).isoformat()}\n")
        f.write("% Script: scripts/make_behavioral_table.py\n")
        f.write("% " + "=" * 70 + "\n\n")
        f.write("% ---- COMPACT TABLE (main paper) ----\n\n")
        f.write(compact_tex)
        f.write("\n\n")
        f.write("% ---- FULL TABLE (supplementary) ----\n\n")
        f.write(full_tex)
        f.write("\n")

    print(f"\nLaTeX tables written to: {tex_path}")

    # Write JSON
    summary = build_json_summary(all_stats, all_overall)
    json_path = OUTPUT_DIR / "behavioral_complete.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary written to:  {json_path}")

    # Also print LaTeX to stdout for review
    print("\n" + "=" * 90)
    print("COMPACT TABLE (main paper)")
    print("=" * 90)
    print(compact_tex)
    print("\n" + "=" * 90)
    print("FULL TABLE (supplementary)")
    print("=" * 90)
    print(full_tex)


if __name__ == "__main__":
    main()
