#!/usr/bin/env python3
"""Analyze De Neys confidence paradigm results for Llama-3.1-8B-Instruct.

R1-Distill is excluded from statistical tests because the <think> token
pushes first_token_prob to 1.0 / entropy to 0.0 uniformly — no variance.

Core analyses
-------------
1. Conflict vs control: first_token_prob and top10_entropy, overall + per category
2. Mann-Whitney U with rank-biserial effect size (r_rb)
3. **De Neys critical test**: among LURED conflict items (verdict == "lure"),
   compare confidence to their MATCHED controls (same matched_pair_id).
   If confidence is still lower, this is "conflict detection without resolution."
4. Confidence gap distribution for conflict items.

Outputs
-------
- ``figures/fig7_confidence.pdf`` — two-panel bar chart
- ``results/confidence/confidence_analysis.json`` — full statistics
- ``docs/confidence_analysis.md`` — narrative findings

Usage::

    python scripts/analyze_confidence.py
    python scripts/analyze_confidence.py --data-dir results_pod/confidence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import stats


# ---------------------------------------------------------------------------
# Project theme (consistent with other analysis scripts)
# ---------------------------------------------------------------------------

COLORS = {
    "conflict": "#1f77b4",
    "control": "#ff7f0e",
    "lured": "#d62728",
    "correct_conflict": "#2ca02c",
}


def set_paper_theme() -> None:
    """Publication-grade rcParams matching the project theme."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_confidence(path: Path) -> dict[str, Any]:
    """Load a confidence JSON and return the full dict."""
    print(f"  Loading {path.name} ...")
    with open(path) as f:
        data = json.load(f)
    print(f"    {data['model']}: {data['n_items']} items")
    return data


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from Mann-Whitney U.

    r_rb = 1 - 2U / (n1 * n2).  Ranges [-1, 1].
    """
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


def _safe_float(x: Any) -> float | None:
    """Convert numpy scalar to Python float, handle None/NaN."""
    if x is None:
        return None
    v = float(x)
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def mann_whitney_with_effect(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    alternative: str = "two-sided",
) -> dict[str, float | None]:
    """Mann-Whitney U with rank-biserial effect size.

    Returns dict with U, p, r_rb, n_x, n_y.
    """
    if len(x) < 2 or len(y) < 2:
        return {"U": None, "p": None, "r_rb": None, "n_x": len(x), "n_y": len(y)}
    u_stat, p_val = stats.mannwhitneyu(x, y, alternative=alternative)
    r_rb = rank_biserial(u_stat, len(x), len(y))
    return {
        "U": _safe_float(u_stat),
        "p": _safe_float(p_val),
        "r_rb": _safe_float(r_rb),
        "n_x": int(len(x)),
        "n_y": int(len(y)),
    }


def significance_stars(p: float | None) -> str:
    if p is None:
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def partition_items(
    results: list[dict[str, Any]],
) -> tuple[list[dict], list[dict]]:
    """Split into conflict and control lists."""
    conflict = [r for r in results if r["conflict"]]
    control = [r for r in results if not r["conflict"]]
    return conflict, control


def extract_metric(items: list[dict], key: str) -> NDArray[np.float64]:
    """Pull a float metric from items, dropping None/null."""
    vals = [r[key] for r in items if r.get(key) is not None]
    return np.array(vals, dtype=np.float64)


def per_category_stats(
    results: list[dict[str, Any]],
    metric: str,
) -> dict[str, dict[str, Any]]:
    """Compute conflict vs control stats per category."""
    categories = sorted(set(r["category"] for r in results))
    out: dict[str, dict] = {}
    for cat in categories:
        conf = extract_metric([r for r in results if r["category"] == cat and r["conflict"]], metric)
        ctrl = extract_metric(
            [r for r in results if r["category"] == cat and not r["conflict"]], metric
        )
        mw = mann_whitney_with_effect(conf, ctrl, alternative="two-sided")
        out[cat] = {
            "conflict_mean": _safe_float(np.mean(conf)) if len(conf) else None,
            "conflict_std": _safe_float(np.std(conf, ddof=1)) if len(conf) > 1 else None,
            "control_mean": _safe_float(np.mean(ctrl)) if len(ctrl) else None,
            "control_std": _safe_float(np.std(ctrl, ddof=1)) if len(ctrl) > 1 else None,
            "n_conflict": int(len(conf)),
            "n_control": int(len(ctrl)),
            **mw,
        }
    return out


def de_neys_critical_test(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """The key De Neys test: lured conflict items vs their matched controls.

    For each conflict item where verdict == "lure", find its matched
    control (same matched_pair_id, conflict == False). Compare
    first_token_prob distributions.

    If the lured items show LOWER first_token_prob than their matched
    controls, the model detects conflict even when it fails to resolve it.
    """
    # Build control lookup
    control_by_pair: dict[str, dict] = {}
    for r in results:
        if not r["conflict"]:
            control_by_pair[r["matched_pair_id"]] = r

    lured = [r for r in results if r["conflict"] and r["verdict"] == "lure"]

    # Paired data
    lured_ftp: list[float] = []
    ctrl_ftp: list[float] = []
    lured_ent: list[float] = []
    ctrl_ent: list[float] = []
    pair_ids: list[str] = []

    for r in lured:
        ctrl = control_by_pair.get(r["matched_pair_id"])
        if ctrl is None:
            continue
        if r["first_token_prob"] is None or ctrl["first_token_prob"] is None:
            continue
        lured_ftp.append(r["first_token_prob"])
        ctrl_ftp.append(ctrl["first_token_prob"])
        lured_ent.append(r["top10_entropy"])
        ctrl_ent.append(ctrl["top10_entropy"])
        pair_ids.append(r["matched_pair_id"])

    lured_ftp_arr = np.array(lured_ftp)
    ctrl_ftp_arr = np.array(ctrl_ftp)
    lured_ent_arr = np.array(lured_ent)
    ctrl_ent_arr = np.array(ctrl_ent)

    n_pairs = len(lured_ftp)

    # Paired Wilcoxon signed-rank test (paired data, non-parametric)
    if n_pairs >= 10:
        ftp_diff = lured_ftp_arr - ctrl_ftp_arr
        ent_diff = lured_ent_arr - ctrl_ent_arr

        # Wilcoxon for first_token_prob: expect lured < control
        w_ftp, p_ftp = stats.wilcoxon(ftp_diff, alternative="less")
        # Wilcoxon for entropy: expect lured > control
        w_ent, p_ent = stats.wilcoxon(ent_diff, alternative="greater")

        # Matched-pair rank-biserial for Wilcoxon: r = 1 - 4T / (n(n+1))
        # where T = min(W+, W-)  ... simpler: use the formula directly
        # r_rb = Z / sqrt(N) from the normal approximation, or compute from W
        # Use the direct formula: r = 1 - (2 * W) / (n * (n + 1) / 2)
        # where W is the smaller sum of ranks
        n_nonzero_ftp = np.sum(ftp_diff != 0)
        n_nonzero_ent = np.sum(ent_diff != 0)

        r_rb_ftp = 1.0 - (2.0 * w_ftp) / (n_nonzero_ftp * (n_nonzero_ftp + 1) / 2) if n_nonzero_ftp > 0 else 0.0
        r_rb_ent = 1.0 - (2.0 * w_ent) / (n_nonzero_ent * (n_nonzero_ent + 1) / 2) if n_nonzero_ent > 0 else 0.0
    else:
        w_ftp = p_ftp = w_ent = p_ent = r_rb_ftp = r_rb_ent = None

    # Also compute unpaired MW for comparison
    mw_ftp = mann_whitney_with_effect(lured_ftp_arr, ctrl_ftp_arr, alternative="two-sided")
    mw_ent = mann_whitney_with_effect(lured_ent_arr, ctrl_ent_arr, alternative="two-sided")

    # Per-pair deltas
    ftp_deltas = (lured_ftp_arr - ctrl_ftp_arr).tolist() if n_pairs else []
    ent_deltas = (lured_ent_arr - ctrl_ent_arr).tolist() if n_pairs else []

    # What fraction of lured items have lower FTP than their control?
    frac_lower_ftp = float(np.mean(lured_ftp_arr < ctrl_ftp_arr)) if n_pairs else None
    frac_higher_ent = float(np.mean(lured_ent_arr > ctrl_ent_arr)) if n_pairs else None

    return {
        "description": (
            "De Neys critical test: lured conflict items vs matched controls. "
            "Tests whether the model shows reduced confidence even on items "
            "where it gives the wrong (lure) answer — conflict detection without resolution."
        ),
        "n_lured_pairs": n_pairs,
        "lured_ftp_mean": _safe_float(np.mean(lured_ftp_arr)) if n_pairs else None,
        "lured_ftp_std": _safe_float(np.std(lured_ftp_arr, ddof=1)) if n_pairs > 1 else None,
        "control_ftp_mean": _safe_float(np.mean(ctrl_ftp_arr)) if n_pairs else None,
        "control_ftp_std": _safe_float(np.std(ctrl_ftp_arr, ddof=1)) if n_pairs > 1 else None,
        "frac_lured_lower_ftp": _safe_float(frac_lower_ftp),
        "frac_lured_higher_entropy": _safe_float(frac_higher_ent),
        "wilcoxon_ftp": {
            "W": _safe_float(w_ftp),
            "p": _safe_float(p_ftp),
            "r_rb": _safe_float(r_rb_ftp),
            "alternative": "less (lured < control)",
        },
        "wilcoxon_entropy": {
            "W": _safe_float(w_ent),
            "p": _safe_float(p_ent),
            "r_rb": _safe_float(r_rb_ent),
            "alternative": "greater (lured > control)",
        },
        "mann_whitney_ftp": mw_ftp,
        "mann_whitney_entropy": mw_ent,
        "per_pair_ftp_delta": ftp_deltas,
        "per_pair_entropy_delta": ent_deltas,
        "per_pair_ids": pair_ids,
        "lured_ftp_values": lured_ftp,
        "ctrl_ftp_values": ctrl_ftp,
    }


def confidence_gap_analysis(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze confidence_gap (lure_token_prob - correct_token_prob) for conflict items."""
    conflict = [r for r in results if r["conflict"] and r.get("confidence_gap") is not None]
    gaps = np.array([r["confidence_gap"] for r in conflict], dtype=np.float64)

    # Split by verdict
    lured_gaps = np.array(
        [r["confidence_gap"] for r in conflict if r["verdict"] == "lure"], dtype=np.float64
    )
    correct_gaps = np.array(
        [r["confidence_gap"] for r in conflict if r["verdict"] == "correct"], dtype=np.float64
    )

    return {
        "description": "confidence_gap = lure_token_prob - correct_token_prob for conflict items",
        "all_conflict": {
            "n": int(len(gaps)),
            "mean": _safe_float(np.mean(gaps)),
            "median": _safe_float(np.median(gaps)),
            "std": _safe_float(np.std(gaps, ddof=1)) if len(gaps) > 1 else None,
            "frac_positive": _safe_float(np.mean(gaps > 0)),
        },
        "lured_items": {
            "n": int(len(lured_gaps)),
            "mean": _safe_float(np.mean(lured_gaps)) if len(lured_gaps) else None,
            "median": _safe_float(np.median(lured_gaps)) if len(lured_gaps) else None,
            "std": _safe_float(np.std(lured_gaps, ddof=1)) if len(lured_gaps) > 1 else None,
        },
        "correct_items": {
            "n": int(len(correct_gaps)),
            "mean": _safe_float(np.mean(correct_gaps)) if len(correct_gaps) else None,
            "median": _safe_float(np.median(correct_gaps)) if len(correct_gaps) else None,
            "std": _safe_float(np.std(correct_gaps, ddof=1)) if len(correct_gaps) > 1 else None,
        },
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def make_figure(
    results: list[dict[str, Any]],
    overall_stats: dict[str, Any],
    fig_path: Path,
) -> None:
    """Two-panel bar chart: first_token_prob and top10_entropy for conflict vs control."""
    set_paper_theme()

    conflict, control = partition_items(results)
    conf_ftp = extract_metric(conflict, "first_token_prob")
    ctrl_ftp = extract_metric(control, "first_token_prob")
    conf_ent = extract_metric(conflict, "top10_entropy")
    ctrl_ent = extract_metric(control, "top10_entropy")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # --- Left panel: first_token_prob ---
    ax = axes[0]
    means = [np.mean(conf_ftp), np.mean(ctrl_ftp)]
    sems = [np.std(conf_ftp, ddof=1) / np.sqrt(len(conf_ftp)),
            np.std(ctrl_ftp, ddof=1) / np.sqrt(len(ctrl_ftp))]
    bars = ax.bar(
        [0, 1], means, yerr=sems,
        color=[COLORS["conflict"], COLORS["control"]],
        capsize=5, width=0.6, edgecolor="black", linewidth=0.5,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Conflict", "Control"])
    ax.set_ylabel("First token probability")
    ax.set_title("Response confidence")
    ax.set_ylim(0, 1.05)

    # Significance annotation
    p_ftp = overall_stats["first_token_prob"]["p"]
    stars = significance_stars(p_ftp)
    if stars != "n.s.":
        y_max = max(m + s for m, s in zip(means, sems)) + 0.04
        ax.plot([0, 0, 1, 1], [y_max - 0.01, y_max, y_max, y_max - 0.01], "k-", linewidth=0.8)
        ax.text(0.5, y_max + 0.01, stars, ha="center", fontsize=11)
    else:
        y_max = max(m + s for m, s in zip(means, sems)) + 0.04
        ax.text(0.5, y_max, "n.s.", ha="center", fontsize=9, color="gray")

    # --- Right panel: top10_entropy ---
    ax = axes[1]
    means = [np.mean(conf_ent), np.mean(ctrl_ent)]
    sems = [np.std(conf_ent, ddof=1) / np.sqrt(len(conf_ent)),
            np.std(ctrl_ent, ddof=1) / np.sqrt(len(ctrl_ent))]
    bars = ax.bar(
        [0, 1], means, yerr=sems,
        color=[COLORS["conflict"], COLORS["control"]],
        capsize=5, width=0.6, edgecolor="black", linewidth=0.5,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Conflict", "Control"])
    ax.set_ylabel("Top-10 token entropy (bits)")
    ax.set_title("Response uncertainty")

    # Significance annotation
    p_ent = overall_stats["top10_entropy"]["p"]
    stars = significance_stars(p_ent)
    if stars != "n.s.":
        y_max = max(m + s for m, s in zip(means, sems)) + 0.06
        ax.plot([0, 0, 1, 1], [y_max - 0.02, y_max, y_max, y_max - 0.02], "k-", linewidth=0.8)
        ax.text(0.5, y_max + 0.02, stars, ha="center", fontsize=11)
    else:
        y_max = max(m + s for m, s in zip(means, sems)) + 0.06
        ax.text(0.5, y_max, "n.s.", ha="center", fontsize=9, color="gray")
    ax.set_ylim(0, y_max + 0.12)

    fig.suptitle("Llama-3.1-8B-Instruct: Confidence under conflict", fontsize=13, y=1.02)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, format="pdf")
    fig.savefig(fig_path.with_suffix(".png"), format="png")
    print(f"  Saved {fig_path} and .png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_markdown(
    overall: dict[str, Any],
    per_cat: dict[str, dict],
    de_neys: dict[str, Any],
    gap: dict[str, Any],
    out_path: Path,
) -> None:
    """Write narrative findings to markdown."""
    lines: list[str] = []
    lines.append("# Confidence Paradigm Analysis (De Neys)")
    lines.append("")
    lines.append("Model: Llama-3.1-8B-Instruct")
    lines.append("")
    lines.append("R1-Distill excluded from confidence tests: `<think>` token drives ")
    lines.append("first_token_prob to 1.0 and entropy to 0.0 uniformly (no variance).")
    lines.append("")

    # Overall
    lines.append("## Overall conflict vs control")
    lines.append("")
    ftp = overall["first_token_prob"]
    ent = overall["top10_entropy"]
    lines.append(
        f"- **First token probability**: conflict {ftp['conflict_mean']:.3f} "
        f"(SD {ftp['conflict_std']:.3f}) vs control {ftp['control_mean']:.3f} "
        f"(SD {ftp['control_std']:.3f})"
    )
    lines.append(
        f"  - Mann-Whitney U = {ftp['U']:.1f}, p = {ftp['p']:.2e}, "
        f"r_rb = {ftp['r_rb']:.3f} {significance_stars(ftp['p'])}"
    )
    lines.append(
        f"- **Top-10 entropy**: conflict {ent['conflict_mean']:.3f} "
        f"(SD {ent['conflict_std']:.3f}) vs control {ent['control_mean']:.3f} "
        f"(SD {ent['control_std']:.3f})"
    )
    lines.append(
        f"  - Mann-Whitney U = {ent['U']:.1f}, p = {ent['p']:.2e}, "
        f"r_rb = {ent['r_rb']:.3f} {significance_stars(ent['p'])}"
    )
    lines.append("")

    # Interpretation
    if ftp["p"] is not None and ftp["p"] < 0.05:
        direction = "lower" if ftp["conflict_mean"] < ftp["control_mean"] else "higher"
        lines.append(
            f"Conflict items elicit significantly {direction} first-token confidence "
            f"than matched controls, consistent with the De Neys prediction that "
            f"conflicting heuristic cues reduce output confidence."
        )
    else:
        lines.append("No significant difference in first-token confidence between conditions.")
    lines.append("")

    # Per category
    lines.append("## Per-category breakdown (first_token_prob)")
    lines.append("")
    lines.append("| Category | Conflict M (SD) | Control M (SD) | U | p | r_rb | Sig |")
    lines.append("|----------|----------------|---------------|---|---|------|-----|")
    for cat, s in sorted(per_cat["first_token_prob"].items()):
        cm = f"{s['conflict_mean']:.3f} ({s['conflict_std']:.3f})" if s["conflict_mean"] is not None else "—"
        ctm = f"{s['control_mean']:.3f} ({s['control_std']:.3f})" if s["control_mean"] is not None else "—"
        u_str = f"{s['U']:.0f}" if s["U"] is not None else "—"
        p_str = f"{s['p']:.3e}" if s["p"] is not None else "—"
        r_str = f"{s['r_rb']:.3f}" if s["r_rb"] is not None else "—"
        sig = significance_stars(s["p"])
        lines.append(f"| {cat} | {cm} | {ctm} | {u_str} | {p_str} | {r_str} | {sig} |")
    lines.append("")

    # De Neys critical test
    lines.append("## De Neys critical test: lured items vs matched controls")
    lines.append("")
    lines.append(
        f"N lured-control pairs: {de_neys['n_lured_pairs']}"
    )
    lines.append(
        f"- Lured FTP: {de_neys['lured_ftp_mean']:.3f} (SD {de_neys['lured_ftp_std']:.3f})"
    )
    lines.append(
        f"- Control FTP: {de_neys['control_ftp_mean']:.3f} (SD {de_neys['control_ftp_std']:.3f})"
    )
    lines.append(
        f"- Fraction where lured < control: {de_neys['frac_lured_lower_ftp']:.1%}"
    )
    wil = de_neys["wilcoxon_ftp"]
    if wil["p"] is not None:
        lines.append(
            f"- Wilcoxon signed-rank (one-sided, lured < control): "
            f"W = {wil['W']:.1f}, p = {wil['p']:.2e}, r_rb = {wil['r_rb']:.3f} "
            f"{significance_stars(wil['p'])}"
        )
    lines.append("")
    if wil["p"] is not None and wil["p"] < 0.05:
        lines.append(
            "**Result**: Lured items show significantly lower first-token confidence "
            "than their matched controls. This is the hallmark De Neys finding: "
            "conflict detection without resolution. The model's initial probability "
            "mass is disrupted by the conflict even on items where it ultimately "
            "gives the heuristic (wrong) answer."
        )
    else:
        lines.append(
            "**Result**: No significant confidence reduction for lured items. "
            "The model does not show the De Neys conflict-detection-without-resolution "
            "signature."
        )
    lines.append("")

    # Entropy for lured items
    lines.append("### Entropy for lured items")
    lines.append(
        f"- Fraction where lured entropy > control: "
        f"{de_neys['frac_lured_higher_entropy']:.1%}"
    )
    wil_ent = de_neys["wilcoxon_entropy"]
    if wil_ent["p"] is not None:
        lines.append(
            f"- Wilcoxon signed-rank (one-sided, lured > control): "
            f"W = {wil_ent['W']:.1f}, p = {wil_ent['p']:.2e}, r_rb = {wil_ent['r_rb']:.3f} "
            f"{significance_stars(wil_ent['p'])}"
        )
    lines.append("")

    # Confidence gap
    lines.append("## Confidence gap distribution")
    lines.append("")
    lines.append(
        f"Across all {gap['all_conflict']['n']} conflict items: "
        f"mean gap = {gap['all_conflict']['mean']:.4f}, "
        f"median = {gap['all_conflict']['median']:.4f}, "
        f"fraction positive (lure > correct) = {gap['all_conflict']['frac_positive']:.1%}"
    )
    if gap["lured_items"]["n"]:
        lines.append(
            f"- Lured items ({gap['lured_items']['n']}): "
            f"mean gap = {gap['lured_items']['mean']:.4f}, "
            f"median = {gap['lured_items']['median']:.4f}"
        )
    if gap["correct_items"]["n"]:
        lines.append(
            f"- Correct items ({gap['correct_items']['n']}): "
            f"mean gap = {gap['correct_items']['mean']:.4f}, "
            f"median = {gap['correct_items']['median']:.4f}"
        )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze De Neys confidence paradigm results")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("results_pod/confidence"),
        help="Directory with confidence JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/confidence"),
        help="Where to write JSON results",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("figures"),
        help="Where to write figure PDF",
    )
    parser.add_argument(
        "--doc-dir",
        type=Path,
        default=Path("docs"),
        help="Where to write markdown report",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("De Neys Confidence Paradigm Analysis")
    print("=" * 60)

    # Load data
    llama_data = load_confidence(args.data_dir / "llama_confidence.json")
    r1_data = load_confidence(args.data_dir / "r1_distill_confidence.json")
    results = llama_data["results"]

    print(f"\n  Llama items: {len(results)}")
    conflict, control = partition_items(results)
    print(f"    Conflict: {len(conflict)}, Control: {len(control)}")
    lured = [r for r in conflict if r["verdict"] == "lure"]
    print(f"    Lured: {len(lured)}, Correct: {sum(1 for r in conflict if r['verdict'] == 'correct')}")

    # R1 sanity check
    r1_ftp = extract_metric(r1_data["results"], "first_token_prob")
    print(f"\n  R1-Distill first_token_prob: mean={np.mean(r1_ftp):.4f}, std={np.std(r1_ftp):.4f}")
    print("    (Confirms <think> token saturates confidence — excluded from tests)")

    # --- 1. Overall Mann-Whitney ---
    print("\n--- Overall conflict vs control ---")
    overall: dict[str, Any] = {}
    for metric in ("first_token_prob", "top10_entropy"):
        conf_vals = extract_metric(conflict, metric)
        ctrl_vals = extract_metric(control, metric)
        mw = mann_whitney_with_effect(conf_vals, ctrl_vals, alternative="two-sided")
        entry = {
            "conflict_mean": _safe_float(np.mean(conf_vals)),
            "conflict_std": _safe_float(np.std(conf_vals, ddof=1)),
            "control_mean": _safe_float(np.mean(ctrl_vals)),
            "control_std": _safe_float(np.std(ctrl_vals, ddof=1)),
            "n_conflict": int(len(conf_vals)),
            "n_control": int(len(ctrl_vals)),
            **mw,
        }
        overall[metric] = entry
        print(
            f"  {metric}: conflict={entry['conflict_mean']:.4f} vs control={entry['control_mean']:.4f} "
            f"| U={entry['U']:.0f}, p={entry['p']:.2e}, r_rb={entry['r_rb']:.3f} "
            f"{significance_stars(entry['p'])}"
        )

    # --- 2. Per-category ---
    print("\n--- Per-category (first_token_prob) ---")
    per_cat = {
        "first_token_prob": per_category_stats(results, "first_token_prob"),
        "top10_entropy": per_category_stats(results, "top10_entropy"),
    }
    for cat, s in sorted(per_cat["first_token_prob"].items()):
        p_str = f"{s['p']:.3e}" if s["p"] is not None else "N/A"
        r_str = f"{s['r_rb']:.3f}" if s["r_rb"] is not None else "N/A"
        print(
            f"  {cat:20s}: conf={s['conflict_mean']:.3f} ctrl={s['control_mean']:.3f} "
            f"p={p_str} r_rb={r_str} {significance_stars(s['p'])}"
        )

    # --- 3. De Neys critical test ---
    print("\n--- De Neys critical test (lured vs matched control) ---")
    de_neys = de_neys_critical_test(results)
    print(f"  N lured pairs: {de_neys['n_lured_pairs']}")
    print(
        f"  Lured FTP: {de_neys['lured_ftp_mean']:.4f} vs Control FTP: {de_neys['control_ftp_mean']:.4f}"
    )
    print(f"  Frac lured < control: {de_neys['frac_lured_lower_ftp']:.1%}")
    wil = de_neys["wilcoxon_ftp"]
    if wil["p"] is not None:
        print(
            f"  Wilcoxon: W={wil['W']:.1f}, p={wil['p']:.2e}, r_rb={wil['r_rb']:.3f} "
            f"{significance_stars(wil['p'])}"
        )

    # --- 4. Confidence gap ---
    print("\n--- Confidence gap distribution ---")
    gap = confidence_gap_analysis(results)
    print(
        f"  All conflict: mean={gap['all_conflict']['mean']:.4f}, "
        f"frac_positive={gap['all_conflict']['frac_positive']:.1%}"
    )

    # --- Save JSON ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "model": llama_data["model"],
        "n_items": llama_data["n_items"],
        "overall_conflict_vs_control": overall,
        "per_category": per_cat,
        "de_neys_critical_test": de_neys,
        "confidence_gap": gap,
        "r1_distill_note": (
            "R1-Distill excluded: <think> token saturates first_token_prob=1.0, "
            "entropy=0.0 uniformly. No variance for statistical tests."
        ),
        "r1_distill_ftp_stats": {
            "mean": _safe_float(np.mean(r1_ftp)),
            "std": _safe_float(np.std(r1_ftp)),
            "min": _safe_float(np.min(r1_ftp)),
            "max": _safe_float(np.max(r1_ftp)),
        },
    }
    json_path = args.output_dir / "confidence_analysis.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved {json_path}")

    # --- Figure ---
    fig_path = args.fig_dir / "fig7_confidence.pdf"
    make_figure(results, overall, fig_path)

    # --- Markdown ---
    md_path = args.doc_dir / "confidence_analysis.md"
    write_markdown(overall, per_cat, de_neys, gap, md_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
