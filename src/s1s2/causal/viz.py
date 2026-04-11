"""Plotting for the causal workstream.

Two canonical figures:

1. **Dose-response line plot** — P(correct) vs alpha for the conflict
   group, no-conflict group, and random-direction control. This is the
   single most informative figure for a causal intervention: the shape
   tells you whether the feature is real, monotonic, and selective.

2. **Ablation bar chart** — baseline vs ablated P(correct) on conflict
   and no-conflict items, with error bars from the bootstrap.

Figures are produced with matplotlib (no seaborn, to keep the dependency
surface small) and saved as PDF by default. Filenames follow the
workstream convention ``{model}_layer{NN}_feature{FFFFFF}_{kind}.pdf``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
from beartype import beartype

# Non-interactive backend by default so the module imports cleanly on
# headless nodes.
matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt

from s1s2.causal.core import AblationResult, CausalCellResult
from s1s2.causal.dose_response import DoseResponseCurve, DoseResponsePoint
from s1s2.utils.logging import get_logger

logger = get_logger("s1s2.causal")


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


_GROUP_STYLES: dict[str, dict[str, str]] = {
    "conflict": {"color": "#c0392b", "marker": "o", "label": "conflict items"},
    "no_conflict": {"color": "#2980b9", "marker": "s", "label": "no-conflict items"},
    "random_control": {
        "color": "#7f8c8d",
        "marker": "x",
        "label": "random direction (control)",
    },
}


@beartype
def _group_points(
    points: Sequence[DoseResponsePoint],
) -> dict[str, list[DoseResponsePoint]]:
    out: dict[str, list[DoseResponsePoint]] = {}
    for p in points:
        out.setdefault(p.group, []).append(p)
    for g in out:
        out[g].sort(key=lambda pt: pt.alpha)
    return out


# ---------------------------------------------------------------------------
# Dose-response line plot
# ---------------------------------------------------------------------------


@beartype
def plot_dose_response(
    curve: DoseResponseCurve,
    *,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Write a dose-response line plot for one feature to disk.

    Error bars use the bootstrap CI stored on each :class:`DoseResponsePoint`.
    The ``alpha=0`` vertical line is drawn as a visual anchor.
    """
    grouped = _group_points(curve.points)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    for group, style in _GROUP_STYLES.items():
        pts = grouped.get(group, [])
        if not pts:
            continue
        xs = np.array([p.alpha for p in pts], dtype=np.float64)
        ys = np.array([p.p_correct for p in pts], dtype=np.float64)
        lo = np.array([p.ci_lower for p in pts], dtype=np.float64)
        hi = np.array([p.ci_upper for p in pts], dtype=np.float64)
        yerr = np.stack([ys - lo, hi - ys], axis=0)
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            capsize=3,
            linewidth=1.5,
            markersize=5,
        )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.set_xlabel(r"steering coefficient $\alpha$")
    ax.set_ylabel("P(correct)")
    if title is None:
        title = f"{curve.model} L{curve.layer} feature {curve.feature_id} — " f"dose response"
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Ablation bar chart
# ---------------------------------------------------------------------------


@beartype
def plot_ablation_bars(
    result: CausalCellResult,
    *,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Bar chart of baseline vs projection-ablation P(correct)."""
    if result.ablation is None:
        raise ValueError(
            f"cell {result.model}/L{result.layer}/F{result.feature_id} "
            "has no ablation result to plot"
        )
    abl: AblationResult = result.ablation

    groups = ["conflict", "no-conflict"]
    baseline = [abl.baseline_p_correct_conflict, abl.baseline_p_correct_no_conflict]
    ablated = [abl.ablated_p_correct_conflict, abl.ablated_p_correct_no_conflict]

    xs = np.arange(len(groups), dtype=np.float64)
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.bar(xs - width / 2, baseline, width=width, label="baseline", color="#2980b9")
    ax.bar(xs + width / 2, ablated, width=width, label="ablated", color="#c0392b")

    for i, (b, a) in enumerate(zip(baseline, ablated, strict=True)):
        ax.text(xs[i] - width / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(xs[i] + width / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels(groups)
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("P(correct)")
    if title is None:
        title = (
            f"{result.model} L{result.layer} feature {result.feature_id} — " "projection ablation"
        )
    ax.set_title(title, fontsize=10)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Summary bar chart across features
# ---------------------------------------------------------------------------


@beartype
def plot_feature_summary_bars(
    results: list[CausalCellResult],
    *,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Summarise a batch of results with a grouped bar chart.

    Bars show the conflict-item slope and the random-control slope per
    feature. The gap between the two is the canonical "selectivity"
    signature — large positive gap = real S2 feature.
    """
    if not results:
        raise ValueError("no results to plot")

    labels = [f"{r.model[:4]}_L{r.layer}_F{r.feature_id}" for r in results]
    conflict_slopes = np.array(
        [float(r.curve.fit.get("conflict", {}).get("slope", 0.0)) for r in results],
        dtype=np.float64,
    )
    random_slopes = np.array(
        [float(r.curve.fit.get("random_control", {}).get("slope", 0.0)) for r in results],
        dtype=np.float64,
    )

    xs = np.arange(len(labels), dtype=np.float64)
    width = 0.35

    fig_w = max(6.0, 0.55 * len(labels) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.2))
    ax.bar(xs - width / 2, conflict_slopes, width=width, label="conflict", color="#c0392b")
    ax.bar(xs + width / 2, random_slopes, width=width, label="random control", color="#7f8c8d")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"slope of P(correct) vs $\alpha$")
    ax.axhline(0.0, color="black", linewidth=0.6)
    if title is None:
        title = "steering slope per feature (feature vs random control)"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


__all__ = [
    "plot_ablation_bars",
    "plot_dose_response",
    "plot_feature_summary_bars",
]
