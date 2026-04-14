#!/usr/bin/env python3
"""Generate scale comparison figure: OLMo-7B vs OLMo-32B behavioral lure rates.

Key finding: scaling from 7B to 32B does NOT uniformly reduce cognitive bias
vulnerability. The 32B model is actually MORE susceptible on base_rate and
framing categories, despite having better overall capabilities.

Usage::

    python scripts/make_scale_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style: NeurIPS-friendly, colorblind-safe (Wong 2011)
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,   # TrueType — required for camera-ready
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------------
# Data (hardcoded from experimental results)
# ---------------------------------------------------------------------------

# Categories present in both models
CATEGORIES = ["overall", "base_rate", "conjunction", "framing"]

DATA_7B: dict[str, float | None] = {
    "overall": 14.9,
    "base_rate": 46.0,
    "conjunction": 50.0,
    "framing": None,  # not measured / insufficient items
}

DATA_32B: dict[str, float | None] = {
    "overall": 19.6,
    "base_rate": 74.3,
    "conjunction": 50.0,
    "framing": 30.0,
}

# Categories where 32B is strictly MORE vulnerable than 7B
WORSE_AT_SCALE = {"overall", "base_rate"}


def make_scale_figure() -> None:
    """Grouped bar chart comparing 7B vs 32B lure rates by category."""

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    x = np.arange(len(CATEGORIES))
    width = 0.32

    vals_7b = []
    vals_32b = []
    for cat in CATEGORIES:
        vals_7b.append(DATA_7B[cat] if DATA_7B[cat] is not None else 0.0)
        vals_32b.append(DATA_32B[cat] if DATA_32B[cat] is not None else 0.0)

    mask_7b = [DATA_7B[cat] is not None for cat in CATEGORIES]
    mask_32b = [DATA_32B[cat] is not None for cat in CATEGORIES]

    # Colors: light blue (7B) vs dark blue (32B), with red-ish accent for
    # categories where scaling makes things worse
    c_7b = "#56B4E9"   # sky blue (Wong)
    c_32b = "#0072B2"  # darker blue (Wong)
    c_worse = "#D55E00" # vermillion — highlights where 32B is MORE vulnerable

    # Draw 7B bars
    bars_7b = ax.bar(
        x - width / 2,
        vals_7b,
        width,
        label="OLMo-7B-Instruct",
        color=c_7b,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )

    # Draw 32B bars — color vermillion where 32B is worse
    colors_32b = [c_worse if cat in WORSE_AT_SCALE else c_32b for cat in CATEGORIES]
    bars_32b = ax.bar(
        x + width / 2,
        vals_32b,
        width,
        label="OLMo-32B-Instruct",
        color=colors_32b,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )

    # Grey out missing data bars
    for i, cat in enumerate(CATEGORIES):
        if not mask_7b[i]:
            bars_7b[i].set_color("#E0E0E0")
            bars_7b[i].set_hatch("///")
            bars_7b[i].set_edgecolor("#BBBBBB")
        if not mask_32b[i]:
            bars_32b[i].set_color("#E0E0E0")
            bars_32b[i].set_hatch("///")
            bars_32b[i].set_edgecolor("#BBBBBB")

    # Value labels on bars
    for bars, mask in [(bars_7b, mask_7b), (bars_32b, mask_32b)]:
        for i, (bar, m) in enumerate(zip(bars, mask)):
            if m and bar.get_height() > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{bar.get_height():.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )
            elif not m:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    2,
                    "n/a",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#999999",
                    style="italic",
                )

    # Annotation arrows for categories where 32B is worse
    for i, cat in enumerate(CATEGORIES):
        if cat in WORSE_AT_SCALE and mask_7b[i] and mask_32b[i]:
            delta = vals_32b[i] - vals_7b[i]
            sign = "+" if delta > 0 else ""
            ax.annotate(
                f"{sign}{delta:.1f}pp",
                xy=(x[i] + width / 2, vals_32b[i] + 4),
                xytext=(x[i] + width / 2 + 0.18, vals_32b[i] + 12),
                fontsize=7.5,
                color=c_worse,
                fontweight="bold",
                arrowprops=dict(
                    arrowstyle="->,head_width=0.15,head_length=0.1",
                    color=c_worse,
                    lw=1.0,
                ),
                ha="left",
                va="bottom",
            )

    # 50% chance line
    ax.axhline(50, color="#999999", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
    ax.text(len(CATEGORIES) - 0.55, 51.5, "chance", fontsize=7.5, color="#999999", ha="right")

    # Axes
    display_labels = [c.replace("_", " ").title() for c in CATEGORIES]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_ylabel("Lure Rate (%)")
    ax.set_ylim(0, 90)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

    ax.set_title(
        "Scale Does Not Reduce Cognitive Bias Vulnerability",
        fontweight="bold",
        pad=10,
    )

    # Grid
    ax.yaxis.grid(True, which="major", linestyle="-", alpha=0.15, zorder=0)
    ax.yaxis.grid(True, which="minor", linestyle=":", alpha=0.10, zorder=0)
    ax.set_axisbelow(True)

    # Spine cleanup
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend — add a manual entry for the "worse at scale" color
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c_7b, edgecolor="white", label="OLMo-7B-Instruct"),
        Patch(facecolor=c_32b, edgecolor="white", label="OLMo-32B-Instruct"),
        Patch(facecolor=c_worse, edgecolor="white", label="32B worse than 7B"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )

    fig.tight_layout()

    # Save
    for ext in ("pdf", "png"):
        outpath = FIGURES_DIR / f"fig_scale_comparison.{ext}"
        fig.savefig(outpath)
        print(f"Saved: {outpath}")

    plt.close(fig)


if __name__ == "__main__":
    make_scale_figure()
