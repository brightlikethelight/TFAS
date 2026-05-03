#!/usr/bin/env python3
"""Generate within-CoT probe trajectory figure for NeurIPS paper.

Shows how probe AUC evolves across positions within the thinking trace,
demonstrating the "deliberation dip" phenomenon: probes lose access to
the S1/S2 distinction mid-reasoning, then recover before the final answer.

Data: Qwen 3-8B THINK, Layer 34.

Usage::

    python scripts/make_within_cot_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style — colorblind-friendly (Wong 2011), 8pt fonts for NeurIPS
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    "pdf.fonttype": 42,  # TrueType — required for NeurIPS
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
})

C_LINE = "#0072B2"          # blue (Wong palette)
C_CHANCE = "#999999"        # gray
C_REFERENCE = "#D55E00"     # vermillion
C_DIP_FILL = "#56B4E9"     # light blue (Wong palette)
C_ANNOTATION = "#333333"    # near-black for text

# ---------------------------------------------------------------------------
# Data — Qwen 3-8B THINK, Layer 34
# ---------------------------------------------------------------------------

POSITIONS = ["P0", "T0", "T25", "T50", "T75", "Tend"]
AUC_VALUES = [0.938, 0.973, 0.791, 0.772, 0.754, 0.971]

# Indices for the deliberation dip shading (T25 through T75)
DIP_START_IDX = 2   # T25
DIP_END_IDX = 4     # T75

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure() -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    x = np.arange(len(POSITIONS))

    # Chance line
    ax.axhline(y=0.5, color=C_CHANCE, linestyle="--", linewidth=0.8,
               label="Chance", zorder=1)

    # P0 reference line
    ax.axhline(y=AUC_VALUES[0], color=C_REFERENCE, linestyle="--",
               linewidth=0.8, alpha=0.7, label=f"P0 baseline ({AUC_VALUES[0]:.3f})",
               zorder=1)

    # Deliberation dip shading
    ax.axvspan(DIP_START_IDX - 0.3, DIP_END_IDX + 0.3,
               color=C_DIP_FILL, alpha=0.15, zorder=0,
               label="Deliberation dip")

    # Main trajectory line
    ax.plot(x, AUC_VALUES, color=C_LINE, linewidth=1.8, marker="o",
            markersize=5, markeredgecolor="white", markeredgewidth=0.6,
            zorder=3)

    # Annotations
    offset_peak = (0, 10)
    offset_min = (0, -14)
    offset_rebound = (0, 10)

    ax.annotate("Peak", xy=(1, AUC_VALUES[1]), xytext=offset_peak,
                textcoords="offset points", fontsize=7, fontweight="bold",
                color=C_ANNOTATION, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color=C_ANNOTATION,
                                lw=0.5, shrinkA=0, shrinkB=2))

    ax.annotate("Minimum", xy=(4, AUC_VALUES[4]), xytext=offset_min,
                textcoords="offset points", fontsize=7, fontweight="bold",
                color=C_ANNOTATION, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color=C_ANNOTATION,
                                lw=0.5, shrinkA=0, shrinkB=2))

    ax.annotate("Rebounds", xy=(5, AUC_VALUES[5]), xytext=offset_rebound,
                textcoords="offset points", fontsize=7, fontweight="bold",
                color=C_ANNOTATION, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color=C_ANNOTATION,
                                lw=0.5, shrinkA=0, shrinkB=2))

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(POSITIONS)
    ax.set_xlabel("Position in generation")
    ax.set_ylabel("Probe AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(-0.4, len(POSITIONS) - 0.6)

    ax.set_title(
        "Within-CoT Probe Trajectory: Deliberation Disrupts Then Resolves",
        fontsize=9, fontweight="bold", pad=8,
    )

    # Legend — compact, bottom-right to avoid data overlap
    ax.legend(loc="lower left", frameon=True, framealpha=0.9,
              edgecolor="#cccccc", fancybox=False)

    fig.tight_layout()

    # Save
    pdf_path = FIGURES_DIR / "fig_within_cot_trajectory.pdf"
    png_path = FIGURES_DIR / "fig_within_cot_trajectory.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    make_figure()
