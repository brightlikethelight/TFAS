#!/usr/bin/env python3
"""Generate dose-response figure for probe-direction steering results.

Plots lure rate vs steering strength (alpha) for probe direction and random
controls, showing the causal effect of the learned S1/S2 direction on
heuristic responding.

Usage::

    python scripts/make_steering_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "results" / "causal" / "probe_steering_llama_l14.json"
FIG_DIR = PROJECT_ROOT / "figures"

# ---------------------------------------------------------------------------
# Style — matches project theme from make_paper_figures.py
# ---------------------------------------------------------------------------


def set_paper_theme() -> None:
    """Publication-grade rcParams: serif, no top/right spines, 300 DPI."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


# Colorblind-friendly palette (Okabe-Ito inspired)
COLOR_PROBE = "#0072B2"  # blue
COLOR_RANDOM = "#999999"  # gray
COLOR_BASELINE = "#555555"  # dark gray


def main() -> None:
    set_paper_theme()

    # -- Load data ----------------------------------------------------------
    with open(DATA_PATH) as f:
        data = json.load(f)

    alphas_data = data["alphas"]
    random_data = data["random_controls"]

    # Sort by alpha value
    alpha_vals = sorted(float(a) for a in alphas_data.keys())
    probe_lure = np.array([alphas_data[str(a)]["lure_rate"] * 100 for a in alpha_vals])

    rand_mean = np.array([random_data[str(a)]["mean_lure_rate"] * 100 for a in alpha_vals])
    rand_std = np.array([random_data[str(a)]["std"] * 100 for a in alpha_vals])

    baseline_lure = alphas_data["0.0"]["lure_rate"] * 100  # 52.5%

    # -- Build figure -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # Random control band (mean +/- 1 SD)
    ax.fill_between(
        alpha_vals,
        rand_mean - rand_std,
        rand_mean + rand_std,
        color=COLOR_RANDOM,
        alpha=0.25,
        label="Random direction (mean $\\pm$ 1 SD)",
        zorder=1,
    )
    ax.plot(
        alpha_vals,
        rand_mean,
        color=COLOR_RANDOM,
        linewidth=1.0,
        linestyle="--",
        zorder=2,
    )

    # Baseline dashed line
    ax.axhline(
        baseline_lure,
        color=COLOR_BASELINE,
        linewidth=0.8,
        linestyle=":",
        zorder=2,
    )
    ax.text(
        4.9,
        baseline_lure + 1.5,
        f"baseline ({baseline_lure:.1f}%)",
        ha="right",
        va="bottom",
        fontsize=7,
        color=COLOR_BASELINE,
    )

    # Probe direction line
    ax.plot(
        alpha_vals,
        probe_lure,
        color=COLOR_PROBE,
        marker="o",
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label="Probe direction",
        zorder=3,
    )

    # -- Axes ---------------------------------------------------------------
    ax.set_xlabel("Steering strength ($\\alpha$)")
    ax.set_ylabel("Lure rate (%)")
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(0, 100)
    ax.set_xticks([-5, -3, -1, 0, 1, 3, 5])

    # -- Annotations (directional labels) -----------------------------------
    ax.annotate(
        "$\\leftarrow$ Toward S1 (heuristic)",
        xy=(-5, 2),
        fontsize=7,
        color="#666666",
        ha="left",
        va="bottom",
    )
    ax.annotate(
        "Toward S2 (deliberative) $\\rightarrow$",
        xy=(5, 2),
        fontsize=7,
        color="#666666",
        ha="right",
        va="bottom",
    )

    # -- Title & legend -----------------------------------------------------
    ax.set_title("Probe-Direction Steering: Causal Effect on Lure Rate")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="none")

    fig.tight_layout()

    # -- Save ---------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / "fig_causal_steering.pdf"
    png_path = FIG_DIR / "fig_causal_steering.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    # -- Summary stats for quick verification -------------------------------
    delta = probe_lure[0] - probe_lure[-1]  # alpha=-5 vs alpha=+5
    print(f"\nProbe lure rate: {probe_lure[0]:.1f}% (alpha=-5) -> {probe_lure[-1]:.1f}% (alpha=+5)")
    print(f"Total swing: {delta:.1f} pp")
    print(f"Random control range: {rand_mean.min():.1f}% - {rand_mean.max():.1f}%")


if __name__ == "__main__":
    main()
