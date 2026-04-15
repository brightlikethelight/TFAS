#!/usr/bin/env python3
"""Dual-panel steering figure: lure rate AND correct rate for both models.

2x2 layout:
  Top-left:     Llama lure rate vs alpha
  Top-right:    Llama correct rate vs alpha
  Bottom-left:  R1 lure rate vs alpha
  Bottom-right: R1 correct rate vs alpha

Shows that positive-alpha steering doesn't just suppress lures -- it redirects
probability mass toward the correct answer, not garbage.

Usage::

    python scripts/make_steering_correctness_figure.py
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
LLAMA_PATH = PROJECT_ROOT / "results" / "causal" / "probe_steering_llama_l14.json"
R1_PATH = PROJECT_ROOT / "results" / "causal" / "probe_steering_r1_l14.json"
FIG_DIR = PROJECT_ROOT / "figures"

# ---------------------------------------------------------------------------
# Style — matches project theme from make_steering_figure.py
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


# Colorblind-friendly palette
COLOR_LURE = "#D55E00"    # vermillion for lure rate
COLOR_CORRECT = "#009E73"  # bluish green for correct rate
COLOR_RANDOM = "#999999"   # gray for random controls
COLOR_BASELINE = "#555555" # dark gray for baseline


def load_steering_data(path: Path) -> dict:
    """Load a steering JSON and extract sorted arrays."""
    with open(path) as f:
        data = json.load(f)

    alphas_data = data["alphas"]
    random_data = data["random_controls"]
    alpha_vals = sorted(float(a) for a in alphas_data.keys())

    result: dict = {"alpha_vals": alpha_vals, "model": data["model"]}

    # Probe direction rates
    for metric in ("lure_rate", "correct_rate", "other_rate"):
        result[f"probe_{metric}"] = np.array(
            [alphas_data[str(a)][metric] * 100 for a in alpha_vals]
        )

    # Random controls for lure rate
    result["rand_lure_mean"] = np.array(
        [random_data[str(a)]["mean_lure_rate"] * 100 for a in alpha_vals]
    )
    result["rand_lure_std"] = np.array(
        [random_data[str(a)]["std"] * 100 for a in alpha_vals]
    )

    # Random controls for correct rate
    result["rand_correct_mean"] = np.array(
        [random_data[str(a)]["mean_correct_rate"] * 100 for a in alpha_vals]
    )
    result["rand_correct_std"] = np.array(
        [random_data[str(a)]["std_correct_rate"] * 100 for a in alpha_vals]
    )

    return result


def plot_panel(
    ax: plt.Axes,
    alpha_vals: list[float],
    probe_vals: np.ndarray,
    rand_mean: np.ndarray,
    rand_std: np.ndarray,
    baseline_val: float,
    color: str,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] = (0, 100),
    show_directional: bool = True,
) -> None:
    """Plot a single steering panel with probe line, random band, baseline."""
    # Random control band
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
        baseline_val,
        color=COLOR_BASELINE,
        linewidth=0.8,
        linestyle=":",
        zorder=2,
    )
    ax.text(
        4.9,
        baseline_val + (ylim[1] - ylim[0]) * 0.03,
        f"baseline ({baseline_val:.1f}%)",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=COLOR_BASELINE,
    )

    # Probe direction line
    ax.plot(
        alpha_vals,
        probe_vals,
        color=color,
        marker="o",
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label="Probe direction",
        zorder=3,
    )

    ax.set_xlabel("Steering strength ($\\alpha$)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(*ylim)
    ax.set_xticks([-5, -3, -1, 0, 1, 3, 5])
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9, edgecolor="none", fontsize=6)

    if show_directional:
        ax.annotate(
            "$\\leftarrow$ S1",
            xy=(-5, ylim[0] + (ylim[1] - ylim[0]) * 0.02),
            fontsize=6.5,
            color="#666666",
            ha="left",
            va="bottom",
        )
        ax.annotate(
            "S2 $\\rightarrow$",
            xy=(5, ylim[0] + (ylim[1] - ylim[0]) * 0.02),
            fontsize=6.5,
            color="#666666",
            ha="right",
            va="bottom",
        )


def print_summary(llama: dict, r1: dict) -> None:
    """Print summary table for alpha=+5 diagnostics."""
    alpha_vals = llama["alpha_vals"]
    idx_5 = alpha_vals.index(5.0)
    idx_0 = alpha_vals.index(0.0)

    print("\n" + "=" * 72)
    print("SUMMARY TABLE: Steering at alpha = +5 vs baseline (alpha = 0)")
    print("=" * 72)

    for name, d in [("Llama-3.1-8B-Instruct", llama), ("R1-Distill-Llama-8B", r1)]:
        lure_0 = d["probe_lure_rate"][idx_0]
        correct_0 = d["probe_correct_rate"][idx_0]
        other_0 = d["probe_other_rate"][idx_0]
        lure_5 = d["probe_lure_rate"][idx_5]
        correct_5 = d["probe_correct_rate"][idx_5]
        other_5 = d["probe_other_rate"][idx_5]

        print(f"\n  {name}")
        print(f"  {'Metric':<16} {'Baseline (a=0)':>14} {'Steered (a=+5)':>14} {'Delta':>10}")
        print(f"  {'-' * 56}")
        print(f"  {'Lure rate':<16} {lure_0:>13.1f}% {lure_5:>13.1f}% {lure_5 - lure_0:>+9.1f} pp")
        print(f"  {'Correct rate':<16} {correct_0:>13.1f}% {correct_5:>13.1f}% {correct_5 - correct_0:>+9.1f} pp")
        print(f"  {'Other rate':<16} {other_0:>13.1f}% {other_5:>13.1f}% {other_5 - other_0:>+9.1f} pp")

        # Key diagnostic: does correct_rate > baseline?
        if correct_5 > correct_0:
            print(f"  --> Correct rate INCREASED by {correct_5 - correct_0:.1f} pp (mass goes to correct, not garbage)")
        else:
            print(f"  --> WARNING: Correct rate did not increase ({correct_5 - correct_0:+.1f} pp)")

    print()


def main() -> None:
    set_paper_theme()

    llama = load_steering_data(LLAMA_PATH)
    r1 = load_steering_data(R1_PATH)

    # -- Build 2x2 figure ---------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    # Row 0: Llama
    alpha_vals = llama["alpha_vals"]
    llama_baseline_lure = llama["probe_lure_rate"][alpha_vals.index(0.0)]
    llama_baseline_correct = llama["probe_correct_rate"][alpha_vals.index(0.0)]

    plot_panel(
        axes[0, 0], alpha_vals,
        llama["probe_lure_rate"],
        llama["rand_lure_mean"], llama["rand_lure_std"],
        llama_baseline_lure,
        COLOR_LURE,
        "Lure rate (%)",
        "Llama-3.1-8B: Lure Rate",
    )

    plot_panel(
        axes[0, 1], alpha_vals,
        llama["probe_correct_rate"],
        llama["rand_correct_mean"], llama["rand_correct_std"],
        llama_baseline_correct,
        COLOR_CORRECT,
        "Correct rate (%)",
        "Llama-3.1-8B: Correct Rate",
    )

    # Row 1: R1
    alpha_vals_r1 = r1["alpha_vals"]
    r1_baseline_lure = r1["probe_lure_rate"][alpha_vals_r1.index(0.0)]
    r1_baseline_correct = r1["probe_correct_rate"][alpha_vals_r1.index(0.0)]

    plot_panel(
        axes[1, 0], alpha_vals_r1,
        r1["probe_lure_rate"],
        r1["rand_lure_mean"], r1["rand_lure_std"],
        r1_baseline_lure,
        COLOR_LURE,
        "Lure rate (%)",
        "R1-Distill-Llama-8B: Lure Rate",
        ylim=(0, 30),
    )

    plot_panel(
        axes[1, 1], alpha_vals_r1,
        r1["probe_correct_rate"],
        r1["rand_correct_mean"], r1["rand_correct_std"],
        r1_baseline_correct,
        COLOR_CORRECT,
        "Correct rate (%)",
        "R1-Distill-Llama-8B: Correct Rate",
        ylim=(40, 80),
    )

    fig.suptitle(
        "Probe-Direction Steering: Lure Rate vs Correct Rate",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # -- Save ---------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / "fig_steering_correctness.pdf"
    png_path = FIG_DIR / "fig_steering_correctness.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    # -- Summary table ------------------------------------------------------
    print_summary(llama, r1)


if __name__ == "__main__":
    main()
