"""
Generate side-by-side steering comparison figure for Llama L14 vs R1-Distill L31.

Llama panel: probe-direction steering with 50-random-direction band (mean ± 1 std).
R1 panel: probe-direction steering only (no random controls available for L31).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.use("Agg")  # non-interactive backend
plt.rcParams["pdf.fonttype"] = 42  # TrueType — required for NeurIPS
plt.rcParams["ps.fonttype"] = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/Users/brightliu/School_Work/TFAS/s1s2")
LLAMA_JSON = ROOT / "results/causal/probe_steering_llama_l14.json"
LLAMA_50RAND_JSON = ROOT / "results/causal/probe_steering_llama_l14_50rand.json"
R1_JSON = ROOT / "results/causal/probe_steering_r1_l31.json"
FIG_PDF = ROOT / "figures/fig_steering_llama_r1_comparison.pdf"
FIG_PNG = ROOT / "figures/fig_steering_llama_r1_comparison.png"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_steering_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_alpha_series(
    data: dict,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return (alphas, lure_pct, correct_pct, other_pct) sorted by alpha."""
    alphas_raw = data["alphas"]
    sorted_keys = sorted(alphas_raw.keys(), key=float)
    alphas = [float(k) for k in sorted_keys]
    lure = [alphas_raw[k]["lure_rate"] * 100 for k in sorted_keys]
    correct = [alphas_raw[k]["correct_rate"] * 100 for k in sorted_keys]
    other = [alphas_raw[k]["other_rate"] * 100 for k in sorted_keys]
    return alphas, lure, correct, other


def extract_rand_band(
    rand_data: dict,
) -> tuple[list[float], list[float], list[float]]:
    """Return (alphas, mean_lure_pct, std_lure_pct) from random_controls block."""
    rc = rand_data["random_controls"]
    sorted_keys = sorted(rc.keys(), key=float)
    alphas = [float(k) for k in sorted_keys]
    means = [rc[k]["mean_lure_rate"] * 100 for k in sorted_keys]
    stds = [rc[k]["std"] * 100 for k in sorted_keys]
    return alphas, means, stds


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
llama_data = load_steering_json(LLAMA_JSON)
llama_rand_data = load_steering_json(LLAMA_50RAND_JSON)
r1_data = load_steering_json(R1_JSON)

# Llama probe-direction series (9 alpha values: -5, -3, -1, -0.5, 0, 0.5, 1, 3, 5)
llama_alphas, llama_lure, llama_correct, llama_other = extract_alpha_series(llama_data)

# R1 probe-direction series (7 alpha values: -5, -3, -1, 0, 1, 3, 5)
r1_alphas, r1_lure, r1_correct, r1_other = extract_alpha_series(r1_data)

# Random band from the 50-direction file (7 alpha values matching r1 set)
rand_alphas, rand_mean_lure, rand_std_lure = extract_rand_band(llama_rand_data)

# Verify R1 alpha=0 correct rate
r1_alpha0_correct = r1_data["alphas"]["0.0"]["correct_rate"]
assert abs(r1_alpha0_correct - 0.8875) < 1e-4, (
    f"Unexpected R1 alpha=0 correct_rate: {r1_alpha0_correct}"
)

# Compute Llama lure swing: max lure - min lure across probe direction
llama_lure_swing = max(llama_lure) - min(llama_lure)  # should be ~37.5pp
r1_lure_swing = max(r1_lure) - min(r1_lure)

print(f"Llama lure swing: {llama_lure_swing:.1f} pp")
print(f"R1 lure swing:    {r1_lure_swing:.2f} pp")


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-paper")
FONT_AXES = 10
FONT_ANNOT = 9
FONT_TICK = 9
FONT_LEGEND = 9

COLOR_LURE = "#d62728"       # red
COLOR_CORRECT = "#1f77b4"    # blue
COLOR_OTHER = "#7f7f7f"      # gray
COLOR_RAND_BAND = "#b0b0b0"  # light gray fill

MARKER_LURE = "o"
MARKER_CORRECT = "s"
MARKER_OTHER = "^"

LW = 1.6   # line width
MS = 5.5   # marker size


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax_llama, ax_r1) = plt.subplots(
    1, 2,
    figsize=(9, 3.5),
    sharey=False,
    constrained_layout=False,
)
fig.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.90, wspace=0.30)

# ---------------------------------------------------------------------------
# Left panel — Llama L14
# ---------------------------------------------------------------------------
ax = ax_llama

# Random-direction band (mean ± 1 std of lure_rate across 50 random dirs)
rand_lo = [m - s for m, s in zip(rand_mean_lure, rand_std_lure)]
rand_hi = [m + s for m, s in zip(rand_mean_lure, rand_std_lure)]
ax.fill_between(
    rand_alphas, rand_lo, rand_hi,
    color=COLOR_RAND_BAND, alpha=0.45, zorder=1,
    label="Random dirs. (mean ± 1 SD, n=50)",
)

# Probe-direction lines
ax.plot(
    llama_alphas, llama_lure,
    color=COLOR_LURE, linestyle="-", marker=MARKER_LURE,
    linewidth=LW, markersize=MS, zorder=3, label="Lure rate",
)
ax.plot(
    llama_alphas, llama_correct,
    color=COLOR_CORRECT, linestyle="--", marker=MARKER_CORRECT,
    linewidth=LW, markersize=MS, zorder=3, label="Correct rate",
)
if any(v > 0 for v in llama_other):
    ax.plot(
        llama_alphas, llama_other,
        color=COLOR_OTHER, linestyle=":", marker=MARKER_OTHER,
        linewidth=LW, markersize=MS, zorder=3, label="Other rate",
    )

# Annotation: 37.5 pp lure swing, double-headed arrow
alpha_min_idx = int(np.argmin(llama_lure))
alpha_max_idx = int(np.argmax(llama_lure))
x_arrow = llama_alphas[alpha_min_idx] + 0.25  # slightly offset from +5
y_lo = llama_lure[alpha_min_idx]   # min lure (~31.25%)
y_hi = llama_lure[alpha_max_idx]   # max lure (~68.75%)
ax.annotate(
    "",
    xy=(x_arrow, y_hi), xytext=(x_arrow, y_lo),
    arrowprops=dict(
        arrowstyle="<->",
        color="black",
        lw=1.2,
        mutation_scale=12,
    ),
    zorder=5,
)
ax.text(
    x_arrow + 0.3, (y_lo + y_hi) / 2,
    f"37.5 pp",
    fontsize=FONT_ANNOT, va="center", ha="left", color="black",
)

ax.set_xlim(-5.6, 5.6)
ax.set_ylim(0, 105)
ax.set_xticks([-5, -3, -1, 0, 1, 3, 5])
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=FONT_TICK)
ax.set_xticklabels([str(a) for a in [-5, -3, -1, 0, 1, 3, 5]], fontsize=FONT_TICK)
ax.set_xlabel("Steering alpha (α)", fontsize=FONT_AXES)
ax.set_ylabel("Response rate", fontsize=FONT_AXES)
ax.set_title("Llama-3.1-8B-Instruct (probe direction, L14)", fontsize=FONT_AXES, pad=4)

# Light gridlines
ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Right panel — R1-Distill L31
# ---------------------------------------------------------------------------
ax = ax_r1

ax.plot(
    r1_alphas, r1_lure,
    color=COLOR_LURE, linestyle="-", marker=MARKER_LURE,
    linewidth=LW, markersize=MS, zorder=3, label="Lure rate",
)
ax.plot(
    r1_alphas, r1_correct,
    color=COLOR_CORRECT, linestyle="--", marker=MARKER_CORRECT,
    linewidth=LW, markersize=MS, zorder=3, label="Correct rate",
)
if any(v > 0 for v in r1_other):
    ax.plot(
        r1_alphas, r1_other,
        color=COLOR_OTHER, linestyle=":", marker=MARKER_OTHER,
        linewidth=LW, markersize=MS, zorder=3, label="Other rate",
    )

# Annotation: tiny range for R1
r1_y_mid = (max(r1_lure) + min(r1_lure)) / 2
ax.annotate(
    f"{r1_lure_swing:.2f} pp range\n(vs. Llama {llama_lure_swing:.1f} pp)",
    xy=(0, r1_y_mid),
    xytext=(1.8, r1_y_mid + 35),
    fontsize=FONT_ANNOT,
    ha="left",
    va="bottom",
    color="black",
    arrowprops=dict(
        arrowstyle="->",
        color="black",
        lw=0.9,
        connectionstyle="arc3,rad=0.25",
    ),
    zorder=5,
)

ax.set_xlim(-5.6, 5.6)
ax.set_ylim(0, 105)
ax.set_xticks([-5, -3, -1, 0, 1, 3, 5])
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=FONT_TICK)
ax.set_xticklabels([str(a) for a in [-5, -3, -1, 0, 1, 3, 5]], fontsize=FONT_TICK)
ax.set_xlabel("Steering alpha (α)", fontsize=FONT_AXES)
ax.set_ylabel("Response rate", fontsize=FONT_AXES)
ax.set_title("R1-Distill-Llama-8B (probe direction, L31)", fontsize=FONT_AXES, pad=4)

ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Shared legend below both panels
# ---------------------------------------------------------------------------
# Build handles manually so legend is shared across panels
legend_handles = [
    plt.Line2D([0], [0], color=COLOR_LURE, linestyle="-", marker=MARKER_LURE,
               linewidth=LW, markersize=MS, label="Lure rate"),
    plt.Line2D([0], [0], color=COLOR_CORRECT, linestyle="--", marker=MARKER_CORRECT,
               linewidth=LW, markersize=MS, label="Correct rate"),
    plt.Line2D([0], [0], color=COLOR_OTHER, linestyle=":", marker=MARKER_OTHER,
               linewidth=LW, markersize=MS, label="Other rate"),
    mpatches.Patch(color=COLOR_RAND_BAND, alpha=0.6,
                   label="Random dirs. (mean ± 1 SD, n=50) [Llama only]"),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    fontsize=FONT_LEGEND,
    frameon=True,
    framealpha=0.9,
    edgecolor="0.8",
    bbox_to_anchor=(0.5, -0.01),
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
FIG_PDF.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(FIG_PDF, bbox_inches="tight", backend="pdf")
print(f"Saved PDF: {FIG_PDF}")

fig.savefig(FIG_PNG, bbox_inches="tight", dpi=300)
print(f"Saved PNG: {FIG_PNG}")

plt.close(fig)
print("Done.")
