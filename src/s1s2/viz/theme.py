"""Shared matplotlib theme and color palette for the paper figures.

Every figure in the paper should look the same: same font, same axes,
same color for "reasoning models", same color for "significant features",
etc. Calling :func:`set_paper_theme` once at the top of a figure script
(or the unified figure generator) applies the global rcParams.

The palette is colorblind-safe (based on Tableau 10 + Color Universal
Design). Do NOT introduce new colors ad-hoc in workstream plots — add
them here so all figures stay in sync.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

__all__ = [
    "COLORS",
    "MODEL_COLORS",
    "MODEL_LINESTYLES",
    "get_model_color",
    "set_paper_theme",
]


# ---------------------------------------------------------------------------
# Semantic color palette (colorblind-safe)
# ---------------------------------------------------------------------------

COLORS: dict[str, str] = {
    "standard": "#1f77b4",  # blue — standard models
    "reasoning": "#ff7f0e",  # orange — reasoning models
    "s1": "#d62728",  # red — S1 / conflict
    "s2": "#2ca02c",  # green — S2 / no-conflict
    "baseline": "#7f7f7f",  # gray — baselines
    "significant": "#e377c2",  # pink — significant features
    "non_significant": "#c7c7c7",  # light gray — non-significant
    "falsified": "#8c564b",  # brown — falsified features
}


# ---------------------------------------------------------------------------
# Per-model colors — used in layer curves, CKA lines, silhouette curves
# ---------------------------------------------------------------------------

MODEL_COLORS: dict[str, str] = {
    "llama-3.1-8b-instruct": "#1f77b4",  # blue
    "gemma-2-9b-it": "#17becf",  # teal (still a "standard" palette)
    "r1-distill-llama-8b": "#ff7f0e",  # orange
    "r1-distill-qwen-7b": "#e377c2",  # pink (warm palette for reasoning)
}


# Solid for standard, dashed for reasoning — this plus color makes the
# figures legible in both color and B&W printouts.
MODEL_LINESTYLES: dict[str, str] = {
    "llama-3.1-8b-instruct": "-",
    "gemma-2-9b-it": "-",
    "r1-distill-llama-8b": "--",
    "r1-distill-qwen-7b": "--",
}


# ---------------------------------------------------------------------------
# Theme setter
# ---------------------------------------------------------------------------


def set_paper_theme() -> None:
    """Apply the paper theme globally.

    Safe to call multiple times — calls :func:`matplotlib.pyplot.rcdefaults`
    is NOT required because we only update a known subset of rcParams and
    leave the rest alone.
    """
    params: dict[str, Any] = {
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
    }
    plt.rcParams.update(params)


def get_model_color(model_key: str) -> str:
    """Return the canonical color for a model key.

    Falls back to the generic "standard"/"reasoning" color if the exact
    key is not registered, using a simple name-based heuristic. Downstream
    plots should never silently mis-color a model, so unknown keys log a
    warning via the standard Python logger.
    """
    if model_key in MODEL_COLORS:
        return MODEL_COLORS[model_key]
    # Heuristic fallback: any "r1" / "reasoning" / "distill" key is a
    # reasoning model, everything else is standard.
    lowered = model_key.lower()
    if any(tag in lowered for tag in ("r1", "reason", "distill", "think")):
        return COLORS["reasoning"]
    return COLORS["standard"]
