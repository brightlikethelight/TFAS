"""Plots for the metacognitive monitoring workstream.

Three figures are generated:

1. **Surprise-feature correlation distribution** — histogram of
   per-feature Spearman rho values, with the ``rho_threshold`` line
   and the count of features above it. Tells you at a glance whether
   Gate 1 will pass.

2. **Specificity scatter** — every difficulty-sensitive feature on
   ``(rho, AUC)`` axes. The Gate 2 region (top-right corner) is
   shaded. Falsified or constant features are drawn in gray.

3. **Gate decision panel** — a small table-like figure listing each
   gate, its criterion, observed metric, and color-coded decision.
   Useful as a single-image summary in the project README.

All plots are headless-safe (matplotlib ``Agg`` backend) and emit a
:class:`matplotlib.figure.Figure` the caller can save or display. We
follow the same conventions as :mod:`s1s2.sae.volcano` so notebooks
can mix-and-match the two viz modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless safe; notebooks can override later

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from beartype import beartype
from matplotlib.figure import Figure

from s1s2.metacog.gates import GateResult
from s1s2.utils.logging import get_logger

logger = get_logger("s1s2.metacog")


# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------


_COLOR_PASS = "#2e7d32"
_COLOR_MARGINAL = "#ed6c02"
_COLOR_FAIL = "#c62828"
_COLOR_NEUTRAL = "#9e9e9e"
_DECISION_TO_COLOR = {
    "go": _COLOR_PASS,
    "marginal": _COLOR_MARGINAL,
    "no_go": _COLOR_FAIL,
}


# ---------------------------------------------------------------------------
# 1. Surprise rho distribution
# ---------------------------------------------------------------------------


@beartype
def plot_rho_distribution(
    df: pd.DataFrame,
    *,
    rho_threshold: float = 0.3,
    title: str = "Surprise-feature correlation",
    out_path: str | Path | None = None,
    figsize: tuple[float, float] = (7.0, 5.0),
    dpi: int = 150,
) -> Figure:
    """Histogram of per-feature Spearman rho values.

    Expects ``df`` with a ``rho`` column. Falls back gracefully on
    empty input.
    """
    if "rho" not in df.columns:
        raise KeyError("plot_rho_distribution requires a 'rho' column")

    rho = df["rho"].to_numpy(dtype=np.float64)
    rho = rho[np.isfinite(rho)]
    if rho.size == 0:
        rho = np.array([0.0])

    n_above = int((rho > rho_threshold).sum())

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.hist(rho, bins=60, color="#1f77b4", alpha=0.8, edgecolor="white")
    ax.axvline(
        rho_threshold,
        color=_COLOR_PASS,
        linestyle="--",
        linewidth=2,
        label=f"rho > {rho_threshold} (n={n_above})",
    )
    ax.axvline(-rho_threshold, color=_COLOR_NEUTRAL, linestyle=":", linewidth=1)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Spearman rho (feature activation vs token surprise)")
    ax.set_ylabel("Number of features")
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        logger.info("wrote %s", out_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Specificity scatter
# ---------------------------------------------------------------------------


@beartype
def plot_specificity_scatter(
    combined_df: pd.DataFrame,
    *,
    rho_threshold: float = 0.3,
    auc_threshold: float = 0.65,
    title: str = "S1/S2 specificity vs difficulty correlation",
    out_path: str | Path | None = None,
    figsize: tuple[float, float] = (7.0, 6.0),
    dpi: int = 150,
    annotate_top_k: int = 5,
) -> Figure:
    """Scatter plot: x = surprise rho, y = matched-pair AUC.

    Expected columns: ``feature_id``, ``rho``, ``auc``,
    ``passes_specificity``, ``is_difficulty_sensitive``. The
    ``combined_df`` produced by
    :func:`s1s2.metacog.difficulty.difficulty_sensitive_features`
    has all of these.
    """
    required = {"feature_id", "rho", "auc"}
    missing = required - set(combined_df.columns)
    if missing:
        raise KeyError(f"plot_specificity_scatter requires columns {required}; missing {missing}")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if combined_df.empty:
        ax.text(
            0.5,
            0.5,
            "No difficulty-sensitive candidates",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            color=_COLOR_NEUTRAL,
        )
        ax.set_xlim(-0.5, 1.0)
        ax.set_ylim(0.4, 1.0)
    else:
        x = combined_df["rho"].to_numpy(dtype=np.float64)
        y = combined_df["auc"].to_numpy(dtype=np.float64)
        passes = (
            combined_df["passes_specificity"].to_numpy(dtype=bool)
            if "passes_specificity" in combined_df.columns
            else np.zeros_like(x, dtype=bool)
        )
        is_meta = (
            combined_df["is_metacognitive"].to_numpy(dtype=bool)
            if "is_metacognitive" in combined_df.columns
            else np.zeros_like(x, dtype=bool)
        )

        # Quadrant shading: Gate 2 region.
        ax.axhspan(
            auc_threshold, 1.0, xmin=(rho_threshold + 0.5) / 1.5, color=_COLOR_PASS, alpha=0.05
        )

        ax.scatter(
            x[~passes],
            y[~passes],
            s=40,
            color=_COLOR_NEUTRAL,
            alpha=0.5,
            label="below specificity",
        )
        ax.scatter(
            x[passes & ~is_meta],
            y[passes & ~is_meta],
            s=70,
            color=_COLOR_MARGINAL,
            edgecolor="black",
            linewidth=0.5,
            label="passes specificity",
        )
        ax.scatter(
            x[passes & is_meta],
            y[passes & is_meta],
            s=110,
            color=_COLOR_PASS,
            edgecolor="black",
            linewidth=0.7,
            label="passes spec + metacog",
            marker="*",
        )

        # Annotate top-K by combined score
        if annotate_top_k > 0:
            for _, row in combined_df.head(annotate_top_k).iterrows():
                ax.annotate(
                    f"#{int(row['feature_id'])}",
                    (float(row["rho"]), float(row["auc"])),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

    ax.axvline(rho_threshold, color=_COLOR_PASS, linestyle="--", linewidth=1)
    ax.axhline(auc_threshold, color=_COLOR_PASS, linestyle="--", linewidth=1)
    ax.set_xlabel("Spearman rho (surprise correlation)")
    ax.set_ylabel("Matched-pair AUC (S1 vs S2)")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        logger.info("wrote %s", out_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Gate decision panel
# ---------------------------------------------------------------------------


@beartype
def plot_gate_panel(
    gate_results: Sequence[GateResult],
    *,
    title: str = "Metacog gate decisions",
    out_path: str | Path | None = None,
    figsize: tuple[float, float] = (8.5, 4.0),
    dpi: int = 150,
) -> Figure:
    """Render a small table-like figure summarizing the 4 gates.

    Each row shows ``Gate i — name``, the criterion, the observed
    metric, and a color-coded decision pill. Designed to be the single
    image you can drop into a lab notebook or paper appendix to
    communicate the headline result.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis("off")
    ax.set_title(title, fontsize=12, loc="left", pad=20)

    n = len(gate_results)
    if n == 0:
        ax.text(0.5, 0.5, "No gate results", ha="center", va="center")
        return fig

    rows = list(reversed(gate_results))  # so Gate 0 is at the bottom
    for i, g in enumerate(rows):
        y = (i + 0.5) / n
        color = _DECISION_TO_COLOR.get(g.decision, _COLOR_NEUTRAL)
        ax.add_patch(
            plt.Rectangle(
                (0.02, y - 0.4 / n),
                0.96,
                0.7 / n,
                facecolor=color,
                alpha=0.10,
                transform=ax.transAxes,
            )
        )
        # Decision pill
        ax.text(
            0.06,
            y,
            f"G{g.gate_id}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color=color,
        )
        ax.text(
            0.13,
            y + 0.02,
            g.name.replace("_", " "),
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
        )
        # Decision text
        ax.text(
            0.92,
            y,
            g.decision.upper().replace("_", " "),
            ha="right",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color=color,
            fontweight="bold",
        )
        # Rationale (smaller)
        ax.text(
            0.13,
            y - 0.04,
            g.rationale,
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            color="#444",
        )

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        logger.info("wrote %s", out_path)
    return fig


# ---------------------------------------------------------------------------
# Convenience: render the full set
# ---------------------------------------------------------------------------


@beartype
def render_all(
    *,
    surprise_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    gate_results: Sequence[GateResult],
    out_dir: str | Path,
    name_prefix: str,
    rho_threshold: float = 0.3,
    auc_threshold: float = 0.65,
) -> dict[str, Path]:
    """Write the three headline figures and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    p1 = out_dir / f"{name_prefix}_rho_dist.png"
    plot_rho_distribution(
        surprise_df,
        rho_threshold=rho_threshold,
        out_path=p1,
        title=f"{name_prefix}: surprise correlation distribution",
    )
    paths["rho_distribution"] = p1

    p2 = out_dir / f"{name_prefix}_specificity_scatter.png"
    plot_specificity_scatter(
        combined_df,
        rho_threshold=rho_threshold,
        auc_threshold=auc_threshold,
        out_path=p2,
        title=f"{name_prefix}: specificity vs surprise",
    )
    paths["specificity_scatter"] = p2

    p3 = out_dir / f"{name_prefix}_gates.png"
    plot_gate_panel(
        list(gate_results),
        out_path=p3,
        title=f"{name_prefix}: gate decisions",
    )
    paths["gate_panel"] = p3

    return paths


__all__ = [
    "plot_gate_panel",
    "plot_rho_distribution",
    "plot_specificity_scatter",
    "render_all",
]
