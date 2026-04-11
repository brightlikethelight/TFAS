"""Volcano plot rendering for SAE differential feature results.

The volcano plot is the canonical visualization of an FDR-corrected
differential experiment: x = log fold change, y = -log10(q-value).
Significant features pop out in the top-left and top-right; most
features sit in a central "cloud" near the origin.

In this codebase we layer the Ma et al. (2026) falsification outcome
on top of the standard plot: features that pass the FDR cutoff *and*
survive falsification are drawn in bold color, while features that
pass FDR but are flagged as spurious (token-level artifacts) are
drawn in gray as a visual reminder that they don't count.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless safe; scripts that want interactive display reset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from beartype import beartype
from matplotlib.figure import Figure

from s1s2.utils.logging import get_logger

logger = get_logger("s1s2.sae")


@beartype
def _neg_log10_q(qvalues: np.ndarray) -> np.ndarray:
    """Compute ``-log10(q)`` with a floor to avoid ``inf`` at q=0."""
    q_floor = np.clip(qvalues.astype(np.float64), 1e-300, 1.0)
    return -np.log10(q_floor)


@beartype
def plot_volcano(
    df: pd.DataFrame,
    title: str,
    out_path: str | Path,
    *,
    fdr_q: float = 0.05,
    annotate_top_k: int = 10,
    ylim: float | None = None,
    xlim: float | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
    dpi: int = 150,
) -> Figure:
    """Render a volcano plot with falsification overlay.

    Parameters
    ----------
    df
        Dataframe with at least columns ``feature_id``, ``log_fc``,
        ``q_value``, and optionally ``is_falsified``. If
        ``is_falsified`` is absent we assume every significant
        feature is "unfalsified" and draw it in the bold color —
        which will make reviewers suspicious, as it should.
    title
        Plot title. Typically something like
        ``"Llama-3.1-8B-Instruct layer 16 (P0, all pairs)"``.
    out_path
        Where to write the figure. The directory is created if missing.
    fdr_q
        The BH FDR threshold (for the horizontal cutoff line).
    annotate_top_k
        Label the top ``k`` non-falsified significant features by
        absolute log fold change with their feature ID.
    ylim, xlim
        Axis limits. ``None`` = autoscale.

    Returns
    -------
    matplotlib.figure.Figure
    """

    required = {"feature_id", "log_fc", "q_value"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"volcano plot requires columns {required}; missing {missing}")

    # Pull arrays.
    feature_ids = df["feature_id"].to_numpy()
    log_fc = df["log_fc"].to_numpy(dtype=np.float64)
    q = df["q_value"].to_numpy(dtype=np.float64)
    neg_log_q = _neg_log10_q(q)

    if "is_falsified" in df.columns:
        is_falsified = df["is_falsified"].to_numpy(dtype=bool)
    else:
        is_falsified = np.zeros_like(feature_ids, dtype=bool)

    significant = q <= fdr_q
    genuine = significant & (~is_falsified)
    spurious = significant & is_falsified
    nonsig = ~significant

    # Figure.
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # Non-significant
    ax.scatter(
        log_fc[nonsig],
        neg_log_q[nonsig],
        s=8,
        c="#cccccc",
        alpha=0.6,
        label=f"n.s. (q>{fdr_q})",
    )
    # Significant but falsified — visible gray, so they're not hidden.
    if spurious.any():
        ax.scatter(
            log_fc[spurious],
            neg_log_q[spurious],
            s=24,
            c="#888888",
            edgecolors="none",
            alpha=0.8,
            marker="x",
            label="significant but falsified (Ma et al.)",
        )
    # Significant and genuine — the only ones that count.
    if genuine.any():
        ax.scatter(
            log_fc[genuine],
            neg_log_q[genuine],
            s=32,
            c="#c03030",
            edgecolors="#400000",
            linewidths=0.4,
            label=f"significant & unfalsified (n={int(genuine.sum())})",
        )

    # Horizontal threshold line
    thresh = -math.log10(max(fdr_q, 1e-300))
    ax.axhline(thresh, ls="--", color="#555555", lw=0.8)
    ax.axvline(0.0, ls="--", color="#555555", lw=0.8)

    ax.set_xlabel("log2 fold change (S1 / S2)")
    ax.set_ylabel(r"$-\log_{10}(q)$")
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(0.0, ylim)
    if xlim is not None:
        ax.set_xlim(-xlim, xlim)
    ax.legend(loc="upper left", frameon=True, fontsize=8)

    # Annotate top-K genuine by |log_fc|
    if annotate_top_k > 0 and genuine.any():
        order = np.argsort(-np.abs(log_fc * genuine))
        labeled = 0
        for idx in order:
            if not genuine[idx]:
                continue
            ax.annotate(
                str(int(feature_ids[idx])),
                xy=(log_fc[idx], neg_log_q[idx]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color="#400000",
            )
            labeled += 1
            if labeled >= annotate_top_k:
                break

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    logger.info("wrote volcano plot to %s", out_path)
    return fig


__all__ = ["plot_volcano"]
