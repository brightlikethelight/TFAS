"""Visualization helpers for the geometry workstream.

All plot functions follow the same contract:

- Accept plain numpy arrays and primitive dicts, not LayerResult objects,
  so they can be invoked from notebooks without re-running the analysis.
- Return a ``matplotlib.figure.Figure`` object. Saving and closing is
  the caller's job — this keeps the functions unit-testable.
- Use matplotlib's default style; no seaborn, no custom RC. Keeps the
  generated PDFs small and the code reproducible.

THE PRIMARY FIGURE in this workstream is
:func:`plot_silhouette_curves` — layer-wise cosine silhouette with
bootstrap CI bands and a permutation-null shadow line. Everything else
supports that.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from jaxtyping import Float, Int

__all__ = [
    "plot_cka_heatmap",
    "plot_cross_model_cka",
    "plot_intrinsic_dim_curves",
    "plot_pca_projection",
    "plot_random_projection_panel",
    "plot_silhouette_curves",
]


# ---------------------------------------------------------------------------
# Silhouette curves — THE flagship figure
# ---------------------------------------------------------------------------


@beartype
def plot_silhouette_curves(
    curves: Mapping[str, Mapping[str, Float[np.ndarray, "k"]]],
    null_line: Float[np.ndarray, "k"] | None = None,
    title: str = "Cosine silhouette by layer",
    ylabel: str = "Cosine silhouette",
) -> mpl_figure.Figure:
    """Layer-wise silhouette curves, one line per model.

    ``curves`` maps ``model_key -> {"layers": np.array(layers),
    "silhouette": np.array, "ci_lower": np.array, "ci_upper": np.array}``.
    The CI bands are drawn as shaded regions. ``null_line`` (optional) is
    drawn as a dashed gray line, typically the 95th-percentile of the
    permutation null — any model curve rising above it is "significant
    clustering".
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, data in curves.items():
        layers = np.asarray(data["layers"])
        sil = np.asarray(data["silhouette"])
        lo = np.asarray(data["ci_lower"])
        hi = np.asarray(data["ci_upper"])
        line, = ax.plot(layers, sil, marker="o", linewidth=2, label=model)
        ax.fill_between(layers, lo, hi, alpha=0.2, color=line.get_color())
    if null_line is not None:
        # Align to the union of layer indices we saw.
        any_layers = np.asarray(next(iter(curves.values()))["layers"])
        if null_line.shape[0] == any_layers.shape[0]:
            ax.plot(
                any_layers,
                null_line,
                linestyle="--",
                color="gray",
                label="null (p=0.05)",
            )
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2D scatter plots
# ---------------------------------------------------------------------------


@beartype
def plot_pca_projection(
    projection: Float[np.ndarray, "n 2"],
    labels: Int[np.ndarray, "n"],
    correctness: np.ndarray | None = None,
    explained_variance: Float[np.ndarray, "2"] | None = None,
    title: str = "PCA projection",
) -> mpl_figure.Figure:
    """2D scatter colored by label, shaped by correctness.

    ``labels`` are class IDs (e.g. 0=S1, 1=S2). ``correctness`` is an
    optional bool/int array; True markers use ``o``, False markers use
    ``x``. The axis labels include the explained variance ratio per PC
    if provided.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    classes = np.unique(labels)
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(classes):
        mask = labels == c
        if correctness is None:
            ax.scatter(
                projection[mask, 0],
                projection[mask, 1],
                s=30,
                color=cmap(i % 10),
                label=f"class {c}",
                alpha=0.7,
                edgecolors="none",
            )
        else:
            correct_mask = mask & correctness.astype(bool)
            wrong_mask = mask & (~correctness.astype(bool))
            ax.scatter(
                projection[correct_mask, 0],
                projection[correct_mask, 1],
                s=30,
                color=cmap(i % 10),
                marker="o",
                label=f"class {c} (correct)",
                alpha=0.7,
                edgecolors="none",
            )
            ax.scatter(
                projection[wrong_mask, 0],
                projection[wrong_mask, 1],
                s=40,
                color=cmap(i % 10),
                marker="x",
                label=f"class {c} (wrong)",
                alpha=0.9,
            )
    if explained_variance is not None and explained_variance.shape[0] >= 2:
        ax.set_xlabel(f"PC1 ({explained_variance[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained_variance[1]*100:.1f}%)")
    else:
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@beartype
def plot_random_projection_panel(
    pca: Float[np.ndarray, "n 2"],
    umap: Float[np.ndarray, "n 2"] | None,
    random_projs: Sequence[np.ndarray],
    labels: Int[np.ndarray, "n"],
    title: str = "Projection panel",
) -> mpl_figure.Figure:
    """4-panel figure: PCA / UMAP / rand-1 / rand-2.

    If ``random_projs`` has only one entry we still plot, stretching the
    single random projection across the remaining panel. If UMAP is
    ``None`` (e.g. unavailable) the panel is labeled "UMAP unavailable"
    and left blank.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    cmap = plt.get_cmap("tab10")
    classes = np.unique(labels)

    def _scatter(ax, proj: np.ndarray, subtitle: str) -> None:
        for i, c in enumerate(classes):
            mask = labels == c
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                s=20,
                color=cmap(i % 10),
                alpha=0.7,
                edgecolors="none",
                label=f"{c}",
            )
        ax.set_title(subtitle, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    _scatter(axes[0], pca, "PCA")
    if umap is not None:
        _scatter(axes[1], umap, "UMAP")
    else:
        axes[1].set_title("UMAP unavailable", fontsize=10)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    # Two random projections side-by-side.
    rand_iter = list(random_projs)[:2]
    for i, proj in enumerate(rand_iter):
        _scatter(axes[2 + i], np.asarray(proj), f"random #{i+1}")
    for i in range(len(rand_iter), 2):
        axes[2 + i].set_title("random (none)", fontsize=10)
        axes[2 + i].set_xticks([])
        axes[2 + i].set_yticks([])

    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CKA heatmaps and curves
# ---------------------------------------------------------------------------


@beartype
def plot_cka_heatmap(
    cka_matrix: Float[np.ndarray, "k k"],
    layer_labels: Sequence[int] | None = None,
    title: str = "Layer-by-layer CKA",
) -> mpl_figure.Figure:
    """Heatmap of a (layers x layers) CKA matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cka_matrix, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
    if layer_labels is not None:
        tick = list(range(len(layer_labels)))
        ax.set_xticks(tick)
        ax.set_yticks(tick)
        ax.set_xticklabels([str(lbl) for lbl in layer_labels], fontsize=8, rotation=90)
        ax.set_yticklabels([str(lbl) for lbl in layer_labels], fontsize=8)
    ax.set_xlabel("layer (model B)")
    ax.set_ylabel("layer (model A)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="CKA")
    fig.tight_layout()
    return fig


@beartype
def plot_cross_model_cka(
    layers: Int[np.ndarray, "k"],
    full: Float[np.ndarray, "k"],
    s1_only: Float[np.ndarray, "k"] | None = None,
    s2_only: Float[np.ndarray, "k"] | None = None,
    model_a: str = "model A",
    model_b: str = "model B",
) -> mpl_figure.Figure:
    """Line plot of layer-matched CKA with optional S1-only / S2-only lines."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, full, marker="o", linewidth=2, label="all problems")
    if s1_only is not None:
        ax.plot(layers, s1_only, marker="s", linewidth=2, label="S1 subset")
    if s2_only is not None:
        ax.plot(layers, s2_only, marker="^", linewidth=2, label="S2 subset")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Linear CKA")
    ax.set_title(f"Layer-matched CKA: {model_a} vs {model_b}")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="black", linewidth=0.5)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Intrinsic-dim curves
# ---------------------------------------------------------------------------


@beartype
def plot_intrinsic_dim_curves(
    curves: Mapping[str, Mapping[str, np.ndarray]],
    metric_key: str = "two_nn",
    title: str = "Intrinsic dimensionality by layer",
) -> mpl_figure.Figure:
    """Per-model ID curves: each model is one line across layers.

    ``curves`` maps ``model_key -> {"layers": np.array, "two_nn":
    np.array, "pr": np.array}``. ``metric_key`` selects which column to
    plot (typically ``'two_nn'`` or ``'pr'``).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, data in curves.items():
        layers = np.asarray(data["layers"])
        y = np.asarray(data[metric_key])
        ax.plot(layers, y, marker="o", linewidth=2, label=model)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(metric_key.replace("_", "-").upper())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig
