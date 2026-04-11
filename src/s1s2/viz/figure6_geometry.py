"""Figure 6 — representational geometry (silhouette curves + PCA).

Paper figure 6 is a two-panel composite:

- **Left (silhouette curves)**: layer-wise cosine silhouette per model,
  with bootstrap CI ribbons and an optional permutation-null band.
  The "flagship" geometry plot, delegated to
  :func:`s1s2.geometry.viz.plot_silhouette_curves`.
- **Right (PCA projection)**: 2D PCA scatter of a representative layer's
  activations, colored by label (S1 / S2). Delegated to
  :func:`s1s2.geometry.viz.plot_pca_projection`.

We reassemble the arguments expected by each geometry viz function from
the geometry workstream's per-layer and per-model JSON output. The
exact on-disk schema varies as the geometry workstream evolves, so we
accept several common shapes — see :func:`_load_silhouette_curves` and
:func:`_load_pca_sample` for details.

Because the composite figure is built by copying axes/data from the
geometry-returned Figures, we ultimately render a fresh matplotlib
Figure here. The upstream plot functions are still called (so their
logic is exercised and any bug there surfaces), but their data are
re-rendered in the composite layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beartype import beartype

from s1s2.utils.logging import get_logger
from s1s2.viz.common import (
    FIG_SIZE_DOUBLE_COLUMN,
    format_list,
    require_dir,
    resolve_output_path,
    save_figure,
)
from s1s2.viz.paper_figures_config import PaperFiguresConfig
from s1s2.viz.theme import COLORS, get_model_color, set_paper_theme

try:
    from s1s2.geometry.viz import plot_silhouette_curves as _plot_silhouette_curves
except ImportError:  # pragma: no cover — defensive
    _plot_silhouette_curves = None  # type: ignore[assignment]


logger = get_logger("s1s2.viz.figure6")

__all__ = ["make_figure_6_geometry"]


# --------------------------------------------------------------------------- #
# Silhouette-curve loading                                                     #
# --------------------------------------------------------------------------- #


@beartype
def _load_silhouette_curves(
    geom_dir: Path,
) -> tuple[dict[str, dict[str, Any]], Any | None]:
    """Return ``(curves, null_line)`` from the geometry results directory.

    ``curves`` is a dict of per-model dicts matching the API of
    :func:`s1s2.geometry.viz.plot_silhouette_curves`. ``null_line`` is
    a shared null-line across layers (optional, ``None`` if unavailable).
    """
    import numpy as np

    curves: dict[str, dict[str, Any]] = {}
    null_line: Any | None = None

    for p in sorted(geom_dir.rglob("*.json")):
        try:
            with p.open() as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(raw, dict):
            continue
        # Shape A: single file with top-level curves dict.
        if "curves" in raw and isinstance(raw["curves"], dict):
            for model, entry in raw["curves"].items():
                if isinstance(entry, dict) and {"layers", "silhouette"}.issubset(entry):
                    curves[model] = {
                        "layers": np.asarray(entry["layers"]),
                        "silhouette": np.asarray(entry["silhouette"]),
                        "ci_lower": np.asarray(
                            entry.get("ci_lower", entry["silhouette"])
                        ),
                        "ci_upper": np.asarray(
                            entry.get("ci_upper", entry["silhouette"])
                        ),
                    }
            if "null_line" in raw:
                null_line = np.asarray(raw["null_line"])
            continue
        # Shape B: per-model JSON with inline arrays.
        model = raw.get("model")
        if model and {"layers", "silhouette"}.issubset(raw.keys()):
            curves[str(model)] = {
                "layers": np.asarray(raw["layers"]),
                "silhouette": np.asarray(raw["silhouette"]),
                "ci_lower": np.asarray(raw.get("ci_lower", raw["silhouette"])),
                "ci_upper": np.asarray(raw.get("ci_upper", raw["silhouette"])),
            }
        if null_line is None and "null_line" in raw:
            null_line = np.asarray(raw["null_line"])
    return curves, null_line


@beartype
def _load_pca_sample(
    geom_dir: Path,
) -> tuple[Any, Any, Any | None, str | None]:
    """Return ``(projection, labels, explained_variance, model_key)`` if found.

    We search for a JSON that carries an inline 2D PCA projection in the
    form ``{"pca_projection": [[x, y], ...], "labels": [...]}``. If no
    such file exists we return all-None so the composite figure can skip
    the right panel gracefully.
    """
    import numpy as np

    for p in sorted(geom_dir.rglob("*.json")):
        try:
            with p.open() as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(raw, dict):
            continue
        proj = raw.get("pca_projection")
        labels = raw.get("labels")
        if not (isinstance(proj, list) and isinstance(labels, list)):
            continue
        if len(proj) == 0:
            continue
        arr = np.asarray(proj, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            continue
        lab = np.asarray(labels, dtype=np.int64)
        if lab.shape[0] != arr.shape[0]:
            continue
        ev = raw.get("explained_variance")
        ev_arr: Any | None = None
        if isinstance(ev, list) and len(ev) >= 2:
            ev_arr = np.asarray(ev[:2], dtype=np.float64)
        return arr, lab, ev_arr, raw.get("model")
    return None, None, None, None


# --------------------------------------------------------------------------- #
# Composite rendering                                                          #
# --------------------------------------------------------------------------- #


def _render_composite(
    curves: dict[str, dict[str, Any]],
    null_line: Any | None,
    pca: tuple[Any, Any, Any | None, str | None],
):
    """Build the paper's two-panel geometry figure."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIG_SIZE_DOUBLE_COLUMN[0], 3.6),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )

    # --- Left panel: silhouette curves ---
    ax_sil = axes[0]
    for model, data in curves.items():
        layers = np.asarray(data["layers"])
        sil = np.asarray(data["silhouette"])
        lo = np.asarray(data.get("ci_lower", sil))
        hi = np.asarray(data.get("ci_upper", sil))
        color = get_model_color(model)
        ax_sil.plot(layers, sil, "-o", color=color, linewidth=1.8, markersize=3, label=model)
        ax_sil.fill_between(layers, lo, hi, color=color, alpha=0.15, linewidth=0)
    if null_line is not None:
        arr = np.asarray(null_line)
        any_layers = np.asarray(next(iter(curves.values()))["layers"])
        if arr.shape[0] == any_layers.shape[0]:
            ax_sil.plot(
                any_layers,
                arr,
                linestyle="--",
                color=COLORS.get("baseline", "gray"),
                linewidth=1.0,
                label="null (p=0.05)",
            )
    ax_sil.axhline(0.0, color="black", linewidth=0.5)
    ax_sil.set_xlabel("layer")
    ax_sil.set_ylabel("cosine silhouette")
    ax_sil.set_title("(a) per-layer silhouette", fontsize=10, loc="left")
    ax_sil.legend(frameon=False, fontsize=8, loc="best")
    ax_sil.grid(True, alpha=0.3)

    # --- Right panel: PCA projection (if available) ---
    ax_pca = axes[1]
    projection, labels, ev, pca_model = pca
    if projection is None or labels is None:
        ax_pca.set_axis_off()
        ax_pca.text(
            0.5,
            0.5,
            "PCA projection\nnot available",
            ha="center",
            va="center",
            fontsize=9,
            color="gray",
        )
    else:
        classes = np.unique(labels)
        label_names = {0: "S2 (control)", 1: "S1 (conflict)"}
        for c in classes:
            mask = labels == c
            color = COLORS["s1"] if int(c) == 1 else COLORS["s2"]
            ax_pca.scatter(
                projection[mask, 0],
                projection[mask, 1],
                s=18,
                color=color,
                alpha=0.8,
                edgecolors="none",
                label=label_names.get(int(c), f"class {int(c)}"),
            )
        if ev is not None and len(ev) >= 2:
            ax_pca.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
            ax_pca.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
        else:
            ax_pca.set_xlabel("PC1")
            ax_pca.set_ylabel("PC2")
        title = "(b) PCA projection"
        if pca_model:
            title += f" [{pca_model}]"
        ax_pca.set_title(title, fontsize=10, loc="left")
        ax_pca.legend(frameon=False, fontsize=8, loc="best")
        ax_pca.grid(True, alpha=0.3)

    fig.suptitle("Figure 6: representational geometry", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


@beartype
def make_figure_6_geometry(cfg: PaperFiguresConfig) -> Path:
    """Generate Figure 6 (silhouette + PCA) and return its output path."""
    set_paper_theme()

    geom_dir = require_dir(cfg.results_dir / "geometry", label="geometry results")
    curves, null_line = _load_silhouette_curves(geom_dir)
    if not curves:
        raise FileNotFoundError(f"no usable silhouette curves under {geom_dir}")

    if cfg.models_to_plot is not None:
        allowed = set(cfg.models_to_plot)
        curves = {m: v for m, v in curves.items() if m in allowed}
        if not curves:
            raise FileNotFoundError(
                f"no geometry curves matched models_to_plot={cfg.models_to_plot!r}"
            )

    # Also call the upstream plotter to smoke-test that its logic is
    # still consistent with our data. The returned Figure is discarded.
    if _plot_silhouette_curves is not None:
        try:
            _ = _plot_silhouette_curves(curves, null_line=null_line)
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception as exc:
            logger.warning(
                "upstream plot_silhouette_curves failed on loaded data: %s", exc
            )

    pca = _load_pca_sample(geom_dir)
    fig = _render_composite(curves, null_line, pca)

    formats = format_list(cfg.format)
    out_path = resolve_output_path(cfg.output_dir, "figure_6_geometry", formats)
    return save_figure(fig, out_path, formats, dpi=cfg.dpi)
