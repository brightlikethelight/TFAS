"""Figure 5 — attention entropy ridge plots (per-layer S1 vs S2).

The attention workstream does not yet ship a ``viz.py`` module; its
result JSONs written by :func:`s1s2.attention.core.save_model_report`
store per-layer summary dicts with keys like ``mean_conflict``,
``mean_noconflict``, ``delta``, ``wilcoxon_p``. We build a compact
double-row ridge figure directly here:

- **Top row**: one small-multiple per model. The x-axis is layer, and
  two lines (conflict / control) show mean per-layer entropy. The gap
  between the lines is the differential signal.
- **Bottom row**: per-layer ``delta = mean_conflict - mean_noconflict``
  overlaid for all models on a single axis. Layers where
  ``wilcoxon_p < 0.05`` are marked with a dot.

We intentionally keep the plot logic inline rather than reaching into
a speculative ``s1s2.attention.viz`` to stay robust to the attention
workstream's API drift. If an attention viz module does land later,
this file is the only place that needs to be retargeted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beartype import beartype

from s1s2.utils.logging import get_logger
from s1s2.viz.common import (
    FIG_SIZE_DOUBLE_ROW,
    format_list,
    require_dir,
    resolve_output_path,
    save_figure,
)
from s1s2.viz.paper_figures_config import PaperFiguresConfig
from s1s2.viz.theme import COLORS, get_model_color, set_paper_theme

logger = get_logger("s1s2.viz.figure5")

__all__ = ["make_figure_5_attention"]


# --------------------------------------------------------------------------- #
# Data extraction                                                              #
# --------------------------------------------------------------------------- #


def _extract_layer_summary(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull a list of per-layer dicts out of an attention result JSON.

    We accept a few common shapes produced by the attention workstream:

    - ``{"layer_summary": [...]}`` — the canonical shape from
      :func:`s1s2.attention.layers.layer_summary`.
    - ``{"layers": [...]}`` — the short form used by older scripts.
    - A top-level list of dicts.
    """
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("layer_summary", "layers", "per_layer"):
        val = payload.get(key)
        if isinstance(val, list) and val and isinstance(val[0], dict):
            return val
    return []


@beartype
def _load_per_model_layer_data(
    attention_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Return ``{model: layer_summary_list}`` from every JSON under ``attention_dir``.

    If a JSON has no ``model`` field we take the file's parent-directory
    name as a fallback identifier (matches the workstream's convention
    of ``attention/{model_key}/*.json``).
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for p in sorted(attention_dir.rglob("*.json")):
        try:
            with p.open() as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        layers = _extract_layer_summary(raw)
        if not layers:
            continue
        # Resolve a model name.
        if isinstance(raw, dict):
            model = str(raw.get("model") or raw.get("model_key") or p.parent.name)
        else:
            model = p.parent.name
        out.setdefault(model, []).extend(layers)
    return out


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #


def _plot_attention_panels(
    per_model: dict[str, list[dict[str, Any]]],
):
    """Render the two-row attention figure and return the Figure."""
    import matplotlib.pyplot as plt
    import numpy as np

    models = sorted(per_model.keys())
    if not models:
        raise ValueError("no attention layer summaries found — nothing to plot")

    n_models = len(models)
    fig, axes = plt.subplots(
        2,
        n_models,
        figsize=(max(FIG_SIZE_DOUBLE_ROW[0], 2.3 * n_models), FIG_SIZE_DOUBLE_ROW[1]),
        squeeze=False,
        gridspec_kw={"height_ratios": [1.3, 1.0]},
    )

    # --- Top row: per-model entropy curves ---
    for col, model in enumerate(models):
        rows = per_model[model]
        rows = sorted(rows, key=lambda r: int(r.get("layer", -1)))
        layers = np.array([int(r.get("layer", 0)) for r in rows])
        conflict = np.array(
            [float(r.get("mean_conflict", float("nan"))) for r in rows]
        )
        control = np.array(
            [float(r.get("mean_noconflict", float("nan"))) for r in rows]
        )
        ax = axes[0, col]
        ax.plot(
            layers,
            conflict,
            "-o",
            color=COLORS["s1"],
            markersize=3,
            linewidth=1.5,
            label="conflict",
        )
        ax.plot(
            layers,
            control,
            "-s",
            color=COLORS["s2"],
            markersize=3,
            linewidth=1.5,
            label="control",
        )
        ax.set_title(model, fontsize=9, color=get_model_color(model))
        ax.set_xlabel("layer")
        if col == 0:
            ax.set_ylabel("mean attention entropy")
        if col == n_models - 1:
            ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- Bottom row: per-layer delta overlay for all models ---
    ax_delta = axes[1, 0]
    # Merge bottom row into a single axis spanning all columns.
    gs = axes[1, 0].get_gridspec()
    for ax in axes[1, :]:
        ax.remove()
    ax_delta = fig.add_subplot(gs[1, :])

    any_sig = False
    for model in models:
        rows = sorted(per_model[model], key=lambda r: int(r.get("layer", -1)))
        layers = np.array([int(r.get("layer", 0)) for r in rows])
        delta = np.array([float(r.get("delta", float("nan"))) for r in rows])
        wp = np.array([float(r.get("wilcoxon_p", 1.0)) for r in rows])
        color = get_model_color(model)
        ax_delta.plot(layers, delta, "-", color=color, linewidth=1.8, label=model)
        mask = (wp < 0.05) & np.isfinite(delta)
        if mask.any():
            any_sig = True
            ax_delta.scatter(
                layers[mask],
                delta[mask],
                s=28,
                facecolor=color,
                edgecolors="black",
                linewidths=0.4,
                zorder=3,
            )
    ax_delta.axhline(0.0, color="black", linewidth=0.6)
    ax_delta.set_xlabel("layer")
    ax_delta.set_ylabel(r"$\Delta$ entropy (conflict - control)")
    legend_extra = "; dots = p<0.05" if any_sig else ""
    ax_delta.set_title(
        f"per-layer differential across models{legend_extra}",
        fontsize=9,
    )
    ax_delta.legend(loc="best", fontsize=8, frameon=True, framealpha=0.9)
    ax_delta.grid(True, alpha=0.3)

    fig.suptitle("Figure 5: attention entropy (S1 vs S2 contrasts)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


@beartype
def make_figure_5_attention(cfg: PaperFiguresConfig) -> Path:
    """Generate Figure 5 (attention entropy) and return its output path."""
    set_paper_theme()

    attn_dir = require_dir(cfg.results_dir / "attention", label="attention results")
    per_model = _load_per_model_layer_data(attn_dir)
    if not per_model:
        raise FileNotFoundError(
            f"no usable attention layer-summary JSONs under {attn_dir}"
        )

    if cfg.models_to_plot is not None:
        allowed = set(cfg.models_to_plot)
        per_model = {m: v for m, v in per_model.items() if m in allowed}
        if not per_model:
            raise FileNotFoundError(
                f"no attention results matched models_to_plot={cfg.models_to_plot!r}"
            )

    fig = _plot_attention_panels(per_model)
    formats = format_list(cfg.format)
    out_path = resolve_output_path(cfg.output_dir, "figure_5_attention", formats)
    return save_figure(fig, out_path, formats, dpi=cfg.dpi)
