"""Figure 4 — causal-intervention bar chart (amplify / ablate / random).

The paper's canonical causal figure is a grouped bar chart showing how
P(correct) changes on conflict items under three interventions:

1. **Amplify**: steering at ``alpha > 0`` with the feature direction.
2. **Ablate**: projection-ablation (i.e. steering at ``alpha = 0``,
   orthogonal component removed).
3. **Random**: steering at ``alpha > 0`` with a random unit vector.

A genuine S2-like feature should produce a meaningful delta on
amplify, a drop on ablate, and near-zero delta on random.

The causal workstream writes per-(model, layer, feature) JSONs via
:func:`s1s2.causal.core.save_cell_result`. Each cell has a dose-response
curve (list of :class:`DoseResponsePoint`) plus an optional
:class:`AblationResult`. This module aggregates across cells and draws
the summary bars — keeping the heavy lifting in
``s1s2.causal.viz.plot_feature_summary_bars`` where applicable, and
falling back to a local bar-chart implementation if that plotter does
not have the exact signature we want.
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
    from s1s2.causal.viz import plot_feature_summary_bars as _plot_summary_bars
except ImportError:  # pragma: no cover — defensive
    _plot_summary_bars = None  # type: ignore[assignment]

logger = get_logger("s1s2.viz.figure4")

__all__ = ["make_figure_4_causal"]


# --------------------------------------------------------------------------- #
# Aggregation from raw JSON cells                                              #
# --------------------------------------------------------------------------- #


def _iter_points(curve_payload: dict[str, Any]) -> list[dict[str, Any]]:
    pts = curve_payload.get("points") or []
    return [p for p in pts if isinstance(p, dict)]


def _delta_on_group(
    points: list[dict[str, Any]],
    group: str,
    *,
    positive: bool = True,
) -> float:
    """Return ``max(p_correct) - p_at_zero`` on the requested group.

    With ``positive=True`` we look at the largest positive alpha's
    P(correct); with ``positive=False`` we look at the smallest negative
    alpha's. The delta is relative to alpha=0 (the baseline steered
    "no-op" — the dose-response pipeline always includes this point).
    """
    xs = [p for p in points if p.get("group") == group]
    if not xs:
        return float("nan")
    zero = next((p for p in xs if abs(float(p.get("alpha", 0.0))) < 1e-9), None)
    p0 = float(zero["p_correct"]) if zero else float("nan")
    if positive:
        signed = [p for p in xs if float(p.get("alpha", 0.0)) > 0.0]
    else:
        signed = [p for p in xs if float(p.get("alpha", 0.0)) < 0.0]
    if not signed:
        return float("nan")
    best = max(signed, key=lambda p: float(p.get("alpha", 0.0)) * (1 if positive else -1))
    return float(best["p_correct"]) - p0


def _ablation_delta(payload: dict[str, Any]) -> float:
    """Return ``baseline - ablated`` on conflict items, or NaN if missing."""
    abl = payload.get("ablation")
    if not isinstance(abl, dict):
        return float("nan")
    base = abl.get("baseline_p_correct_conflict")
    ablt = abl.get("ablated_p_correct_conflict")
    if base is None or ablt is None:
        return float("nan")
    return float(base) - float(ablt)


@beartype
def _load_cell_jsons(causal_dir: Path) -> list[dict[str, Any]]:
    """Load every ``*.json`` under ``causal_dir`` as a dict.

    Files are skipped (not errored) on parse failure. We do not require
    any particular filename convention here — the causal workstream's
    ``save_cell_result`` uses ``{model}_layer{NN}_feature{FFFFFF}.json``
    but we stay agnostic.
    """
    out: list[dict[str, Any]] = []
    for p in sorted(causal_dir.rglob("*.json")):
        try:
            with p.open() as fh:
                payload = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and "curve" in payload:
            out.append(payload)
    return out


@beartype
def _summarize_cells(cells: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Compute per-model mean effects across the three intervention kinds.

    Returns a dict with keys ``models``, ``amplify``, ``ablate``,
    ``random``. Each value list is aligned by index. Missing or NaN
    entries are dropped before the mean so a single crashed run doesn't
    zero-out an otherwise-good model.
    """
    import numpy as np

    per_model: dict[str, dict[str, list[float]]] = {}
    for cell in cells:
        model = str(cell.get("model", "unknown"))
        curve = cell.get("curve") or {}
        points = _iter_points(curve if isinstance(curve, dict) else {})

        amp = _delta_on_group(points, "conflict", positive=True)
        rand = _delta_on_group(points, "random_control", positive=True)
        abl = _ablation_delta(cell)

        bucket = per_model.setdefault(
            model, {"amplify": [], "ablate": [], "random": []}
        )
        if np.isfinite(amp):
            bucket["amplify"].append(amp)
        if np.isfinite(abl):
            bucket["ablate"].append(abl)
        if np.isfinite(rand):
            bucket["random"].append(rand)

    models = sorted(per_model.keys())
    amplify = [float(np.mean(per_model[m]["amplify"])) if per_model[m]["amplify"] else float("nan") for m in models]
    ablate = [float(np.mean(per_model[m]["ablate"])) if per_model[m]["ablate"] else float("nan") for m in models]
    random = [float(np.mean(per_model[m]["random"])) if per_model[m]["random"] else float("nan") for m in models]
    return {
        "models": models,
        "amplify": amplify,
        "ablate": ablate,
        "random": random,
    }


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #


def _plot_grouped_bars(summary: dict[str, list[Any]]):
    """Return a matplotlib Figure with amplify/ablate/random grouped bars."""
    import matplotlib.pyplot as plt
    import numpy as np

    models = list(summary["models"])
    if not models:
        raise ValueError("no causal cells summarised — nothing to plot")

    amplify = np.array(summary["amplify"], dtype=np.float64)
    ablate = np.array(summary["ablate"], dtype=np.float64)
    random = np.array(summary["random"], dtype=np.float64)

    x = np.arange(len(models))
    width = 0.27

    fig, ax = plt.subplots(figsize=FIG_SIZE_DOUBLE_COLUMN)

    bars_amp = ax.bar(
        x - width,
        amplify,
        width=width,
        color=COLORS.get("s2", "#2ca02c"),
        label="amplify (alpha>0)",
    )
    bars_abl = ax.bar(
        x,
        ablate,
        width=width,
        color=COLORS.get("s1", "#d62728"),
        label="ablate",
    )
    bars_rand = ax.bar(
        x + width,
        random,
        width=width,
        color=COLORS.get("baseline", "#7f7f7f"),
        label="random (control)",
    )

    # Outline each bar with its model-specific color so the paper-level
    # identity of the model stays visible at a glance.
    for i, model in enumerate(models):
        mc = get_model_color(model)
        for bar_group in (bars_amp, bars_abl, bars_rand):
            bar_group[i].set_edgecolor(mc)
            bar_group[i].set_linewidth(1.2)

    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(r"$\Delta$ P(correct) on conflict items")
    ax.set_title("Figure 4: causal interventions (per-model mean)")
    ax.legend(loc="best", fontsize=8, frameon=True, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


@beartype
def make_figure_4_causal(cfg: PaperFiguresConfig) -> Path:
    """Generate Figure 4 (causal interventions) from saved causal JSONs."""
    set_paper_theme()

    causal_dir = require_dir(cfg.results_dir / "causal", label="causal results")
    cells = _load_cell_jsons(causal_dir)
    if not cells:
        raise FileNotFoundError(f"no causal-cell JSONs under {causal_dir}")

    if cfg.models_to_plot is not None:
        allowed = set(cfg.models_to_plot)
        cells = [c for c in cells if c.get("model") in allowed]
        if not cells:
            raise FileNotFoundError(
                f"no causal cells matched models_to_plot={cfg.models_to_plot!r}"
            )

    summary = _summarize_cells(cells)
    fig = _plot_grouped_bars(summary)

    formats = format_list(cfg.format)
    out_path = resolve_output_path(cfg.output_dir, "figure_4_causal", formats)
    return save_figure(fig, out_path, formats, dpi=cfg.dpi)
