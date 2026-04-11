"""Figure 2 — layer-wise probe accuracy curves.

This is the paper's headline result figure. Per the scope we delegate
the actual plotting to :func:`s1s2.viz.probe_plots.plot_layer_accuracy_curves`
(the probes workstream's canonical layer-curve plotter) and only add
the shared theme + consistent file naming.

The probe plotter writes its own files via an output-path *stem*, so
we pass it the stem derived from :func:`resolve_output_path` minus the
extension. It writes both a .pdf and a .png; we honor whichever format
the caller asked for by keeping the one(s) they requested.
"""

from __future__ import annotations

from pathlib import Path

from beartype import beartype

from s1s2.utils.logging import get_logger
from s1s2.viz.common import (
    format_list,
    require_dir,
    resolve_output_path,
)
from s1s2.viz.paper_figures_config import PaperFiguresConfig
from s1s2.viz.theme import set_paper_theme

try:
    from s1s2.viz.probe_plots import (
        load_results_tree as _load_probe_results,
    )
    from s1s2.viz.probe_plots import (
        plot_layer_accuracy_curves as _plot_layer_curves,
    )
except ImportError:  # pragma: no cover — defensive
    _load_probe_results = None  # type: ignore[assignment]
    _plot_layer_curves = None  # type: ignore[assignment]

logger = get_logger("s1s2.viz.figure2")

__all__ = ["make_figure_2_probes"]


@beartype
def make_figure_2_probes(cfg: PaperFiguresConfig) -> Path:
    """Generate Figure 2 (layer-wise probe curves) from saved probe JSONs.

    The canonical slice for the paper is ``target="task_type"`` at the
    ``P0`` (final prompt token) position, aggregated over probe type =
    logistic. If the probe workstream has written nothing yet we raise
    :class:`FileNotFoundError` so the main entry point's try/except
    logs a clean "missing input" warning rather than a stack trace.
    """
    if _load_probe_results is None or _plot_layer_curves is None:
        raise RuntimeError(
            "s1s2.viz.probe_plots is unavailable; cannot render Figure 2"
        )

    set_paper_theme()

    probe_dir = require_dir(cfg.results_dir / "probes", label="probe results")
    results = _load_probe_results(probe_dir)
    if not results:
        raise FileNotFoundError(f"no probe JSONs found under {probe_dir}")

    # The probe plotter takes a stem (no suffix) and writes .pdf + .png.
    formats = format_list(cfg.format)
    out_path = resolve_output_path(cfg.output_dir, "figure_2_probes", formats)
    stem = out_path.with_suffix("")

    target = "task_type"
    position = "P0"
    _plot_layer_curves(
        results=results,
        outpath=stem,
        target=target,
        position=position,
        probes=("logistic",),
        show_ci=True,
        show_significance_band=True,
    )
    # plot_layer_accuracy_curves writes both .pdf and .png — return whichever
    # the caller asked for as the "primary" path.
    primary = stem.with_suffix(f".{formats[0]}")
    if not primary.exists():
        # Last-ditch: some edge cases might rename; fall back to the .pdf.
        alt = stem.with_suffix(".pdf")
        if alt.exists():
            return alt
        raise FileNotFoundError(
            f"probe plotter did not produce {primary} (formats={formats})"
        )
    return primary
