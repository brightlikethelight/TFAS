"""Unified figure generator for the paper.

The paper has six main figures:

1. **Figure 1** — benchmark overview (task distribution, behavioral rates).
2. **Figure 2** — layer-wise linear-probe ROC-AUC, one line per model.
3. **Figure 3** — SAE differential feature volcano plot.
4. **Figure 4** — causal-intervention effect bars.
5. **Figure 5** — attention-entropy heatmaps.
6. **Figure 6** — layer-wise cosine silhouette curves (geometry).

Each figure is produced by a workstream-owned plotting function. This
module wires them together so ``python scripts/generate_figures.py`` can
rebuild every paper figure from the per-workstream result JSONs in one
command.

IMPORTANT: this module must remain **defensive**. Workstreams are
developed in parallel; at any given time some may not yet exist, and
some result directories may be empty. We wrap every plot call in
try/except so a single failure does not kill the whole sweep. Failures
are logged as warnings — the final summary tells the user which figures
were produced and which were skipped.
"""

from __future__ import annotations

import importlib
import json
import traceback
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from s1s2.utils.logging import get_logger
from s1s2.viz.theme import COLORS, MODEL_COLORS, set_paper_theme

logger = get_logger("s1s2.viz.figures")


__all__ = [
    "FigureGenerationReport",
    "FigureResult",
    "generate_all_figures",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class FigureResult:
    """Outcome of rendering a single figure."""

    name: str
    output_path: Path | None
    status: str  # one of: "ok", "skipped", "error"
    message: str = ""


@dataclass
class FigureGenerationReport:
    """Aggregated outcome of a full figure-generation sweep."""

    results: list[FigureResult] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.results if r.status == "ok")

    @property
    def n_skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "skipped")

    @property
    def n_error(self) -> int:
        return sum(1 for r in self.results if r.status == "error")

    def log_summary(self) -> None:
        logger.info(
            "figure generation done: %d ok, %d skipped, %d error",
            self.n_ok,
            self.n_skipped,
            self.n_error,
        )
        for r in self.results:
            if r.status == "ok":
                logger.info("  [ok]      %s -> %s", r.name, r.output_path)
            elif r.status == "skipped":
                logger.warning("  [skipped] %s: %s", r.name, r.message)
            else:
                logger.error("  [error]   %s: %s", r.name, r.message)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_optional(module_path: str) -> Any | None:
    """Import a module and return ``None`` on any import-time failure.

    ImportError (module genuinely missing) is logged at debug level so
    normal runs stay quiet. Any other exception (module exists but its
    top-level code crashes — e.g. an AssertionError inside a sibling
    module dragged in by the package ``__init__``) is logged as a
    warning because it usually indicates an upstream bug the user
    should know about, but still returns None so the figure sweep can
    continue.
    """
    try:
        return importlib.import_module(module_path)
    except ImportError as exc:  # module genuinely missing
        logger.debug("optional import %s not available: %s", module_path, exc)
        return None
    except Exception as exc:
        logger.warning(
            "importing %s raised %s: %s",
            module_path,
            type(exc).__name__,
            exc,
        )
        return None


def _get_first_attr(mod: Any, names: Iterable[str]) -> Callable[..., Any] | None:
    """Return the first attribute on ``mod`` that matches any name in ``names``."""
    if mod is None:
        return None
    for name in names:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    return None


def _load_json_tree(root: Path) -> list[dict[str, Any]]:
    """Recursively collect all ``*.json`` files under ``root``.

    Returns an empty list (never None) on missing root; callers then
    skip gracefully.
    """
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(root.rglob("*.json")):
        try:
            with p.open() as fh:
                out.append(json.load(fh))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("failed to parse %s: %s", p, exc)
    return out


def _save_figure(fig: Any, out_path: Path, formats: Iterable[str]) -> Path:
    """Save a matplotlib Figure to the requested formats.

    The primary output path (the one we return) is the first format in
    ``formats``. Additional formats are written as siblings of the primary.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    formats = tuple(formats)
    primary: Path | None = None
    for fmt in formats:
        p = out_path.with_suffix(f".{fmt.lstrip('.')}")
        fig.savefig(p, bbox_inches="tight")
        if primary is None:
            primary = p
    plt.close(fig)
    assert primary is not None
    return primary


def _resolve_plot_fn(
    preferred: list[tuple[str, list[str]]],
) -> Callable[..., Any] | None:
    """Walk a list of ``(module_path, [attr_names])`` and return the first hit."""
    for module_path, names in preferred:
        mod = _import_optional(module_path)
        fn = _get_first_attr(mod, names)
        if fn is not None:
            return fn
    return None


# ---------------------------------------------------------------------------
# Per-figure rendering wrappers. Each returns a FigureResult.
# ---------------------------------------------------------------------------


def _fmt_paths(output_dir: Path, name: str, primary_format: str) -> Path:
    return output_dir / f"{name}.{primary_format.lstrip('.')}"


def _render_figure_1_benchmark_overview(
    results_dir: Path,
    output_dir: Path,
    formats: list[str],
) -> FigureResult:
    """Benchmark summary bar-chart: behavior rates, task-type counts."""
    name = "figure_1_benchmark_overview"
    out_path = _fmt_paths(output_dir, name, formats[0])

    plot_fn = _resolve_plot_fn(
        [
            ("s1s2.benchmark.viz", ["plot_benchmark_overview", "plot_overview"]),
            ("s1s2.viz.benchmark_plots", ["plot_benchmark_overview"]),
        ]
    )
    if plot_fn is None:
        return FigureResult(
            name=name,
            output_path=None,
            status="skipped",
            message="no benchmark-overview plotter found (s1s2.benchmark.viz)",
        )
    bench_results = _load_json_tree(results_dir / "benchmark")
    if not bench_results:
        return FigureResult(
            name=name,
            output_path=None,
            status="skipped",
            message=f"no JSONs under {results_dir / 'benchmark'}",
        )
    try:
        fig = plot_fn(bench_results)
    except Exception as exc:
        return FigureResult(
            name=name,
            output_path=None,
            status="error",
            message=f"{type(exc).__name__}: {exc}",
        )
    primary = _save_figure(fig, out_path, formats)
    return FigureResult(name=name, output_path=primary, status="ok")


def _render_figure_2_probe_layer_curves(
    results_dir: Path,
    output_dir: Path,
    formats: list[str],
) -> FigureResult:
    """Figure 2: layer-wise probe ROC-AUC, one line per (model, probe)."""
    name = "figure_2_probe_layer_curves"
    out_path = _fmt_paths(output_dir, name, formats[0])

    # Preferred path first (scope doc), then existing impl, then fallbacks.
    plot_fn = _resolve_plot_fn(
        [
            ("s1s2.probes.viz", ["plot_layer_curves", "plot_layer_accuracy_curves"]),
            ("s1s2.viz.probe_plots", ["plot_layer_accuracy_curves", "plot_layer_curves"]),
        ]
    )
    if plot_fn is None:
        return FigureResult(
            name=name,
            output_path=None,
            status="skipped",
            message="no probe layer-curve plotter found",
        )

    # The existing implementation takes a 'results' list + an 'outpath' +
    # explicit target/position kwargs, and saves to disk itself. We try the
    # scope-style signature (return a Figure) first, and fall back to the
    # existing save-to-disk contract.
    probe_results = _load_json_tree(results_dir / "probes")
    if not probe_results:
        return FigureResult(
            name=name,
            output_path=None,
            status="skipped",
            message=f"no JSONs under {results_dir / 'probes'}",
        )
    try:
        # Try "return a Figure" signature.
        try:
            fig = plot_fn(probe_results)
        except TypeError:
            # Existing impl: requires target + position + outpath.
            # Use the paper's canonical slice.
            plot_fn(
                results=probe_results,
                outpath=out_path.with_suffix(""),
                target="task_type",
                position="P0",
            )
            return FigureResult(
                name=name,
                output_path=out_path.with_suffix(".pdf"),
                status="ok",
            )
    except Exception as exc:
        return FigureResult(
            name=name,
            output_path=None,
            status="error",
            message=f"{type(exc).__name__}: {exc}",
        )
    primary = _save_figure(fig, out_path, formats)
    return FigureResult(name=name, output_path=primary, status="ok")


def _render_figure_3_sae_volcano(
    results_dir: Path,
    output_dir: Path,
    formats: list[str],
) -> FigureResult:
    """Figure 3: SAE differential-feature volcano plot."""
    name = "figure_3_sae_volcano"
    out_path = _fmt_paths(output_dir, name, formats[0])

    plot_fn = _resolve_plot_fn(
        [
            ("s1s2.sae.volcano", ["plot_volcano"]),
            ("s1s2.sae.viz", ["plot_volcano"]),
        ]
    )
    if plot_fn is None:
        return FigureResult(
            name=name,
            output_path=None,
            status="skipped",
            message="no SAE volcano plotter found",
        )

    # The current plot_volcano signature takes a pandas DataFrame and saves
    # to disk itself — not a Figure-returning function in the scope-doc
    # sense. We load the first SAE JSON with a 'features' table and pass it.
    sae_dir = results_dir / "sae"
    if not sae_dir.exists():
        return FigureResult(
            name=name,
            output_path=None,
            status="skipped",
            message=f"{sae_dir} does not exist",
        )
    json_paths = sorted(sae_dir.rglob("*.json"))
    if not json_paths:
        return FigureResult(
            name=name,
            output_path=None,
            status="skipped",
            message=f"no JSONs under {sae_dir}",
        )

    import pandas as pd

    # Find the first JSON that has a tabular features structure.
    df: pd.DataFrame | None = None
    source: Path | None = None
    required = {"feature_id", "log_fc", "q_value"}
    for p in json_paths:
        try:
            with p.open() as fh:
                raw = json.load(fh)
        except Exception:
            continue
        candidate = raw.get("features") if isinstance(raw, dict) else None
        if isinstance(candidate, list) and candidate:
            first = candidate[0]
            if isinstance(first, dict) and required.issubset(first.keys()):
                df = pd.DataFrame(candidate)
                source = p
                break
    if df is None:
        return FigureResult(
            name=name,
            output_path=None,
            status="skipped",
            message="no SAE JSON contained a 'features' table with log_fc + q_value",
        )

    try:
        # plot_volcano writes to disk and returns the Figure. We want both
        # formats, so we re-save via _save_figure afterwards (plot_volcano
        # writes whatever format the out_path suffix implies).
        fig = plot_fn(
            df,
            title=f"SAE volcano ({source.stem})" if source else "SAE volcano",
            out_path=out_path,
        )
    except Exception as exc:
        return FigureResult(
            name=name,
            output_path=None,
            status="error",
            message=f"{type(exc).__name__}: {exc}",
        )
    primary = _save_figure(fig, out_path, formats)
    return FigureResult(name=name, output_path=primary, status="ok")


def _render_figure_4_causal_intervention(
    results_dir: Path,
    output_dir: Path,
    formats: list[str],
) -> FigureResult:
    """Figure 4: causal-intervention effect bars."""
    name = "figure_4_causal_intervention"
    out_path = _fmt_paths(output_dir, name, formats[0])

    plot_fn = _resolve_plot_fn(
        [
            (
                "s1s2.causal.viz",
                [
                    "plot_intervention_bars",
                    "plot_causal_bars",
                    "plot_ablation_bars",  # currently-shipped name
                ],
            ),
            ("s1s2.viz.causal_plots", ["plot_intervention_bars"]),
        ]
    )
    if plot_fn is None:
        return FigureResult(
            name=name,
            output_path=None,
            status="skipped",
            message="no causal-intervention plotter found",
        )
    causal_results = _load_json_tree(results_dir / "causal")
    if not causal_results:
        return FigureResult(
            name=name,
            output_path=None,
            status="skipped",
            message=f"no JSONs under {results_dir / 'causal'}",
        )
    try:
        fig = plot_fn(causal_results)
    except Exception as exc:
        return FigureResult(
            name=name,
            output_path=None,
            status="error",
            message=f"{type(exc).__name__}: {exc}",
        )
    primary = _save_figure(fig, out_path, formats)
    return FigureResult(name=name, output_path=primary, status="ok")


def _render_figure_5_attention_entropy(
    results_dir: Path,
    output_dir: Path,
    formats: list[str],
) -> FigureResult:
    """Figure 5: attention-entropy heatmap."""
    name = "figure_5_attention_entropy"
    out_path = _fmt_paths(output_dir, name, formats[0])

    plot_fn = _resolve_plot_fn(
        [
            ("s1s2.attention.viz", ["plot_entropy_heatmap", "plot_attention_entropy"]),
            ("s1s2.viz.attention_plots", ["plot_entropy_heatmap"]),
        ]
    )
    if plot_fn is None:
        return FigureResult(
            name=name,
            output_path=None,
            status="skipped",
            message="no attention-entropy plotter found",
        )
    attn_results = _load_json_tree(results_dir / "attention")
    if not attn_results:
        return FigureResult(
            name=name,
            output_path=None,
            status="skipped",
            message=f"no JSONs under {results_dir / 'attention'}",
        )
    try:
        fig = plot_fn(attn_results)
    except Exception as exc:
        return FigureResult(
            name=name,
            output_path=None,
            status="error",
            message=f"{type(exc).__name__}: {exc}",
        )
    primary = _save_figure(fig, out_path, formats)
    return FigureResult(name=name, output_path=primary, status="ok")


def _render_figure_6_geometry_silhouette(
    results_dir: Path,
    output_dir: Path,
    formats: list[str],
) -> FigureResult:
    """Figure 6: layer-wise cosine silhouette curves."""
    name = "figure_6_geometry_silhouette"
    out_path = _fmt_paths(output_dir, name, formats[0])

    plot_fn = _resolve_plot_fn(
        [
            ("s1s2.geometry.viz", ["plot_silhouette_curves"]),
            ("s1s2.viz.geometry_plots", ["plot_silhouette_curves"]),
        ]
    )
    if plot_fn is None:
        return FigureResult(
            name=name,
            output_path=None,
            status="skipped",
            message="no geometry silhouette plotter found",
        )

    # Reassemble the curves-per-model dict from per-model JSONs. The
    # geometry workstream hasn't landed yet so we accept two shapes:
    #   - single-file "summary" JSON with a top-level "curves" key
    #   - per-model JSONs with {"model": ..., "layers": [...],
    #     "silhouette": [...], "ci_lower": [...], "ci_upper": [...]}
    geom_dir = results_dir / "geometry"
    if not geom_dir.exists():
        return FigureResult(
            name=name,
            output_path=None,
            status="skipped",
            message=f"{geom_dir} does not exist",
        )
    import numpy as np

    curves: dict[str, dict[str, np.ndarray]] = {}
    for p in sorted(geom_dir.rglob("*.json")):
        try:
            with p.open() as fh:
                raw = json.load(fh)
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        if "curves" in raw and isinstance(raw["curves"], dict):
            for model, entry in raw["curves"].items():
                if not isinstance(entry, dict):
                    continue
                curves[model] = {k: np.asarray(v) for k, v in entry.items()}
            continue
        model = raw.get("model")
        if model and all(k in raw for k in ("layers", "silhouette", "ci_lower", "ci_upper")):
            curves[model] = {
                "layers": np.asarray(raw["layers"]),
                "silhouette": np.asarray(raw["silhouette"]),
                "ci_lower": np.asarray(raw["ci_lower"]),
                "ci_upper": np.asarray(raw["ci_upper"]),
            }
    if not curves:
        return FigureResult(
            name=name,
            output_path=None,
            status="skipped",
            message=f"no usable silhouette JSONs under {geom_dir}",
        )
    try:
        fig = plot_fn(curves)
    except Exception as exc:
        return FigureResult(
            name=name,
            output_path=None,
            status="error",
            message=f"{type(exc).__name__}: {exc}",
        )
    primary = _save_figure(fig, out_path, formats)
    return FigureResult(name=name, output_path=primary, status="ok")


# ---------------------------------------------------------------------------
# Registry + entry point
# ---------------------------------------------------------------------------


_FIGURE_REGISTRY: dict[str, Callable[..., FigureResult]] = {
    "figure_1_benchmark_overview": _render_figure_1_benchmark_overview,
    "figure_2_probe_layer_curves": _render_figure_2_probe_layer_curves,
    "figure_3_sae_volcano": _render_figure_3_sae_volcano,
    "figure_4_causal_intervention": _render_figure_4_causal_intervention,
    "figure_5_attention_entropy": _render_figure_5_attention_entropy,
    "figure_6_geometry_silhouette": _render_figure_6_geometry_silhouette,
}


def generate_all_figures(
    results_dir: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any] | None = None,
) -> FigureGenerationReport:
    """Generate every paper figure from a results directory.

    Parameters
    ----------
    results_dir
        Root directory containing per-workstream result JSONs. Expected
        layout: ``results_dir/{probes,sae,attention,geometry,causal,benchmark}/...``.
    output_dir
        Where to write the figures. Created if missing.
    config
        Optional Hydra-style config with keys:

        - ``include``: list of figure names to render (default: all known)
        - ``figure_format``: primary format (``"pdf"`` or ``"png"``)
        - ``extra_formats``: additional formats to save alongside

    Returns
    -------
    FigureGenerationReport
        A report describing which figures succeeded, were skipped, or
        errored. Always returns — never raises on plot failures.
    """
    cfg: dict[str, Any] = dict(config or {})
    set_paper_theme()

    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    include = list(cfg.get("include") or _FIGURE_REGISTRY.keys())
    primary_fmt = str(cfg.get("figure_format", "pdf")).lstrip(".")
    extras = [str(f).lstrip(".") for f in cfg.get("extra_formats", [])]
    # Order matters: primary first so _save_figure returns the primary path.
    formats: list[str] = [primary_fmt] + [f for f in extras if f != primary_fmt]

    report = FigureGenerationReport()

    for name in include:
        fn = _FIGURE_REGISTRY.get(name)
        if fn is None:
            report.results.append(
                FigureResult(
                    name=name,
                    output_path=None,
                    status="skipped",
                    message="unknown figure name (not in registry)",
                )
            )
            logger.warning("unknown figure name: %s", name)
            continue
        try:
            result = fn(results_dir, output_dir, formats)
        except Exception as exc:
            # Defensive: even if the renderer itself throws, don't kill the sweep.
            logger.error(
                "renderer %s raised: %s\n%s",
                name,
                exc,
                traceback.format_exc(),
            )
            result = FigureResult(
                name=name,
                output_path=None,
                status="error",
                message=f"{type(exc).__name__}: {exc}",
            )
        report.results.append(result)

    report.log_summary()
    return report


# Silence a lint warning — symbols are referenced above but not used as
# bare names, so importers see them via the explicit __all__.
_ = (COLORS, MODEL_COLORS)
