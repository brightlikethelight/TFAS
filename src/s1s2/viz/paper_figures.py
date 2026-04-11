"""Unified paper-figure generator — the main entry point.

This module owns the "six figures from saved results" pipeline. Each
per-figure module (``figure1_benchmark.py`` through
``figure6_geometry.py``) produces one figure; this file wires them all
together, applies the shared paper theme up front, and reports
successes / skips / failures in a compact summary.

Why a separate module from :mod:`s1s2.viz.figures`?
----------------------------------------------------
An earlier iteration of the viz layer introduced
:func:`s1s2.viz.figures.generate_all_figures`, which pre-dates the
scope spec for the per-figure modules. The two live side-by-side:

- :mod:`s1s2.viz.figures` — general "look up a plot function in any
  workstream and call it" framework. Still useful for ad-hoc calls.
- :mod:`s1s2.viz.paper_figures` — opinionated pipeline that calls the
  six hand-written ``figureN_*`` modules. This is what the paper build
  should use.

Why config-driven?
------------------
We need to (1) run the whole sweep from a single CLI invocation, (2)
skip figures cleanly on partial results, and (3) select which figures
to render during iterative work. A dataclass config beats a pile of
function kwargs for those use cases.

Error-handling contract
-----------------------
Every figure is generated inside its own try/except block. A
``FileNotFoundError`` is reported as a soft skip (missing input); any
other exception is reported as a hard failure with the exception type
and message. Neither aborts the sweep.
"""

from __future__ import annotations

import traceback
from pathlib import Path

from beartype import beartype
from rich.console import Console
from rich.table import Table

from s1s2.utils.logging import get_logger
from s1s2.viz.figure1_benchmark import make_figure_1_benchmark
from s1s2.viz.figure2_probes import make_figure_2_probes
from s1s2.viz.figure3_sae import make_figure_3_sae
from s1s2.viz.figure4_causal import make_figure_4_causal
from s1s2.viz.figure5_attention import make_figure_5_attention
from s1s2.viz.figure6_geometry import make_figure_6_geometry
from s1s2.viz.paper_figures_config import PaperFiguresConfig
from s1s2.viz.theme import set_paper_theme

console = Console()
logger = get_logger("s1s2.viz.paper_figures")


__all__ = [
    "PaperFiguresConfig",
    "make_paper_figures",
]


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #


#: Name -> generator mapping. Each generator takes a
#: :class:`PaperFiguresConfig` and returns a :class:`Path` to the written
#: figure. The names are the keys used in ``cfg.include``.
_FIGURE_GENERATORS: dict[str, object] = {
    "figure_1_benchmark": make_figure_1_benchmark,
    "figure_2_probes": make_figure_2_probes,
    "figure_3_sae": make_figure_3_sae,
    "figure_4_causal": make_figure_4_causal,
    "figure_5_attention": make_figure_5_attention,
    "figure_6_geometry": make_figure_6_geometry,
}


# --------------------------------------------------------------------------- #
# Main entry point                                                             #
# --------------------------------------------------------------------------- #


@beartype
def make_paper_figures(cfg: PaperFiguresConfig) -> dict[str, Path]:
    """Generate all paper figures, returning name -> output path for successes.

    Each figure is generated in its own try/except so a single failure
    does not kill the pipeline. Failures are logged with a clear reason
    and summarized in a rich table at the end.

    Parameters
    ----------
    cfg
        :class:`PaperFiguresConfig` controlling which figures to render,
        where to read results from, and where to write the outputs.

    Returns
    -------
    dict[str, Path]
        Mapping of figure name -> output file path for figures that
        rendered successfully. Figures that were skipped or errored are
        omitted from the return value (they are still reported to the
        console).
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_paper_theme()

    results: dict[str, Path | None] = {}
    messages: dict[str, str] = {}

    for name in cfg.include:
        fn = _FIGURE_GENERATORS.get(name)
        if fn is None:
            console.print(f"[yellow]Skipping unknown figure: {name}[/yellow]")
            results[name] = None
            messages[name] = "unknown figure name"
            continue
        try:
            path = fn(cfg)  # type: ignore[operator]
        except FileNotFoundError as exc:
            console.print(
                f"[yellow]⚠[/yellow] {name}: missing input ({exc})"
            )
            results[name] = None
            messages[name] = f"missing input: {exc}"
            logger.info("skip %s: missing input (%s)", name, exc)
            continue
        except NotImplementedError as exc:
            console.print(
                f"[yellow]⚠[/yellow] {name}: not implemented ({exc})"
            )
            results[name] = None
            messages[name] = f"not implemented: {exc}"
            continue
        except Exception as exc:
            console.print(
                f"[red]✗[/red] {name}: {type(exc).__name__}: {exc}"
            )
            logger.error(
                "figure %s raised %s: %s\n%s",
                name,
                type(exc).__name__,
                exc,
                traceback.format_exc(),
            )
            results[name] = None
            messages[name] = f"{type(exc).__name__}: {exc}"
            continue
        results[name] = path
        console.print(f"[green]✓[/green] {name} -> {path}")

    _print_summary_table(results, messages)
    return {k: v for k, v in results.items() if v is not None}


# --------------------------------------------------------------------------- #
# Summary helpers                                                              #
# --------------------------------------------------------------------------- #


def _print_summary_table(
    results: dict[str, Path | None],
    messages: dict[str, str],
) -> None:
    """Print a compact rich table summarising the sweep.

    We keep this thin so it is trivially suppressable in tests (which
    capture stdout). The table has three columns: figure name, status,
    and either the output path or the failure reason.
    """
    table = Table(title="Paper Figures")
    table.add_column("Figure", style="bold")
    table.add_column("Status")
    table.add_column("Output / Reason")
    for name, path in results.items():
        if path is not None:
            table.add_row(name, "[green]OK[/green]", str(path))
        else:
            reason = messages.get(name, "unknown")
            table.add_row(name, "[red]FAIL[/red]", reason)
    console.print(table)
