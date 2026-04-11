"""Config dataclass for the paper-figures pipeline.

Lives in its own module so that both :mod:`s1s2.viz.paper_figures`
(the pipeline driver) and each per-figure module
(``figure1_benchmark.py`` ... ``figure6_geometry.py``) can import it
without a circular dependency. If this dataclass lived in
``paper_figures.py``, the per-figure modules would need to import
from there, and ``paper_figures.py`` imports from them — a cycle.

Keeping the config here (with no heavy deps) breaks the cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

__all__ = ["PaperFiguresConfig"]


@dataclass
class PaperFiguresConfig:
    """User-facing config for the paper-figure pipeline.

    Fields mirror the CLI flags of ``scripts/generate_figures.py``. See
    :func:`s1s2.viz.paper_figures.make_paper_figures` for the pipeline
    semantics.
    """

    results_dir: Path
    output_dir: Path = Path("figures")
    benchmark_path: Path = Path("data/benchmark/benchmark.jsonl")
    format: Literal["pdf", "png", "both"] = "pdf"
    include: list[str] = field(
        default_factory=lambda: [
            "figure_1_benchmark",
            "figure_2_probes",
            "figure_3_sae",
            "figure_4_causal",
            "figure_5_attention",
            "figure_6_geometry",
        ]
    )
    dpi: int = 300
    models_to_plot: list[str] | None = None  # None = all

    def __post_init__(self) -> None:
        # Coerce Path fields that may arrive as strings from click / YAML.
        self.results_dir = Path(self.results_dir)
        self.output_dir = Path(self.output_dir)
        self.benchmark_path = Path(self.benchmark_path)
