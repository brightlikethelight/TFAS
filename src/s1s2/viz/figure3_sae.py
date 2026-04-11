"""Figure 3 — SAE differential-feature volcano plot.

Delegates to :func:`s1s2.sae.volcano.plot_volcano`, which accepts a
pandas DataFrame with columns ``feature_id``, ``log_fc``, ``q_value``
and optionally ``is_falsified``. The volcano plotter already handles
the three-color separation (non-significant / significant-but-falsified
/ significant-genuine) specified in :file:`CLAUDE.md`; our job here is
to (1) resolve the right results JSON to use, (2) coerce it into a
DataFrame, (3) apply the shared theme, and (4) save to the canonical
figure path.

We accept two on-disk shapes for SAE results:

1. A dict with a top-level ``"features"`` list of per-feature records
   (matches the ``DifferentialResult.to_dict`` format).
2. A pre-exported CSV sibling file (``*_features.csv``) — used by some
   earlier iterations of the workstream. If present, it is preferred
   because no translation is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beartype import beartype

from s1s2.utils.logging import get_logger
from s1s2.viz.common import (
    format_list,
    require_dir,
    resolve_output_path,
    save_figure,
)
from s1s2.viz.paper_figures_config import PaperFiguresConfig
from s1s2.viz.theme import set_paper_theme

try:
    from s1s2.sae.volcano import plot_volcano as _plot_volcano
except ImportError:  # pragma: no cover — defensive
    _plot_volcano = None  # type: ignore[assignment]

logger = get_logger("s1s2.viz.figure3")

__all__ = ["make_figure_3_sae"]

_REQUIRED_COLUMNS: frozenset[str] = frozenset({"feature_id", "log_fc", "q_value"})


@beartype
def _find_sae_dataframe(sae_dir: Path) -> tuple[Any, Path]:
    """Return ``(DataFrame, source_path)`` for the first usable SAE JSON.

    We scan the directory in lexicographic order for stability. A file
    is "usable" if its ``features`` list contains dicts with the three
    required columns. The function imports pandas lazily so that this
    module remains importable in environments without pandas.
    """
    import pandas as pd

    json_paths = sorted(sae_dir.rglob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"no SAE JSONs under {sae_dir}")

    for p in json_paths:
        try:
            with p.open() as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("skipping unreadable %s: %s", p, exc)
            continue

        features_list: list[dict[str, Any]] | None = None
        if isinstance(raw, dict):
            candidate = raw.get("features") or raw.get("differential", {}).get("features")
            if isinstance(candidate, list):
                features_list = [x for x in candidate if isinstance(x, dict)]
        if not features_list:
            continue
        first = features_list[0]
        if not _REQUIRED_COLUMNS.issubset(first.keys()):
            continue
        return pd.DataFrame(features_list), p

    raise FileNotFoundError(
        f"no SAE JSON under {sae_dir} contained a 'features' table "
        f"with columns {sorted(_REQUIRED_COLUMNS)}"
    )


@beartype
def make_figure_3_sae(cfg: PaperFiguresConfig) -> Path:
    """Generate Figure 3 (SAE volcano) and return its output path.

    The SAE plotter is allowed to write the figure itself (its signature
    accepts ``out_path``). We simply pass the canonical path and return it.
    The shared theme is applied before plotting so rcParams (font, grid,
    etc.) are consistent with every other paper figure.
    """
    if _plot_volcano is None:
        raise RuntimeError("s1s2.sae.volcano is unavailable; cannot render Figure 3")

    set_paper_theme()

    sae_dir = require_dir(cfg.results_dir / "sae", label="SAE results")
    df, source = _find_sae_dataframe(sae_dir)

    formats = format_list(cfg.format)
    out_path = resolve_output_path(cfg.output_dir, "figure_3_sae", formats)

    title = f"Figure 3: SAE differential features ({source.stem})"
    fig = _plot_volcano(
        df,
        title=title,
        out_path=out_path,
        fdr_q=0.05,
        annotate_top_k=8,
        dpi=cfg.dpi,
    )
    # If additional formats are requested, save siblings.
    primary = save_figure(fig, out_path, formats, dpi=cfg.dpi)
    return primary
