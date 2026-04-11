"""Layer-level aggregation of per-head attention metrics.

For each layer we report:

- ``mean_entropy`` — mean across query heads (pooled across problems)
- ``max_entropy`` — max across query heads (the most diffuse head)
- ``entropy_spread`` — std across query heads
- Per-condition (conflict / no-conflict) versions of each of the above
- ``S2_head_count`` / ``S1_head_count`` — specialized head counts at this layer
- ``paired_p`` — Wilcoxon signed-rank test on per-problem layer-mean entropy
  between conflict and no-conflict groups (two-sided)

Gemma-2 sliding-window caveat: callers should restrict this function to the
"global" layer subset (or the "sliding_window" subset) via
:func:`s1s2.attention.core.gemma_layer_partition` before interpreting results
— layer 1 (sliding window) and layer 2 (global) have different structural
entropy ceilings.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from beartype import beartype
from jaxtyping import Float
from scipy import stats

from s1s2.attention.core import ModelAttentionData, is_gemma_family
from s1s2.attention.heads import HeadClassification
from s1s2.utils.logging import get_logger

__all__ = [
    "layer_metric_array",
    "layer_statistic_wilcoxon",
    "layer_summary",
]

logger = get_logger(__name__)


@beartype
def layer_metric_array(
    data: ModelAttentionData,
    metric: str,
    position: str | None = None,
) -> Float[np.ndarray, "n_problems n_layers n_heads"]:
    """Return the metric indexed to one position (or means across positions).

    If ``position`` is None, the mean is taken over all selected positions.
    Otherwise uses just the requested position.
    """
    if metric not in data.metrics:
        raise KeyError(f"metric {metric!r} not loaded; have {list(data.metrics.keys())}")
    arr = data.metrics[metric]  # (n_problems, L, H, P)
    if position is None:
        return arr.mean(axis=-1)
    if position not in data.selected_positions:
        raise KeyError(
            f"position {position!r} not in {data.selected_positions}"
        )
    idx = data.selected_positions.index(position)
    return arr[..., idx]


@beartype
def layer_statistic_wilcoxon(
    per_problem_layer_vals: Float[np.ndarray, "n_problems"],
    conflict: np.ndarray,
) -> tuple[float, float]:
    """Two-sided Wilcoxon / Mann-Whitney on per-problem layer values.

    We use Mann-Whitney here because conflict and no-conflict items are
    independent samples from the same layer (matched pairs are a property
    of the benchmark but not guaranteed to align by index). Return (stat, p).
    Returns (nan, 1.0) on degenerate inputs.
    """
    x = per_problem_layer_vals[conflict]
    y = per_problem_layer_vals[~conflict]
    if len(x) == 0 or len(y) == 0:
        return float("nan"), 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            stat, p = stats.mannwhitneyu(
                x.astype(np.float64, copy=False),
                y.astype(np.float64, copy=False),
                alternative="two-sided",
            )
        except ValueError:
            return float("nan"), 1.0
    return float(stat), float(p)


@beartype
def layer_summary(
    data: ModelAttentionData,
    head_classifs: list[HeadClassification],
    metric: str = "entropy",
    position: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate per-head metrics across heads, per layer.

    Returns a list with one dict per layer::

        {
          "layer": int,
          "layer_type": "global" | "sliding_window" | "standard",
          "metric": str,
          "position": str | "mean_across_positions",
          "mean": float,
          "max": float,
          "spread": float,
          "mean_conflict": float,
          "mean_noconflict": float,
          "delta": float,                  # mean_conflict - mean_noconflict
          "wilcoxon_p": float,
          "S2_head_count": int,
          "S1_head_count": int,
          "mixed_head_count": int,
          "non_specialized_head_count": int,
        }

    ``layer_type`` is documented for Gemma-2 (even=global, odd=sliding_window)
    and "standard" otherwise.
    """
    arr = layer_metric_array(data, metric=metric, position=position)  # (N, L, H)
    n_problems, n_layers, n_heads = arr.shape

    # Per-problem layer value = mean over heads at that layer
    per_problem_layer = arr.mean(axis=2)  # (N, L)

    counts: dict[int, dict[str, int]] = {
        L: {"S2": 0, "S1": 0, "mixed": 0, "non": 0} for L in range(n_layers)
    }
    for c in head_classifs:
        if c.layer < 0 or c.layer >= n_layers:
            continue
        if c.classification == "S2_specialized":
            counts[c.layer]["S2"] += 1
        elif c.classification == "S1_specialized":
            counts[c.layer]["S1"] += 1
        elif c.classification == "mixed":
            counts[c.layer]["mixed"] += 1
        else:
            counts[c.layer]["non"] += 1

    pos_label = position if position is not None else "mean_across_positions"
    use_gemma = is_gemma_family(data.family)

    out: list[dict[str, Any]] = []
    for L in range(n_layers):
        layer_vals = arr[:, L, :]  # (N, H)
        per_prob = per_problem_layer[:, L]  # (N,)
        m_conflict = (
            float(per_prob[data.conflict].mean()) if data.conflict.any() else float("nan")
        )
        m_noconflict = (
            float(per_prob[~data.conflict].mean())
            if (~data.conflict).any()
            else float("nan")
        )
        stat, p = layer_statistic_wilcoxon(per_prob, data.conflict)

        layer_type = "standard"
        if use_gemma:
            layer_type = "global" if (L % 2 == 0) else "sliding_window"

        out.append(
            {
                "layer": int(L),
                "layer_type": layer_type,
                "metric": metric,
                "position": pos_label,
                "mean": float(layer_vals.mean()),
                "max": float(layer_vals.max()),
                "spread": float(layer_vals.std(ddof=1)) if layer_vals.size > 1 else 0.0,
                "mean_conflict": m_conflict,
                "mean_noconflict": m_noconflict,
                "delta": float(m_conflict - m_noconflict)
                if np.isfinite(m_conflict) and np.isfinite(m_noconflict)
                else float("nan"),
                "wilcoxon_statistic": float(stat),
                "wilcoxon_p": float(p),
                "S2_head_count": int(counts[L]["S2"]),
                "S1_head_count": int(counts[L]["S1"]),
                "mixed_head_count": int(counts[L]["mixed"]),
                "non_specialized_head_count": int(counts[L]["non"]),
            }
        )
    return out
