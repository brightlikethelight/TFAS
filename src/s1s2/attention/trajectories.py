"""Temporal entropy trajectories across token positions.

The activation cache stores attention metrics at a fixed set of canonical
positions (``P0``, ``P2``, ``T0``, ``T25``, ``T50``, ``T75``, ``Tend``). For
reasoning models these span the thinking trace; for non-reasoning models only
``P0``/``P2`` are valid. We treat the T-positions as a SPARSE trajectory: five
samples of entropy across the think trace.

Features computed per (head, problem) from the sparse trajectory:

- ``slope`` — least-squares slope of entropy vs. normalized position index
- ``peak_position`` — which T-position carried the max entropy
- ``mean`` / ``std`` — summary statistics of the trajectory

Group-level comparison between S1 and S2 problems:

- Mann-Whitney U test on slopes (hypothesis: S2 slope < S1 slope, as S2
  "converges" during the thinking trace)

Callers that need full per-token trajectories must extend the extraction
pipeline — that's out of scope for this workstream.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from beartype import beartype
from jaxtyping import Float
from scipy import stats

from s1s2.attention.core import ModelAttentionData
from s1s2.utils.logging import get_logger

__all__ = [
    "T_POSITION_LABELS",
    "compute_trajectories",
    "select_t_positions",
    "trajectory_features",
]

logger = get_logger(__name__)

# Canonical ordering of the thinking-trace sampling points. Not all of these
# are necessarily present in a given ModelAttentionData.selected_positions.
T_POSITION_LABELS: tuple[str, ...] = ("T0", "T25", "T50", "T75", "Tend")


@beartype
def select_t_positions(
    selected_positions: list[str],
) -> tuple[list[str], list[int]]:
    """Return (kept_labels, indices_into_selected_positions) in canonical order.

    Keeps only the subset of selected positions that are part of the T-series,
    in canonical T-order (T0, T25, T50, T75, Tend). If there are zero of them,
    returns ([], []).
    """
    kept_labels: list[str] = []
    indices: list[int] = []
    for t in T_POSITION_LABELS:
        if t in selected_positions:
            kept_labels.append(t)
            indices.append(selected_positions.index(t))
    return kept_labels, indices


@beartype
def trajectory_features(
    trajectories: Float[np.ndarray, "n_problems n_t"],
) -> dict[str, np.ndarray]:
    """Compute per-problem trajectory features.

    Uses position index in ``[0, n_t - 1]`` normalized to ``[0, 1]`` as the
    x-axis. A closed-form least-squares slope is used (no scipy roundtrip per
    problem). Returns a dict with keys: ``slope``, ``peak_position``, ``mean``,
    ``std``.
    """
    n_problems, n_t = trajectories.shape
    if n_t < 2:
        # Slope undefined; return zeros and NaN peaks.
        return {
            "slope": np.zeros((n_problems,), dtype=np.float64),
            "peak_position": np.zeros((n_problems,), dtype=np.int32),
            "mean": trajectories.mean(axis=1) if n_t > 0 else np.zeros(n_problems),
            "std": np.zeros(n_problems),
        }
    x = np.linspace(0.0, 1.0, num=n_t, dtype=np.float64)
    x_mean = float(x.mean())
    x_centered = x - x_mean
    denom = float(np.sum(x_centered**2))
    y = trajectories.astype(np.float64, copy=False)
    y_mean = y.mean(axis=1, keepdims=True)
    num = np.sum(x_centered[None, :] * (y - y_mean), axis=1)
    slope = num / denom if denom > 0 else np.zeros(n_problems)
    peak = np.argmax(y, axis=1).astype(np.int32)
    return {
        "slope": slope.astype(np.float64),
        "peak_position": peak,
        "mean": y.mean(axis=1).astype(np.float64),
        "std": y.std(axis=1, ddof=1 if n_t > 1 else 0).astype(np.float64),
    }


@beartype
def _slope_comparison(
    slopes: Float[np.ndarray, "n_problems"], conflict: np.ndarray
) -> dict[str, float]:
    """Mann-Whitney U on slopes between conflict and no-conflict groups.

    Hypothesis: S2 problems (conflict=True? see note below) show steeper
    (more negative) entropy decrease during the thinking trace.

    Note on direction: "conflict" = the S1-eliciting task setup. The
    interesting prediction is that S2 *responses* (deliberation) correspond
    to entropy decreases. The mapping from task-condition to S2-response is
    indirect; that's why the ground-truth head classification
    (``heads.classify_heads``) uses the full metric profile rather than a
    single slope test. Here we just report the descriptive comparison.
    """
    x = slopes[conflict]
    y = slopes[~conflict]
    nan = {
        "slope_mean_conflict": float("nan"),
        "slope_mean_noconflict": float("nan"),
        "delta_mean": float("nan"),
        "mw_p": 1.0,
        "n_conflict": float(len(x)),
        "n_noconflict": float(len(y)),
    }
    if len(x) == 0 or len(y) == 0:
        return nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        except ValueError:
            return nan
    return {
        "slope_mean_conflict": float(x.mean()),
        "slope_mean_noconflict": float(y.mean()),
        "delta_mean": float(x.mean() - y.mean()),
        "mw_p": float(p),
        "n_conflict": float(len(x)),
        "n_noconflict": float(len(y)),
    }


@beartype
def compute_trajectories(
    data: ModelAttentionData, metric: str = "entropy"
) -> dict[str, Any]:
    """Compute temporal trajectory features across all (layer, head) cells.

    If there are fewer than 2 T-positions available (e.g. standard model with
    only P0/P2 extracted) this returns ``{"available": False, "reason": ...}``
    and the rest of the report should just skip trajectory analysis.

    Returns
    -------
    ``{
        "available": True,
        "metric": "entropy",
        "t_positions": list[str],
        "n_t": int,
        "per_head": list of per-(layer, head) dicts with slope/mean/delta/p,
        "mean_trajectory_conflict": (n_layers, n_heads, n_t) nested lists,
        "mean_trajectory_noconflict": same,
    }``
    """
    if metric not in data.metrics:
        return {
            "available": False,
            "reason": f"metric {metric!r} not loaded",
        }

    t_labels, t_indices = select_t_positions(data.selected_positions)
    if len(t_labels) < 2:
        return {
            "available": False,
            "reason": (
                f"need >=2 T-positions to compute trajectories; "
                f"have {t_labels}"
            ),
            "metric": metric,
            "t_positions": t_labels,
        }

    arr = data.metrics[metric][..., t_indices]  # (N, L, H, nT)
    n_problems, n_layers, n_heads, n_t = arr.shape
    conflict = data.conflict

    # Per-head statistics
    per_head: list[dict[str, Any]] = []
    # Vectorize across (L, H): compute slope per (L, H, problem) in one pass.
    # Flatten (L, H) for a single vectorized slope calculation.
    flat = arr.transpose(1, 2, 0, 3).reshape(n_layers * n_heads, n_problems, n_t)
    # Shape: (L*H, N, nT)
    x = np.linspace(0.0, 1.0, num=n_t, dtype=np.float64)
    x_centered = x - x.mean()
    denom = float(np.sum(x_centered**2))

    y_mean = flat.mean(axis=2, keepdims=True)  # (L*H, N, 1)
    num = np.sum(x_centered[None, None, :] * (flat - y_mean), axis=2)  # (L*H, N)
    slopes = num / denom if denom > 0 else np.zeros_like(num)  # (L*H, N)

    # Per-(L,H) group comparisons.
    for i in range(n_layers * n_heads):
        layer = i // n_heads
        head = i % n_heads
        head_slopes = slopes[i]
        comp = _slope_comparison(head_slopes, conflict)
        per_head.append(
            {
                "layer": int(layer),
                "head": int(head),
                "slope_mean_conflict": comp["slope_mean_conflict"],
                "slope_mean_noconflict": comp["slope_mean_noconflict"],
                "delta_mean": comp["delta_mean"],
                "mw_p": comp["mw_p"],
            }
        )

    # Population-level trajectory: mean over problems within each condition.
    # Shape: (L, H, nT)
    if conflict.any():
        mean_traj_c = arr[conflict].mean(axis=0)
    else:
        mean_traj_c = np.full((n_layers, n_heads, n_t), np.nan, dtype=np.float32)
    if (~conflict).any():
        mean_traj_nc = arr[~conflict].mean(axis=0)
    else:
        mean_traj_nc = np.full((n_layers, n_heads, n_t), np.nan, dtype=np.float32)

    return {
        "available": True,
        "metric": metric,
        "t_positions": t_labels,
        "n_t": int(n_t),
        "per_head": per_head,
        "mean_trajectory_conflict": mean_traj_c.astype(float).tolist(),
        "mean_trajectory_noconflict": mean_traj_nc.astype(float).tolist(),
    }
