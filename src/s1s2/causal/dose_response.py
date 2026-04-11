"""Dose-response curve fitting for steering experiments.

For a given (model, layer, feature), the dose-response experiment sweeps
``alpha`` through a fixed grid (e.g. ``[-5, -3, -1, -0.5, 0, 0.5, 1, 3, 5]``)
and measures P(correct) on three item groups:

1. Conflict items (S1 lure present).
2. No-conflict control items (matched on surface form / difficulty).
3. Random-direction control: steered with a random unit vector rather
   than the feature direction.

The causal signature of a real S2-like feature is a monotonic increase
in P(correct) on conflict items with positive ``alpha``, little or no
change on no-conflict items, and near-zero change under random control.

This module holds the aggregation / curve-fit logic. The actual steering
loop lives in :mod:`s1s2.causal.core`; this file is pure post-processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from beartype import beartype
from jaxtyping import Float

from s1s2.utils.logging import get_logger
from s1s2.utils.stats import bootstrap_ci

logger = get_logger("s1s2.causal")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class DoseResponsePoint:
    """One measurement at one alpha on one item group."""

    alpha: float
    group: str  # "conflict" | "no_conflict" | "random_control"
    n: int
    p_correct: float
    ci_lower: float
    ci_upper: float
    # Optional: for random controls, the per-seed breakdown.
    per_seed: list[float] = field(default_factory=list)


@dataclass
class DoseResponseCurve:
    """A complete dose-response curve for one feature at one (model, layer)."""

    model: str
    layer: int
    feature_id: int
    alphas: list[float]
    points: list[DoseResponsePoint]
    fit: dict[str, Any]  # curve-fit parameters

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "layer": int(self.layer),
            "feature_id": int(self.feature_id),
            "alphas": [float(a) for a in self.alphas],
            "points": [
                {
                    "alpha": float(p.alpha),
                    "group": p.group,
                    "n": int(p.n),
                    "p_correct": float(p.p_correct),
                    "ci_lower": float(p.ci_lower),
                    "ci_upper": float(p.ci_upper),
                    "per_seed": [float(x) for x in p.per_seed],
                }
                for p in self.points
            ],
            "fit": _as_py(self.fit),
        }


def _as_py(obj: Any) -> Any:
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_as_py(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _as_py(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_as_py(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Aggregation from per-item correctness
# ---------------------------------------------------------------------------


@beartype
def aggregate_p_correct(
    correct: Float[np.ndarray, "n"] | np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(p_correct, ci_lower, ci_upper)`` for a 0/1 accuracy vector.

    Bootstrap percentile CI matches the rest of the codebase.
    """
    vec = np.asarray(correct, dtype=np.float64).ravel()
    if vec.size == 0:
        return float("nan"), float("nan"), float("nan")
    point, lo, hi = bootstrap_ci(
        vec,
        statistic=lambda v: float(np.mean(v)),
        n_resamples=int(n_bootstrap),
        confidence=0.95,
        seed=int(seed),
    )
    return point, lo, hi


@beartype
def build_curve(
    *,
    model: str,
    layer: int,
    feature_id: int,
    alphas: list[float],
    conflict_correct_by_alpha: dict[float, np.ndarray],
    no_conflict_correct_by_alpha: dict[float, np.ndarray],
    random_correct_by_alpha_seed: dict[float, dict[int, np.ndarray]],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> DoseResponseCurve:
    """Assemble a :class:`DoseResponseCurve` from the per-alpha measurement dicts.

    ``conflict_correct_by_alpha[alpha]`` is a per-problem 0/1 vector of
    correctness on conflict items at that alpha; likewise for
    ``no_conflict_correct_by_alpha``. ``random_correct_by_alpha_seed[alpha][seed]``
    is the same vector for the random-direction control; we average
    across seeds for the point estimate and keep the per-seed values for
    provenance.
    """
    points: list[DoseResponsePoint] = []

    for alpha in alphas:
        # Conflict group.
        vec_c = conflict_correct_by_alpha.get(alpha)
        if vec_c is not None:
            p, lo, hi = aggregate_p_correct(vec_c, n_bootstrap=n_bootstrap, seed=seed)
            points.append(
                DoseResponsePoint(
                    alpha=float(alpha),
                    group="conflict",
                    n=len(vec_c),
                    p_correct=p,
                    ci_lower=lo,
                    ci_upper=hi,
                )
            )

        # No-conflict group.
        vec_nc = no_conflict_correct_by_alpha.get(alpha)
        if vec_nc is not None:
            p, lo, hi = aggregate_p_correct(vec_nc, n_bootstrap=n_bootstrap, seed=seed)
            points.append(
                DoseResponsePoint(
                    alpha=float(alpha),
                    group="no_conflict",
                    n=len(vec_nc),
                    p_correct=p,
                    ci_lower=lo,
                    ci_upper=hi,
                )
            )

        # Random-direction control: mean across seeds of per-problem accuracy.
        seed_map = random_correct_by_alpha_seed.get(alpha, {})
        per_seed_mean: list[float] = []
        if seed_map:
            all_vecs = []
            for _s, vec in sorted(seed_map.items()):
                if vec.size == 0:
                    continue
                per_seed_mean.append(float(np.mean(vec)))
                all_vecs.append(vec)
            if all_vecs:
                combined = np.concatenate(all_vecs).astype(np.float64)
                p, lo, hi = aggregate_p_correct(combined, n_bootstrap=n_bootstrap, seed=seed + 1)
                points.append(
                    DoseResponsePoint(
                        alpha=float(alpha),
                        group="random_control",
                        n=int(combined.size),
                        p_correct=p,
                        ci_lower=lo,
                        ci_upper=hi,
                        per_seed=per_seed_mean,
                    )
                )

    fit = fit_curve(points)

    return DoseResponseCurve(
        model=model,
        layer=int(layer),
        feature_id=int(feature_id),
        alphas=[float(a) for a in alphas],
        points=points,
        fit=fit,
    )


# ---------------------------------------------------------------------------
# Curve fitting & shape checks
# ---------------------------------------------------------------------------


@beartype
def _slope_and_monotonicity(xs: np.ndarray, ys: np.ndarray) -> dict[str, float]:
    """Cheap curve summary: least-squares slope + Spearman-like monotonicity.

    We deliberately avoid fitting sigmoids or similar parametric forms
    because the dose-response grid is sparse (9 points) and the asymptotic
    regime is the least-informative part. Instead we report:

    * ``slope`` — OLS slope of P(correct) on alpha. This is the primary
      effect size.
    * ``r`` — Pearson correlation between alpha and P(correct).
    * ``monotonic_positive`` — fraction of adjacent (alpha_i, alpha_{i+1})
      pairs where the measured P(correct) is non-decreasing.
    """
    if xs.size < 2:
        return {"slope": 0.0, "intercept": 0.0, "r": 0.0, "monotonic_positive": 0.0}

    x_mean = float(xs.mean())
    y_mean = float(ys.mean())
    cov = float(((xs - x_mean) * (ys - y_mean)).mean())
    x_var = float(((xs - x_mean) ** 2).mean())
    y_var = float(((ys - y_mean) ** 2).mean())
    slope = cov / x_var if x_var > 0 else 0.0
    intercept = y_mean - slope * x_mean
    denom = (x_var * y_var) ** 0.5
    r = cov / denom if denom > 0 else 0.0

    # Monotonicity: fraction of adjacent pairs that do not decrease (alpha sorted).
    order = np.argsort(xs)
    ys_sorted = ys[order]
    diffs = np.diff(ys_sorted)
    monotonic_positive = float(np.mean(diffs >= 0)) if diffs.size > 0 else 0.0

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r),
        "monotonic_positive": float(monotonic_positive),
    }


@beartype
def fit_curve(points: list[DoseResponsePoint]) -> dict[str, Any]:
    """Summarise the three curves (conflict / no-conflict / random control).

    We compute per-group slopes and a headline "selectivity" number that
    captures the difference between the conflict-item slope and the
    random-direction slope. A large positive selectivity at positive
    alpha is the signature of a real S2 feature.
    """
    by_group: dict[str, list[DoseResponsePoint]] = {}
    for p in points:
        by_group.setdefault(p.group, []).append(p)

    out: dict[str, Any] = {}
    for group, pts in by_group.items():
        xs = np.array([p.alpha for p in pts], dtype=np.float64)
        ys = np.array([p.p_correct for p in pts], dtype=np.float64)
        out[group] = _slope_and_monotonicity(xs, ys)

    # Headline effect: slope_conflict - slope_random. If the feature steering
    # is doing something the random direction isn't, this is large.
    slope_conflict = float(out.get("conflict", {}).get("slope", 0.0))
    slope_random = float(out.get("random_control", {}).get("slope", 0.0))
    slope_no_conflict = float(out.get("no_conflict", {}).get("slope", 0.0))
    out["selectivity_vs_random"] = float(slope_conflict - slope_random)
    out["selectivity_vs_no_conflict"] = float(slope_conflict - slope_no_conflict)
    return out


# ---------------------------------------------------------------------------
# Signature check
# ---------------------------------------------------------------------------


@beartype
def is_canonical_s2_signature(
    curve: DoseResponseCurve,
    *,
    min_conflict_slope: float = 0.02,
    max_no_conflict_slope_magnitude: float = 0.02,
    max_random_slope_magnitude: float = 0.01,
) -> bool:
    """Return ``True`` iff the curve has a 'real S2 feature' shape.

    The three criteria (all must hold):

    1. Conflict-item slope is clearly positive (>= ``min_conflict_slope``).
    2. No-conflict slope magnitude is small.
    3. Random-direction slope magnitude is small.

    Default thresholds are conservative; tune via the config for specific
    experiments.
    """
    fit = curve.fit
    slope_c = float(fit.get("conflict", {}).get("slope", 0.0))
    slope_nc = abs(float(fit.get("no_conflict", {}).get("slope", 0.0)))
    slope_r = abs(float(fit.get("random_control", {}).get("slope", 0.0)))
    return (
        slope_c >= min_conflict_slope
        and slope_nc <= max_no_conflict_slope_magnitude
        and slope_r <= max_random_slope_magnitude
    )


__all__ = [
    "DoseResponseCurve",
    "DoseResponsePoint",
    "aggregate_p_correct",
    "build_curve",
    "fit_curve",
    "is_canonical_s2_signature",
]
