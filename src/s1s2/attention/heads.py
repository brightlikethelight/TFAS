"""Per-head differential testing and multi-metric consensus classification.

For each (layer, head, position, metric), we run a Mann-Whitney U test
comparing the metric's distribution between conflict (S1-eliciting) and
no-conflict items. Effect sizes are rank-biserial correlation (non-parametric)
and Cohen's d (for comparability with the parametric literature). BH-FDR
correction is applied **jointly** over heads x metrics x positions inside
:mod:`s1s2.attention.core`.

A head is classified as S2-specialized (following Fartale et al., 2025) iff:

1. It is significant in at least ``min_significant`` metrics, AND
2. Effect size |r_rb| on entropy meets ``entropy_effect_threshold`` (medium),
   AND
3. The direction (sign of the effect, mapped through
   :data:`s1s2.attention.core.METRIC_DIRECTIONS`) is consistent across all
   significant metrics and points to S2.

Same criteria with flipped sign define S1-specialized. The "mixed" bucket
covers heads with 3+ significant metrics but inconsistent directions (often
heads that show unusual patterns at some positions).

GQA non-independence: query heads in the same KV group share the key/value
projection, so their attention distributions are correlated. We additionally
report a KV-group aggregate where we median-pool metric values across heads
in the same group and rerun the tests. This is the more conservative report.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from beartype import beartype
from jaxtyping import Bool, Float
from scipy import stats

from s1s2.attention.core import (
    METRIC_DIRECTIONS,
    METRIC_NAMES,
    ModelAttentionData,
)
from s1s2.utils.logging import get_logger
from s1s2.utils.stats import cohens_d, rank_biserial

__all__ = [
    "HeadClassification",
    "classify_heads",
    "differential_test_vector",
    "head_classifications_to_records",
    "kv_group_classify",
    "kv_group_median_pool",
    "run_all_head_differential_tests",
    "run_head_differential_tests_for_metric",
]

logger = get_logger(__name__)

Classification = Literal[
    "S2_specialized",
    "S1_specialized",
    "mixed",
    "non_specialized",
]


# --------------------------------------------------------------------------- #
# Classification dataclass                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class HeadClassification:
    """Summary of one head's specialization.

    Attributes
    ----------
    layer, head : int
        Layer index and query-head index.
    classification : one of the four labels
    n_significant_metrics : number of metrics where ``significant`` after FDR
    entropy_effect : rank-biserial effect size on entropy (signed; positive
        means S2-direction for entropy, i.e. higher conflict entropy)
    entropy_p : raw p-value on entropy (before FDR)
    metric_results : per-metric dict of ``{p_value, q_value, effect_size_rb,
        cohens_d, direction, significant}``. ``direction`` is +1 if the effect
        points to S2 for that metric (using :data:`METRIC_DIRECTIONS`), -1 if
        S1, 0 if no effect.
    position : position label the classification was computed at (we pick
        the position with the strongest per-head evidence; see
        :func:`classify_heads` for how).
    kv_group : KV-group index this head belongs to (Llama: head // 4).
    """

    layer: int
    head: int
    classification: Classification
    n_significant_metrics: int
    entropy_effect: float
    entropy_p: float
    position: str
    kv_group: int
    metric_results: dict[str, dict[str, float]] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Per-vector differential test                                                #
# --------------------------------------------------------------------------- #


@beartype
def differential_test_vector(
    values: Float[np.ndarray, "n"],
    conflict: Bool[np.ndarray, "n"],
) -> dict[str, float]:
    """Run a single Mann-Whitney U test + effect sizes.

    Parameters
    ----------
    values : one scalar per problem
    conflict : same length, True = S1-eliciting problem

    Returns
    -------
    dict with ``{p_value, u_statistic, effect_size_rb, cohens_d,
    mean_conflict, mean_noconflict, median_conflict, median_noconflict,
    n_conflict, n_noconflict}``. Returns NaNs and p=1.0 if either group is
    empty or if both groups are constant (Mann-Whitney undefined).
    """
    x = values[conflict]
    y = values[~conflict]
    n_x = int(x.size)
    n_y = int(y.size)
    nan_dict: dict[str, float] = {
        "p_value": 1.0,
        "u_statistic": float("nan"),
        "effect_size_rb": 0.0,
        "cohens_d": 0.0,
        "mean_conflict": float(np.nan if n_x == 0 else np.mean(x)),
        "mean_noconflict": float(np.nan if n_y == 0 else np.mean(y)),
        "median_conflict": float(np.nan if n_x == 0 else np.median(x)),
        "median_noconflict": float(np.nan if n_y == 0 else np.median(y)),
        "n_conflict": float(n_x),
        "n_noconflict": float(n_y),
    }
    if n_x == 0 or n_y == 0:
        return nan_dict

    x_f = x.astype(np.float64, copy=False)
    y_f = y.astype(np.float64, copy=False)

    # scipy warns when there are ties; we do not care — U is still valid.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            u_stat, p_value = stats.mannwhitneyu(x_f, y_f, alternative="two-sided")
        except ValueError:
            # All values identical → U test undefined.
            return nan_dict

    # rank_biserial uses mannwhitneyu internally, so this will succeed.
    rb = rank_biserial(x_f, y_f)
    d = cohens_d(x_f, y_f)

    return {
        "p_value": float(p_value),
        "u_statistic": float(u_stat),
        "effect_size_rb": float(rb),
        "cohens_d": float(d),
        "mean_conflict": float(np.mean(x_f)),
        "mean_noconflict": float(np.mean(y_f)),
        "median_conflict": float(np.median(x_f)),
        "median_noconflict": float(np.median(y_f)),
        "n_conflict": float(n_x),
        "n_noconflict": float(n_y),
    }


# --------------------------------------------------------------------------- #
# Run tests across a (n_problems, L, H, P) metric tensor                      #
# --------------------------------------------------------------------------- #


@beartype
def run_head_differential_tests_for_metric(
    metric_array: Float[np.ndarray, "n_problems n_layers n_heads n_positions"],
    conflict: Bool[np.ndarray, "n_problems"],
    positions: list[str],
    metric_name: str,
    group_size: int,
) -> pd.DataFrame:
    """Run per-head differential tests for one metric, across all layers/positions.

    Vectorizes the test loop: we compute medians, means, and the U statistic
    in bulk over ``(layer, head, position)`` rather than scalar-testing each
    head. scipy's ``mannwhitneyu`` supports array inputs via ``axis``, which
    gives a ~100x speedup vs the Python for-loop on Llama-scale (5k tests).

    Returns a long-format dataframe with columns::

        layer, head, kv_group, position, metric, p_value, u_statistic,
        effect_size_rb, cohens_d, mean_conflict, mean_noconflict,
        median_conflict, median_noconflict, n_conflict, n_noconflict, direction

    ``direction`` is +1 / 0 / -1 indicating whether the observed effect is
    in the S2 direction for this metric (using
    :data:`s1s2.attention.core.METRIC_DIRECTIONS`).
    """
    n_problems, n_layers, n_heads, n_positions = metric_array.shape
    if conflict.shape != (n_problems,):
        raise ValueError(
            f"conflict shape {conflict.shape} != (n_problems={n_problems},)"
        )
    if len(positions) != n_positions:
        raise ValueError(
            f"len(positions)={len(positions)} != n_positions={n_positions}"
        )
    if metric_name not in METRIC_DIRECTIONS:
        raise KeyError(f"unknown metric {metric_name!r}")

    n_c = int(conflict.sum())
    n_nc = int((~conflict).sum())
    if n_c == 0 or n_nc == 0:
        # Return empty frame in that case — nothing to test.
        return pd.DataFrame(
            columns=[
                "layer", "head", "kv_group", "position", "metric",
                "p_value", "u_statistic", "effect_size_rb", "cohens_d",
                "mean_conflict", "mean_noconflict",
                "median_conflict", "median_noconflict",
                "n_conflict", "n_noconflict", "direction",
            ]
        )

    x = metric_array[conflict].astype(np.float64, copy=False)  # (n_c, L, H, P)
    y = metric_array[~conflict].astype(np.float64, copy=False)  # (n_nc, L, H, P)

    # Bulk Mann-Whitney U: scipy accepts (n, ...) with axis=0, outputting
    # shape (...,) for both u and p.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            u_stat, p_value = stats.mannwhitneyu(
                x, y, alternative="two-sided", axis=0
            )
        except ValueError:
            u_stat = np.full((n_layers, n_heads, n_positions), np.nan)
            p_value = np.ones((n_layers, n_heads, n_positions), dtype=np.float64)

    # rank-biserial: r = 1 - 2U/(n_x * n_y). Sign: positive means x stochastically
    # larger than y (i.e. conflict > no-conflict).
    rb = 1.0 - (2.0 * u_stat) / (n_c * n_nc)

    # Cohen's d vectorized. ddof=1 to match scalar helper.
    mean_x = x.mean(axis=0)
    mean_y = y.mean(axis=0)
    var_x = x.var(axis=0, ddof=1) if n_c > 1 else np.zeros_like(mean_x)
    var_y = y.var(axis=0, ddof=1) if n_nc > 1 else np.zeros_like(mean_y)
    pooled_var = ((n_c - 1) * var_x + (n_nc - 1) * var_y) / max(n_c + n_nc - 2, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(pooled_var > 0, (mean_x - mean_y) / np.sqrt(pooled_var), 0.0)

    # Replace NaNs in U/p from constant-column edge cases: treat as "no effect".
    u_stat = np.where(np.isnan(u_stat), 0.5 * n_c * n_nc, u_stat)
    p_value = np.where(np.isnan(p_value), 1.0, p_value)
    rb = np.where(np.isnan(rb), 0.0, rb)

    median_x = np.median(x, axis=0)
    median_y = np.median(y, axis=0)

    # Flatten to a long-format dataframe.
    L, H, P = n_layers, n_heads, n_positions
    layer_grid, head_grid, pos_grid = np.meshgrid(
        np.arange(L), np.arange(H), np.arange(P), indexing="ij"
    )

    direction_sign = METRIC_DIRECTIONS[metric_name]
    # The observed effect is in the S2 direction if sign(mean_x - mean_y) == direction_sign.
    effect_sign_obs = np.sign(mean_x - mean_y).astype(np.int8)
    direction = effect_sign_obs * direction_sign  # +1 = S2-dir, -1 = S1-dir, 0 = flat

    kv_groups = head_grid // max(group_size, 1)
    positions_arr = np.array(positions)[pos_grid]

    df = pd.DataFrame(
        {
            "layer": layer_grid.ravel().astype(np.int32),
            "head": head_grid.ravel().astype(np.int32),
            "kv_group": kv_groups.ravel().astype(np.int32),
            "position": positions_arr.ravel(),
            "metric": metric_name,
            "p_value": p_value.ravel().astype(np.float64),
            "u_statistic": u_stat.ravel().astype(np.float64),
            "effect_size_rb": rb.ravel().astype(np.float64),
            "cohens_d": d.ravel().astype(np.float64),
            "mean_conflict": mean_x.ravel().astype(np.float64),
            "mean_noconflict": mean_y.ravel().astype(np.float64),
            "median_conflict": median_x.ravel().astype(np.float64),
            "median_noconflict": median_y.ravel().astype(np.float64),
            "n_conflict": float(n_c),
            "n_noconflict": float(n_nc),
            "direction": direction.ravel().astype(np.int8),
        }
    )
    return df


@beartype
def run_all_head_differential_tests(
    data: ModelAttentionData,
    metrics: tuple[str, ...] = METRIC_NAMES,
) -> pd.DataFrame:
    """Run differential tests for all metrics on a :class:`ModelAttentionData`.

    Returns a single concatenated long-format dataframe.
    """
    parts: list[pd.DataFrame] = []
    for m in metrics:
        if m not in data.metrics:
            logger.debug("metric %s not loaded; skipping", m)
            continue
        part = run_head_differential_tests_for_metric(
            metric_array=data.metrics[m],
            conflict=data.conflict,
            positions=list(data.selected_positions),
            metric_name=m,
            group_size=data.group_size,
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=0, ignore_index=True)


# --------------------------------------------------------------------------- #
# Multi-metric consensus head classification                                  #
# --------------------------------------------------------------------------- #


@beartype
def _choose_representative_position(
    sub: pd.DataFrame,
) -> str:
    """Pick the position with the strongest entropy evidence for this head.

    "Strongest" = smallest entropy q-value at this head. Ties broken by
    largest |entropy_effect|. If the head has no entropy rows (shouldn't
    happen but is defensive), falls back to the first row's position.
    """
    ent = sub[sub["metric"] == "entropy"]
    if ent.empty:
        return str(sub.iloc[0]["position"])
    # Smallest q, then largest |effect|.
    ent_sorted = ent.assign(
        abs_rb=ent["effect_size_rb"].abs()
    ).sort_values(
        by=["q_value", "abs_rb"], ascending=[True, False]
    )
    return str(ent_sorted.iloc[0]["position"])


@beartype
def classify_heads(
    df: pd.DataFrame,
    n_layers: int,
    n_heads: int,
    min_significant: int = 3,
    entropy_effect_threshold: float = 0.3,
) -> list[HeadClassification]:
    """Apply the Fartale et al. (2025) multi-metric consensus rule.

    Iterates over every (layer, head) cell. For that head, picks the most
    informative position (smallest entropy q-value) and reads one row per
    metric at that position. Then applies the 3-of-5 + |r_rb|>=0.3 + consistent
    direction rule.

    ``df`` is expected to have ``q_value`` and ``significant`` columns already
    attached (via :func:`s1s2.attention.core._apply_bh_in_place`).
    """
    if df.empty:
        return []
    required = {
        "layer", "head", "position", "metric", "p_value", "q_value",
        "significant", "effect_size_rb", "direction",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns {missing}")

    group_size = _infer_group_size(df)

    results: list[HeadClassification] = []
    # Pre-index for speed.
    grouped = df.groupby(["layer", "head"], sort=True)
    for (layer, head), sub in grouped:
        # Pick representative position for this head (smallest entropy q)
        rep_pos = _choose_representative_position(sub)
        sub_at_pos = sub[sub["position"] == rep_pos]

        per_metric: dict[str, dict[str, float]] = {}
        entropy_effect = 0.0
        entropy_p = 1.0
        entropy_q = 1.0
        for _, row in sub_at_pos.iterrows():
            m = str(row["metric"])
            per_metric[m] = {
                "p_value": float(row["p_value"]),
                "q_value": float(row["q_value"]),
                "effect_size_rb": float(row["effect_size_rb"]),
                "cohens_d": float(row["cohens_d"]) if "cohens_d" in row else float("nan"),
                "direction": int(row["direction"]),
                "significant": bool(row["significant"]),
            }
            if m == "entropy":
                entropy_effect = float(row["effect_size_rb"])
                entropy_p = float(row["p_value"])
                entropy_q = float(row["q_value"])

        # Count significant metrics at this position
        sig_metrics = [m for m, mr in per_metric.items() if mr["significant"]]
        n_sig = len(sig_metrics)

        # Consensus criteria:
        #  1. >= min_significant significant metrics
        #  2. |entropy effect size| >= threshold
        #  3. All significant metrics point the same way (S1 or S2)
        classification: Classification = "non_specialized"
        if n_sig >= min_significant:
            dirs = [per_metric[m]["direction"] for m in sig_metrics]
            all_s2 = all(d == +1 for d in dirs)
            all_s1 = all(d == -1 for d in dirs)
            if abs(entropy_effect) >= entropy_effect_threshold:
                if all_s2:
                    classification = "S2_specialized"
                elif all_s1:
                    classification = "S1_specialized"
                else:
                    classification = "mixed"
            else:
                # 3+ significant but entropy effect too small → mixed
                classification = "mixed"

        results.append(
            HeadClassification(
                layer=int(layer),
                head=int(head),
                classification=classification,
                n_significant_metrics=int(n_sig),
                entropy_effect=float(entropy_effect),
                entropy_p=float(entropy_p),
                position=rep_pos,
                kv_group=int(head) // max(group_size, 1),
                metric_results=per_metric,
            )
        )
    return results


def _infer_group_size(df: pd.DataFrame) -> int:
    """Infer GQA group size from the df's ``head`` / ``kv_group`` columns."""
    if "kv_group" not in df.columns:
        return 1
    # group_size = unique head counts per kv_group, assume uniform
    first_group = df[df["kv_group"] == df["kv_group"].iloc[0]]
    return max(int(first_group["head"].nunique()), 1)


@beartype
def head_classifications_to_records(
    classifs: list[HeadClassification],
) -> list[dict[str, Any]]:
    """Convert a list of :class:`HeadClassification` to JSON-serializable dicts."""
    return [asdict(c) for c in classifs]


# --------------------------------------------------------------------------- #
# KV-group aggregate (more conservative for GQA)                              #
# --------------------------------------------------------------------------- #


@beartype
def kv_group_median_pool(
    metric_array: Float[np.ndarray, "n_problems n_layers n_heads n_positions"],
    group_size: int,
) -> Float[np.ndarray, "n_problems n_layers n_kv_groups n_positions"]:
    """Collapse query heads sharing a KV group via median.

    Median (rather than mean) is used because it's more robust to one
    outlier query head in a group, which matches the "conservative"
    framing of the KV-group analysis.
    """
    if group_size <= 1:
        return metric_array
    n_problems, n_layers, n_heads, n_positions = metric_array.shape
    if n_heads % group_size != 0:
        raise ValueError(
            f"n_heads={n_heads} not divisible by group_size={group_size}"
        )
    n_groups = n_heads // group_size
    reshaped = metric_array.reshape(n_problems, n_layers, n_groups, group_size, n_positions)
    return np.median(reshaped, axis=3)


@beartype
def kv_group_classify(
    data: ModelAttentionData,
    metrics: tuple[str, ...] = METRIC_NAMES,
    q: float = 0.05,
    min_significant: int = 3,
    entropy_effect_threshold: float = 0.3,
) -> list[HeadClassification]:
    """Run the full classification at KV-group granularity.

    For each metric, we collapse query heads in the same KV group by median,
    then rerun the per-head pipeline (tests + consensus) treating the pooled
    groups as "heads" of count ``n_kv_heads``. The returned
    :class:`HeadClassification` objects have ``head`` set to the KV-group
    index and ``kv_group`` set identically (so downstream code can treat the
    records uniformly).
    """
    group_size = data.group_size
    if group_size <= 1:
        # No pooling needed; just return the per-head classification.
        from s1s2.attention.core import _apply_bh_in_place

        df = run_all_head_differential_tests(data, metrics=metrics)
        df = _apply_bh_in_place(df, q=q)
        return classify_heads(
            df,
            n_layers=data.n_layers,
            n_heads=data.n_heads,
            min_significant=min_significant,
            entropy_effect_threshold=entropy_effect_threshold,
        )

    pooled_metrics: dict[str, np.ndarray] = {}
    for m in metrics:
        if m not in data.metrics:
            continue
        pooled_metrics[m] = kv_group_median_pool(data.metrics[m], group_size=group_size)

    # Build a temporary dataframe of tests at KV-group granularity.
    parts: list[pd.DataFrame] = []
    for m, arr in pooled_metrics.items():
        part = run_head_differential_tests_for_metric(
            metric_array=arr,
            conflict=data.conflict,
            positions=list(data.selected_positions),
            metric_name=m,
            group_size=1,  # already pooled → each "head" is its own group
        )
        parts.append(part)
    if not parts:
        return []
    df = pd.concat(parts, axis=0, ignore_index=True)

    # Apply BH-FDR jointly across all KV-group tests
    from s1s2.attention.core import _apply_bh_in_place

    df = _apply_bh_in_place(df, q=q)

    classifs = classify_heads(
        df,
        n_layers=data.n_layers,
        n_heads=data.n_kv_heads,
        min_significant=min_significant,
        entropy_effect_threshold=entropy_effect_threshold,
    )
    # Annotate the records as KV-group-scope: head index == kv_group index.
    for c in classifs:
        c.kv_group = int(c.head)
    return classifs
