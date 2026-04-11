"""Differential activation analysis on SAE feature activations.

For every feature ``i`` in the loaded SAE we ask: does activation on
conflict (S1-eliciting) problems differ from activation on matched
no-conflict controls? The test is Mann-Whitney U (non-parametric,
no normality assumption, robust to the ReLU-induced spikes at zero
that SAE codes exhibit), effect size is the rank-biserial correlation,
and p-values are corrected across all features with Benjamini-Hochberg
at q=0.05.

Per CLAUDE.md we ALSO compute the analysis on the matched-difficulty
subset to control for the difficulty confound. The matched-subset
result is the one to quote if the naive and matched disagree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from beartype import beartype
from jaxtyping import Bool, Float
from scipy import stats

from s1s2.sae.loaders import SAEHandle
from s1s2.utils.logging import get_logger
from s1s2.utils.stats import bh_fdr

logger = get_logger("s1s2.sae")


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class DifferentialResult:
    """Output of one differential-analysis pass on all SAE features.

    ``df`` has one row per feature and columns:

    - ``feature_id``: int SAE feature index
    - ``mean_S1``, ``mean_S2``: mean feature activation in each group
    - ``log_fc``: base-2 log fold change, with epsilon smoothing
    - ``effect_size``: rank-biserial correlation (positive => S1 > S2)
    - ``u_stat``: Mann-Whitney U statistic
    - ``p_value``: raw two-sided p-value
    - ``q_value``: BH-adjusted q-value
    - ``significant``: ``q_value <= fdr_q``

    ``n_S1`` / ``n_S2`` record the sample sizes so downstream readers
    can sanity-check statistical power before quoting a feature as
    significant.
    """

    df: pd.DataFrame
    n_S1: int
    n_S2: int
    fdr_q: float
    subset: str  # "all" or "matched_pairs"


# ---------------------------------------------------------------------------
# Encoding helper
# ---------------------------------------------------------------------------


@beartype
def encode_batched(
    sae: SAEHandle,
    activations: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Run activations through the SAE encoder in small batches.

    Returns a float32 ``(n, n_features)`` numpy array. We batch because
    some Gemma Scope checkpoints have ~1M features, and encoding all
    problems at once OOMs even on CPU.
    """
    if activations.ndim != 2:
        raise ValueError(f"expected 2D activations, got shape {activations.shape}")
    n = activations.shape[0]
    out = np.empty((n, sae.n_features), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        x = torch.from_numpy(activations[start:end].astype(np.float32, copy=False))
        with torch.no_grad():
            z = sae.encode(x)
        out[start:end] = z.detach().float().cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


@beartype
def _log_fc(mean_a: np.ndarray, mean_b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Base-2 log fold change with symmetric epsilon smoothing.

    A pure ``log2(a / b)`` blows up when ``b == 0`` (which is extremely
    common on SAE codes with ReLU). Adding ``eps`` to both numerator and
    denominator bounds the fold change to a reasonable range and keeps
    the volcano plot readable.
    """
    return np.log2((mean_a + eps) / (mean_b + eps))


@beartype
def _rank_biserial_from_u(u: float, n_x: int, n_y: int) -> float:
    """Rank-biserial correlation given the Mann-Whitney U statistic.

    ``r_rb = 1 - 2U / (n_x * n_y)``, sign positive when group X is
    stochastically larger. We compute it from U directly rather than
    recomputing the ranks, which is a meaningful speedup given we call
    it once per feature.
    """
    nm = n_x * n_y
    if nm == 0:
        return 0.0
    return float(1.0 - 2.0 * u / nm)


@beartype
def differential_activation(
    feature_activations: Float[np.ndarray, "n_problems n_features"],
    conflict: Bool[np.ndarray, "n_problems"],
    fdr_q: float = 0.05,
    feature_ids: Sequence[int] | None = None,
    subset_label: str = "all",
) -> DifferentialResult:
    """Run Mann-Whitney U + BH-FDR on every SAE feature.

    Parameters
    ----------
    feature_activations
        The SAE-encoded activations, shape ``(n_problems, n_features)``.
        Typically the output of :func:`encode_batched` on the P0 slice.
    conflict
        Boolean array: ``True`` for S1-eliciting items (conflict) and
        ``False`` for the no-conflict controls.
    fdr_q
        FDR target for BH correction across features.
    feature_ids
        Optional explicit feature IDs to report in the dataframe. If
        ``None``, we use ``range(n_features)``. Pass explicit IDs when
        you're running on a pre-selected subset of features.
    subset_label
        String tag written into ``DifferentialResult.subset``. Typical
        values: ``"all"`` and ``"matched_pairs"``.
    """

    if feature_activations.ndim != 2:
        raise ValueError(
            f"expected 2D feature activations, got shape {feature_activations.shape}"
        )
    if feature_activations.shape[0] != conflict.shape[0]:
        raise ValueError(
            f"problem-axis mismatch: {feature_activations.shape[0]} "
            f"vs {conflict.shape[0]}"
        )

    n_problems, n_features = feature_activations.shape
    s1_idx = np.where(conflict)[0]
    s2_idx = np.where(~conflict)[0]
    n_s1, n_s2 = len(s1_idx), len(s2_idx)

    if n_s1 < 3 or n_s2 < 3:
        logger.warning(
            "Differential analysis: very small group sizes (n_S1=%d, n_S2=%d). "
            "Mann-Whitney p-values will be unreliable.",
            n_s1,
            n_s2,
        )

    act_s1 = feature_activations[s1_idx]
    act_s2 = feature_activations[s2_idx]
    mean_s1 = act_s1.mean(axis=0)
    mean_s2 = act_s2.mean(axis=0)
    log_fc = _log_fc(mean_s1, mean_s2)

    p_values = np.ones(n_features, dtype=np.float64)
    u_stats = np.zeros(n_features, dtype=np.float64)
    effect_sizes = np.zeros(n_features, dtype=np.float64)

    # SAE codes are heavily zero-inflated. Features that are *always*
    # zero in both groups produce a degenerate U = 0.5 * n_s1 * n_s2
    # with no statistical meaning. We special-case those to avoid
    # polluting the multiple-comparisons pool.
    col_max = feature_activations.max(axis=0)
    col_min = feature_activations.min(axis=0)
    is_constant = col_max == col_min

    for i in range(n_features):
        if is_constant[i]:
            # Feature never activates or activates identically — degenerate.
            u_stats[i] = 0.5 * n_s1 * n_s2
            p_values[i] = 1.0
            effect_sizes[i] = 0.0
            continue
        try:
            u, p = stats.mannwhitneyu(
                act_s1[:, i], act_s2[:, i], alternative="two-sided"
            )
        except ValueError:
            u, p = 0.5 * n_s1 * n_s2, 1.0
        u_stats[i] = float(u)
        p_values[i] = float(p)
        effect_sizes[i] = _rank_biserial_from_u(float(u), n_s1, n_s2)

    rejected, q_values = bh_fdr(p_values.astype(np.float64), q=fdr_q)

    if feature_ids is None:
        ids: np.ndarray = np.arange(n_features, dtype=np.int64)
    else:
        ids = np.asarray(list(feature_ids), dtype=np.int64)
        if ids.shape[0] != n_features:
            raise ValueError(
                f"feature_ids length {ids.shape[0]} != n_features {n_features}"
            )

    df = pd.DataFrame(
        {
            "feature_id": ids,
            "mean_S1": mean_s1.astype(np.float64),
            "mean_S2": mean_s2.astype(np.float64),
            "log_fc": log_fc.astype(np.float64),
            "effect_size": effect_sizes,
            "u_stat": u_stats,
            "p_value": p_values,
            "q_value": q_values,
            "significant": rejected.astype(bool),
        }
    )

    n_sig = int(df["significant"].sum())
    logger.info(
        "Differential (%s): n_S1=%d n_S2=%d significant=%d/%d at q<=%.3f",
        subset_label,
        n_s1,
        n_s2,
        n_sig,
        n_features,
        fdr_q,
    )
    return DifferentialResult(
        df=df,
        n_S1=n_s1,
        n_S2=n_s2,
        fdr_q=fdr_q,
        subset=subset_label,
    )


# ---------------------------------------------------------------------------
# Matched-pair subset helpers
# ---------------------------------------------------------------------------


@beartype
def matched_pair_subset(
    feature_activations: Float[np.ndarray, "n_problems n_features"],
    conflict: Bool[np.ndarray, "n_problems"],
    matched_pair_id: np.ndarray,
    correct: Bool[np.ndarray, "n_problems"] | None = None,
) -> tuple[
    Float[np.ndarray, "n_pairs_times_two n_features"],
    Bool[np.ndarray, "n_pairs_times_two"],
]:
    """Restrict activations to conflict/control pairs linked by ``matched_pair_id``.

    The matched-pair contrast is the single cleanest confound control
    we have — each S1 item shares surface form and intended difficulty
    with its S2 control. If ``correct`` is provided, we further restrict
    to conflict items the model answered correctly (the "model resisted
    the lure" sample), matching the prompt-spec.

    Returns a fresh activation matrix and a matching conflict array,
    both re-ordered so every pair sits adjacent. Orphaned rows (IDs
    with only one member of the pair, e.g. items with empty
    ``matched_pair_id``) are dropped.
    """

    # Normalize IDs to hashable byte-strings / Python strings.
    if matched_pair_id.dtype == object:
        ids = np.asarray(
            [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
             for s in matched_pair_id]
        )
    else:
        ids = matched_pair_id.astype(str)

    # Group row indices by matched_pair_id.
    by_pair: dict[str, dict[str, int]] = {}
    for row_idx, pid in enumerate(ids):
        if pid == "" or pid == "b''":
            continue
        kind = "S1" if conflict[row_idx] else "S2"
        by_pair.setdefault(pid, {})[kind] = row_idx

    keep_rows: list[int] = []
    keep_is_conflict: list[bool] = []
    for pid, members in by_pair.items():
        if "S1" not in members or "S2" not in members:
            continue
        s1_row = members["S1"]
        s2_row = members["S2"]
        # If caller passed a correctness mask, require S1 item to be answered
        # correctly (= model resisted the lure). The control is always kept.
        if correct is not None and not bool(correct[s1_row]):
            continue
        keep_rows.extend([s1_row, s2_row])
        keep_is_conflict.extend([True, False])

    if len(keep_rows) == 0:
        logger.warning("matched_pair_subset: no pairs remain after filtering.")
        return (
            np.empty((0, feature_activations.shape[1]), dtype=feature_activations.dtype),
            np.empty((0,), dtype=bool),
        )

    rows = np.asarray(keep_rows, dtype=np.int64)
    return feature_activations[rows], np.asarray(keep_is_conflict, dtype=bool)


__all__ = [
    "DifferentialResult",
    "differential_activation",
    "encode_batched",
    "matched_pair_subset",
]
