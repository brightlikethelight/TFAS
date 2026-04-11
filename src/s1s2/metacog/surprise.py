"""Token-surprise correlation with SAE features.

The metacognitive monitoring hypothesis is that an LLM has an internal
"difficulty detector" — a direction or SAE feature whose activation
tracks how hard the model is finding the current token. We
operationalize "hard" as **token surprise** in bits (the model's own
``-log2 p_t`` over its emitted token), and we look for SAE features
whose per-problem activation correlates with surprise across the
benchmark corpus.

A *difficulty-sensitive feature* is one with Spearman ``rho > 0.3``
between its activation and a per-problem aggregate of token surprise
(by default, mean surprise over the answer span — at the answer
position the model has either committed to S1 or worked through to S2,
so its own uncertainty is the cleanest difficulty signal we have).

Multiple comparisons across (feature, layer) combinations are corrected
with Benjamini-Hochberg FDR per :func:`s1s2.utils.stats.bh_fdr`.

Why Spearman, not Pearson
-------------------------
SAE codes are heavily zero-inflated and right-skewed; Pearson
correlations conflate that long tail with signal. Spearman is robust to
ties (the many zeros) and to monotone non-linearities, which is exactly
what we want when the question is "do these two quantities co-vary,"
not "do they vary linearly."

Aggregation over the trace
--------------------------
The HDF5 cache stores per-position surprises (``/by_position``) and an
optional full-trace ragged array (``/full_trace_offsets`` /
``/full_trace_values``). For the headline analysis we aggregate the
full trace into a per-problem scalar. The aggregation is configurable:

- ``"mean_full"`` — mean surprise over the entire generation
- ``"max_full"`` — max surprise (the single most-uncertain token)
- ``"mean_answer"`` — mean over the final-answer slice (P2-only)
- ``"top_decile"`` — mean of the top-10% most-surprising tokens
- ``"by_position:<label>"`` — exact value at one canonical position

The default is ``"mean_full"`` because it's most stable to noise while
still tracking the signal we care about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from beartype import beartype
from jaxtyping import Float
from scipy import stats

from s1s2.utils.logging import get_logger
from s1s2.utils.stats import bh_fdr

logger = get_logger("s1s2.metacog")


SurpriseAggregation = Literal[
    "mean_full",
    "max_full",
    "top_decile_full",
    "mean_answer",
    "by_position:P0",
    "by_position:P2",
    "by_position:T0",
    "by_position:T25",
    "by_position:T50",
    "by_position:T75",
    "by_position:Tend",
    "by_position:Tswitch",
]


# ---------------------------------------------------------------------------
# Surprise aggregation
# ---------------------------------------------------------------------------


@beartype
def aggregate_surprise(
    by_position: Float[np.ndarray, "n_problems n_positions"],
    position_labels: list[str],
    full_trace_offsets: np.ndarray | None = None,
    full_trace_values: np.ndarray | None = None,
    method: str = "mean_full",
) -> Float[np.ndarray, "n_problems"]:
    """Reduce per-token / per-position surprise to one scalar per problem.

    The ``by_position`` array is always available; the ragged full trace
    is only available if extraction was run with ``full_trace=True``. We
    silently fall back to ``by_position[:, P2]`` when a full-trace
    method is requested but the trace isn't on disk — the caller can
    inspect ``method_used`` if it cares (logged at INFO).

    The shape contract on the return value is ``(n_problems,)``,
    float32, with NaNs replaced by 0.0 (a problem with no surprise
    information is treated as "not difficult", which is a conservative
    bias for the downstream Spearman correlation).
    """

    n_problems = by_position.shape[0]
    out = np.zeros(n_problems, dtype=np.float32)

    if method.startswith("by_position:"):
        label = method.split(":", 1)[1]
        if label not in position_labels:
            raise KeyError(f"position {label!r} not in {position_labels}")
        idx = position_labels.index(label)
        out[:] = by_position[:, idx].astype(np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    has_trace = full_trace_offsets is not None and full_trace_values is not None
    if not has_trace:
        # Fall back to averaging over canonical positions where valid
        # (skipping zero entries which correspond to invalid positions)
        logger.info(
            "aggregate_surprise: no ragged trace on disk; "
            "falling back to mean over by_position rows"
        )
        # Treat exactly-zero entries as "missing" — they correspond to
        # positions that weren't valid for the problem (the extractor
        # writes 0 for invalid).
        masked = np.where(by_position > 0, by_position, np.nan)
        with np.errstate(invalid="ignore"):
            if method == "max_full":
                out[:] = np.nanmax(masked, axis=1)
            else:
                out[:] = np.nanmean(masked, axis=1)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return out

    offsets = np.asarray(full_trace_offsets, dtype=np.int64)
    values = np.asarray(full_trace_values, dtype=np.float32)
    if offsets.shape[0] != n_problems + 1:
        raise ValueError(
            f"full_trace_offsets has length {offsets.shape[0]}; "
            f"expected n_problems+1 = {n_problems + 1}"
        )

    for i in range(n_problems):
        start, end = int(offsets[i]), int(offsets[i + 1])
        if end <= start:
            continue
        chunk = values[start:end]
        if chunk.size == 0:
            continue
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size == 0:
            continue
        if method == "mean_full":
            out[i] = float(chunk.mean())
        elif method == "max_full":
            out[i] = float(chunk.max())
        elif method == "top_decile_full":
            k = max(1, int(np.ceil(0.1 * chunk.size)))
            # Top-k by magnitude.
            top = np.partition(chunk, -k)[-k:]
            out[i] = float(top.mean())
        elif method == "mean_answer":
            # Approximate "answer span" as the last 20% of the trace —
            # without explicit char/token spans on the trace, this is
            # the best we can do. Reasoning models will mostly be
            # measuring post-</think> tokens.
            k = max(1, int(np.ceil(0.2 * chunk.size)))
            tail = chunk[-k:]
            out[i] = float(tail.mean())
        else:
            raise ValueError(
                f"unknown surprise aggregation method {method!r}; "
                "valid: mean_full, max_full, top_decile_full, mean_answer, by_position:<label>"
            )

    return out


# ---------------------------------------------------------------------------
# Correlation result container
# ---------------------------------------------------------------------------


@dataclass
class SurpriseCorrelationResult:
    """Output of :func:`surprise_feature_correlation` over all features.

    The ``df`` carries one row per feature with the columns:

    - ``feature_id``: SAE feature index
    - ``rho``: Spearman correlation between activation and surprise
    - ``p_value``: raw two-sided p-value
    - ``q_value``: BH-adjusted q-value across all features in this layer
    - ``mean_activation``: mean activation across the corpus
    - ``frac_active``: fraction of problems where the feature was non-zero
    - ``is_difficulty_sensitive``: ``rho > rho_threshold`` AND
      ``q_value <= fdr_q``
    - ``is_constant``: feature never moved (excluded from FDR pool)

    A "difficulty-sensitive" feature is the candidate set for further
    metacognitive analysis (specificity, confidently-wrong, causal).
    """

    df: pd.DataFrame
    n_problems: int
    aggregation: str
    rho_threshold: float
    fdr_q: float
    layer: int
    model_key: str

    @property
    def difficulty_sensitive_ids(self) -> np.ndarray:
        return self.df.loc[self.df["is_difficulty_sensitive"], "feature_id"].to_numpy()

    @property
    def n_difficulty_sensitive(self) -> int:
        return int(self.df["is_difficulty_sensitive"].sum())


# ---------------------------------------------------------------------------
# Core correlation
# ---------------------------------------------------------------------------


@beartype
def surprise_feature_correlation(
    feature_activations: Float[np.ndarray, "n_problems n_features"],
    surprise: Float[np.ndarray, "n_problems"],
    *,
    rho_threshold: float = 0.3,
    fdr_q: float = 0.05,
    layer: int = -1,
    model_key: str = "",
    aggregation: str = "mean_full",
    feature_ids: np.ndarray | None = None,
) -> SurpriseCorrelationResult:
    """Spearman-correlate every SAE feature with per-problem surprise.

    The shape of ``feature_activations`` is ``(n_problems, n_features)``;
    we test the column corresponding to each feature against the
    ``(n_problems,)`` surprise vector.

    Constant columns (which would produce NaN p-values from
    ``scipy.stats.spearmanr``) are explicitly skipped: they get
    ``p_value=1`` and ``rho=0``, and are excluded from the BH-FDR
    pool so they don't drag down the significance threshold for
    everything else. The ``is_constant`` boolean column lets the
    caller see exactly how many were skipped.
    """

    if feature_activations.shape[0] != surprise.shape[0]:
        raise ValueError(
            f"problem-axis mismatch: features={feature_activations.shape[0]} "
            f"vs surprise={surprise.shape[0]}"
        )

    n_problems, n_features = feature_activations.shape

    rhos = np.zeros(n_features, dtype=np.float64)
    p_values = np.ones(n_features, dtype=np.float64)
    mean_activation = feature_activations.mean(axis=0).astype(np.float64)
    frac_active = (feature_activations > 1e-8).mean(axis=0).astype(np.float64)

    col_max = feature_activations.max(axis=0)
    col_min = feature_activations.min(axis=0)
    is_constant = col_max == col_min

    surprise_var = float(surprise.var())
    if surprise_var == 0.0:
        logger.warning(
            "surprise vector is constant — no feature can correlate with it. "
            "Returning trivial result."
        )

    # spearmanr supports a vectorized 2D mode if we feed it a single
    # column of surprise stacked next to the feature matrix, but the
    # vectorized mode doesn't accept "skip constant columns", so a
    # straight Python loop with the per-feature path is both simpler
    # and less error-prone.
    for i in range(n_features):
        if is_constant[i] or surprise_var == 0.0:
            rhos[i] = 0.0
            p_values[i] = 1.0
            continue
        try:
            rho, p = stats.spearmanr(feature_activations[:, i], surprise)
        except Exception:  # pragma: no cover - degenerate input
            rho, p = 0.0, 1.0
        if np.isnan(rho) or np.isnan(p):
            rho, p = 0.0, 1.0
        rhos[i] = float(rho)
        p_values[i] = float(p)

    # BH-FDR over the non-constant pool only.
    q_values = np.ones(n_features, dtype=np.float64)
    pool_mask = ~is_constant
    if pool_mask.any() and surprise_var > 0.0:
        rejected_pool, q_pool = bh_fdr(p_values[pool_mask], q=fdr_q)
        q_values[pool_mask] = q_pool

    if feature_ids is None:
        ids = np.arange(n_features, dtype=np.int64)
    else:
        ids = np.asarray(feature_ids, dtype=np.int64)
        if ids.shape[0] != n_features:
            raise ValueError(f"feature_ids length {ids.shape[0]} != n_features {n_features}")

    is_difficulty_sensitive = (rhos > rho_threshold) & (q_values <= fdr_q) & (~is_constant)

    df = pd.DataFrame(
        {
            "feature_id": ids,
            "rho": rhos,
            "p_value": p_values,
            "q_value": q_values,
            "mean_activation": mean_activation,
            "frac_active": frac_active,
            "is_constant": is_constant,
            "is_difficulty_sensitive": is_difficulty_sensitive,
        }
    )

    n_diff = int(is_difficulty_sensitive.sum())
    logger.info(
        "[%s/L%02d] surprise-feature corr: n_problems=%d n_features=%d "
        "n_constant=%d n_difficulty_sensitive=%d (rho>%.2f, q<=%.3f)",
        model_key or "model",
        layer,
        n_problems,
        n_features,
        int(is_constant.sum()),
        n_diff,
        rho_threshold,
        fdr_q,
    )

    return SurpriseCorrelationResult(
        df=df,
        n_problems=n_problems,
        aggregation=aggregation,
        rho_threshold=rho_threshold,
        fdr_q=fdr_q,
        layer=layer,
        model_key=model_key,
    )


@beartype
def merge_correlation_results(
    results: list[SurpriseCorrelationResult],
    fdr_q: float = 0.05,
    rho_threshold: float = 0.3,
) -> pd.DataFrame:
    """Concatenate per-layer results and re-apply BH-FDR globally.

    When the metacog scan covers multiple layers, the layer-internal
    BH correction is the right call for *that* layer's significance,
    but to count features for Gate 1 ("at least N features survive
    rho > 0.3 and FDR") we want one global FDR pool. This helper
    rebuilds the global ``q_value`` column on the concatenated dataframe.

    Returns the concatenated dataframe with an additional ``layer``
    column and ``model_key`` column for downstream pivoting.
    """

    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        df = r.df.copy()
        df["layer"] = r.layer
        df["model_key"] = r.model_key
        rows.append(df)
    big = pd.concat(rows, ignore_index=True)

    # Re-do BH on the concatenation, dropping constants from the pool.
    pool = ~big["is_constant"].to_numpy(dtype=bool)
    p = big["p_value"].to_numpy(dtype=np.float64)
    q_global = np.ones_like(p)
    if pool.any():
        _, q_pool = bh_fdr(p[pool], q=fdr_q)
        q_global[pool] = q_pool
    big["q_value_global"] = q_global
    big["is_difficulty_sensitive_global"] = (
        (big["rho"].to_numpy() > rho_threshold) & (q_global <= fdr_q) & pool
    )
    return big


__all__ = [
    "SurpriseAggregation",
    "SurpriseCorrelationResult",
    "aggregate_surprise",
    "merge_correlation_results",
    "surprise_feature_correlation",
]
