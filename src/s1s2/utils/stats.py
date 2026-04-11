"""Shared statistical helpers.

Functions here are used across all five workstreams. They implement the
NON-NEGOTIABLE statistical standards from CLAUDE.md:

- Benjamini-Hochberg FDR correction
- Permutation tests with the North et al. (2002) +1 correction
- Bootstrap confidence intervals
- Effect sizes (Cohen's d, rank-biserial correlation, AUC)
- Hewitt & Liang selectivity (computed elsewhere; this just provides primitives)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from beartype import beartype
from jaxtyping import Bool, Float
from scipy import stats

# ----- Multiple comparisons -----


@beartype
def bh_fdr(
    pvalues: Float[np.ndarray, "n"],
    q: float = 0.05,
) -> tuple[Bool[np.ndarray, "n"], Float[np.ndarray, "n"]]:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvalues : array of raw p-values
    q : false discovery rate target (default 0.05)

    Returns
    -------
    rejected : boolean array, True where the null is rejected at FDR q
    qvalues : adjusted p-values

    Notes
    -----
    Uses the standard BH procedure (independence or positive dependence).
    For arbitrary dependence, use Benjamini-Yekutieli, which divides by
    the harmonic number sum_{i=1}^n 1/i. Most LLM probing settings satisfy
    PRDS so BH is appropriate.
    """
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    # qvalues[i] = min over j>=i of (n / (j+1)) * pvalues_sorted[j]
    inv_rank = np.arange(1, n + 1, dtype=np.float64)
    raw = ranked * n / inv_rank
    # Make monotone non-decreasing from the right
    raw = np.minimum.accumulate(raw[::-1])[::-1]
    qvalues = np.empty_like(raw)
    qvalues[order] = np.clip(raw, 0.0, 1.0)
    rejected = qvalues <= q
    return rejected, qvalues


# ----- Permutation tests -----


@beartype
def permutation_test_two_sample(
    x: Float[np.ndarray, "n"],
    y: Float[np.ndarray, "m"],
    statistic: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_permutations: int = 10_000,
    seed: int | None = 0,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Two-sample permutation test with North +1 correction.

    Returns (observed_statistic, p_value).

    Default statistic is the difference in means. The +1 correction
    (Phipson & Smyth 2010 / North et al. 2002) prevents zero p-values:
    ``p = (n_extreme + 1) / (n_perms + 1)``.
    """
    rng = np.random.default_rng(seed)
    if statistic is None:
        def statistic(a, b):  # type: ignore[no-redef]
            return float(np.mean(a) - np.mean(b))

    obs = float(statistic(x, y))
    pooled = np.concatenate([x, y])
    n = len(x)

    perm_stats = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        rng.shuffle(pooled)
        perm_stats[i] = statistic(pooled[:n], pooled[n:])

    if alternative == "two-sided":
        n_extreme = int(np.sum(np.abs(perm_stats) >= abs(obs)))
    elif alternative == "greater":
        n_extreme = int(np.sum(perm_stats >= obs))
    elif alternative == "less":
        n_extreme = int(np.sum(perm_stats <= obs))
    else:
        raise ValueError(f"alternative must be 'two-sided' | 'greater' | 'less'; got {alternative}")

    pvalue = (n_extreme + 1) / (n_permutations + 1)
    return obs, pvalue


# ----- Effect sizes -----


@beartype
def cohens_d(x: Float[np.ndarray, "n"], y: Float[np.ndarray, "m"]) -> float:
    """Cohen's d with pooled standard deviation."""
    nx, ny = len(x), len(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_var = ((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2)
    if pooled_var == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled_var))


@beartype
def rank_biserial(x: Float[np.ndarray, "n"], y: Float[np.ndarray, "m"]) -> float:
    """Rank-biserial correlation from a Mann-Whitney U test.

    Convention: positive means ``x`` stochastically larger than ``y``.

    ``scipy.stats.mannwhitneyu(x, y)`` returns ``U_x``, the count of pairs where
    ``x_i > y_j`` (with ties getting 0.5). When x is stochastically larger this
    count is high, so the rank-biserial in the convention "positive iff x > y"
    is ``2*U_x / (n*m) - 1`` (range ``[-1, 1]``).
    """
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(2.0 * u / (len(x) * len(y)) - 1.0)


# ----- Bootstrap CIs -----


@beartype
def bootstrap_ci(
    data: Float[np.ndarray, "n"],
    statistic: Callable[[np.ndarray], float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """Bootstrap percentile CI.

    Returns (point_estimate, lower, upper) at the requested confidence level.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_stats = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_stats[i] = statistic(data[idx])
    point = float(statistic(data))
    alpha = 1 - confidence
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return point, lower, upper


@beartype
def paired_bootstrap_ci_diff(
    x: Float[np.ndarray, "n"],
    y: Float[np.ndarray, "n"],
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for the difference statistic(x_boot, y_boot).

    Use when comparing two probes / two models on the same problems — the
    pairing matters and shouldn't be broken.
    """
    if len(x) != len(y):
        raise ValueError(f"paired bootstrap requires equal lengths, got {len(x)} vs {len(y)}")
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_stats = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_stats[i] = statistic(x[idx], y[idx])
    point = float(statistic(x, y))
    alpha = 1 - confidence
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return point, lower, upper


# ----- Misc -----


@beartype
def gini_coefficient(values: Float[np.ndarray, "n"]) -> float:
    """Gini coefficient. Range [0, 1]. 0 = perfect equality, 1 = max inequality.

    Used for attention distributions: Gini is naturally scale-invariant
    (uniform over 10 tokens vs 1000 both give Gini=0), unlike entropy which
    requires explicit normalization by log2(t).
    """
    if values.size == 0:
        return 0.0
    # Negative values are not meaningful for attention weights
    if np.any(values < 0):
        raise ValueError("Gini coefficient requires non-negative values")
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    if cumsum[-1] == 0:
        return 0.0
    return float((2 * np.sum((np.arange(1, n + 1)) * sorted_vals)) / (n * cumsum[-1]) - (n + 1) / n)


@beartype
def shannon_entropy_bits(probs: Float[np.ndarray, "..."]) -> Float[np.ndarray, "..."]:
    """Shannon entropy in bits, computed along the last axis.

    For a 1-D input ``(n,)`` this returns a 0-D ndarray (so consumers can rely on
    a numpy return type even for the rank-1 case). Numerically stable: clamps
    probabilities at ``1e-12`` before ``log2``.
    """
    safe = np.clip(np.asarray(probs), 1e-12, 1.0)
    out = -np.sum(safe * np.log2(safe), axis=-1)
    # Sum along the last axis of a 1-D array returns a numpy scalar; rewrap as
    # a 0-D ndarray so the jaxtyping return annotation holds.
    return np.asarray(out)
