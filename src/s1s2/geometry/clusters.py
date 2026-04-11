"""Cluster separation metrics for residual-stream activations.

**Primary metric is cosine silhouette**. In 4096-dim Euclidean space,
pairwise distances concentrate (all pairs look equidistant), making
Euclidean silhouette uninformative. Cosine silhouette is scale-invariant
along each vector and empirically cleaner for transformer residuals.

Every public function here accepts ``(X, labels)`` and returns either a
scalar metric or a ``(metric, ci_lower, ci_upper)`` triple. Bootstrap is
resample-with-replacement over the rows of ``X`` (and the matching
labels). The permutation null is label-shuffle, which is the correct
null for "are these labels related to the structure of X?"

Two robustness checks alongside silhouette:

- Calinski-Harabasz (variance-ratio criterion, higher = better)
- Davies-Bouldin (intra/inter cluster ratio, lower = better)

A +1 correction (North et al. 2002) is applied to permutation p-values:
``p = (n_extreme + 1) / (n_perms + 1)``. This prevents zero p-values.
"""

from __future__ import annotations

import numpy as np
from beartype import beartype
from jaxtyping import Float, Int
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

__all__ = [
    "SilhouetteResult",
    "calinski_harabasz",
    "cosine_silhouette",
    "cosine_silhouette_with_ci",
    "davies_bouldin",
    "silhouette_permutation_test",
]


# ---------------------------------------------------------------------------
# Core silhouette
# ---------------------------------------------------------------------------


@beartype
def cosine_silhouette(
    X: Float[np.ndarray, "n d"],
    labels: Int[np.ndarray, "n"],
) -> float:
    """Silhouette score with cosine distance.

    Returns ``0.0`` when fewer than two distinct labels or fewer than
    ``n_samples=3`` items: silhouette is undefined in those cases and
    returning 0 is cleaner than propagating NaN through the rest of the
    pipeline.
    """
    if X.shape[0] < 3:
        return 0.0
    uniq = np.unique(labels)
    if uniq.size < 2:
        return 0.0
    # sklearn's silhouette expects float64 and raises if any cluster is a
    # singleton (< 2 members). Guard against that.
    sizes = np.array([int(np.sum(labels == u)) for u in uniq])
    if (sizes < 2).any():
        return 0.0
    return float(silhouette_score(X.astype(np.float64, copy=False), labels, metric="cosine"))


@beartype
def calinski_harabasz(
    X: Float[np.ndarray, "n d"],
    labels: Int[np.ndarray, "n"],
) -> float:
    """Calinski-Harabasz score (variance-ratio criterion). Higher is better."""
    if X.shape[0] < 3:
        return 0.0
    uniq = np.unique(labels)
    if uniq.size < 2:
        return 0.0
    sizes = np.array([int(np.sum(labels == u)) for u in uniq])
    if (sizes < 2).any():
        return 0.0
    return float(calinski_harabasz_score(X.astype(np.float64, copy=False), labels))


@beartype
def davies_bouldin(
    X: Float[np.ndarray, "n d"],
    labels: Int[np.ndarray, "n"],
) -> float:
    """Davies-Bouldin score. Lower is better. ``inf`` if undefined."""
    if X.shape[0] < 3:
        return float("inf")
    uniq = np.unique(labels)
    if uniq.size < 2:
        return float("inf")
    sizes = np.array([int(np.sum(labels == u)) for u in uniq])
    if (sizes < 2).any():
        return float("inf")
    return float(davies_bouldin_score(X.astype(np.float64, copy=False), labels))


# ---------------------------------------------------------------------------
# Bootstrap CI for cosine silhouette
# ---------------------------------------------------------------------------


@beartype
def cosine_silhouette_with_ci(
    X: Float[np.ndarray, "n d"],
    labels: Int[np.ndarray, "n"],
    n_bootstrap: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap the cosine silhouette score.

    Returns ``(silhouette, ci_lower, ci_upper)``. The bootstrap resamples
    rows of ``X`` with replacement. If a resample happens to produce a
    degenerate labelling (singleton cluster) the silhouette is 0 for that
    resample — this is the conservative choice, not an exclusion.
    """
    point = cosine_silhouette(X, labels)
    if n_bootstrap <= 0:
        return point, point, point
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    X64 = X.astype(np.float64, copy=False)
    boot = np.empty(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        boot[i] = cosine_silhouette(X64[idx], labels[idx])
    alpha = 1.0 - float(confidence)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Permutation test on silhouette
# ---------------------------------------------------------------------------


@beartype
def silhouette_permutation_test(
    X: Float[np.ndarray, "n d"],
    labels: Int[np.ndarray, "n"],
    n_permutations: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, Float[np.ndarray, "k"]]:
    """Label-shuffle permutation test for cosine silhouette.

    Returns ``(observed, p_value, null_distribution)``. The observed
    statistic is the cosine silhouette at the true labels; the null is
    the distribution of silhouettes at shuffled labels. One-sided
    (alternative: "observed is larger than null"). Uses the North +1
    correction.
    """
    obs = cosine_silhouette(X, labels)
    if n_permutations <= 0:
        return obs, 1.0, np.zeros(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    X64 = X.astype(np.float64, copy=False)
    null = np.empty(int(n_permutations), dtype=np.float64)
    labels_copy = labels.copy()
    for i in range(int(n_permutations)):
        rng.shuffle(labels_copy)
        null[i] = cosine_silhouette(X64, labels_copy)
    n_extreme = int(np.sum(null >= obs))
    p = (n_extreme + 1) / (int(n_permutations) + 1)
    return obs, float(p), null


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


class SilhouetteResult:
    """Everything we compute for one (X, labels) pair.

    Intentionally a plain class — it's serialized to JSON via
    :meth:`to_dict` without needing the dataclass machinery.
    """

    def __init__(
        self,
        silhouette: float,
        ci_lower: float,
        ci_upper: float,
        permutation_p: float,
        null_p95: float,
        null_mean: float,
        calinski_harabasz: float,
        davies_bouldin: float,
        n_samples: int,
        n_labels: int,
    ) -> None:
        self.silhouette = silhouette
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
        self.permutation_p = permutation_p
        self.null_p95 = null_p95
        self.null_mean = null_mean
        self.calinski_harabasz = calinski_harabasz
        self.davies_bouldin = davies_bouldin
        self.n_samples = n_samples
        self.n_labels = n_labels

    def to_dict(self) -> dict[str, float | int]:
        return {
            "silhouette": float(self.silhouette),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "permutation_p": float(self.permutation_p),
            "null_p95": float(self.null_p95),
            "null_mean": float(self.null_mean),
            "calinski_harabasz": float(self.calinski_harabasz),
            "davies_bouldin": float(self.davies_bouldin),
            "n_samples": int(self.n_samples),
            "n_labels": int(self.n_labels),
        }


@beartype
def compute_silhouette_result(
    X: Float[np.ndarray, "n d"],
    labels: Int[np.ndarray, "n"],
    n_bootstrap: int = 1000,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> SilhouetteResult:
    """One-shot compute + serialize helper for orchestration code."""
    point, lo, hi = cosine_silhouette_with_ci(
        X, labels, n_bootstrap=n_bootstrap, seed=seed
    )
    obs, p, null = silhouette_permutation_test(
        X, labels, n_permutations=n_permutations, seed=seed + 1
    )
    # obs should equal point, but use it directly so the result is always
    # self-consistent even if clip-guards trigger in one path and not the other.
    ch = calinski_harabasz(X, labels)
    db = davies_bouldin(X, labels)
    return SilhouetteResult(
        silhouette=float(obs),
        ci_lower=float(lo),
        ci_upper=float(hi),
        permutation_p=float(p),
        null_p95=float(np.percentile(null, 95)) if null.size else 0.0,
        null_mean=float(null.mean()) if null.size else 0.0,
        calinski_harabasz=float(ch),
        davies_bouldin=float(db),
        n_samples=int(X.shape[0]),
        n_labels=int(np.unique(labels).size),
    )
