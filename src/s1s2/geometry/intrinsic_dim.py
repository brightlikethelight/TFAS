"""Intrinsic dimensionality estimators.

Two complementary tools:

- :func:`two_nn_intrinsic_dim` — Facco et al. 2017. For each point,
  compute the ratio of its second to first nearest-neighbor distance,
  :math:`\\mu_i = r_2 / r_1`. Under the assumption of locally uniform
  density, the empirical CDF of :math:`\\mu` satisfies
  :math:`F(\\mu) = 1 - \\mu^{-d}`, so a linear regression of
  :math:`-\\log(1-F)` on :math:`\\log\\mu` has slope :math:`d`. This is
  the "local-scale" intrinsic dimension — it estimates the dimension of
  the manifold on which the data locally lives, independently of the
  ambient dimension.

- :func:`participation_ratio` — single-number effective dimension
  derived from the PCA eigenspectrum:
  :math:`\\text{PR} = (\\sum \\lambda_i)^2 / \\sum \\lambda_i^2`. PR
  equals ``d`` for isotropic data and 1 for rank-1 data. Gao et al.
  2017 advocate this as "effective dimension" for neural population
  codes.

For the s1s2 project, PR captures how "spread out" the activations are
in principal directions; Two-NN captures manifold structure. We report
both at every layer because they dissociate in meaningful ways (e.g. a
highly anisotropic low-PR Gaussian can still have a high Two-NN ID if it
has 50 weakly-varying dimensions).
"""

from __future__ import annotations

import numpy as np
from beartype import beartype
from jaxtyping import Float
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "participation_ratio",
    "two_nn_intrinsic_dim",
    "two_nn_with_discard",
]


# ---------------------------------------------------------------------------
# Two-NN intrinsic dimension
# ---------------------------------------------------------------------------


@beartype
def two_nn_intrinsic_dim(
    X: Float[np.ndarray, "n d"],
    fraction: float = 0.9,
    metric: str = "euclidean",
) -> float:
    """Two-NN intrinsic dimensionality estimator.

    Parameters
    ----------
    X : (n, d) array
    fraction : fraction of the lowest-:math:`\\mu` points to use in the
        regression. Facco et al. recommend 0.9 (drop the top 10% as
        outliers where local-density assumption breaks).
    metric : nearest-neighbor metric. ``'euclidean'`` is the original
        formulation; ``'cosine'`` is sometimes used for normalized LLM
        embeddings but the CDF identity still holds under local
        log-uniformity.

    Returns ``d`` as a float. Returns ``0.0`` if fewer than 3 samples or
    if all :math:`\\mu_i` are zero/one (degenerate).
    """
    n = X.shape[0]
    if n < 3:
        return 0.0

    nn = NearestNeighbors(n_neighbors=3, metric=metric)
    nn.fit(X.astype(np.float64, copy=False))
    dists, _ = nn.kneighbors(X, n_neighbors=3)
    # dists[:, 0] is self-distance (0.0), [:, 1] is 1st NN, [:, 2] is 2nd NN.
    r1 = dists[:, 1]
    r2 = dists[:, 2]

    # Filter out duplicates / degenerate neighborhoods where r1 == 0.
    valid = (r1 > 0) & (r2 > 0)
    if not valid.any():
        return 0.0
    mu = r2[valid] / r1[valid]
    mu = mu[mu > 1.0]  # mu >= 1 by construction; strict inequality for log.
    if mu.size < 3:
        return 0.0

    # Empirical CDF of sorted mu: F_i = (i + 1) / n. We want -log(1 - F_i)
    # regressed on log(mu_i), slope = d.
    mu_sorted = np.sort(mu)
    m = mu_sorted.size
    # Drop the top (1 - fraction) fraction to avoid the sparse tail. The
    # usual convention: keep points with F <= fraction.
    k = max(3, int(np.floor(float(fraction) * m)))
    mu_fit = mu_sorted[:k]
    F = np.arange(1, k + 1, dtype=np.float64) / float(m)
    y = -np.log(1.0 - F)
    x = np.log(mu_fit)
    # Guard against -inf at F->1 (shouldn't happen after the top-fraction cut,
    # but be defensive).
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return 0.0
    x = x[finite]
    y = y[finite]
    # Closed-form regression through the origin (as in the Facco paper):
    # d = sum(x * y) / sum(x * x)
    denom = float((x * x).sum())
    if denom == 0.0:
        return 0.0
    d_hat = float((x * y).sum() / denom)
    return max(0.0, d_hat)


@beartype
def two_nn_with_discard(
    X: Float[np.ndarray, "n d"],
    fraction: float = 0.9,
    n_bootstrap: int = 0,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Two-NN with an optional nonparametric bootstrap CI.

    Returns ``(d_hat, ci_lower, ci_upper)``. With ``n_bootstrap=0``
    (default) the CI is set to the point estimate.
    """
    point = two_nn_intrinsic_dim(X, fraction=fraction)
    if n_bootstrap <= 0:
        return point, point, point
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    X64 = X.astype(np.float64, copy=False)
    boot = np.empty(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        boot[i] = two_nn_intrinsic_dim(X64[idx], fraction=fraction)
    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Participation ratio
# ---------------------------------------------------------------------------


@beartype
def participation_ratio(
    X: Float[np.ndarray, "n d"],
    center: bool = True,
) -> float:
    """Participation ratio from the PCA eigenspectrum.

    .. math::

        \\text{PR} = \\frac{(\\sum_i \\lambda_i)^2}{\\sum_i \\lambda_i^2}

    where :math:`\\lambda_i` are the eigenvalues of the centered
    covariance matrix. Bounded in ``[1, min(n, d)]``. Equals the full
    rank ``d`` for an isotropic Gaussian and 1 for rank-1 data.

    We compute the eigenvalues via SVD on the centered matrix, which is
    the numerically stable way to get them without materializing
    ``(d x d)`` when ``d`` is large.
    """
    if X.shape[0] < 2:
        return 1.0
    X64 = X.astype(np.float64, copy=False)
    if center:
        X64 = X64 - X64.mean(axis=0, keepdims=True)
    # svd gives singular values s; eigenvalues of the (n-1)-scaled
    # covariance are s^2 / (n - 1). The (n - 1) scaling cancels in the PR
    # ratio so we can skip it.
    s = np.linalg.svd(X64, full_matrices=False, compute_uv=False)
    lam = s.astype(np.float64) ** 2
    total = float(lam.sum())
    sq = float((lam * lam).sum())
    if sq <= 0.0:
        return 1.0
    return float((total * total) / sq)
