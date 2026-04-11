"""Centered Kernel Alignment (CKA) for comparing representations.

Linear CKA is the standard form: given two representations
:math:`X \\in \\mathbb{R}^{n \\times d_X}` and :math:`Y \\in
\\mathbb{R}^{n \\times d_Y}` over the same ``n`` items, CKA measures how
similar the two representation subspaces are, in [0, 1], invariant to
orthogonal transforms and isotropic scaling.

We implement two algebraically equivalent forms:

- :func:`linear_cka` — the naive Gram-matrix form, ``O(n^2 * max(dX, dY))``,
  numerically well-behaved and easy to verify.
- :func:`linear_cka_fast` — HSIC via feature-feature products, ``O(n *
  dX * dY)``. Faster when ``n`` is large relative to ``d``.

Both produce identical results up to float64 rounding (tested).

The flagship usage in this workstream is **layer-matched CKA between a
reasoning model and its non-reasoning counterpart** (R1-Distill vs base
Llama). See :func:`layer_matched_cka` for the orchestrating helper.
"""

from __future__ import annotations

import numpy as np
from beartype import beartype
from jaxtyping import Float

__all__ = [
    "layer_matched_cka",
    "linear_cka",
    "linear_cka_fast",
    "within_model_cka",
]


# ---------------------------------------------------------------------------
# Core CKA
# ---------------------------------------------------------------------------


@beartype
def linear_cka(
    X: Float[np.ndarray, "n d1"],
    Y: Float[np.ndarray, "n d2"],
) -> float:
    """Linear CKA via centered Gram matrices.

    Returns a scalar in ``[0, 1]``. 1 = representations span the same
    subspace (up to orthogonal rotation and scaling); 0 = orthogonal.
    Matches the definition in Kornblith et al. 2019 (eq. 4):

        CKA(X, Y) = HSIC(K_X, K_Y) / sqrt(HSIC(K_X, K_X) * HSIC(K_Y, K_Y))

    where ``K_X = X X^T``, ``K_Y = Y Y^T`` are sample-sample Gram matrices
    of column-centered X, Y, and the (n-1)^2 normalisation cancels.
    Equivalent to ``tr(K_X K_Y) / sqrt(tr(K_X K_X) * tr(K_Y K_Y))``.

    Algebraically equivalent to :func:`linear_cka_fast`; this version is
    O(n^2 * max(d1, d2)) and is preferred when ``n`` is small.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"CKA requires matched rows: X has {X.shape[0]}, Y has {Y.shape[0]}"
        )
    if X.shape[0] < 2:
        return 0.0
    X64 = (
        X.astype(np.float64, copy=False)
        - X.astype(np.float64, copy=False).mean(axis=0, keepdims=True)
    )
    Y64 = (
        Y.astype(np.float64, copy=False)
        - Y.astype(np.float64, copy=False).mean(axis=0, keepdims=True)
    )

    K_X = X64 @ X64.T  # (n, n) — sample-sample gram of X
    K_Y = Y64 @ Y64.T  # (n, n) — sample-sample gram of Y

    # Frobenius inner products: tr(A B) = (A * B).sum() for symmetric A, B.
    num = float((K_X * K_Y).sum())
    den_x = float((K_X * K_X).sum())
    den_y = float((K_Y * K_Y).sum())
    if den_x <= 0.0 or den_y <= 0.0:
        return 0.0
    return float(num / (np.sqrt(den_x) * np.sqrt(den_y)))


@beartype
def linear_cka_fast(
    X: Float[np.ndarray, "n d1"],
    Y: Float[np.ndarray, "n d2"],
) -> float:
    """Linear CKA via feature-feature products.

    Algebraically equivalent to :func:`linear_cka` but faster when
    ``n >> max(d1, d2)``. Uses the identity::

        ||X^T Y||_F^2 = tr(X^T Y Y^T X) = ||XYT||_F^2

    so we can compute everything from ``(d1 x d2)``-sized matrices
    instead of ``(n x n)``.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"CKA requires matched rows: X has {X.shape[0]}, Y has {Y.shape[0]}"
        )
    if X.shape[0] < 2:
        return 0.0
    X64 = X.astype(np.float64, copy=False) - X.astype(np.float64, copy=False).mean(axis=0, keepdims=True)
    Y64 = Y.astype(np.float64, copy=False) - Y.astype(np.float64, copy=False).mean(axis=0, keepdims=True)

    XtY = X64.T @ Y64  # (d1, d2)
    XtX = X64.T @ X64  # (d1, d1)
    YtY = Y64.T @ Y64  # (d2, d2)

    num = float((XtY * XtY).sum())
    den_x = float((XtX * XtX).sum())
    den_y = float((YtY * YtY).sum())
    if den_x <= 0.0 or den_y <= 0.0:
        return 0.0
    return float(num / (np.sqrt(den_x) * np.sqrt(den_y)))


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------


@beartype
def layer_matched_cka(
    acts_a: list[Float[np.ndarray, "n d"]],
    acts_b: list[Float[np.ndarray, "n d"]],
    mask: np.ndarray | None = None,
    use_fast: bool = True,
) -> Float[np.ndarray, "k"]:
    """CKA for each matched layer pair.

    ``acts_a[l]`` and ``acts_b[l]`` are expected to be the residual stream
    at layer ``l`` for the same set of problems across two models. If the
    two models have different layer counts, the caller is responsible for
    resampling (linear interpolation by depth fraction is typical).

    ``mask`` optionally selects a subset of rows to compute CKA over — e.g.
    the S1 subset or the S2 subset. Must be a 1D boolean array of length
    ``n``.

    Returns a ``(n_layers,)`` float64 array.
    """
    if len(acts_a) != len(acts_b):
        raise ValueError(
            f"layer count mismatch: {len(acts_a)} vs {len(acts_b)}"
        )
    fn = linear_cka_fast if use_fast else linear_cka
    out = np.empty(len(acts_a), dtype=np.float64)
    for l, (A, B) in enumerate(zip(acts_a, acts_b, strict=True)):
        if mask is not None:
            A = A[mask]
            B = B[mask]
        out[l] = fn(A, B)
    return out


@beartype
def within_model_cka(
    acts: list[Float[np.ndarray, "n d"]],
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    use_fast: bool = True,
) -> Float[np.ndarray, "k"]:
    """Per-layer CKA between two subsets of rows within a single model.

    Use case: CKA(S1 activations, S2 activations) at each layer. The
    layer where this drops is where the model's internal representation
    of S1- vs S2-leaning problems diverges.

    Both masks must be 1D boolean arrays of length ``n_problems``. They
    may overlap (e.g. difficulty-matched subsets) but typically won't.
    """
    fn = linear_cka_fast if use_fast else linear_cka
    out = np.empty(len(acts), dtype=np.float64)
    for l, arr in enumerate(acts):
        A = arr[mask_a]
        B = arr[mask_b]
        if A.shape[0] != B.shape[0]:
            # CKA requires matched rows. If the two subsets are different
            # sizes we truncate to the smaller size to keep things
            # well-defined. Caller should pass matched-size masks when
            # strictness matters.
            m = min(A.shape[0], B.shape[0])
            A = A[:m]
            B = B[:m]
        out[l] = fn(A, B)
    return out
