"""Dimensionality reduction / projection utilities.

Three projection families, all reducing ``(n_samples, d_in)`` activations
to ``(n_samples, d_out)``:

- :func:`pca_project` — sklearn PCA wrapper, returns the projected
  activations plus per-component explained-variance ratios.
- :func:`umap_project` — UMAP with defaults chosen for high-dim LLM
  activations (cosine metric, ``n_neighbors=30``, ``min_dist=0.1``).
- :func:`random_projection` — multiple Gaussian random projections, the
  non-negotiable control against UMAP-hallucinated structure. If 100
  random 2D projections don't show the same clusters UMAP does, UMAP is
  lying.

All randomness is driven by ``np.random.default_rng(seed)`` so runs are
bit-reproducible.
"""

from __future__ import annotations

import numpy as np
from beartype import beartype
from jaxtyping import Float
from sklearn.decomposition import PCA

__all__ = [
    "RandomProjectionPanel",
    "pca_project",
    "random_projection",
    "umap_project",
]


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


@beartype
def pca_project(
    X: Float[np.ndarray, "n d"],
    n_components: int = 2,
    seed: int = 0,
) -> tuple[Float[np.ndarray, "n k"], Float[np.ndarray, "k"]]:
    """PCA-project ``X`` to ``n_components`` dimensions.

    Returns ``(projection, explained_variance_ratio)``. The second value is
    the per-component explained variance ratio, sorted descending, and is
    what you want in axis labels on 2D plots (``f"PC1 ({ev[0]:.1%})"``).
    """
    n_samples, d = X.shape
    k = min(int(n_components), n_samples, d)
    if k < 1:
        raise ValueError(f"n_components too small: {k} from X.shape={X.shape}")
    pca = PCA(n_components=k, random_state=seed)
    proj = pca.fit_transform(X.astype(np.float64, copy=False))
    return proj.astype(np.float64, copy=False), np.asarray(
        pca.explained_variance_ratio_, dtype=np.float64
    )


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------


@beartype
def umap_project(
    X: Float[np.ndarray, "n d"],
    n_components: int = 2,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
    seed: int = 0,
) -> Float[np.ndarray, "n k"]:
    """UMAP embedding with defaults from the geometry planning research.

    Imported lazily so that environments without ``umap-learn`` can still
    use the rest of the geometry module. Raises :class:`ImportError` with
    an actionable message if UMAP is missing.
    """
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "umap-learn is required for umap_project; install with "
            "`pip install umap-learn`"
        ) from e

    n_samples = X.shape[0]
    # UMAP's k-NN graph needs n_neighbors < n_samples.
    nn = max(2, min(int(n_neighbors), n_samples - 1))
    reducer = umap.UMAP(
        n_components=int(n_components),
        n_neighbors=nn,
        min_dist=float(min_dist),
        metric=metric,
        random_state=int(seed),
    )
    emb = reducer.fit_transform(X.astype(np.float32, copy=False))
    return np.asarray(emb, dtype=np.float64)


# ---------------------------------------------------------------------------
# Random projection
# ---------------------------------------------------------------------------


@beartype
def random_projection(
    X: Float[np.ndarray, "n d"],
    n_components: int = 2,
    n_seeds: int = 100,
    seed: int = 0,
    normalize: bool = True,
) -> list[Float[np.ndarray, "n k"]]:
    """Produce ``n_seeds`` independent Gaussian random projections of ``X``.

    Each projection uses a fresh ``(d, n_components)`` Gaussian matrix with
    entries ``~ N(0, 1/d)`` (the Johnson-Lindenstrauss-style convention
    where ``E[R^T R] = I_k``). When ``normalize=True`` (default) the
    projected coordinates are then standardised per-axis to unit variance,
    which makes "visible cluster magnitude" comparable across choices of
    ``d`` and across projections — and makes the random-projection control
    interpretable on the same axis scale as PCA/UMAP plots.

    **Why a list and not an average?** The control we want is "does
    the structure you see in UMAP ALSO show up in random projections?"
    That is a per-projection question, not an aggregate. Viz and
    cluster-detection code can iterate over the list and decide.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    k = int(n_components)
    if d <= 0:
        raise ValueError("input has zero feature dimension")
    scale = 1.0 / np.sqrt(d)
    X64 = X.astype(np.float64, copy=False)
    projections: list[np.ndarray] = []
    for _ in range(int(n_seeds)):
        R = rng.standard_normal((d, k)) * scale
        proj = X64 @ R
        if normalize:
            std = proj.std(axis=0, keepdims=True)
            std = np.where(std > 1e-12, std, 1.0)
            proj = proj / std
        projections.append(proj)
    return projections


# ---------------------------------------------------------------------------
# Random projection panel: summary object for viz
# ---------------------------------------------------------------------------


class RandomProjectionPanel:
    """Bundle of a PCA, UMAP, and 3 random projections for plotting.

    This is the canonical "4-panel figure" in :mod:`s1s2.geometry.viz`:
    side-by-side PCA / UMAP / random-1 / random-2 / random-3. Constructed
    once, passed to :func:`viz.plot_random_projection_panel`. Kept in its
    own class so ``viz`` does not accidentally re-fit UMAP when re-plotting.
    """

    def __init__(
        self,
        pca: Float[np.ndarray, "n 2"],
        pca_ev: Float[np.ndarray, "2"],
        umap: Float[np.ndarray, "n 2"] | None,
        random_projs: list[np.ndarray],
    ) -> None:
        self.pca = pca
        self.pca_ev = pca_ev
        self.umap = umap
        self.random_projs = random_projs

    @classmethod
    def from_activations(
        cls,
        X: Float[np.ndarray, "n d"],
        *,
        umap_n_neighbors: int = 30,
        umap_min_dist: float = 0.1,
        umap_metric: str = "cosine",
        n_random: int = 3,
        seed: int = 0,
        skip_umap: bool = False,
    ) -> RandomProjectionPanel:
        """Fit all projections in one call.

        ``skip_umap=True`` lets callers opt out of UMAP for the panel (useful
        for unit tests where ``umap-learn`` may not be installed).
        """
        pca_proj, ev = pca_project(X, n_components=2, seed=seed)
        try:
            if skip_umap:
                umap_proj: np.ndarray | None = None
            else:
                umap_proj = umap_project(
                    X,
                    n_components=2,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    metric=umap_metric,
                    seed=seed,
                )
        except ImportError:
            umap_proj = None
        rps = random_projection(X, n_components=2, n_seeds=n_random, seed=seed + 1)
        return cls(pca=pca_proj, pca_ev=ev, umap=umap_proj, random_projs=rps)
