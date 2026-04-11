"""Linear separability with the ``d >> N`` confound mitigated.

**The problem.** With 4096-dim residual streams and 400-500 problems,
any two random classes are linearly separable (Cover's theorem: in
:math:`d` dimensions, :math:`n < 2d` generic points are separable with
probability 1). So reporting "linear SVM accuracy = 100% therefore
representations encode X" is uninformative — it would be 100% for
random labels too.

**The fix, both of which we implement.**

1. **PCA pre-reduction.** Project to 50-100 PCs (covering ~95% variance),
   then fit a linear SVM. This brings the effective dimensionality below
   the sample count, so accuracy is meaningful.

2. **Margin comparison vs shuffled labels.** Even when both real and
   shuffled labels achieve perfect train accuracy, the MARGIN of the
   real-label SVM should be larger (the real labels have a "natural"
   separating hyperplane with wider tolerance). Margin is computed as
   :math:`1 / \\|w\\|` where :math:`w` is the SVM weight vector. Compare
   the real margin to the null distribution from shuffled labels.

The :func:`linear_separability_with_d_gg_n_fix` function runs both and
reports the full picture: cross-validated accuracy in PCA-reduced space
(the primary number), cross-validated accuracy in the ambient space (the
"look how trivially separable everything is" number), real margin in
ambient space, and the shuffled-label null.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from beartype import beartype
from jaxtyping import Float, Int
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC

__all__ = [
    "SeparabilityResult",
    "_cv_accuracy",
    "_svm_margin",
    "linear_separability_with_d_gg_n_fix",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_svm(seed: int) -> LinearSVC:
    """LinearSVC with stable defaults.

    ``C=1.0`` is the sklearn default; ``dual='auto'`` lets sklearn pick
    the faster of dual/primal given ``(n, d)``; increased ``max_iter``
    because convergence on 4096-dim data is slow.
    """
    return LinearSVC(
        C=1.0,
        loss="squared_hinge",
        dual="auto",
        max_iter=5000,
        tol=1e-4,
        random_state=int(seed),
    )


def _svm_margin(svm: LinearSVC) -> float:
    """SVM margin: :math:`1 / \\|w\\|_2`.

    For a fitted LinearSVC, ``svm.coef_`` has shape ``(1, d)`` for binary
    classification and ``(k, d)`` for one-vs-rest k-class. We return the
    smallest per-class margin as the conservative number. Returns
    ``inf`` if ``w`` is zero (a degenerate fit).
    """
    w = np.asarray(svm.coef_, dtype=np.float64)
    norms = np.linalg.norm(w, axis=1)
    # Guard against zero weights.
    norms = np.where(norms > 0, norms, np.inf)
    return float(1.0 / norms.min())


def _cv_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    seed: int,
) -> float:
    """Stratified k-fold CV accuracy for a fresh LinearSVC."""
    n_classes = int(np.unique(y).size)
    if n_classes < 2:
        return float("nan")
    # Guard against folds where a class has too few samples.
    min_class = int(min(int(np.sum(y == c)) for c in np.unique(y)))
    k = max(2, min(int(n_folds), min_class))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed))
    scores = cross_val_score(
        _make_svm(seed),
        X.astype(np.float64, copy=False),
        y,
        cv=skf,
        scoring="accuracy",
        n_jobs=1,
    )
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SeparabilityResult:
    """Everything we compute about linear separability for one (X, y).

    Primary fields for reporting:
    - ``pca_cv_accuracy`` — the meaningful accuracy (d << N after PCA).
    - ``pca_cv_accuracy_shuffled_mean`` — chance after PCA.
    - ``margin_real`` and ``margin_shuffled_mean`` / ``margin_p`` — the
      margin-based check in ambient space.
    """

    n_samples: int
    d_ambient: int
    pca_dim: int
    explained_variance_ratio: float
    ambient_cv_accuracy: float
    pca_cv_accuracy: float
    margin_real: float
    margin_shuffled_mean: float
    margin_shuffled_std: float
    margin_p: float
    ambient_cv_accuracy_shuffled_mean: float
    ambient_cv_accuracy_shuffled_std: float
    pca_cv_accuracy_shuffled_mean: float
    pca_cv_accuracy_shuffled_std: float
    pca_accuracy_p: float
    n_shuffles: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": int(self.n_samples),
            "d_ambient": int(self.d_ambient),
            "pca_dim": int(self.pca_dim),
            "explained_variance_ratio": float(self.explained_variance_ratio),
            "ambient_cv_accuracy": float(self.ambient_cv_accuracy),
            "pca_cv_accuracy": float(self.pca_cv_accuracy),
            "margin_real": float(self.margin_real),
            "margin_shuffled_mean": float(self.margin_shuffled_mean),
            "margin_shuffled_std": float(self.margin_shuffled_std),
            "margin_p": float(self.margin_p),
            "ambient_cv_accuracy_shuffled_mean": float(
                self.ambient_cv_accuracy_shuffled_mean
            ),
            "ambient_cv_accuracy_shuffled_std": float(
                self.ambient_cv_accuracy_shuffled_std
            ),
            "pca_cv_accuracy_shuffled_mean": float(self.pca_cv_accuracy_shuffled_mean),
            "pca_cv_accuracy_shuffled_std": float(self.pca_cv_accuracy_shuffled_std),
            "pca_accuracy_p": float(self.pca_accuracy_p),
            "n_shuffles": int(self.n_shuffles),
        }


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


@beartype
def linear_separability_with_d_gg_n_fix(
    X: Float[np.ndarray, "n d"],
    y: Int[np.ndarray, "n"],
    pca_dim: int = 50,
    n_shuffles: int = 100,
    n_folds: int = 5,
    seed: int = 0,
) -> SeparabilityResult:
    """Fit the full separability diagnostic.

    Steps:

    1. Fit PCA to ``pca_dim`` components (clamped to ``min(n, d,
       pca_dim)``), record cumulative explained variance.
    2. CV-accuracy of a LinearSVC in BOTH ambient and PCA spaces.
    3. Fit a LinearSVC on the full dataset (ambient) and record its margin.
    4. For ``n_shuffles`` random label permutations: repeat steps 2 and 3.
    5. Report real values, the shuffled null, and two one-sided
       p-values: one on the PCA accuracy (``P[acc_shuffle >= acc_real]``)
       and one on the margin (``P[margin_shuffle >= margin_real]``).

    The PCA-accuracy null typically sits around chance for sensible
    targets — that's the desired behavior. The ambient-accuracy null
    typically sits at 1.0 — that's the d>>N pathology we are trying
    to demonstrate.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    if n < 4:
        raise ValueError(f"need at least 4 samples for CV, got n={n}")
    if np.unique(y).size < 2:
        raise ValueError("need at least two distinct labels for separability")

    X64 = X.astype(np.float64, copy=False)

    # --- PCA ---
    k = max(1, min(int(pca_dim), n - 1, d))
    pca = PCA(n_components=k, random_state=int(seed))
    X_pca = pca.fit_transform(X64)
    ev_ratio = float(np.sum(pca.explained_variance_ratio_))

    # --- Real CV accuracies ---
    ambient_cv = _cv_accuracy(X64, y, n_folds=n_folds, seed=seed)
    pca_cv = _cv_accuracy(X_pca, y, n_folds=n_folds, seed=seed)

    # --- Real SVM margin in ambient space ---
    svm_ambient = _make_svm(seed)
    svm_ambient.fit(X64, y)
    margin_real = _svm_margin(svm_ambient)

    # --- Shuffled null ---
    n_shuffles = int(n_shuffles)
    shuf_ambient_cv = np.empty(n_shuffles, dtype=np.float64)
    shuf_pca_cv = np.empty(n_shuffles, dtype=np.float64)
    shuf_margin = np.empty(n_shuffles, dtype=np.float64)
    y_perm = y.copy()
    for i in range(n_shuffles):
        rng.shuffle(y_perm)
        try:
            shuf_ambient_cv[i] = _cv_accuracy(
                X64, y_perm, n_folds=n_folds, seed=seed + 1 + i
            )
        except ValueError:
            shuf_ambient_cv[i] = float("nan")
        try:
            shuf_pca_cv[i] = _cv_accuracy(
                X_pca, y_perm, n_folds=n_folds, seed=seed + 1 + i
            )
        except ValueError:
            shuf_pca_cv[i] = float("nan")
        svm_s = _make_svm(seed + 1 + i)
        try:
            svm_s.fit(X64, y_perm)
            shuf_margin[i] = _svm_margin(svm_s)
        except ValueError:
            shuf_margin[i] = float("nan")

    # Drop NaNs before summarizing.
    ambient_ok = shuf_ambient_cv[np.isfinite(shuf_ambient_cv)]
    pca_ok = shuf_pca_cv[np.isfinite(shuf_pca_cv)]
    margin_ok = shuf_margin[np.isfinite(shuf_margin)]

    def _p_one_sided(real: float, null: np.ndarray) -> float:
        if null.size == 0:
            return 1.0
        n_extreme = int(np.sum(null >= real))
        return (n_extreme + 1) / (null.size + 1)

    return SeparabilityResult(
        n_samples=n,
        d_ambient=d,
        pca_dim=k,
        explained_variance_ratio=ev_ratio,
        ambient_cv_accuracy=ambient_cv,
        pca_cv_accuracy=pca_cv,
        margin_real=margin_real,
        margin_shuffled_mean=float(margin_ok.mean()) if margin_ok.size else float("nan"),
        margin_shuffled_std=float(margin_ok.std(ddof=1)) if margin_ok.size > 1 else 0.0,
        margin_p=_p_one_sided(margin_real, margin_ok),
        ambient_cv_accuracy_shuffled_mean=float(ambient_ok.mean()) if ambient_ok.size else float("nan"),
        ambient_cv_accuracy_shuffled_std=float(ambient_ok.std(ddof=1)) if ambient_ok.size > 1 else 0.0,
        pca_cv_accuracy_shuffled_mean=float(pca_ok.mean()) if pca_ok.size else float("nan"),
        pca_cv_accuracy_shuffled_std=float(pca_ok.std(ddof=1)) if pca_ok.size > 1 else 0.0,
        pca_accuracy_p=_p_one_sided(pca_cv, pca_ok),
        n_shuffles=n_shuffles,
    )
