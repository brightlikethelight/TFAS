"""Probe classes: MassMean, Logistic, MLP, CCS.

All probes inherit from :class:`Probe` and expose a common minimal interface:

* ``fit(X_train, y_train)``     — train the probe on float32 activations and binary labels.
* ``predict_proba(X) -> ndarray`` — return ``p(y=1 | x)`` shaped ``(n_samples,)``.
* ``score(X, y) -> dict``       — compute ROC-AUC, balanced accuracy, F1, MCC.

Design notes
------------
* Labels are always binary ``{0, 1}``. Multiclass is out of scope (all four probing
  targets are binary).
* We NEVER subclass ``sklearn.base.BaseEstimator`` — the probes participate in
  a bespoke cross-validation loop in :mod:`s1s2.probes.core` and adopting sklearn's
  estimator interface would tempt users to pass them to grid-search etc., where the
  Hewitt-Liang control would be silently bypassed.
* ``LogisticRegressionProbe`` uses ``LogisticRegressionCV`` internally for the
  L2 penalty choice. That means ``fit`` performs an inner 5-fold CV to pick C,
  giving us true nested CV when wrapped by the outer k-fold splitter in
  :class:`s1s2.probes.core.ProbeRunner`.
* ``MLPProbe`` uses PyTorch and respects the CUDA device if available, else CPU.
* ``CCSProbe`` is the Burns et al. (2022) unsupervised probe. It requires *paired*
  contrast statements — in our setting we pair ``(resid, -resid)`` which reduces
  CCS to finding a direction along which the projection distribution is bimodal.
  This is a weaker formulation than the original paper (which uses yes/no
  completions) but is the most we can do from a bare residual cache. CCS results
  should be treated as exploratory.

References
----------
* Alain & Bengio (2018). Understanding intermediate layers using linear classifier probes.
* Hewitt & Liang (2019). Designing and interpreting probes with control tasks.
* Burns et al. (2022). Discovering latent knowledge in language models without supervision.
* Marks & Tegmark (2023). The geometry of truth — mass-mean probes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from beartype import beartype
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch import nn

__all__ = [
    "CCSProbe",
    "LogisticRegressionProbe",
    "MLPProbe",
    "MassMeanProbe",
    "Probe",
    "get_probe_class",
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Probe(ABC):
    """Abstract base for all probes.

    Subclasses must implement :meth:`fit` and :meth:`predict_proba`. :meth:`score`
    is provided automatically and returns a dict of ROC-AUC, balanced accuracy,
    F1, and Matthews correlation.
    """

    name: str = "probe"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Probe:
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Hard predictions at the given probability threshold."""
        return (self.predict_proba(X) >= threshold).astype(np.int8)

    @beartype
    def score(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Return the canonical metric bundle.

        ROC-AUC is robust to class imbalance; balanced accuracy corrects
        naive accuracy for class skew; F1 is the standard binary F1; MCC
        ties it all together (it is the only metric that is symmetric under
        class swap and rewards predicting both classes). We always report
        all four so reviewers can pick their favourite.
        """
        if not self._fitted:
            raise RuntimeError(f"{self.name}: score() called before fit()")
        proba = self.predict_proba(X)
        hard = (proba >= 0.5).astype(np.int8)
        # ROC-AUC is undefined if y is single-class. Return 0.5 (chance) in that
        # edge case so downstream plotting doesn't break.
        if len(np.unique(y)) < 2:
            auc = 0.5
        else:
            auc = float(roc_auc_score(y, proba))
        return {
            "roc_auc": auc,
            "balanced_accuracy": float(balanced_accuracy_score(y, hard)),
            "f1": float(f1_score(y, hard, zero_division=0)),
            "mcc": float(matthews_corrcoef(y, hard)),
        }

    def weight_vector(self) -> np.ndarray | None:
        """Return the probe's linear direction if it has one, else None.

        Used by the self-correction trajectory alignment analysis (cosine of
        two probe directions). Subclasses override where meaningful.
        """
        return None


# ---------------------------------------------------------------------------
# Mass-mean probe
# ---------------------------------------------------------------------------


class MassMeanProbe(Probe):
    """Zero-parameter probe: classify by cosine similarity to class means.

    The "weight" direction is ``mu1 - mu0``, normalised. This is the Marks &
    Tegmark "geometry of truth" baseline and exists here as a sanity check:
    if a real signal exists, even this trivial probe should pick it up.

    We report the *sigmoided cosine* as the predicted probability. A temperature
    of 4.0 keeps the output well away from the saturation regions so AUC is
    well-defined.
    """

    name = "mass_mean"

    def __init__(self, temperature: float = 4.0, seed: int = 0) -> None:
        super().__init__(seed)
        self.temperature = temperature
        self.direction_: np.ndarray | None = None
        self.bias_: float = 0.0

    @beartype
    def fit(self, X: np.ndarray, y: np.ndarray) -> MassMeanProbe:
        if X.ndim != 2:
            raise ValueError(f"expected (n, d) activations, got shape {X.shape}")
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.int64, copy=False)
        mu0 = X[y == 0].mean(axis=0)
        mu1 = X[y == 1].mean(axis=0)
        direction = mu1 - mu0
        norm = np.linalg.norm(direction)
        if norm == 0:
            # Classes have identical means — probe is trivially chance.
            self.direction_ = np.zeros_like(direction)
        else:
            self.direction_ = direction / norm
        # Midpoint threshold along the direction (project the two means).
        self.bias_ = -0.5 * float(self.direction_ @ (mu0 + mu1))
        self._fitted = True
        return self

    @beartype
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.direction_ is None:
            raise RuntimeError("MassMeanProbe: predict_proba before fit")
        X = X.astype(np.float32, copy=False)
        logits = X @ self.direction_ + self.bias_
        # sigmoid with temperature scaling
        return 1.0 / (1.0 + np.exp(-self.temperature * logits))

    def weight_vector(self) -> np.ndarray | None:
        return None if self.direction_ is None else self.direction_.copy()


# ---------------------------------------------------------------------------
# Logistic regression probe (primary result)
# ---------------------------------------------------------------------------


class LogisticRegressionProbe(Probe):
    """L2-regularised logistic regression with nested CV for C.

    Uses ``sklearn.linear_model.LogisticRegressionCV`` with:
    * L2 penalty (we're probing low-dim subspaces — L1 would zero-out dims)
    * 20 Cs log-spaced in ``[1e-4, 1e4]``
    * 5-fold internal CV for hyperparameter selection
    * balanced class weights (important for ``correctness`` / ``bias_susceptible``
      which can be imbalanced on any given model)
    * LBFGS solver (fastest for small n, dense d)

    The result is the PRIMARY probe metric in the paper.
    """

    name = "logistic"

    def __init__(
        self,
        Cs: np.ndarray | None = None,
        cv: int = 5,
        max_iter: int = 5000,
        seed: int = 0,
    ) -> None:
        super().__init__(seed)
        self.Cs = np.logspace(-4, 4, 20) if Cs is None else Cs
        self.cv = cv
        self.max_iter = max_iter
        self.clf: LogisticRegressionCV | None = None

    @beartype
    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegressionProbe:
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.int64, copy=False)
        # If only one class is present we can't fit anything meaningful — emit
        # a degenerate "predict the majority" fallback so the caller doesn't
        # crash. This situation arises rarely (e.g. a fold with zero positives)
        # but we don't want a whole run to abort for one pathological fold.
        if len(np.unique(y)) < 2:
            self.clf = None
            self._majority_ = int(np.bincount(y).argmax())
            self._fitted = True
            return self
        clf = LogisticRegressionCV(
            Cs=self.Cs,
            cv=self.cv,
            penalty="l2",
            solver="lbfgs",
            max_iter=self.max_iter,
            class_weight="balanced",
            scoring="roc_auc",
            refit=True,
            random_state=self.seed,
            n_jobs=1,  # keep deterministic
        )
        clf.fit(X, y)
        self.clf = clf
        self._fitted = True
        return self

    @beartype
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LogisticRegressionProbe: predict_proba before fit")
        if self.clf is None:
            # Degenerate fallback branch.
            proba = np.full(len(X), float(getattr(self, "_majority_", 0)))
            return proba
        X = X.astype(np.float32, copy=False)
        return self.clf.predict_proba(X)[:, 1].astype(np.float64)

    def weight_vector(self) -> np.ndarray | None:
        if self.clf is None:
            return None
        # LogisticRegressionCV stores coef_ as (1, d) for binary tasks.
        w = self.clf.coef_.ravel()
        n = float(np.linalg.norm(w))
        return w / n if n > 0 else w


# ---------------------------------------------------------------------------
# MLP probe
# ---------------------------------------------------------------------------


class _MLPModule(nn.Module):
    """The actual nn.Module. Kept private so Probe stays a pure NumPy-facing API."""

    def __init__(self, d: int, hidden: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPProbe(Probe):
    """Nonlinear probe. Quantifies how much *nonlinear* decoding buys you.

    If ``MLPProbe`` beats ``LogisticRegressionProbe`` by >5pp AUC on a given
    layer, the representation encodes the target nonlinearly at that layer
    (think XOR-like features). If it ties, the signal is linearly decodable
    and the extra capacity adds nothing.

    Defaults per the task brief: d -> 256 -> 1, dropout 0.2, AdamW lr=1e-3,
    weight_decay=1e-4, batch_size=32, 100 epochs with early-stopping patience=10
    on held-out validation AUC (we split 90/10 from the training fold).
    """

    name = "mlp"

    def __init__(
        self,
        hidden: int = 256,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        max_epochs: int = 100,
        patience: int = 10,
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(seed)
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: _MLPModule | None = None

    def _make_model(self, d: int) -> _MLPModule:
        torch.manual_seed(self.seed)
        return _MLPModule(d=d, hidden=self.hidden, dropout=self.dropout).to(self.device)

    @beartype
    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPProbe:
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        if len(np.unique(y)) < 2:
            # Degenerate fold — same fallback as the logistic probe.
            self.model = None
            self._majority_ = int(y.round().astype(int).mean() >= 0.5)
            self._fitted = True
            return self

        n, d = X.shape
        # Internal 90/10 split for early stopping. Stratified by y.
        rng = np.random.default_rng(self.seed)
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]
        rng.shuffle(idx_pos)
        rng.shuffle(idx_neg)
        n_val_pos = max(1, int(round(0.1 * len(idx_pos))))
        n_val_neg = max(1, int(round(0.1 * len(idx_neg))))
        val_idx = np.concatenate([idx_pos[:n_val_pos], idx_neg[:n_val_neg]])
        train_idx = np.concatenate([idx_pos[n_val_pos:], idx_neg[n_val_neg:]])
        if train_idx.size == 0 or val_idx.size == 0:
            train_idx = np.arange(n)
            val_idx = np.arange(n)
        X_tr = torch.from_numpy(X[train_idx]).to(self.device)
        y_tr = torch.from_numpy(y[train_idx]).to(self.device)
        X_va = torch.from_numpy(X[val_idx]).to(self.device)
        y_va = y[val_idx]

        model = self._make_model(d)
        opt = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # Class-balanced BCE: pos weight = n_neg / n_pos.
        n_pos = float((y_tr == 1).sum().item())
        n_neg = float((y_tr == 0).sum().item())
        pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_auc = -np.inf
        best_state: dict[str, torch.Tensor] | None = None
        bad_epochs = 0

        for _ in range(self.max_epochs):
            model.train()
            perm = torch.randperm(len(X_tr), device=self.device)
            for start in range(0, len(X_tr), self.batch_size):
                batch = perm[start : start + self.batch_size]
                if len(batch) == 0:
                    continue
                logits = model(X_tr[batch])
                loss = loss_fn(logits, y_tr[batch])
                opt.zero_grad()
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(X_va).detach().cpu().numpy()
            val_proba = 1.0 / (1.0 + np.exp(-val_logits))
            if len(np.unique(y_va)) < 2:
                val_auc = 0.5
            else:
                val_auc = float(roc_auc_score(y_va, val_proba))
            if val_auc > best_auc + 1e-6:
                best_auc = val_auc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        self._fitted = True
        return self

    @beartype
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("MLPProbe: predict_proba before fit")
        if self.model is None:
            return np.full(len(X), float(getattr(self, "_majority_", 0)))
        X = X.astype(np.float32, copy=False)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).to(self.device)).detach().cpu().numpy()
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)


# ---------------------------------------------------------------------------
# CCS probe (Burns et al. 2022) — correctness-only
# ---------------------------------------------------------------------------


class CCSProbe(Probe):
    """Contrast-Consistent Search probe (Burns et al. 2022).

    The original formulation requires a paired ``(yes_activation, no_activation)``
    per example — typically the final-token residual from the model being asked
    the claim framed positively vs negatively. Our HDF5 cache does not store
    such paired completions, so we use the common single-sample substitute:
    we treat ``(x, -x)`` as the contrast pair, which reduces CCS to finding a
    direction ``theta`` such that ``sigmoid(theta . x)`` is bimodal and
    ``sigmoid(theta . x) + sigmoid(-theta . x) ~ 1`` (trivially satisfied).
    The loss reduces to the confidence term alone:

    .. math::
        L = - E [ ( \\max(p, 1-p) - 0.5 )^2 ]

    i.e. we encourage the sigmoid output to push toward 0 or 1. With 10 random
    restarts (``n_restarts``) we pick the direction with lowest final loss,
    then align its sign with the training labels (since the probe is
    unsupervised, sign is ambiguous).

    *Caveat*: this is strictly weaker than the paired-completion CCS from the
    paper. Results should be reported as exploratory. We only run this probe
    on the ``correctness`` target (the only one that maps cleanly to the
    "truth" framing CCS was designed for).
    """

    name = "ccs"

    def __init__(
        self,
        n_restarts: int = 10,
        max_epochs: int = 1000,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(seed)
        self.n_restarts = n_restarts
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.theta_: torch.Tensor | None = None
        self.bias_: torch.Tensor | None = None
        self.sign_: int = 1

    def _train_one(self, X: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        torch.manual_seed(seed)
        d = X.shape[1]
        theta = nn.Parameter(torch.randn(d, device=self.device) / (d**0.5))
        bias = nn.Parameter(torch.zeros(1, device=self.device))
        opt = torch.optim.AdamW(
            [theta, bias], lr=self.lr, weight_decay=self.weight_decay
        )
        last_loss = float("inf")
        for _ in range(self.max_epochs):
            logits = X @ theta + bias
            p = torch.sigmoid(logits)
            # Informative loss: push away from 0.5.
            conf = torch.minimum(p, 1 - p)
            loss = (conf**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
        return theta.detach(), bias.detach(), last_loss

    @beartype
    def fit(self, X: np.ndarray, y: np.ndarray) -> CCSProbe:
        X_np = X.astype(np.float32, copy=False)
        # Normalize: zero-mean, unit-norm per feature (Burns et al. normalize
        # contrast pairs; here we just standardize to keep optimization stable).
        self._mean_ = X_np.mean(axis=0)
        X_centered = X_np - self._mean_
        X_t = torch.from_numpy(X_centered).to(self.device)

        best_loss = float("inf")
        best_theta: torch.Tensor | None = None
        best_bias: torch.Tensor | None = None
        for r in range(self.n_restarts):
            theta, bias, loss = self._train_one(X_t, seed=self.seed + r)
            if loss < best_loss:
                best_loss = loss
                best_theta = theta
                best_bias = bias

        self.theta_ = best_theta
        self.bias_ = best_bias
        self._fitted = True

        # Sign alignment: since CCS is unsupervised, whichever side is "1" is
        # arbitrary. Use the training labels just to align the sign — this does
        # NOT turn it into a supervised probe (we don't fit theta on the labels).
        if len(np.unique(y)) >= 2 and self.theta_ is not None and self.bias_ is not None:
            proba_raw = self._proba_raw(X_centered)
            auc_pos = float(roc_auc_score(y, proba_raw))
            self.sign_ = 1 if auc_pos >= 0.5 else -1
        return self

    def _proba_raw(self, X_centered: np.ndarray) -> np.ndarray:
        assert self.theta_ is not None and self.bias_ is not None
        X_t = torch.from_numpy(X_centered).to(self.device)
        with torch.no_grad():
            logits = (X_t @ self.theta_ + self.bias_).cpu().numpy()
        return 1.0 / (1.0 + np.exp(-logits))

    @beartype
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("CCSProbe: predict_proba before fit")
        X_centered = X.astype(np.float32, copy=False) - self._mean_
        p = self._proba_raw(X_centered)
        if self.sign_ < 0:
            p = 1.0 - p
        return p.astype(np.float64)

    def weight_vector(self) -> np.ndarray | None:
        if self.theta_ is None:
            return None
        w = self.theta_.detach().cpu().numpy() * self.sign_
        n = float(np.linalg.norm(w))
        return w / n if n > 0 else w


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


#: Registry of probe classes keyed by their canonical short name. ``logreg`` is
#: kept as an alias of ``logistic`` because the public task brief and the
#: results JSON use ``logreg`` while internal history uses ``logistic``; rather
#: than break either, we resolve both.
_PROBE_REGISTRY: dict[str, type[Probe]] = {
    "mass_mean": MassMeanProbe,
    "logistic": LogisticRegressionProbe,
    "logreg": LogisticRegressionProbe,
    "mlp": MLPProbe,
    "ccs": CCSProbe,
}


@beartype
def get_probe_class(name: str) -> type[Probe]:
    """Resolve a string probe name to its class."""
    if name not in _PROBE_REGISTRY:
        raise ValueError(
            f"Unknown probe {name!r}; expected one of {sorted(_PROBE_REGISTRY)}"
        )
    return _PROBE_REGISTRY[name]
