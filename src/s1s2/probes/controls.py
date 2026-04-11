"""Hewitt & Liang (2019) control tasks.

The control task swaps the real labels for random labels with the *same marginal
distribution*. If a probe can still fit the random labels well, the probe is too
expressive: whatever accuracy we see on the real task is mostly a property of the
classifier, not of the representation.

The canonical metric is **selectivity**:

.. math::
    \\text{selectivity} = \\text{real\\_accuracy} - \\text{control\\_accuracy}

Anything below ~5 percentage points should be reported with strong caveats —
it's probe expressiveness, not representational structure.

Two control-task variants are used in the literature:

1. **Permutation control** (used here). Shuffle the labels across items,
   preserving the base rate. A probe that fits these random labels well is
   memorising item identity.
2. **Random per-type control** (Hewitt & Liang's original). Map each unique
   token/item *type* to a random label, so duplicates keep consistent labels.
   We don't have enough duplicate items to do this cleanly, so we default to
   permutation.

This module owns only the helper :func:`run_control_task`. The real vs control
delta is computed in :mod:`s1s2.probes.core`.
"""

from __future__ import annotations

import numpy as np
from beartype import beartype

from s1s2.probes.probes import Probe, get_probe_class

__all__ = [
    "run_control_task",
    "shuffled_labels",
]


@beartype
def shuffled_labels(y: np.ndarray, seed: int) -> np.ndarray:
    """Return a permutation of ``y`` that preserves the marginal distribution."""
    rng = np.random.default_rng(seed)
    y_shuf = y.copy()
    rng.shuffle(y_shuf)
    return y_shuf


@beartype
def run_control_task(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    probe_name: str,
    n_seeds: int = 5,
    base_seed: int = 0,
    probe_kwargs: dict | None = None,
) -> dict[str, float]:
    """Train ``n_seeds`` control probes with shuffled labels and return mean metrics.

    Parameters
    ----------
    X_train, X_test
        Float32 activations for the two folds.
    y_train, y_test
        True labels (used only to preserve the base rate of the shuffle and to
        evaluate against — we shuffle TRAIN labels but evaluate on the TRUE
        test labels, matching Hewitt & Liang's protocol).
    probe_name
        Registry key from :mod:`s1s2.probes.probes`.
    n_seeds
        Number of shuffle restarts. The returned metrics are the mean across
        seeds. 5 is a reasonable trade-off between noise and compute.
    base_seed
        Seed offset; restart ``k`` uses ``base_seed + k``.
    probe_kwargs
        Keyword arguments forwarded to the probe constructor.

    Returns
    -------
    dict with keys ``control_roc_auc``, ``control_balanced_accuracy``,
    ``control_f1``, ``control_mcc``, ``control_std_roc_auc``,
    ``control_n_seeds``.
    """
    probe_cls = get_probe_class(probe_name)
    probe_kwargs = dict(probe_kwargs or {})
    aucs: list[float] = []
    baccs: list[float] = []
    f1s: list[float] = []
    mccs: list[float] = []
    for k in range(n_seeds):
        seed = base_seed + k
        y_train_shuf = shuffled_labels(y_train, seed=seed)
        probe: Probe = probe_cls(**{**probe_kwargs, "seed": seed})
        probe.fit(X_train, y_train_shuf)
        metrics = probe.score(X_test, y_test)
        aucs.append(metrics["roc_auc"])
        baccs.append(metrics["balanced_accuracy"])
        f1s.append(metrics["f1"])
        mccs.append(metrics["mcc"])
    return {
        "control_roc_auc": float(np.mean(aucs)),
        "control_balanced_accuracy": float(np.mean(baccs)),
        "control_f1": float(np.mean(f1s)),
        "control_mcc": float(np.mean(mccs)),
        "control_std_roc_auc": float(np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0),
        "control_n_seeds": int(n_seeds),
    }
