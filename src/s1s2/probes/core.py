"""ProbeRunner: orchestrates training, CV, controls, bootstrap, permutation.

This is the workhorse of the probing workstream. Given a configured
:class:`ProbeRunner`, the caller hands it an ``(X, y)`` bundle plus target
metadata, and gets back a fully populated result dict ready to serialize.

Pipeline per (model, layer, position, target):

1. Build target (handled by the caller via :mod:`s1s2.probes.targets`).
2. Split with :class:`StratifiedKFold` over a cantor-paired ``(target,
   stratify_key)`` so folds preserve both target-label and task-category
   balance.
3. For each configured probe type, for each outer fold:
    a. Fit on the training fold.
    b. Score on the test fold.
    c. Run the Hewitt & Liang control probe (same classifier, shuffled labels).
4. Aggregate across folds into a single metric dict per probe.
5. Run a permutation test on the real vs shuffled labels (label-perm null of
   the concatenated test predictions).
6. Bootstrap-CI the concatenated test predictions.
7. (Optionally) run leave-one-category-out transfer.

Everything stochastic respects a master seed.

Outputs are plain dicts — serialization is the caller's job.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from beartype import beartype
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from s1s2.probes.controls import run_control_task
from s1s2.probes.probes import Probe, get_probe_class
from s1s2.probes.targets import TargetData
from s1s2.utils import io as ioh
from s1s2.utils.logging import get_logger
from s1s2.utils.stats import bh_fdr, bootstrap_ci

__all__ = [
    "FoldResult",
    "LayerResult",
    "ProbeResult",
    "ProbeRunner",
    "RunnerConfig",
    "apply_bh_across_layers",
    "git_sha",
    "layer_result_to_dict",
    "load_layer_activations",
    "loco_split_iter",
    "make_stratify_key",
    "primary_probe_name",
    "save_layer_result",
]


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """Knobs that govern a run.

    Kept as a plain dataclass (not a Hydra config) so :class:`ProbeRunner`
    stays independent of Hydra and is trivially unit-testable.

    The CLI in :mod:`s1s2.probes.cli` is responsible for translating Hydra
    config (``configs/probe.yaml``) into one of these.
    """

    probes: tuple[str, ...] = ("mass_mean", "logistic", "mlp")
    n_folds: int = 5
    n_seeds: int = 3
    control_enabled: bool = True
    control_n_shuffles: int = 3
    n_permutations: int = 1000
    n_bootstrap: int = 1000
    permutation_alpha: float = 0.05
    run_loco: bool = True
    loco_targets: tuple[str, ...] = (
        "task_type",
        "correctness",
        "bias_susceptible",
        "processing_mode",
    )
    probe_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    seed: int = 0
    #: Apply ``StandardScaler`` (fit on train-fold only) before every probe
    #: fit. NON-NEGOTIABLE per the task brief — turning it off is a bug unless
    #: debugging a probe's scale invariance.
    standardize: bool = True


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """Per-fold metrics plus concatenable arrays for pooled analysis."""

    fold_idx: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    real_metrics: dict[str, float]
    control_metrics: dict[str, float] | None
    test_proba: np.ndarray
    test_y: np.ndarray


@dataclass
class ProbeResult:
    """Aggregate per-probe metrics across all folds."""

    name: str
    fold_metrics: list[dict[str, float]]
    pooled_y: np.ndarray
    pooled_proba: np.ndarray
    control_metrics: list[dict[str, float]]
    summary: dict[str, float]
    permutation_p: float | None = None
    permutation_null_auc: np.ndarray | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (without raw vectors)."""
        out = {
            "summary": self.summary,
            "fold_metrics": self.fold_metrics,
            "control_metrics": self.control_metrics,
        }
        if self.permutation_p is not None:
            out["permutation_p"] = float(self.permutation_p)
        return out


@dataclass
class LayerResult:
    """Everything a per-layer probe run produces."""

    model: str
    layer: int
    position: str
    target: str
    n_problems: int
    n_train_per_fold: list[int]
    n_test_per_fold: list[int]
    probes: dict[str, ProbeResult]
    loco: dict[str, Any] | None
    config: dict[str, Any]
    git_sha: str
    elapsed_s: float


# ---------------------------------------------------------------------------
# Stratified CV helpers
# ---------------------------------------------------------------------------


@beartype
def make_stratify_key(y: np.ndarray, aux: np.ndarray) -> np.ndarray:
    """Combine a binary target label and an auxiliary key into a single strata.

    We hash ``aux`` to dense ints then cantor-pair with ``y``. This gives
    :class:`StratifiedKFold` something to stratify on that preserves both
    target-label balance AND the auxiliary balance (typically task category).

    Cantor pair: ``(a, b) -> (a + b) * (a + b + 1) / 2 + b``. Safe for
    non-negative ints in reasonable ranges.
    """
    if len(y) != len(aux):
        raise ValueError(f"length mismatch: {len(y)} vs {len(aux)}")
    # Encode aux to dense ints.
    _, aux_int = np.unique(aux, return_inverse=True)
    y_int = y.astype(np.int64)
    a = aux_int.astype(np.int64)
    # Cantor pairing — outputs a unique non-negative int per (y, a) pair.
    key = (y_int + a) * (y_int + a + 1) // 2 + a
    return key


# ---------------------------------------------------------------------------
# Git SHA helper
# ---------------------------------------------------------------------------


def git_sha() -> str:
    """Return the short git SHA of the current checkout, or ``"unknown"``."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Probe selection helper
# ---------------------------------------------------------------------------


@beartype
def primary_probe_name(probes: tuple[str, ...] | list[str]) -> str | None:
    """Return the canonical "primary" probe name from a list of probes.

    The primary probe is the one whose results we lead with in figures and
    summaries — for the s1s2 project this is logistic regression. If
    logistic isn't in the list we fall back to MLP, then mass-mean, then
    the first probe. Returns ``None`` only if the list is empty.

    Used for downstream analyses that need a single probe to focus on
    (LOCO transfer, BH-FDR significance flagging, etc.).
    """
    if not probes:
        return None
    preferred_order = ("logistic", "mlp", "mass_mean", "ccs")
    for name in preferred_order:
        if name in probes:
            return name
    return probes[0]


# ---------------------------------------------------------------------------
# LOCO split iterator
# ---------------------------------------------------------------------------


@beartype
def loco_split_iter(
    y: np.ndarray,
    group_id: np.ndarray,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(held_out_group, train_idx, test_idx)`` tuples.

    Each iteration trains on all items whose group is not the held-out one and
    tests on the held-out items. Groups with only one class are skipped (can't
    test AUC on a single-class fold).
    """
    splits: list[tuple[str, np.ndarray, np.ndarray]] = []
    uniq = np.unique(group_id)
    for g in uniq:
        test_mask = group_id == g
        train_mask = ~test_mask
        y_test = y[test_mask]
        y_train = y[train_mask]
        if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
            continue
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]
        splits.append((str(g), train_idx, test_idx))
    return splits


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


class ProbeRunner:
    """Train one probe bundle at one (layer, position, target).

    Usage::

        runner = ProbeRunner(config)
        layer_result = runner.run(
            X=X, target_data=td, model="llama...", layer=16, position="P0",
        )
    """

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config

    # -- Fold iteration --------------------------------------------------

    def _make_folds(
        self, y: np.ndarray, stratify_key: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return a list of ``(train_idx, test_idx)`` pairs.

        Uses cantor-paired ``(y, aux)`` as the stratification key. Falls back
        to stratifying on ``y`` alone if the combined key has too few items
        per stratum for k-fold to proceed.
        """
        key = make_stratify_key(y, stratify_key)
        # Some strata may have <n_folds members — sklearn will error in that
        # case. Merge rare strata into a single "other" bucket.
        counts = dict(zip(*np.unique(key, return_counts=True), strict=True))
        min_per_fold = self.config.n_folds
        rare = {v for v, c in counts.items() if c < min_per_fold}
        if rare:
            key = np.where(np.isin(key, list(rare)), -1, key)
        # If even after merging we don't have enough samples per stratum, fall
        # back to just y.
        counts = dict(zip(*np.unique(key, return_counts=True), strict=True))
        if any(c < min_per_fold for c in counts.values()):
            key = y
        skf = StratifiedKFold(
            n_splits=self.config.n_folds, shuffle=True, random_state=self.config.seed
        )
        return list(skf.split(np.zeros_like(y), key))

    # -- Per-probe loop --------------------------------------------------

    def _run_one_probe(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        folds: list[tuple[np.ndarray, np.ndarray]],
    ) -> ProbeResult:
        probe_cls = get_probe_class(name)
        probe_kwargs = dict(self.config.probe_kwargs.get(name, {}))

        fold_metrics: list[dict[str, float]] = []
        pooled_y: list[np.ndarray] = []
        pooled_proba: list[np.ndarray] = []
        control_metrics: list[dict[str, float]] = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            # --- Standardize on the train fold only (no leakage) -----------
            # Fitting on the full X would leak test statistics into the train
            # features via the mean/std. We keep the scaled arrays local to
            # this fold so downstream probes all see the same transformation.
            X_train_raw = X[train_idx]
            X_test_raw = X[test_idx]
            if self.config.standardize:
                scaler = StandardScaler(with_mean=True, with_std=True)
                scaler.fit(X_train_raw)
                X_train = scaler.transform(X_train_raw).astype(np.float32, copy=False)
                X_test = scaler.transform(X_test_raw).astype(np.float32, copy=False)
            else:
                X_train = X_train_raw
                X_test = X_test_raw

            # Multi-seed average for the probe fit itself. For stateless/tiny
            # variance probes (MassMean, Logistic) this is cheap.
            seed_aucs: list[float] = []
            seed_metrics: list[dict[str, float]] = []
            seed_probas: list[np.ndarray] = []
            for s in range(self.config.n_seeds):
                seed = self.config.seed + 100 * fold_idx + s
                probe: Probe = probe_cls(**{**probe_kwargs, "seed": seed})
                probe.fit(X_train, y[train_idx])
                metrics = probe.score(X_test, y[test_idx])
                seed_aucs.append(metrics["roc_auc"])
                seed_metrics.append(metrics)
                seed_probas.append(probe.predict_proba(X_test))

            # Average the probas across seeds to get a stable pooled prediction.
            avg_proba = np.mean(np.stack(seed_probas, axis=0), axis=0)
            y_test = y[test_idx]
            mean_metrics = {
                k: float(np.mean([m[k] for m in seed_metrics]))
                for k in seed_metrics[0]
            }
            mean_metrics["roc_auc_std"] = float(
                np.std([m["roc_auc"] for m in seed_metrics], ddof=1)
                if len(seed_metrics) > 1
                else 0.0
            )
            fold_metrics.append(mean_metrics)
            pooled_y.append(y_test)
            pooled_proba.append(avg_proba)

            if self.config.control_enabled:
                ctrl = run_control_task(
                    X_train,
                    X_test,
                    y[train_idx],
                    y[test_idx],
                    probe_name=name,
                    n_seeds=self.config.control_n_shuffles,
                    base_seed=self.config.seed + 1000 * fold_idx,
                    probe_kwargs=probe_kwargs,
                )
                control_metrics.append(ctrl)

        pooled_y_arr = np.concatenate(pooled_y)
        pooled_proba_arr = np.concatenate(pooled_proba)

        summary = self._summarize_probe(
            fold_metrics, control_metrics, pooled_y_arr, pooled_proba_arr
        )

        # Permutation test on the pooled predictions: shuffle pooled_y, compute
        # AUC, compare to the observed AUC.
        perm_p, perm_null = self._permutation_test(pooled_y_arr, pooled_proba_arr)
        summary["permutation_p"] = float(perm_p)
        summary["permutation_null_auc_95"] = float(np.percentile(perm_null, 95))

        return ProbeResult(
            name=name,
            fold_metrics=fold_metrics,
            pooled_y=pooled_y_arr,
            pooled_proba=pooled_proba_arr,
            control_metrics=control_metrics,
            summary=summary,
            permutation_p=float(perm_p),
            permutation_null_auc=perm_null,
        )

    # -- Aggregation helpers -------------------------------------------

    def _summarize_probe(
        self,
        fold_metrics: list[dict[str, float]],
        control_metrics: list[dict[str, float]],
        pooled_y: np.ndarray,
        pooled_proba: np.ndarray,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for k in ("roc_auc", "balanced_accuracy", "f1", "mcc"):
            vals = np.array([m[k] for m in fold_metrics])
            out[f"{k}_mean"] = float(vals.mean())
            out[f"{k}_std"] = float(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        # Primary metric: pooled AUC is more stable than mean-of-fold AUC on
        # small folds.
        if len(np.unique(pooled_y)) >= 2:
            out["roc_auc"] = float(roc_auc_score(pooled_y, pooled_proba))
        else:
            out["roc_auc"] = 0.5
        # Bootstrap CI on pooled AUC.
        if len(pooled_y) > 5 and len(np.unique(pooled_y)) >= 2:
            idx = np.arange(len(pooled_y))
            stat = lambda i: (  # noqa: E731
                float(roc_auc_score(pooled_y[i.astype(int)], pooled_proba[i.astype(int)]))
                if len(np.unique(pooled_y[i.astype(int)])) >= 2
                else 0.5
            )
            point, lo, hi = bootstrap_ci(
                idx.astype(np.float64),
                stat,
                n_resamples=self.config.n_bootstrap,
                confidence=0.95,
                seed=self.config.seed,
            )
            out["roc_auc_ci_lower"] = float(lo)
            out["roc_auc_ci_upper"] = float(hi)
        else:
            out["roc_auc_ci_lower"] = float(out["roc_auc"])
            out["roc_auc_ci_upper"] = float(out["roc_auc"])
        # Hewitt & Liang selectivity.
        if control_metrics:
            ctrl_auc = float(np.mean([m["control_roc_auc"] for m in control_metrics]))
            out["control_roc_auc"] = ctrl_auc
            out["selectivity"] = float(out["roc_auc"] - ctrl_auc)
        return out

    def _permutation_test(
        self, y: np.ndarray, proba: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Label-permutation null for ROC-AUC.

        Hypothesis: ``AUC(y, proba) > AUC(shuffle(y), proba)``. One-sided.
        Returns (p, null_distribution). Uses the North et al. +1 correction.
        """
        if len(np.unique(y)) < 2:
            return 1.0, np.full(self.config.n_permutations, 0.5)
        obs = float(roc_auc_score(y, proba))
        rng = np.random.default_rng(self.config.seed)
        null = np.empty(self.config.n_permutations, dtype=np.float64)
        y_copy = y.copy()
        for i in range(self.config.n_permutations):
            rng.shuffle(y_copy)
            if len(np.unique(y_copy)) < 2:
                null[i] = 0.5
                continue
            null[i] = float(roc_auc_score(y_copy, proba))
        n_extreme = int(np.sum(null >= obs))
        p = (n_extreme + 1) / (self.config.n_permutations + 1)
        return p, null

    # -- LOCO --------------------------------------------------------

    def _run_loco(
        self,
        probe_name: str,
        X: np.ndarray,
        target_data: TargetData,
    ) -> dict[str, Any]:
        """Train on all-but-one group, test on the held out group.

        Also applies per-fold ``StandardScaler`` on the training side only, so
        the transfer metric is directly comparable to the CV metric.
        """
        probe_cls = get_probe_class(probe_name)
        probe_kwargs = dict(self.config.probe_kwargs.get(probe_name, {}))
        splits = loco_split_iter(target_data.y, target_data.group_id)
        per_group: list[dict[str, Any]] = []
        aucs: list[float] = []
        for held_out, tr_idx, te_idx in splits:
            X_tr_raw = X[tr_idx]
            X_te_raw = X[te_idx]
            if self.config.standardize:
                scaler = StandardScaler(with_mean=True, with_std=True)
                scaler.fit(X_tr_raw)
                X_tr = scaler.transform(X_tr_raw).astype(np.float32, copy=False)
                X_te = scaler.transform(X_te_raw).astype(np.float32, copy=False)
            else:
                X_tr = X_tr_raw
                X_te = X_te_raw
            probe: Probe = probe_cls(**{**probe_kwargs, "seed": self.config.seed})
            probe.fit(X_tr, target_data.y[tr_idx])
            metrics = probe.score(X_te, target_data.y[te_idx])
            per_group.append({"held_out": held_out, **metrics})
            aucs.append(metrics["roc_auc"])
        return {
            "probe": probe_name,
            "per_group": per_group,
            "mean_roc_auc": float(np.mean(aucs)) if aucs else 0.5,
            "std_roc_auc": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            "n_groups": len(per_group),
        }

    # -- Public entry point -----------------------------------------

    @beartype
    def run(
        self,
        X: np.ndarray,
        target_data: TargetData,
        model: str,
        layer: int,
        position: str,
    ) -> LayerResult:
        """Run the full probe bundle on one (model, layer, position, target)."""
        started = time.time()
        if X.shape[0] != target_data.mask.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but target mask has "
                f"{target_data.mask.shape[0]}; apply the mask before calling run()"
            )
        X_masked = X[target_data.mask]
        y = target_data.y
        if X_masked.shape[0] != y.shape[0]:
            raise ValueError(
                f"masked X ({X_masked.shape[0]}) != len(y) ({y.shape[0]})"
            )

        folds = self._make_folds(y, target_data.stratify_key)

        probes_result: dict[str, ProbeResult] = {}
        for probe_name in self.config.probes:
            # CCS is only meaningful on the correctness target — see the
            # docstring in probes.CCSProbe.
            if probe_name == "ccs" and target_data.target != "correctness":
                continue
            pr = self._run_one_probe(probe_name, X_masked, y, folds)
            probes_result[probe_name] = pr

        loco_out: dict[str, Any] | None = None
        if self.config.run_loco and target_data.target in self.config.loco_targets:
            # Use the primary probe (logistic) for LOCO — matches the task brief.
            primary = primary_probe_name(self.config.probes)
            if primary is not None:
                loco_out = self._run_loco(primary, X_masked, target_data)

        # Record n_train/n_test counts from the folds.
        n_train = [len(t) for t, _ in folds]
        n_test = [len(t) for _, t in folds]

        return LayerResult(
            model=model,
            layer=layer,
            position=position,
            target=target_data.target,
            n_problems=int(X_masked.shape[0]),
            n_train_per_fold=n_train,
            n_test_per_fold=n_test,
            probes=probes_result,
            loco=loco_out,
            config={
                "probes": list(self.config.probes),
                "n_folds": self.config.n_folds,
                "n_seeds": self.config.n_seeds,
                "control_enabled": self.config.control_enabled,
                "control_n_shuffles": self.config.control_n_shuffles,
                "n_permutations": self.config.n_permutations,
                "n_bootstrap": self.config.n_bootstrap,
                "seed": self.config.seed,
            },
            git_sha=git_sha(),
            elapsed_s=float(time.time() - started),
        )


# ---------------------------------------------------------------------------
# Serialization + BH-FDR across layers
# ---------------------------------------------------------------------------


def _as_py(obj: Any) -> Any:
    """Recursively coerce numpy scalars / arrays into JSON-native types."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_as_py(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _as_py(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_as_py(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


@beartype
def layer_result_to_dict(result: LayerResult) -> dict[str, Any]:
    """Flatten a :class:`LayerResult` into a JSON-serialisable dict.

    Mirrors the schema in the task brief: each probe entry reports ``auc_mean``,
    ``auc_std``, ``selectivity``, bootstrap CI, and (for the primary probe) the
    permutation p-value. LOCO transfer, config hash, and git SHA are top-level.
    """
    # Build the per-probe summary blob. We map the internal LogReg name
    # ``logistic`` to the brief's ``logreg`` when writing to disk so the schema
    # matches across both code paths.
    probes_out: dict[str, Any] = {}
    brief_name_map = {"logistic": "logreg"}
    for name, pr in result.probes.items():
        out_name = brief_name_map.get(name, name)
        summary = dict(pr.summary)
        entry: dict[str, Any] = {
            "auc_mean": summary.get("roc_auc", summary.get("roc_auc_mean", float("nan"))),
            "auc_std": summary.get("roc_auc_std", float("nan")),
            "balanced_accuracy_mean": summary.get("balanced_accuracy_mean", float("nan")),
            "f1_mean": summary.get("f1_mean", float("nan")),
            "mcc_mean": summary.get("mcc_mean", float("nan")),
            "selectivity": summary.get("selectivity", float("nan")),
            "control_auc_mean": summary.get("control_roc_auc", float("nan")),
            "ci": [
                summary.get("roc_auc_ci_lower", float("nan")),
                summary.get("roc_auc_ci_upper", float("nan")),
            ],
            "p_perm": summary.get("permutation_p", float("nan")),
            "n_folds": len(pr.fold_metrics),
        }
        if "permutation_p_bh" in summary:
            entry["p_perm_bh"] = summary["permutation_p_bh"]
        if "significant_bh" in summary:
            entry["significant_bh"] = summary["significant_bh"]
        probes_out[out_name] = entry

    # Ensure all brief-defined probe keys exist (None if we didn't run them),
    # so consumers can rely on the schema shape.
    for key in ("mass_mean", "logreg", "mlp", "ccs"):
        probes_out.setdefault(key, None)

    out: dict[str, Any] = {
        "model": result.model,
        "target": result.target,
        "layer": int(result.layer),
        "position": result.position,
        "n_samples": int(result.n_problems),
        "n_train_per_fold": list(result.n_train_per_fold),
        "n_test_per_fold": list(result.n_test_per_fold),
        "probes": probes_out,
        "leave_one_category_out": result.loco,
        "config": result.config,
        "config_hash": result.config.get("config_hash", ""),
        "git_sha": result.git_sha,
        "wandb_run": result.config.get("wandb_run"),
        "elapsed_s": float(result.elapsed_s),
    }
    return _as_py(out)


@beartype
def save_layer_result(
    result: LayerResult, results_dir: str | Path, flat: bool = True
) -> Path:
    """Write a :class:`LayerResult` to disk.

    By default (``flat=True``) the filename follows the task brief:
    ``{model}_{target}_layer{NN}_{position}.json`` in a single directory. Set
    ``flat=False`` for the older ``{model}/{target}/layer_NN_pos_P.json``
    nested layout if a downstream tool depends on it.
    """
    rd = Path(results_dir)
    if flat:
        rd.mkdir(parents=True, exist_ok=True)
        fname = (
            f"{result.model}_{result.target}_layer{result.layer:02d}_"
            f"{result.position}.json"
        )
        path = rd / fname
    else:
        rd_nested = rd / result.model / result.target
        rd_nested.mkdir(parents=True, exist_ok=True)
        path = rd_nested / f"layer_{result.layer:02d}_pos_{result.position}.json"
    with path.open("w") as fh:
        json.dump(layer_result_to_dict(result), fh, indent=2, allow_nan=True)
    return path


@beartype
def apply_bh_across_layers(
    results: list[LayerResult],
    probe_name: str = "logistic",
    q: float = 0.05,
) -> list[LayerResult]:
    """Annotate per-layer results with a BH-FDR-corrected p-value.

    Expects ``results`` to be the collection of layer results for ONE (model,
    target, position) combination. The correction is applied across layers for
    the selected ``probe_name``. Mutates the ``summary`` dicts in place and
    returns ``results`` for chaining.
    """
    pvals: list[float] = []
    idxs: list[int] = []
    for i, r in enumerate(results):
        pr = r.probes.get(probe_name)
        if pr is None or pr.permutation_p is None:
            continue
        pvals.append(float(pr.permutation_p))
        idxs.append(i)
    if not pvals:
        return results
    rejected, q_adj = bh_fdr(np.array(pvals, dtype=np.float64), q=q)
    for j, i in enumerate(idxs):
        pr = results[i].probes[probe_name]
        pr.summary["permutation_p_bh"] = float(q_adj[j])
        pr.summary["significant_bh"] = bool(rejected[j])
    return results


# ---------------------------------------------------------------------------
# Reading activations for a run (thin wrapper around utils.io)
# ---------------------------------------------------------------------------


@beartype
def load_layer_activations(
    activations_path: str | Path,
    model_key_hdf5: str,
    layer: int,
    position: str,
) -> tuple[np.ndarray, bool]:
    """Return ``(X, is_position_valid)`` for one layer+position.

    Returns ``X`` as float32 and a flag indicating whether the position is
    valid for this model (non-reasoning models have T-positions but they are
    invalid). Callers should skip the layer if ``is_position_valid`` is False.
    """
    with ioh.open_activations(activations_path) as f:
        labels = ioh.position_labels(f, model_key_hdf5)
        valid = ioh.position_valid(f, model_key_hdf5)
        if position not in labels:
            raise KeyError(f"position {position!r} not in {labels}")
        pos_idx = labels.index(position)
        # "valid" is (n_problems, n_positions). We need any-valid — if none of
        # the problems has this position valid, the position was not extracted.
        is_valid = bool(valid[:, pos_idx].any())
        if not is_valid:
            return np.empty((0, 0), dtype=np.float32), False
        arr = ioh.get_residual(f, model_key_hdf5, layer=layer, position=position)
    return arr.astype(np.float32, copy=False), True
