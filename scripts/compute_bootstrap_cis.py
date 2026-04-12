#!/usr/bin/env python3
"""Compute bootstrap 95% CIs for logistic-regression probe AUCs.

Standalone script intended to run on the pod where activations live.
No project imports required -- uses only sklearn, h5py, numpy, scipy.

For each model in the HDF5 file, for each layer, trains a logistic
regression probe (LogisticRegressionCV, inner 3-fold, class_weight=balanced)
under 5-fold stratified outer CV. Bootstrap CIs are computed by resampling
the pooled test-fold predictions 1000 times.

Four probe conditions per layer:
  1. vulnerable  -- base_rate, conjunction, syllogism (conflict items)
  2. immune      -- crt, arithmetic, framing, anchoring (conflict items)
  3. all         -- all categories (conflict items)
  4. permuted    -- Hewitt-Liang control (vulnerable categories, shuffled labels)

Selectivity = vulnerable_AUC - permuted_AUC.

Usage:
    python scripts/compute_bootstrap_cis.py \
        --h5-path data/activations/llama31_8b_instruct.h5 \
        --output-dir results/bootstrap_cis/ \
        --n-bootstrap 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── Category definitions ────────────────────────────────────────────────

VULNERABLE_CATEGORIES: list[str] = ["base_rate", "conjunction", "syllogism"]
IMMUNE_CATEGORIES: list[str] = ["crt", "arithmetic", "framing", "anchoring"]

# ── HDF5 helpers (standalone -- no project imports) ─────────────────────


def load_problem_metadata(f: h5py.File) -> dict[str, np.ndarray]:
    """Load per-problem metadata arrays from the HDF5 file."""
    out: dict[str, np.ndarray] = {}
    for key in ("id", "category", "correct_answer", "lure_answer"):
        raw = f[f"/problems/{key}"][:]
        out[key] = np.array(
            [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in raw]
        )
    out["conflict"] = f["/problems/conflict"][:]
    return out


def list_models(f: h5py.File) -> list[str]:
    return list(f["/models"].keys())


def model_n_layers(f: h5py.File, model_key: str) -> int:
    return int(f[f"/models/{model_key}/metadata"].attrs["n_layers"])


def get_residual_p0(f: h5py.File, model_key: str, layer: int) -> np.ndarray:
    """Load residual activations at position P0 for a single layer.

    Returns shape (n_problems, hidden_dim).
    """
    arr = f[f"/models/{model_key}/residual/layer_{layer:02d}"][:]
    # Position index: find P0
    labels_raw = f[f"/models/{model_key}/position_index/labels"][:]
    labels = [
        s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in labels_raw
    ]
    if "P0" not in labels:
        raise KeyError(f"P0 not found in position labels: {labels}")
    p0_idx = labels.index("P0")
    return arr[:, p0_idx, :]


# ── Core probing + bootstrap ────────────────────────────────────────────


def train_and_predict_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_outer_folds: int = 5,
    n_inner_cv: int = 3,
    max_iter: int = 5000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Train logistic regression under stratified outer CV.

    Returns (pooled_y_true, pooled_proba) -- the concatenated test-fold
    labels and predicted probabilities across all outer folds.
    """
    skf = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=seed)
    pooled_y: list[np.ndarray] = []
    pooled_proba: list[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # StandardScaler fit on train fold only (no leakage)
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train).astype(np.float32, copy=False)
        X_test = scaler.transform(X_test).astype(np.float32, copy=False)

        # Skip degenerate folds
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            pooled_y.append(y_test)
            pooled_proba.append(np.full(len(y_test), 0.5))
            continue

        clf = LogisticRegressionCV(
            Cs=np.logspace(-4, 4, 20),
            cv=n_inner_cv,
            penalty="l2",
            solver="lbfgs",
            max_iter=max_iter,
            class_weight="balanced",
            scoring="roc_auc",
            refit=True,
            random_state=seed + fold_idx,
            n_jobs=1,
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]

        pooled_y.append(y_test)
        pooled_proba.append(proba)

    return np.concatenate(pooled_y), np.concatenate(pooled_proba)


def bootstrap_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Compute bootstrap percentile CI for ROC-AUC.

    Returns dict with keys: auc, ci_lower, ci_upper, std.
    """
    n = len(y_true)
    if n < 2 or len(np.unique(y_true)) < 2:
        return {"auc": 0.5, "ci_lower": 0.5, "ci_upper": 0.5, "std": 0.0}

    observed_auc = float(roc_auc_score(y_true, y_proba))
    rng = np.random.default_rng(seed)
    boot_aucs = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_boot = y_true[idx]
        p_boot = y_proba[idx]
        if len(np.unique(y_boot)) < 2:
            boot_aucs[i] = 0.5
        else:
            boot_aucs[i] = roc_auc_score(y_boot, p_boot)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_aucs, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_aucs, 100 * (1 - alpha / 2)))
    std = float(np.std(boot_aucs, ddof=1))

    return {
        "auc": observed_auc,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": std,
    }


def run_permuted_control(
    X: np.ndarray,
    y: np.ndarray,
    n_shuffles: int = 3,
    n_outer_folds: int = 5,
    n_inner_cv: int = 3,
    max_iter: int = 5000,
    seed: int = 0,
) -> float:
    """Hewitt-Liang permuted-label control: mean AUC across shuffle restarts.

    Shuffles the *training* labels while evaluating against *true* test labels,
    matching the protocol in controls.py.
    """
    aucs: list[float] = []
    for k in range(n_shuffles):
        shuf_seed = seed + 10000 + k
        rng = np.random.default_rng(shuf_seed)
        y_shuffled = y.copy()
        rng.shuffle(y_shuffled)

        skf = StratifiedKFold(
            n_splits=n_outer_folds, shuffle=True, random_state=seed
        )
        fold_aucs: list[float] = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train_shuf = y_shuffled[train_idx]
            y_test_true = y[test_idx]

            scaler = StandardScaler(with_mean=True, with_std=True)
            scaler.fit(X_train)
            X_train = scaler.transform(X_train).astype(np.float32, copy=False)
            X_test = scaler.transform(X_test).astype(np.float32, copy=False)

            if len(np.unique(y_train_shuf)) < 2 or len(np.unique(y_test_true)) < 2:
                fold_aucs.append(0.5)
                continue

            clf = LogisticRegressionCV(
                Cs=np.logspace(-4, 4, 20),
                cv=n_inner_cv,
                penalty="l2",
                solver="lbfgs",
                max_iter=max_iter,
                class_weight="balanced",
                scoring="roc_auc",
                refit=True,
                random_state=shuf_seed + fold_idx,
                n_jobs=1,
            )
            clf.fit(X_train, y_train_shuf)
            proba = clf.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test_true, proba))
            fold_aucs.append(auc)

        aucs.append(float(np.mean(fold_aucs)))

    return float(np.mean(aucs))


# ── Subset builders ─────────────────────────────────────────────────────


def build_subset(
    X_all: np.ndarray,
    conflict: np.ndarray,
    categories: np.ndarray,
    category_list: list[str] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Filter to conflict items in the given categories.

    Returns (X_subset, y_subset) where y = conflict flag, or None if
    fewer than 10 items or only one class present.
    """
    if category_list is not None:
        cat_mask = np.isin(categories, category_list)
    else:
        cat_mask = np.ones(len(categories), dtype=bool)

    mask = cat_mask
    X_sub = X_all[mask]
    y_sub = conflict[mask].astype(np.int64)

    if len(y_sub) < 10 or len(np.unique(y_sub)) < 2:
        return None
    return X_sub, y_sub


# ── Main pipeline ───────────────────────────────────────────────────────


def run_model(
    f: h5py.File,
    model_key: str,
    meta: dict[str, np.ndarray],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Run all probes for one model across all layers."""
    n_layers = model_n_layers(f, model_key)
    categories = meta["category"]
    conflict = meta["conflict"].astype(np.int64)

    conditions = {
        "vulnerable": VULNERABLE_CATEGORIES,
        "immune": IMMUNE_CATEGORIES,
        "all": None,  # all categories
    }

    results: dict[str, Any] = {
        "model": model_key,
        "n_layers": n_layers,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "conditions": {},
    }

    for cond_name, cat_list in conditions.items():
        print(f"\n  Condition: {cond_name}")
        layer_results: list[dict[str, Any]] = []

        for layer in range(n_layers):
            t0 = time.time()
            X_p0 = get_residual_p0(f, model_key, layer)
            subset = build_subset(X_p0, conflict, categories, cat_list)

            if subset is None:
                print(f"    Layer {layer:2d}: SKIPPED (insufficient data)")
                layer_results.append({
                    "layer": layer,
                    "skipped": True,
                    "reason": "insufficient data",
                })
                continue

            X, y = subset
            n_pos = int(y.sum())
            n_neg = int(len(y) - n_pos)

            # Real probe
            pooled_y, pooled_proba = train_and_predict_cv(
                X, y, n_outer_folds=5, n_inner_cv=3, max_iter=5000, seed=seed,
            )
            boot = bootstrap_auc(
                pooled_y, pooled_proba,
                n_resamples=n_bootstrap, seed=seed,
            )

            entry: dict[str, Any] = {
                "layer": layer,
                "skipped": False,
                "n_samples": len(y),
                "n_conflict": n_pos,
                "n_no_conflict": n_neg,
                "auc": boot["auc"],
                "ci_lower": boot["ci_lower"],
                "ci_upper": boot["ci_upper"],
                "std": boot["std"],
            }

            # Permuted-label control only for the vulnerable condition
            if cond_name == "vulnerable":
                ctrl_auc = run_permuted_control(
                    X, y,
                    n_shuffles=3,
                    n_outer_folds=5,
                    n_inner_cv=3,
                    max_iter=5000,
                    seed=seed,
                )
                selectivity = boot["auc"] - ctrl_auc
                entry["control_auc"] = ctrl_auc
                entry["selectivity"] = selectivity
            elapsed = time.time() - t0
            entry["elapsed_s"] = round(elapsed, 2)

            ci_str = f"[{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]"
            sel_str = ""
            if cond_name == "vulnerable":
                sel_str = f"  sel={entry['selectivity']:.3f}"
            print(
                f"    Layer {layer:2d}: AUC={boot['auc']:.3f} "
                f"CI={ci_str}{sel_str}  ({elapsed:.1f}s)"
            )

            layer_results.append(entry)

        results["conditions"][cond_name] = layer_results

    return results


def print_summary_table(results: dict[str, Any]) -> None:
    """Print a compact summary table to stdout."""
    model = results["model"]
    print(f"\n{'='*80}")
    print(f"  BOOTSTRAP CI SUMMARY: {model}")
    print(f"{'='*80}")

    for cond_name in ("vulnerable", "immune", "all"):
        layers = results["conditions"].get(cond_name, [])
        if not layers:
            continue
        print(f"\n  --- {cond_name.upper()} ---")
        header = f"  {'Layer':>5s}  {'AUC':>6s}  {'CI_lo':>6s}  {'CI_hi':>6s}  {'Std':>6s}"
        if cond_name == "vulnerable":
            header += f"  {'Ctrl':>6s}  {'Sel':>6s}"
        print(header)
        print(f"  {'-'*len(header)}")

        for entry in layers:
            if entry.get("skipped"):
                print(f"  {entry['layer']:5d}  {'---':>6s}  {'---':>6s}  {'---':>6s}  {'---':>6s}")
                continue
            row = (
                f"  {entry['layer']:5d}"
                f"  {entry['auc']:6.3f}"
                f"  {entry['ci_lower']:6.3f}"
                f"  {entry['ci_upper']:6.3f}"
                f"  {entry['std']:6.3f}"
            )
            if cond_name == "vulnerable":
                row += f"  {entry.get('control_auc', float('nan')):6.3f}"
                row += f"  {entry.get('selectivity', float('nan')):+6.3f}"
            print(row)

    # Peak layer for vulnerable condition
    vuln = results["conditions"].get("vulnerable", [])
    valid = [e for e in vuln if not e.get("skipped")]
    if valid:
        best = max(valid, key=lambda e: e["auc"])
        print(f"\n  Peak vulnerable layer: {best['layer']} "
              f"(AUC={best['auc']:.3f} "
              f"[{best['ci_lower']:.3f}, {best['ci_upper']:.3f}]"
              f", selectivity={best.get('selectivity', float('nan')):.3f})")
    print()


# ── JSON serialization helper ───────────────────────────────────────────


def as_python(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [as_python(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): as_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_python(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ── CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute bootstrap 95% CIs for probe AUCs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--h5-path",
        type=str,
        required=True,
        help="Path to the HDF5 activation file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/bootstrap_cis",
        help="Directory to write JSON results. Default: results/bootstrap_cis/",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples. Default: 1000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Master random seed. Default: 0.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run only this model key. Default: all models in the HDF5.",
    )
    args = parser.parse_args()

    h5_path = Path(args.h5_path)
    if not h5_path.exists():
        print(f"ERROR: HDF5 file not found: {h5_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {h5_path}")
    print(f"Output:  {output_dir}")
    print(f"Bootstrap resamples: {args.n_bootstrap}")
    print(f"Seed: {args.seed}")

    with h5py.File(h5_path, "r") as f:
        meta = load_problem_metadata(f)
        models = [args.model] if args.model else list_models(f)
        n_problems = len(meta["category"])
        n_conflict = int(meta["conflict"].sum())
        print(f"\nProblems: {n_problems} total, {n_conflict} conflict")

        vuln_mask = np.isin(meta["category"], VULNERABLE_CATEGORIES)
        imm_mask = np.isin(meta["category"], IMMUNE_CATEGORIES)
        print(f"Vulnerable categories ({len(VULNERABLE_CATEGORIES)}): "
              f"{int(vuln_mask.sum())} items "
              f"({int((meta['conflict'][vuln_mask]).sum())} conflict)")
        print(f"Immune categories ({len(IMMUNE_CATEGORIES)}): "
              f"{int(imm_mask.sum())} items "
              f"({int((meta['conflict'][imm_mask]).sum())} conflict)")

        for model_key in models:
            print(f"\n{'='*60}")
            print(f"Model: {model_key}")
            print(f"{'='*60}")
            t_start = time.time()

            results = run_model(
                f, model_key, meta,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )

            results["elapsed_total_s"] = round(time.time() - t_start, 1)
            results["h5_path"] = str(h5_path)

            # Save JSON
            out_path = output_dir / f"{model_key}_bootstrap_cis.json"
            with open(out_path, "w") as fp:
                json.dump(as_python(results), fp, indent=2, allow_nan=True)
            print(f"\nSaved: {out_path}")

            print_summary_table(results)

    print("Done.")


if __name__ == "__main__":
    main()
