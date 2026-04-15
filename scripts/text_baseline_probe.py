"""Text-only baseline probe: can prompt text alone distinguish conflict from control?

This is the highest-ROI confound control. If a sentence-transformer embedding
of the raw prompt text achieves AUC comparable to the activation probe, then
the activation probe may simply be detecting surface text features rather than
a genuine internal "deliberation" representation.

Protocol mirrors the activation probes:
  - 5-fold stratified CV (stratified by category x label)
  - LogisticRegressionCV (L2, same as activation probes)
  - Hewitt & Liang control (shuffled-label baseline)
  - Permutation test (1000 shuffles, North et al. +1 correction)
  - Bootstrap CI (1000 resamples)
  - Cross-prediction: train on vulnerable text, test on immune text

Runs entirely on CPU. No GPU or HDF5 files needed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_PATH = Path(__file__).resolve().parent.parent / "data" / "benchmark" / "benchmark.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "probes"
RESULTS_PATH = RESULTS_DIR / "text_baseline_probe.json"

VULNERABLE_CATEGORIES = {"base_rate", "conjunction", "syllogism"}
IMMUNE_CATEGORIES = {"crt", "arithmetic", "framing", "anchoring"}

N_FOLDS = 5
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 1000
SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_benchmark() -> list[dict]:
    """Load benchmark items from JSONL."""
    items = []
    with open(BENCHMARK_PATH) as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items


# ---------------------------------------------------------------------------
# Stratified CV key (matches activation probe protocol)
# ---------------------------------------------------------------------------


def make_stratify_key(y: np.ndarray, categories: np.ndarray) -> np.ndarray:
    """Cantor-pair (label, category) for stratified splitting."""
    _, cat_int = np.unique(categories, return_inverse=True)
    y_int = y.astype(np.int64)
    a = cat_int.astype(np.int64)
    return (y_int + a) * (y_int + a + 1) // 2 + a


# ---------------------------------------------------------------------------
# Core probe evaluation
# ---------------------------------------------------------------------------


def evaluate_probe(
    X: np.ndarray,
    y: np.ndarray,
    categories: np.ndarray,
    label: str = "all",
) -> dict:
    """Run 5-fold CV logistic regression + controls + permutation test + bootstrap CI."""
    n_samples = len(y)
    n_pos = int(y.sum())
    n_neg = n_samples - n_pos

    if n_samples < 10 or n_pos < 3 or n_neg < 3:
        return {
            "label": label,
            "n_samples": n_samples,
            "n_conflict": n_pos,
            "n_control": n_neg,
            "skipped": True,
            "reason": "too few samples for CV",
        }

    strat_key = make_stratify_key(y, categories)

    # Handle rare strata: merge any with fewer members than n_folds
    counts = dict(zip(*np.unique(strat_key, return_counts=True), strict=True))
    rare = {v for v, c in counts.items() if c < N_FOLDS}
    if rare:
        strat_key = np.where(np.isin(strat_key, list(rare)), -1, strat_key)
    # If still problematic, fall back to y alone
    counts = dict(zip(*np.unique(strat_key, return_counts=True), strict=True))
    if any(c < N_FOLDS for c in counts.values()):
        strat_key = y

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # --- Real probe ---
    pooled_y = []
    pooled_proba = []
    fold_aucs = []

    for train_idx, test_idx in skf.split(X, strat_key):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegressionCV(
            Cs=10, cv=3, scoring="roc_auc", max_iter=2000,
            random_state=SEED, solver="lbfgs", class_weight="balanced",
        )
        clf.fit(X_train, y[train_idx])
        proba = clf.predict_proba(X_test)[:, 1]

        pooled_y.append(y[test_idx])
        pooled_proba.append(proba)
        fold_aucs.append(roc_auc_score(y[test_idx], proba))

    pooled_y_arr = np.concatenate(pooled_y)
    pooled_proba_arr = np.concatenate(pooled_proba)
    observed_auc = roc_auc_score(pooled_y_arr, pooled_proba_arr)

    # --- Hewitt & Liang control (shuffled labels) ---
    control_aucs = []
    rng_ctrl = np.random.default_rng(SEED + 9999)
    for _ in range(3):
        y_shuf = y.copy()
        rng_ctrl.shuffle(y_shuf)
        strat_shuf = make_stratify_key(y_shuf, categories)
        counts_shuf = dict(zip(*np.unique(strat_shuf, return_counts=True), strict=True))
        rare_shuf = {v for v, c in counts_shuf.items() if c < N_FOLDS}
        if rare_shuf:
            strat_shuf = np.where(np.isin(strat_shuf, list(rare_shuf)), -1, strat_shuf)
        counts_shuf = dict(zip(*np.unique(strat_shuf, return_counts=True), strict=True))
        if any(c < N_FOLDS for c in counts_shuf.values()):
            strat_shuf = y_shuf

        ctrl_pooled_y = []
        ctrl_pooled_proba = []
        for train_idx, test_idx in skf.split(X, strat_shuf):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            clf = LogisticRegressionCV(
                Cs=10, cv=3, scoring="roc_auc", max_iter=2000,
                random_state=SEED, solver="lbfgs", class_weight="balanced",
            )
            clf.fit(X_train, y_shuf[train_idx])
            proba = clf.predict_proba(X_test)[:, 1]
            ctrl_pooled_y.append(y_shuf[test_idx])
            ctrl_pooled_proba.append(proba)
        ctrl_y = np.concatenate(ctrl_pooled_y)
        ctrl_p = np.concatenate(ctrl_pooled_proba)
        if len(np.unique(ctrl_y)) >= 2:
            control_aucs.append(roc_auc_score(ctrl_y, ctrl_p))
        else:
            control_aucs.append(0.5)
    control_auc = float(np.mean(control_aucs))
    selectivity = observed_auc - control_auc

    # --- Permutation test ---
    rng_perm = np.random.default_rng(SEED)
    null_aucs = np.empty(N_PERMUTATIONS)
    y_perm = pooled_y_arr.copy()
    for i in range(N_PERMUTATIONS):
        rng_perm.shuffle(y_perm)
        if len(np.unique(y_perm)) < 2:
            null_aucs[i] = 0.5
        else:
            null_aucs[i] = roc_auc_score(y_perm, pooled_proba_arr)
    n_extreme = int(np.sum(null_aucs >= observed_auc))
    perm_p = (n_extreme + 1) / (N_PERMUTATIONS + 1)

    # --- Bootstrap CI ---
    rng_boot = np.random.default_rng(SEED + 1)
    boot_aucs = np.empty(N_BOOTSTRAP)
    n = len(pooled_y_arr)
    for i in range(N_BOOTSTRAP):
        idx = rng_boot.integers(0, n, size=n)
        if len(np.unique(pooled_y_arr[idx])) < 2:
            boot_aucs[i] = 0.5
        else:
            boot_aucs[i] = roc_auc_score(pooled_y_arr[idx], pooled_proba_arr[idx])
    ci_lower = float(np.percentile(boot_aucs, 2.5))
    ci_upper = float(np.percentile(boot_aucs, 97.5))

    return {
        "label": label,
        "n_samples": n_samples,
        "n_conflict": n_pos,
        "n_control": n_neg,
        "auc": float(observed_auc),
        "auc_ci_lower": ci_lower,
        "auc_ci_upper": ci_upper,
        "auc_fold_mean": float(np.mean(fold_aucs)),
        "auc_fold_std": float(np.std(fold_aucs, ddof=1)),
        "control_auc": control_auc,
        "selectivity": float(selectivity),
        "permutation_p": float(perm_p),
        "permutation_null_95th": float(np.percentile(null_aucs, 95)),
    }


# ---------------------------------------------------------------------------
# Cross-prediction: train on vulnerable, test on immune (and vice versa)
# ---------------------------------------------------------------------------


def cross_predict(
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    label: str,
) -> dict:
    """Train on one subset, test on another. No CV -- single split."""
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    n_train = len(y_train)
    n_test = len(y_test)

    if n_test < 5 or len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
        return {"label": label, "skipped": True, "reason": "insufficient samples"}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegressionCV(
        Cs=10, cv=3, scoring="roc_auc", max_iter=2000,
        random_state=SEED, solver="lbfgs", class_weight="balanced",
    )
    clf.fit(X_train_s, y_train)
    proba = clf.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, proba)

    # Bootstrap CI on the transfer AUC
    rng_boot = np.random.default_rng(SEED + 2)
    boot_aucs = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng_boot.integers(0, n_test, size=n_test)
        if len(np.unique(y_test[idx])) < 2:
            boot_aucs[i] = 0.5
        else:
            boot_aucs[i] = roc_auc_score(y_test[idx], proba[idx])

    return {
        "label": label,
        "n_train": n_train,
        "n_test": n_test,
        "auc": float(auc),
        "auc_ci_lower": float(np.percentile(boot_aucs, 2.5)),
        "auc_ci_upper": float(np.percentile(boot_aucs, 97.5)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()

    # 1. Load benchmark
    print("Loading benchmark...")
    items = load_benchmark()
    print(f"  {len(items)} items loaded")

    prompts = [item["prompt"] for item in items]
    labels = np.array([int(item["conflict"]) for item in items])
    categories = np.array([item["category"] for item in items])

    # 2. Encode with sentence-transformer
    print("Encoding prompts with all-MiniLM-L6-v2...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(prompts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"  Embedding shape: {embeddings.shape}")

    # 3. Define subsets
    vuln_mask = np.array([c in VULNERABLE_CATEGORIES for c in categories])
    immune_mask = np.array([c in IMMUNE_CATEGORIES for c in categories])

    print(f"  Vulnerable: {vuln_mask.sum()} items ({VULNERABLE_CATEGORIES})")
    print(f"  Immune:     {immune_mask.sum()} items ({IMMUNE_CATEGORIES})")

    # 4. Evaluate: all categories
    print("\n--- All categories (470 items) ---")
    result_all = evaluate_probe(embeddings, labels, categories, label="all_categories")
    print(f"  AUC = {result_all['auc']:.3f} [{result_all['auc_ci_lower']:.3f}, {result_all['auc_ci_upper']:.3f}]")
    print(f"  Control AUC = {result_all['control_auc']:.3f}, Selectivity = {result_all['selectivity']:.3f}")
    print(f"  Permutation p = {result_all['permutation_p']:.4f}")

    # 5. Evaluate: vulnerable only
    print(f"\n--- Vulnerable categories only ({vuln_mask.sum()} items) ---")
    result_vuln = evaluate_probe(
        embeddings[vuln_mask], labels[vuln_mask], categories[vuln_mask],
        label="vulnerable_only",
    )
    print(f"  AUC = {result_vuln['auc']:.3f} [{result_vuln['auc_ci_lower']:.3f}, {result_vuln['auc_ci_upper']:.3f}]")
    print(f"  Control AUC = {result_vuln['control_auc']:.3f}, Selectivity = {result_vuln['selectivity']:.3f}")
    print(f"  Permutation p = {result_vuln['permutation_p']:.4f}")

    # 6. Evaluate: immune only
    print(f"\n--- Immune categories only ({immune_mask.sum()} items) ---")
    result_immune = evaluate_probe(
        embeddings[immune_mask], labels[immune_mask], categories[immune_mask],
        label="immune_only",
    )
    print(f"  AUC = {result_immune['auc']:.3f} [{result_immune['auc_ci_lower']:.3f}, {result_immune['auc_ci_upper']:.3f}]")
    print(f"  Control AUC = {result_immune['control_auc']:.3f}, Selectivity = {result_immune['selectivity']:.3f}")
    print(f"  Permutation p = {result_immune['permutation_p']:.4f}")

    # 7. Per-category evaluation
    print("\n--- Per-category results ---")
    unique_cats = sorted(np.unique(categories))
    per_category = {}
    for cat in unique_cats:
        cat_mask = categories == cat
        n_cat = cat_mask.sum()
        n_conf = labels[cat_mask].sum()
        if n_conf < 3 or (n_cat - n_conf) < 3:
            print(f"  {cat}: SKIPPED (too few samples)")
            per_category[cat] = {"skipped": True, "n_samples": int(n_cat)}
            continue
        result_cat = evaluate_probe(
            embeddings[cat_mask], labels[cat_mask], categories[cat_mask],
            label=f"category_{cat}",
        )
        per_category[cat] = result_cat
        vuln_tag = " [V]" if cat in VULNERABLE_CATEGORIES else (" [I]" if cat in IMMUNE_CATEGORIES else "")
        print(f"  {cat}{vuln_tag}: AUC = {result_cat['auc']:.3f} "
              f"[{result_cat['auc_ci_lower']:.3f}, {result_cat['auc_ci_upper']:.3f}] "
              f"sel={result_cat['selectivity']:.3f} p={result_cat['permutation_p']:.4f}")

    # 8. Cross-prediction: train vulnerable -> test immune
    print("\n--- Cross-prediction ---")
    xp_vuln_to_immune = cross_predict(
        embeddings, labels, vuln_mask, immune_mask,
        label="train_vulnerable_test_immune",
    )
    print(f"  Vulnerable -> Immune: AUC = {xp_vuln_to_immune['auc']:.3f} "
          f"[{xp_vuln_to_immune['auc_ci_lower']:.3f}, {xp_vuln_to_immune['auc_ci_upper']:.3f}]")

    xp_immune_to_vuln = cross_predict(
        embeddings, labels, immune_mask, vuln_mask,
        label="train_immune_test_vulnerable",
    )
    print(f"  Immune -> Vulnerable: AUC = {xp_immune_to_vuln['auc']:.3f} "
          f"[{xp_immune_to_vuln['auc_ci_lower']:.3f}, {xp_immune_to_vuln['auc_ci_upper']:.3f}]")

    # 9. Assemble results
    elapsed = time.time() - t0
    results = {
        "description": (
            "Text-only baseline probe: logistic regression on sentence-transformer "
            "embeddings (all-MiniLM-L6-v2) of raw prompt text. Tests whether prompt "
            "text alone can distinguish conflict from control items."
        ),
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": int(embeddings.shape[1]),
        "protocol": {
            "cv_folds": N_FOLDS,
            "n_permutations": N_PERMUTATIONS,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
            "classifier": "LogisticRegressionCV(Cs=10, cv=3, class_weight=balanced)",
            "control": "Hewitt & Liang (3 shuffled-label runs)",
        },
        "results": {
            "all_categories": result_all,
            "vulnerable_only": result_vuln,
            "immune_only": result_immune,
            "per_category": per_category,
        },
        "cross_prediction": {
            "train_vulnerable_test_immune": xp_vuln_to_immune,
            "train_immune_test_vulnerable": xp_immune_to_vuln,
        },
        "interpretation": _interpret(result_all, result_vuln, result_immune,
                                     xp_vuln_to_immune, xp_immune_to_vuln),
        "elapsed_s": float(elapsed),
    }

    # 10. Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Elapsed: {elapsed:.1f}s")


def _interpret(
    all_res: dict, vuln_res: dict, immune_res: dict,
    xp_v2i: dict, xp_i2v: dict,
) -> dict:
    """Generate interpretive summary comparing text baseline to activation probes."""
    # Reference activation probe AUCs from existing results
    # Llama peak AUC ≈ 0.999 (layer 14), vulnerable within ≈ 0.999
    # Cross-prediction vulnerable->immune ≈ 0.38-0.57 for activation probes

    text_auc = all_res.get("auc", 0.5)
    text_vuln = vuln_res.get("auc", 0.5)
    text_immune = immune_res.get("auc", 0.5)
    text_xp = xp_v2i.get("auc", 0.5)

    return {
        "text_all_auc": text_auc,
        "text_vulnerable_auc": text_vuln,
        "text_immune_auc": text_immune,
        "text_cross_pred_vuln_to_immune": text_xp,
        "note": (
            "Compare these values to activation probe AUCs. "
            "If text AUC is close to activation AUC, the probe may be "
            "detecting surface text features. If text AUC << activation AUC, "
            "the activation probe captures information beyond text surface form. "
            "Cross-prediction: text features should transfer better across "
            "categories than activation features if they capture shared "
            "structural cues (e.g., 'costs X more than' in conflict items)."
        ),
        "reference_activation_aucs": {
            "llama_peak_all": 0.999,
            "llama_vulnerable_within": 0.999,
            "llama_cross_vuln_to_immune_l14": 0.378,
            "r1_peak_all": 0.929,
        },
    }


if __name__ == "__main__":
    main()
