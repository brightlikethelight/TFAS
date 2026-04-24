#!/usr/bin/env python3
"""Probe-score tracking under continuous steering for R1-Distill-Llama-8B.

Mechanistic diagnostic for a behaviorally inert steering result: when
continuous probe-direction steering at alpha=+5 fails to reduce the lure
rate on R1-Distill, this script distinguishes two hypotheses:

  H1 (mechanical failure): The hook fires but R1's residual stream dynamics
     actively cancel the perturbation — the probe score at P0 does NOT shift
     with alpha. The model is informationally resistant.

  H2 (behavioral decoupling): The probe score DOES shift with alpha (steering
     lands in representation space), but the shifted representation still
     produces the same behavior. The probe direction is not causally
     sufficient for behavior change in R1.

What this script measures:
  For each of the 80 vulnerable conflict items, at each alpha in {-5,-3,0,+3,+5}:
  1. Run model.generate() with continuous SteeringHook at alpha.
  2. BEFORE generation, extract the residual stream at L31, position P0
     (last prompt token) via a separate extraction hook on the STEERED forward pass.
  3. Compute probe_score = activation @ probe_weight (dot product in original space).
  4. Record (item_id, alpha, probe_score, verdict).

Key design decision on P0 capture:
  P0 is the last token of the prompt (just before the model generates anything).
  Under continuous steering, the hook fires during the prefill pass and during
  every generation step. We capture P0 during the prefill pass by registering
  a separate read-only extraction hook that stores the last-position hidden state
  after the steered layer output. The extraction hook is layer-specific and
  reads from the same position the SteeringHook writes to, so we capture the
  STEERED representation (post-hook), which is exactly what feeds forward into
  subsequent layers and ultimately drives behavior.

Probe loading strategy:
  1. Look for a saved probe in /workspace/s1s2/results/probes/
     (checkpoint format: JSON with coef and intercept fields, or numpy .npy).
  2. If the H5 activation cache is available at /workspace/s1s2/data/activations/,
     retrain the probe in-process (same logic as run_probe_steering.py).
  3. If neither is available, exit with a clear error.

Usage:
    python scripts/r1_probe_score_tracking.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --benchmark /workspace/s1s2/data/benchmark/benchmark.jsonl \
        --target-layer 31 \
        --output /workspace/s1s2/results/causal/r1_probe_score_tracking_l31.json \
        --cache-dir /workspace/hf_cache

    # Smoke test (2 items, 3 alphas):
    python scripts/r1_probe_score_tracking.py \\
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
        --benchmark /workspace/s1s2/data/benchmark/benchmark.jsonl \\
        --smoke-test

Standalone: no Hydra, no W&B. Designed to run directly on a GPU pod.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Ensure src/ importable without editable install
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
H5_MODEL_KEY = "deepseek-ai_DeepSeek-R1-Distill-Llama-8B"

VULNERABLE_CATEGORIES: list[str] = ["base_rate", "conjunction", "syllogism"]

# Default pod paths — callers can override via CLI flags
DEFAULT_H5_PATH = "/workspace/s1s2/data/activations/r1_distill_llama.h5"
DEFAULT_PROBE_DIR = "/workspace/s1s2/results/probes"
DEFAULT_OUTPUT = "/workspace/s1s2/results/causal/r1_probe_score_tracking_l31.json"
DEFAULT_CACHE_DIR = "/workspace/hf_cache"

DEFAULT_ALPHAS: list[float] = [-5.0, -3.0, 0.0, 3.0, 5.0]
SMOKE_ALPHAS: list[float] = [-3.0, 0.0, 3.0]

TARGET_LAYER_DEFAULT = 31
MAX_NEW_TOKENS = 2048


# ======================================================================== #
#  Probe loading / training                                                  #
# ======================================================================== #


def load_h5_metadata(f: Any) -> dict[str, np.ndarray]:
    """Load per-problem metadata arrays from the HDF5 file.

    Decodes bytes to str so downstream code can compare with Python strings.
    """
    out: dict[str, np.ndarray] = {}
    for key in ("id", "category", "correct_answer", "lure_answer"):
        raw = f[f"/problems/{key}"][:]
        out[key] = np.array(
            [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in raw]
        )
    out["conflict"] = f["/problems/conflict"][:]
    return out


def load_residual_p0(f: Any, model_key: str, layer: int) -> np.ndarray:
    """Load residual activations at position P0 for a single layer.

    Returns shape (n_problems, hidden_dim).
    """
    arr = f[f"/models/{model_key}/residual/layer_{layer:02d}"][:]
    labels_raw = f[f"/models/{model_key}/position_index/labels"][:]
    labels = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in labels_raw]
    if "P0" not in labels:
        raise KeyError(f"P0 not found in position labels: {labels}")
    p0_idx = labels.index("P0")
    return arr[:, p0_idx, :]


def train_probe_from_h5(
    h5_path: str,
    model_key: str,
    target_layer: int,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Train logistic regression probe on conflict vs control, vulnerable cats.

    Returns:
        direction: probe weight vector in original activation space, shape (hidden_dim,).
            Convention: positive dot product = more conflict-like (S1-like).
            Negated so that positive alpha steers toward S2 (correct).
        cv_auc: cross-validated ROC-AUC of the probe.
    """
    import h5py
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    print(f"  Loading activations from {h5_path}")
    with h5py.File(h5_path, "r") as f:
        meta = load_h5_metadata(f)
        X_all = load_residual_p0(f, model_key, target_layer)

    categories = meta["category"]
    conflict = meta["conflict"].astype(np.int64)

    vuln_mask = np.isin(categories, VULNERABLE_CATEGORIES)
    X = X_all[vuln_mask]
    y = conflict[vuln_mask]

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    print(f"  Probe data: {len(y)} items ({n_pos} conflict / {n_neg} control), "
          f"layer {target_layer}")

    if n_pos < 5 or n_neg < 5:
        raise ValueError(
            f"Insufficient probe data: {n_pos} conflict, {n_neg} control. Need >= 5 each."
        )

    # 5-fold stratified CV to get pooled AUC
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pooled_y: list[np.ndarray] = []
    pooled_proba: list[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_te = scaler.transform(X_te).astype(np.float32)

        clf = LogisticRegressionCV(
            Cs=np.logspace(-4, 4, 20),
            cv=3,
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced",
            scoring="roc_auc",
            refit=True,
            random_state=seed + fold_idx,
            n_jobs=1,
        )
        clf.fit(X_tr, y_tr)
        pooled_y.append(y_te)
        pooled_proba.append(clf.predict_proba(X_te)[:, 1])

    cv_auc = float(roc_auc_score(np.concatenate(pooled_y), np.concatenate(pooled_proba)))

    # Refit on all data to extract coef_ for steering direction
    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X).astype(np.float32)
    clf_full = LogisticRegressionCV(
        Cs=np.logspace(-4, 4, 20),
        cv=3,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        class_weight="balanced",
        scoring="roc_auc",
        refit=True,
        random_state=seed,
        n_jobs=1,
    )
    clf_full.fit(X_scaled, y)

    # coef_ shape: (1, hidden_dim), class 1 = conflict.
    # Negate so positive projection = S2-like. Back-transform through scaler so
    # the vector lives in the original (unscaled) activation space.
    raw_coef = -clf_full.coef_[0].copy()
    direction = raw_coef / scaler_full.scale_  # shape (hidden_dim,)

    print(f"  Probe CV AUC: {cv_auc:.4f}")
    print(f"  Direction norm: {np.linalg.norm(direction):.4f}")
    print(f"  Best regularization C: {clf_full.C_[0]:.4f}")

    return direction.astype(np.float32), cv_auc


def try_load_probe_from_checkpoint(
    probe_dir: str,
    model_key: str,
    target_layer: int,
) -> np.ndarray | None:
    """Try to load a pre-saved probe weight vector.

    Looks for files named:
      <probe_dir>/<model_key>_l<layer>_coef.npy   (numpy array, shape [hidden_dim])
      <probe_dir>/<model_key>_l<layer>_probe.json  (JSON with "coef" list field)

    Returns the coef vector (float32, shape [hidden_dim]) or None if not found.
    """
    probe_path = Path(probe_dir)
    prefix = f"{model_key}_l{target_layer}"

    npy_path = probe_path / f"{prefix}_coef.npy"
    if npy_path.exists():
        coef = np.load(str(npy_path)).astype(np.float32)
        print(f"  Loaded probe coef from {npy_path} (shape {coef.shape})")
        return coef

    json_path = probe_path / f"{prefix}_probe.json"
    if json_path.exists():
        with open(json_path) as fh:
            data = json.load(fh)
        coef = np.array(data["coef"], dtype=np.float32)
        print(f"  Loaded probe coef from {json_path} (shape {coef.shape})")
        return coef

    # Also accept any JSON with "direction" or "coef_" fields (other naming conventions)
    for candidate in sorted(probe_path.glob(f"*{target_layer}*.json")):
        try:
            with open(candidate) as fh:
                data = json.load(fh)
            for field in ("direction", "coef_", "coef", "weight"):
                if field in data:
                    coef = np.array(data[field], dtype=np.float32)
                    if coef.ndim == 1 and coef.shape[0] in (3584, 4096):
                        print(f"  Loaded probe coef from {candidate} (field='{field}', "
                              f"shape {coef.shape})")
                        return coef
        except Exception:
            continue

    return None


def get_probe_direction(
    probe_dir: str,
    h5_path: str | None,
    model_key: str,
    target_layer: int,
    seed: int,
) -> tuple[np.ndarray, float | None]:
    """Obtain probe weight vector, by loading checkpoint or retraining.

    Returns:
        direction: float32 array shape [hidden_dim]
        cv_auc: float if retrained, None if loaded from checkpoint
    """
    # Try checkpoint first
    coef = try_load_probe_from_checkpoint(probe_dir, model_key, target_layer)
    if coef is not None:
        return coef, None

    # Fall back to retraining from H5
    if h5_path is None or not Path(h5_path).exists():
        raise FileNotFoundError(
            f"No probe checkpoint found in {probe_dir} and H5 cache not available at "
            f"{h5_path}. Cannot obtain probe direction.\n"
            "Either:\n"
            "  (a) save a probe checkpoint there first, or\n"
            "  (b) provide --h5-path pointing to the activation H5 file."
        )

    print("  No checkpoint found — retraining probe from H5 activations.")
    direction, cv_auc = train_probe_from_h5(h5_path, model_key, target_layer, seed=seed)
    return direction, cv_auc


# ======================================================================== #
#  Benchmark loading                                                         #
# ======================================================================== #


def load_conflict_items_vulnerable(benchmark_path: str) -> list[dict[str, Any]]:
    """Load conflict items from vulnerable categories only."""
    items: list[dict[str, Any]] = []
    with open(benchmark_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("conflict") and item.get("category") in VULNERABLE_CATEGORIES:
                items.append(item)
    return items


# ======================================================================== #
#  Response classification                                                   #
# ======================================================================== #


def classify_response(
    text: str,
    correct_answer: str,
    lure_answer: str,
    answer_pattern: str = "",
    lure_pattern: str = "",
) -> str:
    """Classify model output as 'correct', 'lure', or 'other'.

    Strips <think>...</think> blocks first (R1-Distill emits these).
    Tries regex patterns before substring matching.
    """
    clean = text
    if "<think>" in clean and "</think>" in clean:
        te = clean.index("</think>")
        clean = clean[te + len("</think>"):].strip()

    if answer_pattern:
        try:
            if re.search(answer_pattern, clean, re.IGNORECASE):
                return "correct"
        except re.error:
            pass
    if lure_pattern:
        try:
            if re.search(lure_pattern, clean, re.IGNORECASE):
                return "lure"
        except re.error:
            pass

    if correct_answer and re.search(re.escape(correct_answer), clean, re.IGNORECASE):
        return "correct"
    if lure_answer and re.search(re.escape(lure_answer), clean, re.IGNORECASE):
        return "lure"

    return "other"


# ======================================================================== #
#  Core experiment: probe-score tracking                                     #
# ======================================================================== #


def run_probe_score_tracking(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    probe_direction: np.ndarray,
    target_layer: int,
    alphas: list[float],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[dict[str, Any]]:
    """Run probe-score tracking experiment across items and alphas.

    For each (item, alpha) pair:
      1. Attach the SteeringHook to layer `target_layer`.
      2. Register a separate read-only extraction hook on the SAME layer that
         captures the steered residual stream at the last prompt position P0.
      3. Run model.generate() with both hooks active.
      4. Compute probe_score = activation_p0 @ probe_direction.
      5. Classify the generated response.

    The extraction hook fires on every forward call of the layer. During
    model.generate(), that includes the prefill pass (all prompt tokens) and
    each autoregressive decode step (single token). We capture the P0 activation
    from the PREFILL pass only: that's the unique forward call where the sequence
    length equals the prompt length. We detect this by tracking sequence length
    in the hook closure.

    Returns list of per-item-per-alpha dicts with fields:
      item_id, category, alpha, probe_score, verdict, prompt_len
    """
    import torch
    from s1s2.causal.steering import SteeringHook, normalize_direction

    # Normalize probe direction to unit vector for the steering hook.
    # For the probe score we use the UNNORMALIZED direction (the actual coef_
    # in activation space) so the score is in interpretable units.
    probe_direction_np = probe_direction.copy()
    probe_direction_t = torch.from_numpy(probe_direction_np)

    results: list[dict[str, Any]] = []
    n_items = len(items)

    for item_idx, item in enumerate(items):
        if item_idx > 0 and item_idx % 10 == 0:
            print(f"  [{item_idx}/{n_items}] processing item {item_idx}...")

        # Build prompt once; prompt_len is constant across alphas for this item.
        messages = [{"role": "user", "content": item["prompt"]}]
        if item.get("system_prompt"):
            messages.insert(0, {"role": "system", "content": item["system_prompt"]})

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        prompt_len = int(inputs.input_ids.shape[1])

        for alpha in alphas:
            # State captured by the extraction hook closure.
            captured: dict[str, Any] = {"p0_activation": None, "captured": False}

            def make_extraction_hook(
                state: dict[str, Any], p_len: int
            ):
                """Factory so we get a fresh closure per (item, alpha)."""

                def _hook(module: Any, inputs: Any, output: Any) -> None:
                    # Skip if we've already captured P0 (only want prefill pass).
                    if state["captured"]:
                        return
                    # Extract hidden states from layer output tuple.
                    hidden = output[0] if isinstance(output, tuple) else output
                    if not isinstance(hidden, torch.Tensor):
                        return
                    seq_len = hidden.shape[1]
                    # Prefill pass: seq_len == prompt length.
                    # Decode steps: seq_len == 1. Capture only the prefill.
                    if seq_len == p_len:
                        # P0 is the last token of the prompt.
                        p0 = hidden[0, -1, :].detach().float().cpu().numpy()
                        state["p0_activation"] = p0
                        state["captured"] = True

                return _hook

            # Hook registration order matters: PyTorch runs forward hooks in
            # registration order. We want to capture the POST-steering activation
            # (what actually feeds into subsequent layers), so the extraction hook
            # must be registered AFTER the SteeringHook enters its context.
            # For alpha=0 (no steering hook), order doesn't matter.
            inner = getattr(model, "model", model)
            layer_module = inner.layers[target_layer]

            if alpha == 0.0:
                # Baseline: register extraction hook alone, no steering.
                extr_handle = layer_module.register_forward_hook(
                    make_extraction_hook(captured, prompt_len)
                )
                try:
                    with torch.no_grad():
                        out = model.generate(
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                        )
                finally:
                    extr_handle.remove()
            else:
                # Enter SteeringHook first so it registers its forward hook first.
                # Then register the extraction hook — it fires second, capturing
                # the steered output (after the steering mutation has been applied).
                steer_hook = SteeringHook(
                    model,
                    layer=target_layer,
                    direction=probe_direction_t,
                    alpha=alpha,
                )
                with steer_hook:
                    extr_handle = layer_module.register_forward_hook(
                        make_extraction_hook(captured, prompt_len)
                    )
                    try:
                        with torch.no_grad():
                            out = model.generate(
                                inputs.input_ids,
                                attention_mask=inputs.attention_mask,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                            )
                    finally:
                        extr_handle.remove()

            # Decode generated text (only new tokens, not the prompt).
            gen_ids = out[0][prompt_len:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=False)

            verdict = classify_response(
                response,
                item["correct_answer"],
                item["lure_answer"],
                item.get("answer_pattern", ""),
                item.get("lure_pattern", ""),
            )

            # Compute probe score in original activation space.
            if captured["p0_activation"] is not None:
                probe_score = float(
                    np.dot(captured["p0_activation"], probe_direction_np)
                )
                # Also compute normalized score (cosine with unit direction).
                unit_dir = probe_direction_np / (np.linalg.norm(probe_direction_np) + 1e-12)
                probe_score_normalized = float(
                    np.dot(captured["p0_activation"], unit_dir)
                )
            else:
                # Should not happen for a correctly-shaped model; emit NaN as sentinel.
                probe_score = float("nan")
                probe_score_normalized = float("nan")
                print(
                    f"  WARNING: P0 activation not captured for item "
                    f"{item.get('id', item_idx)}, alpha={alpha}. "
                    "Check layer index and model architecture."
                )

            results.append(
                {
                    "item_id": item.get("id", f"item_{item_idx}"),
                    "item_idx": item_idx,
                    "category": item.get("category", ""),
                    "alpha": float(alpha),
                    "probe_score": probe_score,
                    "probe_score_normalized": probe_score_normalized,
                    "verdict": verdict,
                    "prompt_len": prompt_len,
                }
            )

    return results


# ======================================================================== #
#  Aggregation and reporting                                                  #
# ======================================================================== #


def aggregate_results(
    results: list[dict[str, Any]],
    alphas: list[float],
    probe_norm: float,
) -> dict[str, Any]:
    """Compute per-alpha summary statistics and the key diagnostic.

    The diagnostic is:
      - If probe_score shifts with alpha -> H2 (behavioral decoupling).
      - If probe_score is flat across alphas -> H1 (mechanical failure / cancellation).

    A slope significantly different from zero (in the expected direction) is
    evidence for H2. Near-zero slope is evidence for H1.
    """
    from collections import defaultdict

    by_alpha: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_alpha[row["alpha"]].append(row)

    per_alpha_stats: dict[str, dict[str, Any]] = {}
    for alpha in sorted(alphas):
        rows = by_alpha[alpha]
        if not rows:
            continue
        scores = [r["probe_score"] for r in rows if not np.isnan(r["probe_score"])]
        scores_norm = [
            r["probe_score_normalized"]
            for r in rows
            if not np.isnan(r["probe_score_normalized"])
        ]
        verdicts = [r["verdict"] for r in rows]
        n = len(rows)
        n_lure = sum(1 for v in verdicts if v == "lure")
        n_correct = sum(1 for v in verdicts if v == "correct")

        per_alpha_stats[str(alpha)] = {
            "alpha": alpha,
            "n_items": n,
            "mean_probe_score": float(np.mean(scores)) if scores else float("nan"),
            "std_probe_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "mean_probe_score_normalized": (
                float(np.mean(scores_norm)) if scores_norm else float("nan")
            ),
            "std_probe_score_normalized": (
                float(np.std(scores_norm, ddof=1)) if len(scores_norm) > 1 else 0.0
            ),
            "lure_rate": n_lure / n if n > 0 else float("nan"),
            "correct_rate": n_correct / n if n > 0 else float("nan"),
            "n_lure": n_lure,
            "n_correct": n_correct,
            "n_other": n - n_lure - n_correct,
        }

    # Compute slope of probe_score ~ alpha via least-squares as a summary stat.
    # Also compute expected slope under no cancellation: if steering adds alpha*unit_dir
    # to hidden, then probe_score increases by alpha * (||direction|| * cos_theta)
    # where cos_theta is the angle between the unit steer direction and probe_direction.
    # Since we steer along the SAME direction as the probe (they're the same vector,
    # just normalized for steering), the expected slope is ||probe_direction||.
    # Observed slope << expected_slope => cancellation.
    alpha_vals = sorted(alphas)
    mean_scores = [
        per_alpha_stats[str(a)]["mean_probe_score"]
        for a in alpha_vals
        if str(a) in per_alpha_stats and not np.isnan(per_alpha_stats[str(a)]["mean_probe_score"])
    ]
    alpha_vals_clean = [
        a
        for a in alpha_vals
        if str(a) in per_alpha_stats and not np.isnan(per_alpha_stats[str(a)]["mean_probe_score"])
    ]

    observed_slope: float | None = None
    expected_slope: float = probe_norm  # ||probe_direction||

    if len(alpha_vals_clean) >= 2:
        # Simple OLS slope
        A = np.array(alpha_vals_clean)
        B = np.array(mean_scores)
        A_centered = A - A.mean()
        observed_slope = float(np.dot(A_centered, B) / (np.dot(A_centered, A_centered) + 1e-12))

    cancellation_ratio: float | None = None
    if observed_slope is not None and expected_slope > 0:
        # Ratio of observed to expected slope.
        # ~1.0 => no cancellation (H2). ~0.0 => full cancellation (H1).
        cancellation_ratio = observed_slope / expected_slope

    # Baseline (alpha=0) probe score
    baseline_score: float | None = None
    if "0.0" in per_alpha_stats:
        baseline_score = per_alpha_stats["0.0"]["mean_probe_score"]

    return {
        "per_alpha": per_alpha_stats,
        "probe_direction_norm": probe_norm,
        "expected_slope_per_unit_alpha": expected_slope,
        "observed_slope_probe_score_per_unit_alpha": observed_slope,
        "cancellation_ratio": cancellation_ratio,
        "baseline_probe_score_alpha0": baseline_score,
        "diagnosis": _diagnosis(cancellation_ratio),
    }


def _diagnosis(cancellation_ratio: float | None) -> str:
    """Plain-language mechanistic diagnosis from the cancellation ratio.

    > 0.7: Steering lands in representation space (H2 — behavioral decoupling).
    0.3-0.7: Partial cancellation — mixed evidence.
    < 0.3: Strong cancellation (H1 — R1 dynamics cancel the perturbation).
    """
    if cancellation_ratio is None:
        return "INCONCLUSIVE: insufficient data to compute slope."
    if cancellation_ratio > 0.7:
        return (
            "H2 (BEHAVIORAL DECOUPLING): Probe score shifts with alpha as expected "
            f"(ratio={cancellation_ratio:.3f}). The steering reaches the representation, "
            "but R1's behavior does not respond. The probe direction is not causally "
            "sufficient — something downstream decouples representation from behavior."
        )
    elif cancellation_ratio > 0.3:
        return (
            f"PARTIAL CANCELLATION (ratio={cancellation_ratio:.3f}): R1 dynamics partially "
            "cancel the steering perturbation. Mixed H1/H2 evidence. Consider stronger alphas "
            "or multi-layer steering."
        )
    else:
        return (
            "H1 (MECHANICAL CANCELLATION): Probe score barely shifts with alpha "
            f"(ratio={cancellation_ratio:.3f}). R1's residual stream dynamics actively "
            "cancel the perturbation. The model is mechanistically resistant to this "
            "steering approach at this layer. Consider: (a) deeper layers, "
            "(b) multi-layer clamping, (c) SAE-feature steering."
        )


def print_summary(agg: dict[str, Any]) -> None:
    """Print a human-readable summary of the probe-score tracking results."""
    print(f"\n{'='*70}")
    print("  PROBE-SCORE TRACKING RESULTS")
    print(f"{'='*70}")
    print(f"  Probe direction norm:        {agg['probe_direction_norm']:.4f}")
    print(f"  Expected slope (no cancel):  {agg['expected_slope_per_unit_alpha']:.4f}")
    if agg["observed_slope_probe_score_per_unit_alpha"] is not None:
        print(f"  Observed slope:              "
              f"{agg['observed_slope_probe_score_per_unit_alpha']:.4f}")
    if agg["cancellation_ratio"] is not None:
        print(f"  Cancellation ratio:          {agg['cancellation_ratio']:.4f}")
    if agg["baseline_probe_score_alpha0"] is not None:
        print(f"  Baseline probe score (α=0):  {agg['baseline_probe_score_alpha0']:.4f}")
    print()
    print("  Per-alpha summary:")
    print(f"  {'alpha':>6}  {'mean_score':>12}  {'std':>8}  {'lure_rate':>10}  "
          f"{'correct_rate':>12}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*12}")
    for alpha_str, stats in sorted(
        agg["per_alpha"].items(), key=lambda x: float(x[0])
    ):
        a = stats["alpha"]
        ms = stats["mean_probe_score"]
        sd = stats["std_probe_score"]
        lr = stats["lure_rate"]
        cr = stats["correct_rate"]
        print(f"  {a:>+6.1f}  {ms:>12.4f}  {sd:>8.4f}  {lr:>10.3f}  {cr:>12.3f}")

    print()
    print(f"  DIAGNOSIS: {agg['diagnosis']}")
    print(f"{'='*70}")


# ======================================================================== #
#  CLI                                                                       #
# ======================================================================== #


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe-score tracking under continuous steering for R1-Distill-Llama-8B. "
            "Diagnoses whether behavioral inertia under steering is due to "
            "mechanical cancellation (H1) or behavioral decoupling (H2)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID,
        help=f"HuggingFace model ID. Default: {MODEL_ID}",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Path to benchmark.jsonl file.",
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        default=TARGET_LAYER_DEFAULT,
        help=f"Layer to steer and extract probe score from. Default: {TARGET_LAYER_DEFAULT}",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default=",".join(str(a) for a in DEFAULT_ALPHAS),
        help="Comma-separated steering strengths. Default: -5,-3,0,3,5",
    )
    parser.add_argument(
        "--probe-dir",
        type=str,
        default=DEFAULT_PROBE_DIR,
        help=f"Directory to search for saved probe checkpoints. Default: {DEFAULT_PROBE_DIR}",
    )
    parser.add_argument(
        "--h5-path",
        type=str,
        default=DEFAULT_H5_PATH,
        help=(
            f"Path to HDF5 activation cache (for probe retraining if no checkpoint). "
            f"Default: {DEFAULT_H5_PATH}"
        ),
    )
    parser.add_argument(
        "--h5-model-key",
        type=str,
        default=H5_MODEL_KEY,
        help=f"Model key in the HDF5 file. Default: {H5_MODEL_KEY}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"HuggingFace model cache dir. Default: {DEFAULT_CACHE_DIR}",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"Max tokens to generate per item. Default: {MAX_NEW_TOKENS}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Default: 0",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on 2 items with 3 alphas (-3,0,3) for quick validation.",
    )
    args = parser.parse_args()

    # Seed RNGs
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    alphas = [float(x.strip()) for x in args.alphas.split(",")]
    if args.smoke_test:
        alphas = SMOKE_ALPHAS
        print("  SMOKE TEST MODE: 2 items, alphas = [-3, 0, +3]")

    print(f"\n{'='*70}")
    print("  PROBE-SCORE TRACKING UNDER CONTINUOUS STEERING")
    print(f"{'='*70}")
    print(f"  Model:        {args.model}")
    print(f"  Benchmark:    {args.benchmark}")
    print(f"  Target layer: {args.target_layer}")
    print(f"  Alphas:       {alphas}")
    print(f"  Max tokens:   {args.max_new_tokens}")
    print(f"  Probe dir:    {args.probe_dir}")
    print(f"  H5 path:      {args.h5_path}")
    print(f"  Output:       {args.output}")
    print(f"  Seed:         {args.seed}")
    print(f"  Smoke test:   {args.smoke_test}")
    print()

    # ── Phase 1: Load or train probe ─────────────────────────────────────────
    t_probe_start = time.time()
    print("[Phase 1] Obtaining probe direction...")
    probe_direction, probe_auc = get_probe_direction(
        probe_dir=args.probe_dir,
        h5_path=args.h5_path,
        model_key=args.h5_model_key,
        target_layer=args.target_layer,
        seed=args.seed,
    )
    probe_norm = float(np.linalg.norm(probe_direction))
    t_probe = time.time() - t_probe_start
    print(f"  Done in {t_probe:.1f}s. "
          f"hidden_dim={probe_direction.shape[0]}, norm={probe_norm:.4f}, "
          f"probe_auc={probe_auc if probe_auc is not None else 'checkpoint'}")

    # ── Phase 2: Load model ───────────────────────────────────────────────────
    print(f"\n[Phase 2] Loading model: {args.model}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    t_load = time.time() - t_load_start
    print(f"  Loaded in {t_load:.1f}s")
    if torch.cuda.is_available():
        print(f"  VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ── Phase 3: Load benchmark items ────────────────────────────────────────
    print(f"\n[Phase 3] Loading conflict items from benchmark...")
    items = load_conflict_items_vulnerable(args.benchmark)
    print(f"  Found {len(items)} conflict items in vulnerable categories "
          f"({VULNERABLE_CATEGORIES})")

    if not items:
        print("ERROR: No conflict items found. Check benchmark path and category names.",
              file=sys.stderr)
        sys.exit(1)

    if args.smoke_test:
        items = items[:2]
        print(f"  SMOKE TEST: truncated to {len(items)} items")

    # ── Phase 4: Probe-score tracking experiment ──────────────────────────────
    print(f"\n[Phase 4] Running probe-score tracking experiment...")
    print(f"  {len(items)} items x {len(alphas)} alphas = "
          f"{len(items) * len(alphas)} forward passes")
    print(f"  (Each forward pass generates up to {args.max_new_tokens} tokens — "
          "expect this to be slow for R1)\n")

    t_exp_start = time.time()
    raw_results = run_probe_score_tracking(
        model=model,
        tokenizer=tokenizer,
        items=items,
        probe_direction=probe_direction,
        target_layer=args.target_layer,
        alphas=alphas,
        max_new_tokens=args.max_new_tokens,
    )
    t_exp = time.time() - t_exp_start
    print(f"\n  Experiment complete in {t_exp:.1f}s "
          f"({t_exp / max(len(raw_results), 1):.1f}s per forward pass)")

    # ── Phase 5: Aggregate and diagnose ──────────────────────────────────────
    print("\n[Phase 5] Aggregating results...")
    agg = aggregate_results(raw_results, alphas, probe_norm)
    print_summary(agg)

    # ── Phase 6: Save results ─────────────────────────────────────────────────
    output_data: dict[str, Any] = {
        "model": args.model,
        "target_layer": args.target_layer,
        "alphas": alphas,
        "n_items": len(items),
        "vulnerable_categories": VULNERABLE_CATEGORIES,
        "probe_direction_norm": probe_norm,
        "probe_auc": probe_auc,
        "probe_hidden_dim": int(probe_direction.shape[0]),
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "dtype": "bfloat16",
            "smoke_test": args.smoke_test,
            "h5_path": args.h5_path,
            "probe_dir": args.probe_dir,
        },
        "per_item_results": raw_results,
        "aggregate": agg,
        "elapsed_s": {
            "probe_loading": round(t_probe, 1),
            "model_loading": round(t_load, 1),
            "experiment": round(t_exp, 1),
            "total": round(t_probe + t_load + t_exp, 1),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(output_data, fh, indent=2, allow_nan=True)
    print(f"\n  Results saved to: {out_path}")
    print(f"  Total elapsed: {t_probe + t_load + t_exp:.1f}s")
    print()


if __name__ == "__main__":
    main()
