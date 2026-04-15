#!/usr/bin/env python3
"""Probe-direction steering: causal evidence that S1/S2 probe direction is functional.

Train a logistic regression probe on conflict vs control (vulnerable categories only)
at a target layer, extract the coef_ weight vector as the "S1/S2 direction," then steer
the model along that direction at varying alphas. If the direction is causally relevant:
  - Positive alpha (toward S2) should REDUCE lure rate on conflict items.
  - Negative alpha (toward S1) should INCREASE lure rate.
  - Random unit-vector controls should show no systematic effect.

This is the strongest causal claim we can make: the representation the probe finds
actually mediates the behavior.

Standalone script, no Hydra. Deployable directly to a GPU pod.

Supports two steering positions via --steer-position:
  - "continuous" (default): hook stays active during generation, so the
    steering perturbation is applied at EVERY token position (prompt +
    all generated tokens). This is the right choice for reasoning models
    (R1-Distill) where the model generates 500+ thinking tokens before
    the answer -- a prompt-only signal would wash out.
  - "prompt": steering is applied only during the prompt forward pass.
    The hook is removed before model.generate() produces new tokens.
    This isolates the effect to the initial representation.

Usage:
    python scripts/run_probe_steering.py \
        --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --h5-path data/activations/llama31_8b_instruct.h5 \
        --benchmark data/benchmark/benchmark.jsonl \
        --target-layer 14 \
        --alphas "-5,-3,-1,-0.5,0,0.5,1,3,5" \
        --n-random-controls 5 \
        --max-new-tokens 128 \
        --output results/causal/probe_steering_llama_l14.json \
        --cache-dir /workspace/hf_cache

    # For reasoning models (temporal washout experiment):
    python scripts/run_probe_steering.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --h5-path data/activations/r1_distill_llama.h5 \
        --target-layer 14 --max-new-tokens 2048 \
        --steer-position continuous \
        --output results/causal/probe_steering_r1_continuous_l14.json \
        --cache-dir /workspace/hf_cache
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Ensure src/ importable without editable install
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from s1s2.utils.seed import set_global_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VULNERABLE_CATEGORIES: list[str] = ["base_rate", "conjunction", "syllogism"]


# ======================================================================== #
#  Phase 1 — Probe training (CPU, from H5 activations)                     #
# ======================================================================== #


def load_h5_metadata(f: h5py.File) -> dict[str, np.ndarray]:
    """Load per-problem metadata arrays from the HDF5 file."""
    out: dict[str, np.ndarray] = {}
    for key in ("id", "category", "correct_answer", "lure_answer"):
        raw = f[f"/problems/{key}"][:]
        out[key] = np.array(
            [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in raw]
        )
    out["conflict"] = f["/problems/conflict"][:]
    return out


def load_residual_p0(f: h5py.File, model_key: str, layer: int) -> np.ndarray:
    """Load residual activations at position P0 for a single layer.

    Returns shape (n_problems, hidden_dim).
    """
    arr = f[f"/models/{model_key}/residual/layer_{layer:02d}"][:]
    labels_raw = f[f"/models/{model_key}/position_index/labels"][:]
    labels = [
        s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in labels_raw
    ]
    if "P0" not in labels:
        raise KeyError(f"P0 not found in position labels: {labels}")
    p0_idx = labels.index("P0")
    return arr[:, p0_idx, :]


def train_probe_and_get_direction(
    h5_path: str,
    model_key: str,
    target_layer: int,
    seed: int = 0,
) -> tuple[np.ndarray, float, int]:
    """Train LogisticRegressionCV on conflict vs control, vulnerable cats only.

    Returns:
        direction: coef_ weight vector, shape (hidden_dim,), NOT yet normalized
        auc: cross-validated ROC-AUC of the probe
        hidden_dim: dimensionality
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    with h5py.File(h5_path, "r") as f:
        meta = load_h5_metadata(f)
        X_all = load_residual_p0(f, model_key, target_layer)

    categories = meta["category"]
    conflict = meta["conflict"].astype(np.int64)

    # Filter to vulnerable categories only
    vuln_mask = np.isin(categories, VULNERABLE_CATEGORIES)
    X = X_all[vuln_mask]
    y = conflict[vuln_mask]

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    print(f"  Probe data: {len(y)} items ({n_pos} conflict, {n_neg} control) "
          f"from {VULNERABLE_CATEGORIES}")

    if n_pos < 5 or n_neg < 5:
        raise ValueError(
            f"Insufficient data for probe: {n_pos} conflict, {n_neg} control. "
            "Need at least 5 of each."
        )

    # Stratified CV to get pooled test-fold AUC
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pooled_y: list[np.ndarray] = []
    pooled_proba: list[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

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
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        pooled_y.append(y_test)
        pooled_proba.append(proba)

    all_y = np.concatenate(pooled_y)
    all_proba = np.concatenate(pooled_proba)
    cv_auc = float(roc_auc_score(all_y, all_proba))

    # Refit on full data to get the final coef_
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

    # coef_ is shape (1, hidden_dim) for binary classification.
    # Convention: class 1 = conflict. Positive coef_ direction = more conflict-like (S1).
    # We NEGATE so that positive alpha pushes TOWARD S2 (away from conflict).
    raw_coef = -clf_full.coef_[0].copy()  # shape (hidden_dim,), negated

    # Account for the scaler: the probe was trained on standardized data.
    # coef_ in original space = coef_scaled / std (ignoring intercept for direction).
    direction = raw_coef / scaler_full.scale_

    print(f"  Probe CV AUC: {cv_auc:.4f}")
    print(f"  Direction norm (pre-normalize): {np.linalg.norm(direction):.4f}")
    print(f"  Best C: {clf_full.C_[0]:.4f}")

    return direction.astype(np.float32), cv_auc, X.shape[1]


# ======================================================================== #
#  Phase 2 — Behavioral eval under steering                                #
# ======================================================================== #


def load_conflict_items_vulnerable(benchmark_path: str) -> list[dict[str, Any]]:
    """Load conflict items from vulnerable categories as raw dicts.

    Using raw dicts instead of BenchmarkItem to keep the behavioral eval
    loop lightweight and avoid import complications on the pod.
    """
    items: list[dict[str, Any]] = []
    with open(benchmark_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item["conflict"] and item["category"] in VULNERABLE_CATEGORIES:
                items.append(item)
    return items


def classify_response(
    text: str,
    correct_answer: str,
    lure_answer: str,
    answer_pattern: str,
    lure_pattern: str,
) -> str:
    """Classify a model response as 'correct', 'lure', or 'other'.

    Tries regex patterns first (more precise), then falls back to substring
    matching if patterns fail.
    """
    # Strip thinking traces from reasoning models
    clean = text
    if "<think>" in clean and "</think>" in clean:
        te = clean.index("</think>")
        clean = clean[te + len("</think>"):].strip()

    # Try answer_pattern / lure_pattern (regex) first
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

    # Fallback to substring
    if correct_answer and re.search(re.escape(correct_answer), clean, re.IGNORECASE):
        return "correct"
    if lure_answer and re.search(re.escape(lure_answer), clean, re.IGNORECASE):
        return "lure"

    return "other"


def run_behavioral_eval(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    *,
    max_new_tokens: int = 128,
    hook_ctx: Any = None,
    steer_position: str = "continuous",
) -> dict[str, Any]:
    """Run behavioral eval on items, optionally under a steering hook context.

    Args:
        steer_position: "continuous" keeps the hook active during the entire
            model.generate() call (prompt encoding + all generated tokens).
            "prompt" applies the hook only during a manual prompt forward pass,
            removes it, then runs generation without steering. This matters for
            reasoning models where 500+ thinking tokens can wash out a
            prompt-only signal.

    Returns dict with lure_rate, correct_rate, other_rate, n, and per-item details.
    """
    import torch

    verdicts: list[str] = []

    if steer_position == "continuous":
        # Original behavior: hook wraps the whole generation loop.
        ctx = hook_ctx if hook_ctx is not None else _NullContext()

        with ctx:
            for item in items:
                messages = [{"role": "user", "content": item["prompt"]}]
                if item.get("system_prompt"):
                    messages.insert(0, {"role": "system", "content": item["system_prompt"]})

                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    out = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )

                n_gen = out.shape[1] - inputs.input_ids.shape[1]
                response = tokenizer.decode(
                    out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False
                )

                verdict = classify_response(
                    response,
                    item["correct_answer"],
                    item["lure_answer"],
                    item.get("answer_pattern", ""),
                    item.get("lure_pattern", ""),
                )
                verdicts.append(verdict)

    elif steer_position == "prompt":
        # Prompt-only: steer during a prefill forward pass, then generate
        # without the hook so generated tokens are unperturbed.
        for item in items:
            messages = [{"role": "user", "content": item["prompt"]}]
            if item.get("system_prompt"):
                messages.insert(0, {"role": "system", "content": item["system_prompt"]})

            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            # Step 1: Run a steered prefill pass to produce perturbed KV cache.
            ctx = hook_ctx if hook_ctx is not None else _NullContext()
            with ctx:
                with torch.no_grad():
                    prefill_out = model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        use_cache=True,
                    )
            # Hook is now removed (exited context manager).

            # Step 2: Generate from the perturbed KV cache, no steering.
            past_kv = prefill_out.past_key_values
            # The last token's logits determine the first generated token.
            last_token_logits = prefill_out.logits[:, -1:, :]
            first_token_id = last_token_logits.argmax(dim=-1)  # greedy

            with torch.no_grad():
                out = model.generate(
                    input_ids=first_token_id,
                    attention_mask=torch.cat(
                        [inputs.attention_mask,
                         torch.ones(1, 1, device=model.device, dtype=inputs.attention_mask.dtype)],
                        dim=1,
                    ),
                    past_key_values=past_kv,
                    max_new_tokens=max_new_tokens - 1,  # already generated 1 token
                    do_sample=False,
                )

            # Reconstruct full generated sequence for decoding.
            full_gen_ids = torch.cat(
                [first_token_id, out[:, 1:]], dim=1
            )
            response = tokenizer.decode(full_gen_ids[0], skip_special_tokens=False)

            verdict = classify_response(
                response,
                item["correct_answer"],
                item["lure_answer"],
                item.get("answer_pattern", ""),
                item.get("lure_pattern", ""),
            )
            verdicts.append(verdict)

    else:
        raise ValueError(
            f"Unknown steer_position={steer_position!r}. Expected 'continuous' or 'prompt'."
        )

    n = len(verdicts)
    n_lure = sum(1 for v in verdicts if v == "lure")
    n_correct = sum(1 for v in verdicts if v == "correct")
    n_other = n - n_lure - n_correct

    return {
        "lure_rate": n_lure / n if n > 0 else 0.0,
        "correct_rate": n_correct / n if n > 0 else 0.0,
        "other_rate": n_other / n if n > 0 else 0.0,
        "n": n,
        "n_lure": n_lure,
        "n_correct": n_correct,
        "n_other": n_other,
    }


class _NullContext:
    """No-op context manager for the baseline (alpha=0) condition."""

    def __enter__(self) -> _NullContext:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ======================================================================== #
#  Phase 3 — Dose-response sweep                                           #
# ======================================================================== #


def run_dose_response(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    direction: np.ndarray,
    target_layer: int,
    alphas: list[float],
    *,
    max_new_tokens: int = 128,
    n_random_controls: int = 5,
    random_seed: int = 42,
    steer_position: str = "continuous",
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Sweep alphas for both the probe direction and random controls.

    Args:
        steer_position: "continuous" or "prompt". Passed through to
            run_behavioral_eval. See that function's docstring for details.

    Returns:
        probe_results: {alpha_str: {lure_rate, correct_rate, ...}}
        random_results: {alpha_str: {mean_lure_rate, std, ...}}
    """
    import torch
    from s1s2.causal.steering import SteeringHook, random_unit_direction

    hidden_dim = direction.shape[0]

    # Normalize the probe direction
    d_norm = float(np.linalg.norm(direction))
    if d_norm == 0.0:
        raise ValueError("Probe coef_ is zero vector -- probe failed.")
    direction_unit = direction / d_norm
    direction_t = torch.from_numpy(direction_unit.astype(np.float32))

    print(f"\n{'='*70}")
    print(f"  PROBE-DIRECTION STEERING SWEEP")
    print(f"  Layer {target_layer}, {len(items)} conflict items, "
          f"{len(alphas)} alphas, {n_random_controls} random controls")
    print(f"  Steer position: {steer_position}")
    print(f"{'='*70}")

    # --- Probe direction sweep ---
    probe_results: dict[str, dict[str, Any]] = {}
    for alpha in alphas:
        t0 = time.time()
        if alpha == 0.0:
            # Baseline: no hook
            res = run_behavioral_eval(
                model, tokenizer, items,
                max_new_tokens=max_new_tokens,
                hook_ctx=None,
                steer_position=steer_position,
            )
        else:
            hook = SteeringHook(
                model, layer=target_layer, direction=direction_t, alpha=alpha,
            )
            res = run_behavioral_eval(
                model, tokenizer, items,
                max_new_tokens=max_new_tokens,
                hook_ctx=hook,
                steer_position=steer_position,
            )
        elapsed = time.time() - t0
        probe_results[str(float(alpha))] = res
        lr = res["lure_rate"]
        cr = res["correct_rate"]
        print(f"  alpha={alpha:+5.1f}  lure={lr:.3f}  correct={cr:.3f}  "
              f"({elapsed:.1f}s)")

    # --- Random-direction controls ---
    print(f"\n  Random-direction controls ({n_random_controls} directions):")
    random_results: dict[str, dict[str, Any]] = {}

    for alpha in alphas:
        lure_rates: list[float] = []
        correct_rates: list[float] = []
        t0 = time.time()

        for seed_i in range(n_random_controls):
            # Deterministic seed per (alpha, seed_i)
            rnd_seed = random_seed + seed_i * 101 + int(abs(alpha) * 1000)
            rnd_dir = random_unit_direction(hidden_dim, seed=rnd_seed)

            if alpha == 0.0:
                # All random controls at alpha=0 are identical to baseline
                res = probe_results["0.0"]
            else:
                hook = SteeringHook(
                    model, layer=target_layer, direction=rnd_dir, alpha=alpha,
                )
                res = run_behavioral_eval(
                    model, tokenizer, items,
                    max_new_tokens=max_new_tokens,
                    hook_ctx=hook,
                    steer_position=steer_position,
                )
            lure_rates.append(res["lure_rate"])
            correct_rates.append(res["correct_rate"])

        elapsed = time.time() - t0
        random_results[str(float(alpha))] = {
            "mean_lure_rate": float(np.mean(lure_rates)),
            "std_lure_rate": float(np.std(lure_rates, ddof=1)) if len(lure_rates) > 1 else 0.0,
            "mean_correct_rate": float(np.mean(correct_rates)),
            "std_correct_rate": float(np.std(correct_rates, ddof=1)) if len(correct_rates) > 1 else 0.0,
            "lure_rates": lure_rates,
            "correct_rates": correct_rates,
            "n_directions": n_random_controls,
        }
        mlr = float(np.mean(lure_rates))
        slr = float(np.std(lure_rates, ddof=1)) if len(lure_rates) > 1 else 0.0
        print(f"  alpha={alpha:+5.1f}  mean_lure={mlr:.3f} +/- {slr:.3f}  ({elapsed:.1f}s)")

    return probe_results, random_results


# ======================================================================== #
#  Phase 4 — Visualization                                                 #
# ======================================================================== #


def make_figure(
    alphas: list[float],
    probe_results: dict[str, dict[str, Any]],
    random_results: dict[str, dict[str, Any]],
    model_name: str,
    target_layer: int,
    figure_path: str,
    steer_position: str = "continuous",
) -> None:
    """Dose-response figure: probe direction vs random controls."""
    sorted_alphas = sorted(alphas)
    alpha_strs = [str(float(a)) for a in sorted_alphas]

    probe_lure = [probe_results[a]["lure_rate"] for a in alpha_strs]
    probe_correct = [probe_results[a]["correct_rate"] for a in alpha_strs]

    rnd_lure_mean = [random_results[a]["mean_lure_rate"] for a in alpha_strs]
    rnd_lure_std = [random_results[a]["std_lure_rate"] for a in alpha_strs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left panel: lure rate ---
    ax1.plot(sorted_alphas, probe_lure, "o-", color="#d62728", lw=2.5,
             markersize=8, label="Probe direction", zorder=5)
    ax1.plot(sorted_alphas, rnd_lure_mean, "s--", color="#7f7f7f", lw=1.5,
             markersize=6, label="Random control (mean)", zorder=3)
    ax1.fill_between(
        sorted_alphas,
        [m - s for m, s in zip(rnd_lure_mean, rnd_lure_std)],
        [m + s for m, s in zip(rnd_lure_mean, rnd_lure_std)],
        alpha=0.2, color="#7f7f7f", label="Random control +/- 1 SD",
    )
    ax1.set_xlabel(r"Steering strength $\alpha$", fontsize=12)
    ax1.set_ylabel("Lure rate (conflict items)", fontsize=12)
    ax1.set_title("Lure rate vs steering strength", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(y=probe_lure[sorted_alphas.index(0.0)] if 0.0 in sorted_alphas else 0.5,
                ls=":", color="black", alpha=0.3, lw=1)
    ax1.axvline(x=0, ls=":", color="black", alpha=0.3, lw=1)
    ax1.grid(True, alpha=0.15)

    # --- Right panel: correct rate ---
    ax2.plot(sorted_alphas, probe_correct, "o-", color="#2ca02c", lw=2.5,
             markersize=8, label="Probe direction", zorder=5)
    rnd_correct_mean = [random_results[a]["mean_correct_rate"] for a in alpha_strs]
    rnd_correct_std = [random_results[a]["std_correct_rate"] for a in alpha_strs]
    ax2.plot(sorted_alphas, rnd_correct_mean, "s--", color="#7f7f7f", lw=1.5,
             markersize=6, label="Random control (mean)", zorder=3)
    ax2.fill_between(
        sorted_alphas,
        [m - s for m, s in zip(rnd_correct_mean, rnd_correct_std)],
        [m + s for m, s in zip(rnd_correct_mean, rnd_correct_std)],
        alpha=0.2, color="#7f7f7f", label="Random control +/- 1 SD",
    )
    ax2.set_xlabel(r"Steering strength $\alpha$", fontsize=12)
    ax2.set_ylabel("Correct rate (conflict items)", fontsize=12)
    ax2.set_title("Correct rate vs steering strength", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.axhline(y=probe_correct[sorted_alphas.index(0.0)] if 0.0 in sorted_alphas else 0.5,
                ls=":", color="black", alpha=0.3, lw=1)
    ax2.axvline(x=0, ls=":", color="black", alpha=0.3, lw=1)
    ax2.grid(True, alpha=0.15)

    short_model = model_name.split("/")[-1]
    pos_label = "continuous" if steer_position == "continuous" else "prompt-only"
    fig.suptitle(
        f"Probe-direction steering: {short_model}, layer {target_layer} ({pos_label})\n"
        f"(+alpha = toward S2 / away from conflict-typical representation)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    print(f"\n  Figure saved: {figure_path}")
    plt.close(fig)


# ======================================================================== #
#  CLI entry point                                                         #
# ======================================================================== #


def hf_model_key(model_id: str) -> str:
    """Convert HuggingFace model ID to the HDF5 key convention: / -> _."""
    return model_id.replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe-direction steering experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model ID (e.g. unsloth/Meta-Llama-3.1-8B-Instruct)")
    parser.add_argument("--h5-path", type=str, required=True,
                        help="Path to HDF5 activation file.")
    parser.add_argument("--h5-model-key", type=str, default=None,
                        help="Model key in the HDF5 file. Default: auto from --model.")
    parser.add_argument("--benchmark", type=str, required=True,
                        help="Path to benchmark.jsonl.")
    parser.add_argument("--target-layer", type=int, required=True,
                        help="Layer to steer (0-indexed).")
    parser.add_argument("--alphas", type=str, default="-5,-3,-1,-0.5,0,0.5,1,3,5",
                        help="Comma-separated steering strengths.")
    parser.add_argument("--n-random-controls", type=int, default=5,
                        help="Number of random-direction control baselines.")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Max tokens to generate per item.")
    parser.add_argument("--steer-position", type=str, default="continuous",
                        choices=["continuous", "prompt"],
                        help="Where to apply steering. 'continuous' (default): hook "
                             "stays active during generation, steering every token. "
                             "'prompt': hook fires only during prompt prefill, then "
                             "is removed before generation. Use 'continuous' for "
                             "reasoning models to avoid temporal washout.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path for output JSON.")
    parser.add_argument("--figure-dir", type=str, default=None,
                        help="Dir for figures. Default: figures/ relative to project root.")
    parser.add_argument("--cache-dir", type=str, default="/workspace/hf_cache",
                        help="HuggingFace cache directory.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype. Default: bfloat16.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run on 3 items with 2 alphas for quick validation.")
    args = parser.parse_args()

    set_global_seed(args.seed, deterministic_torch=False)

    alphas = [float(x.strip()) for x in args.alphas.split(",")]
    h5_model_key = args.h5_model_key or hf_model_key(args.model)

    if args.figure_dir:
        fig_dir = Path(args.figure_dir)
    else:
        fig_dir = _ROOT / "figures"

    print(f"{'='*70}")
    print(f"  PROBE-DIRECTION STEERING EXPERIMENT")
    print(f"{'='*70}")
    print(f"  Model:        {args.model}")
    print(f"  H5 path:      {args.h5_path}")
    print(f"  H5 model key: {h5_model_key}")
    print(f"  Benchmark:    {args.benchmark}")
    print(f"  Target layer: {args.target_layer}")
    print(f"  Alphas:       {alphas}")
    print(f"  Random dirs:  {args.n_random_controls}")
    print(f"  Max tokens:   {args.max_new_tokens}")
    print(f"  Steer pos:    {args.steer_position}")
    print(f"  Seed:         {args.seed}")
    print(f"  Smoke test:   {args.smoke_test}")
    print()

    # ── Phase 1: Train probe and extract direction ────────────────────
    t_probe = time.time()
    print("[Phase 1] Training probe and extracting S1/S2 direction...")
    direction, probe_auc, hidden_dim = train_probe_and_get_direction(
        h5_path=args.h5_path,
        model_key=h5_model_key,
        target_layer=args.target_layer,
        seed=args.seed,
    )
    t_probe = time.time() - t_probe
    print(f"  Probe training: {t_probe:.1f}s, AUC={probe_auc:.4f}, "
          f"hidden_dim={hidden_dim}")

    # ── Phase 2: Load model ───────────────────────────────────────────
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    print(f"\n[Phase 2] Loading model: {args.model}")
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
    )
    model.eval()
    t_load = time.time() - t_load
    print(f"  Model loaded in {t_load:.1f}s")
    if torch.cuda.is_available():
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── Phase 3: Load benchmark items ─────────────────────────────────
    items = load_conflict_items_vulnerable(args.benchmark)
    print(f"\n[Phase 3] Loaded {len(items)} conflict items from vulnerable categories")

    if args.smoke_test:
        items = items[:3]
        alphas = [a for a in alphas if a in (-3.0, 0.0, 3.0)]
        if 0.0 not in alphas:
            alphas.insert(0, 0.0)
        print(f"  SMOKE TEST: {len(items)} items, alphas={alphas}")

    if not items:
        print("ERROR: No conflict items found in vulnerable categories.", file=sys.stderr)
        sys.exit(1)

    # ── Phase 4: Dose-response sweep ──────────────────────────────────
    print(f"\n[Phase 4] Running dose-response sweep...")
    t_sweep = time.time()
    probe_results, random_results = run_dose_response(
        model=model,
        tokenizer=tokenizer,
        items=items,
        direction=direction,
        target_layer=args.target_layer,
        alphas=alphas,
        max_new_tokens=args.max_new_tokens,
        n_random_controls=args.n_random_controls,
        random_seed=args.seed + 7777,
        steer_position=args.steer_position,
    )
    t_sweep = time.time() - t_sweep
    print(f"\n  Sweep completed in {t_sweep:.1f}s")

    # ── Phase 5: Save results ─────────────────────────────────────────
    short_model = args.model.split("/")[-1].lower().replace("-", "_")
    figure_name = f"fig_probe_steering_{short_model}_l{args.target_layer}.pdf"
    figure_path = str(fig_dir / figure_name)

    output_data: dict[str, Any] = {
        "model": args.model,
        "h5_model_key": h5_model_key,
        "target_layer": args.target_layer,
        "direction": "probe_coef",
        "probe_train_auc": probe_auc,
        "hidden_dim": hidden_dim,
        "n_conflict_items": len(items),
        "vulnerable_categories": VULNERABLE_CATEGORIES,
        "alphas": {k: v for k, v in probe_results.items()},
        "random_controls": {
            k: {
                "mean_lure_rate": v["mean_lure_rate"],
                "std": v["std_lure_rate"],
                "mean_correct_rate": v["mean_correct_rate"],
                "std_correct_rate": v["std_correct_rate"],
                "lure_rates": v["lure_rates"],
                "correct_rates": v["correct_rates"],
                "n_directions": v["n_directions"],
            }
            for k, v in random_results.items()
        },
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "n_random_controls": args.n_random_controls,
            "steer_position": args.steer_position,
            "seed": args.seed,
            "dtype": args.dtype,
            "smoke_test": args.smoke_test,
        },
        "figure_path": figure_path,
        "elapsed_s": {
            "probe_training": round(t_probe, 1),
            "model_loading": round(t_load, 1),
            "steering_sweep": round(t_sweep, 1),
            "total": round(t_probe + t_load + t_sweep, 1),
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(output_data, fh, indent=2, allow_nan=True)
    print(f"\n  Results saved: {args.output}")

    # ── Phase 6: Figure ───────────────────────────────────────────────
    print("\n[Phase 6] Generating figure...")
    make_figure(
        alphas=alphas,
        probe_results=probe_results,
        random_results=random_results,
        model_name=args.model,
        target_layer=args.target_layer,
        figure_path=figure_path,
        steer_position=args.steer_position,
    )

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Probe AUC:           {probe_auc:.4f}")
    print(f"  Baseline lure rate:  {probe_results.get('0.0', {}).get('lure_rate', 'N/A')}")

    sorted_alphas = sorted(alphas)
    min_alpha = str(float(sorted_alphas[0]))
    max_alpha = str(float(sorted_alphas[-1]))
    if min_alpha in probe_results and max_alpha in probe_results:
        lr_neg = probe_results[min_alpha]["lure_rate"]
        lr_pos = probe_results[max_alpha]["lure_rate"]
        delta = lr_neg - lr_pos
        print(f"  Lure rate at alpha={sorted_alphas[0]:+.1f}: {lr_neg:.3f}")
        print(f"  Lure rate at alpha={sorted_alphas[-1]:+.1f}: {lr_pos:.3f}")
        print(f"  Delta (neg - pos):   {delta:+.3f}")
        if delta > 0.10:
            print(f"  >>> STRONG causal effect: probe direction modulates behavior")
        elif delta > 0.05:
            print(f"  >>> Moderate causal effect")
        else:
            print(f"  >>> Weak or no causal effect")

    baseline_key = "0.0"
    if baseline_key in random_results:
        rnd_bl = random_results[baseline_key]["mean_lure_rate"]
        rnd_max = random_results[max_alpha]["mean_lure_rate"]
        rnd_min = random_results[min_alpha]["mean_lure_rate"]
        rnd_range = abs(rnd_min - rnd_max)
        print(f"  Random control range: {rnd_range:.3f} (expect ~0)")

    print(f"\n  Output: {args.output}")
    print(f"  Figure: {figure_path}")
    print()


if __name__ == "__main__":
    main()
