#!/usr/bin/env python3
"""Steering-specificity controls: (a) probe direction on non-conflict items,
(b) mean-difference direction on conflict items.

Gemini/GPT reviewer feedback demanded these two controls:
    1. Specificity: steering along the probe direction on CONTROL (non-conflict)
       items should show null effect. If steering also flips control-item
       behavior, the probe direction captures something other than processing
       mode (e.g., generic "answer first-instinct-thing"-ness).
    2. Mean-difference baseline: CogBias uses h_conflict - h_control mean-diff
       steering. We compare probe-weight steering (ours) vs mean-diff steering
       on the same items. If our probe-weight direction produces a larger
       causal handle than mean-diff, we have a stronger methodological claim.

Reuses load_h5_metadata, load_residual_p0, run_behavioral_eval, classify_response
from run_probe_steering.py by importing the module.

Usage (Llama, fastest to run because no CoT):
    python scripts/run_steering_specificity.py \
        --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --h5-path data/activations/llama31_8b_instruct.h5 \
        --benchmark data/benchmark/benchmark.jsonl \
        --target-layer 14 \
        --alphas "-5,-3,-1,0,1,3,5" \
        --output results/causal/steering_specificity_llama_l14.json \
        --cache-dir /workspace/hf_cache
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

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from s1s2.utils.seed import set_global_seed
import run_probe_steering as rps


VULNERABLE_CATEGORIES = rps.VULNERABLE_CATEGORIES


def load_control_items_vulnerable(benchmark_path: str) -> list[dict[str, Any]]:
    """Load NON-conflict (control) items from vulnerable categories.

    These are the matched-pair controls for the conflict items: same surface
    form, same category, but no S1/S2 conflict (normative answer is obvious).
    Steering on these items tests specificity.
    """
    items: list[dict[str, Any]] = []
    with open(benchmark_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if (not item["conflict"]) and item["category"] in VULNERABLE_CATEGORIES:
                items.append(item)
    return items


def compute_mean_diff_direction(
    h5_path: str,
    model_key: str,
    target_layer: int,
) -> np.ndarray:
    """Compute h_conflict_mean - h_control_mean on vulnerable categories.

    This is the CogBias / Rimsky-style "steering vector": the mean activation
    difference between conflict and control items. A simpler-than-probe
    direction used as a baseline.
    """
    with h5py.File(h5_path, "r") as f:
        meta = rps.load_h5_metadata(f)
        X_all = rps.load_residual_p0(f, model_key, target_layer)

    categories = meta["category"]
    conflict = meta["conflict"].astype(np.int64)
    vuln_mask = np.isin(categories, VULNERABLE_CATEGORIES)
    X = X_all[vuln_mask]
    y = conflict[vuln_mask]

    X_conflict = X[y == 1]
    X_control = X[y == 0]
    mean_conflict = X_conflict.mean(axis=0)
    mean_control = X_control.mean(axis=0)

    # NEGATE so positive alpha pushes away from conflict (matches probe convention).
    direction = -(mean_conflict - mean_control)
    return direction.astype(np.float32)


def run_sweep_one_config(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    direction: np.ndarray,
    target_layer: int,
    alphas: list[float],
    label: str,
    *,
    max_new_tokens: int = 128,
    steer_position: str = "continuous",
) -> dict[str, dict[str, Any]]:
    """Run a single (direction, items) sweep over alphas. No random controls."""
    import torch
    from s1s2.causal.steering import SteeringHook

    # Normalize
    d_norm = float(np.linalg.norm(direction))
    if d_norm == 0.0:
        raise ValueError(f"[{label}] Direction is zero vector.")
    direction_unit = direction / d_norm
    direction_t = torch.from_numpy(direction_unit.astype(np.float32))

    print(f"\n{'='*70}")
    print(f"  SWEEP: {label}")
    print(f"  Layer {target_layer}, {len(items)} items, {len(alphas)} alphas")
    print(f"{'='*70}")

    out: dict[str, dict[str, Any]] = {}
    for alpha in alphas:
        t0 = time.time()
        if alpha == 0.0:
            res = rps.run_behavioral_eval(
                model, tokenizer, items,
                max_new_tokens=max_new_tokens,
                hook_ctx=None, steer_position=steer_position,
            )
        else:
            hook = SteeringHook(
                model, layer=target_layer, direction=direction_t, alpha=alpha,
            )
            res = rps.run_behavioral_eval(
                model, tokenizer, items,
                max_new_tokens=max_new_tokens,
                hook_ctx=hook, steer_position=steer_position,
            )
        elapsed = time.time() - t0
        out[str(float(alpha))] = res
        print(f"  alpha={alpha:+5.1f}  lure={res['lure_rate']:.3f}  "
              f"correct={res['correct_rate']:.3f}  ({elapsed:.1f}s)")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--h5-model-key", default=None)
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--target-layer", type=int, default=14)
    parser.add_argument("--alphas", type=str, default="-5,-3,-1,0,1,3,5")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steer-position", type=str, default="continuous",
                        choices=["continuous", "prompt"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="/workspace/hf_cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    set_global_seed(args.seed, deterministic_torch=False)
    alphas = [float(x.strip()) for x in args.alphas.split(",")]
    h5_model_key = args.h5_model_key or rps.hf_model_key(args.model)

    print(f"{'='*70}")
    print(f"  STEERING SPECIFICITY EXPERIMENT")
    print(f"{'='*70}")
    print(f"  Model:        {args.model}")
    print(f"  Benchmark:    {args.benchmark}")
    print(f"  Target layer: {args.target_layer}")
    print(f"  Alphas:       {alphas}")

    # Phase 1: directions
    print("\n[Phase 1] Training probe direction...")
    probe_dir, probe_auc, hidden_dim = rps.train_probe_and_get_direction(
        h5_path=args.h5_path, model_key=h5_model_key,
        target_layer=args.target_layer, seed=args.seed,
    )
    print(f"\n[Phase 1b] Computing mean-difference direction...")
    meandiff_dir = compute_mean_diff_direction(
        h5_path=args.h5_path, model_key=h5_model_key,
        target_layer=args.target_layer,
    )
    cos_sim = float(
        np.dot(probe_dir, meandiff_dir)
        / (np.linalg.norm(probe_dir) * np.linalg.norm(meandiff_dir))
    )
    print(f"  cos(probe, meandiff) = {cos_sim:.4f}")

    # Phase 2: items
    print("\n[Phase 2] Loading items...")
    conflict_items = rps.load_conflict_items_vulnerable(args.benchmark)
    control_items = load_control_items_vulnerable(args.benchmark)
    print(f"  {len(conflict_items)} conflict items, {len(control_items)} control items")

    if args.smoke_test:
        conflict_items = conflict_items[:3]
        control_items = control_items[:3]
        alphas = [-3.0, 0.0, 3.0]

    # Phase 3: model
    print("\n[Phase 3] Loading model...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir,
        torch_dtype=dtype_map[args.dtype], device_map="auto",
    )
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s")
    if torch.cuda.is_available():
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Phase 4: four sweeps
    sweeps = {
        "probe_on_conflict": (probe_dir, conflict_items,
                              "Probe direction on CONFLICT items (baseline reproduction)"),
        "probe_on_control": (probe_dir, control_items,
                             "Probe direction on CONTROL items (specificity test)"),
        "meandiff_on_conflict": (meandiff_dir, conflict_items,
                                 "Mean-diff direction on CONFLICT items (CogBias baseline)"),
        "meandiff_on_control": (meandiff_dir, control_items,
                                "Mean-diff direction on CONTROL items (specificity of mean-diff)"),
    }
    results: dict[str, Any] = {
        "model": args.model,
        "h5_model_key": h5_model_key,
        "target_layer": args.target_layer,
        "probe_cv_auc": probe_auc,
        "hidden_dim": hidden_dim,
        "vulnerable_categories": VULNERABLE_CATEGORIES,
        "n_conflict_items": len(conflict_items),
        "n_control_items": len(control_items),
        "cos_probe_meandiff": cos_sim,
        "alphas": alphas,
        "sweeps": {},
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "steer_position": args.steer_position,
            "seed": args.seed,
            "dtype": args.dtype,
            "smoke_test": args.smoke_test,
        },
    }

    for sweep_key, (direction, items, label) in sweeps.items():
        sweep_result = run_sweep_one_config(
            model, tokenizer, items, direction, args.target_layer, alphas,
            label, max_new_tokens=args.max_new_tokens,
            steer_position=args.steer_position,
        )
        results["sweeps"][sweep_key] = sweep_result

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote results to {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for sweep_key in sweeps:
        print(f"\n  {sweep_key}:")
        for alpha_str, res in results["sweeps"][sweep_key].items():
            print(f"    alpha={float(alpha_str):+5.1f}  "
                  f"lure={res['lure_rate']:.3f}  correct={res['correct_rate']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
