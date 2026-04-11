#!/usr/bin/env python3
"""Estimate compute cost for the full s1s2 pipeline.

Based on empirical timings from planning docs and standard A100/H100 rates.
Prints a breakdown table with estimated wall time and dollar cost per stage.

Usage:
    python deploy/cost_estimate.py
    python deploy/cost_estimate.py --n-problems 284 --n-models 4
    python deploy/cost_estimate.py --platform fasrc  # free (allocation-based)
"""
from __future__ import annotations

import argparse


# ---- Cost model -------------------------------------------------------------

# RunPod hourly rates (USD) — ranges as of 2026-Q1
RUNPOD_RATES: dict[str, tuple[float, float]] = {
    "A100-80GB": (0.82, 1.39),
    "H100-80GB": (1.25, 2.69),
}

# FASRC/Kempner is allocation-based (SUs), not dollar-billed.
# 1 SU ~= 1 CPU-core-hour. GPU hours consume SUs at a higher rate.
# For cost comparison purposes we treat it as "free" (covered by PI allocation).
FASRC_GPU_SU_RATE: float = 10.0  # SUs per GPU-hour (approximate)


def estimate_pipeline(
    n_problems: int = 284,
    n_models: int = 4,
    include_attention: bool = True,
) -> list[dict[str, str | float]]:
    """Return per-stage estimates as a list of dicts.

    Timing model (empirical, A100-80GB baseline):
    - Model loading: ~30s per model (BF16 8B param model)
    - Extraction per problem: ~2s (forward pass + hook capture)
    - Probe training: CPU-bound, ~5 min total
    - SAE encoding: ~2 hours (encode activations through SAE)
    - Attention re-extraction: ~4 hours (needs output_attentions=True, slower)
    - Geometry analysis: CPU-bound, ~10 min

    These are conservative estimates. Actual times may be lower on H100.
    """
    stages: list[dict[str, str | float]] = []

    # 1. Model download (one-time, not GPU time)
    # ~16GB per 8B model in BF16, ~64GB total. Depends on network speed.
    stages.append({
        "stage": "Model download",
        "gpu_hours": 0.0,
        "wall_minutes": 30.0,
        "notes": f"{n_models} models, ~16GB each. Network-bound.",
    })

    # 2. Activation extraction
    # Per model: load (30s) + n_problems * 2s/problem
    extract_per_model_s = 30 + n_problems * 2.0
    extract_total_s = extract_per_model_s * n_models
    extract_hours = extract_total_s / 3600
    stages.append({
        "stage": "Extraction (residual stream)",
        "gpu_hours": extract_hours,
        "wall_minutes": extract_total_s / 60,
        "notes": f"{n_models} models x {n_problems} problems @ ~2s/problem",
    })

    # 3. Probes (CPU-bound)
    stages.append({
        "stage": "Probes (linear probing)",
        "gpu_hours": 0.0,
        "wall_minutes": 5.0,
        "notes": "CPU-only. Logistic regression + mass-mean on cached activations.",
    })

    # 4. SAE encoding
    # Encoding all residual activations through SAE is compute-intensive
    sae_hours = 2.0
    stages.append({
        "stage": "SAE analysis",
        "gpu_hours": sae_hours,
        "wall_minutes": sae_hours * 60,
        "notes": "Encode activations through Llama Scope + Gemma Scope SAEs.",
    })

    # 5. Attention analysis (optional, expensive)
    if include_attention:
        attn_hours = 4.0
        stages.append({
            "stage": "Attention extraction + analysis",
            "gpu_hours": attn_hours,
            "wall_minutes": attn_hours * 60,
            "notes": "Re-extract with output_attentions=True. Eager attn, no Flash.",
        })

    # 6. Geometry (CPU-bound)
    stages.append({
        "stage": "Geometry analysis",
        "gpu_hours": 0.0,
        "wall_minutes": 10.0,
        "notes": "PCA + silhouette + separability on cached activations.",
    })

    return stages


def print_estimate(
    stages: list[dict[str, str | float]],
    gpu_type: str = "A100-80GB",
    platform: str = "runpod",
) -> None:
    """Print a formatted cost estimate table."""
    total_gpu_hours = sum(float(s["gpu_hours"]) for s in stages)
    total_wall_min = sum(float(s["wall_minutes"]) for s in stages)

    print()
    print("=" * 80)
    print(f"  s1s2 Pipeline Cost Estimate — {platform.upper()} ({gpu_type})")
    print("=" * 80)
    print()
    print(f"  {'Stage':<35} {'GPU hrs':>8} {'Wall min':>9}   Notes")
    print(f"  {'-'*35} {'-'*8} {'-'*9}   {'-'*30}")

    for s in stages:
        gpu_h = float(s["gpu_hours"])
        wall_m = float(s["wall_minutes"])
        gpu_str = f"{gpu_h:.2f}" if gpu_h > 0 else "—"
        print(f"  {s['stage']:<35} {gpu_str:>8} {wall_m:>9.0f}   {s['notes']}")

    print(f"  {'-'*35} {'-'*8} {'-'*9}")
    print(f"  {'TOTAL':<35} {total_gpu_hours:>8.2f} {total_wall_min:>9.0f}")
    print()

    if platform == "runpod":
        rate_lo, rate_hi = RUNPOD_RATES.get(gpu_type, (1.0, 2.0))
        # You pay for wall time, not just GPU time (pod is running throughout)
        wall_hours = total_wall_min / 60
        cost_lo = wall_hours * rate_lo
        cost_hi = wall_hours * rate_hi
        print(f"  RunPod {gpu_type} rate: ${rate_lo:.2f}—${rate_hi:.2f}/hr")
        print(f"  Estimated wall time: {wall_hours:.1f} hours")
        print(f"  Estimated cost: ${cost_lo:.2f}—${cost_hi:.2f}")
        print()
        print(f"  Recommendation: Use a single A100-80GB pod. At ~{wall_hours:.0f}h wall time,")
        print(f"  total cost is ${cost_lo:.0f}—${cost_hi:.0f}. An H100 is faster but")
        print(f"  the workload is memory-bound, so the speedup is marginal.")
    elif platform == "fasrc":
        sus = total_gpu_hours * FASRC_GPU_SU_RATE
        print(f"  FASRC GPU SU rate: ~{FASRC_GPU_SU_RATE:.0f} SUs per GPU-hour")
        print(f"  Estimated SU cost: ~{sus:.0f} SUs")
        print(f"  (Covered by PI allocation — no dollar cost.)")
        print()
        print(f"  Note: Kempner gpu_requeue has 7hr max wall time.")
        if total_wall_min > 7 * 60:
            n_jobs = int(total_wall_min / (7 * 60)) + 1
            print(f"  Pipeline exceeds 7hr — will need ~{n_jobs} preemptible job submissions.")
            print(f"  Use deploy/checkpoint_extract.py for automatic resume.")
        else:
            print(f"  Pipeline fits within a single 7hr job.")

    print()
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-problems", type=int, default=284, help="Number of benchmark problems")
    parser.add_argument("--n-models", type=int, default=4, help="Number of models")
    parser.add_argument(
        "--platform",
        choices=["runpod", "fasrc"],
        default="runpod",
        help="Deployment platform (default: runpod)",
    )
    parser.add_argument(
        "--gpu-type",
        choices=["A100-80GB", "H100-80GB"],
        default="A100-80GB",
        help="GPU type for cost calculation",
    )
    parser.add_argument(
        "--skip-attention",
        action="store_true",
        help="Exclude attention re-extraction from estimate",
    )
    args = parser.parse_args()

    stages = estimate_pipeline(
        n_problems=args.n_problems,
        n_models=args.n_models,
        include_attention=not args.skip_attention,
    )
    print_estimate(stages, gpu_type=args.gpu_type, platform=args.platform)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
