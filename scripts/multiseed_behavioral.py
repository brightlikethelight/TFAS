#!/usr/bin/env python3
"""Multi-seed robustness analysis for behavioral lure rates.

Addresses the reproducibility concern: syllogism lure rates for Llama
showed 52% -> 4% instability across runs. This script runs the same model
on the same benchmark N times with different generation seeds (under
do_sample=True) and reports per-category mean +/- std of lure rates.
Categories with >10pp std are flagged as unstable.

Usage:
    python scripts/multiseed_behavioral.py \
        --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --max-new-tokens 256

    python scripts/multiseed_behavioral.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --seeds 0,42,123,777,999 \
        --max-new-tokens 4096
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = "0,42,123"
TEMPERATURE = 0.7
TOP_P = 0.95
DEFAULT_BENCHMARK = "data/benchmark/benchmark.jsonl"
INSTABILITY_THRESHOLD_PP = 10.0  # flag categories with std > 10 percentage points


# ---------------------------------------------------------------------------
# Scoring (self-contained to avoid import issues on pod)
# ---------------------------------------------------------------------------

def score_response(text: str, correct_answer: str, lure_answer: str,
                   answer_pattern: str, lure_pattern: str,
                   is_conflict: bool) -> str:
    """Classify response as correct / lure / other_wrong / refusal.

    Mirrors the logic in s1s2.extract.scoring but is self-contained so
    the script runs without installing the package.
    """
    if text is None:
        text = ""

    # Refusal check
    refusal_phrases = (
        "i cannot", "i can't help", "i refuse", "i will not", "i won't",
        "as an ai", "as a language model",
    )
    lowered = text.lower()
    if any(p in lowered for p in refusal_phrases):
        return "refusal"

    def _compile(pat: str) -> re.Pattern[str] | None:
        if not pat:
            return None
        try:
            return re.compile(pat, flags=re.IGNORECASE)
        except re.error:
            return re.compile(re.escape(pat), flags=re.IGNORECASE)

    ans_re = _compile(answer_pattern)
    lure_re = _compile(lure_pattern) if lure_pattern else None

    ans_match = ans_re.search(text) if ans_re else None
    lure_match = lure_re.search(text) if lure_re else None

    matched_correct = ans_match is not None
    matched_lure = (lure_match is not None) and is_conflict

    if matched_correct:
        if matched_lure and lure_match is not None and lure_match.start() > ans_match.start():
            return "lure"
        return "correct"
    if matched_lure:
        return "lure"
    return "other_wrong"


def split_thinking(text: str) -> tuple[str, str]:
    """Split <think>...</think> trace from answer text."""
    if "<think>" in text and "</think>" in text:
        ts = text.index("<think>") + len("<think>")
        te = text.index("</think>")
        return text[ts:te].strip(), text[te + len("</think>"):].strip()
    return "", text.strip()


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------

def run_single_seed(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    seed: int,
    max_new_tokens: int,
    device: str,
) -> list[dict[str, Any]]:
    """Run behavioral eval on all items with a specific generation seed.

    Sets torch.manual_seed(seed) before EACH item's generation call
    to ensure the seed fully controls that item's sampling trajectory.
    """
    results: list[dict[str, Any]] = []

    for i, item in enumerate(items):
        messages = [{"role": "user", "content": item["prompt"]}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        # Seed immediately before generation for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.time() - t0
        n_gen = out.shape[1] - inputs.input_ids.shape[1]
        response = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False,
        )

        _thinking, answer = split_thinking(response)
        verdict = score_response(
            answer,
            item["correct_answer"],
            item.get("lure_answer", ""),
            item.get("answer_pattern", re.escape(item["correct_answer"])),
            item.get("lure_pattern", re.escape(item.get("lure_answer", ""))),
            item["conflict"],
        )

        results.append({
            "id": item["id"],
            "category": item["category"],
            "conflict": item["conflict"],
            "verdict": verdict,
            "n_tokens": n_gen,
            "elapsed_s": round(elapsed, 1),
        })

        if (i + 1) % 20 == 0 or i == 0:
            print(f"    [seed={seed}] {i+1}/{len(items)} "
                  f"last={item['id'][:35]:35s} -> {verdict}")

    return results


# ---------------------------------------------------------------------------
# Per-category lure rate computation
# ---------------------------------------------------------------------------

def compute_per_category_lure_rates(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute lure rate for each category from a single seed's results."""
    by_cat: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_conflict": 0, "n_lured": 0, "n_correct": 0, "n_total": 0}
    )
    for r in results:
        cat = r["category"]
        by_cat[cat]["n_total"] += 1
        if r["conflict"]:
            by_cat[cat]["n_conflict"] += 1
            if r["verdict"] == "lure":
                by_cat[cat]["n_lured"] += 1
            elif r["verdict"] == "correct":
                by_cat[cat]["n_correct"] += 1

    out: dict[str, dict[str, Any]] = {}
    for cat, counts in sorted(by_cat.items()):
        nc = counts["n_conflict"]
        out[cat] = {
            "n_conflict": nc,
            "n_lured": counts["n_lured"],
            "lure_rate": counts["n_lured"] / max(nc, 1),
            "correct_rate": counts["n_correct"] / max(nc, 1),
            "n_total": counts["n_total"],
        }
    return out


# ---------------------------------------------------------------------------
# Cross-seed aggregation
# ---------------------------------------------------------------------------

def aggregate_across_seeds(
    all_seed_cat_rates: list[dict[str, dict[str, Any]]],
    seeds: list[int],
) -> dict[str, dict[str, Any]]:
    """Compute mean +/- std of lure rates across seeds per category.

    Returns a dict keyed by category with aggregated statistics and
    an instability flag.
    """
    # Collect all categories seen across any seed
    all_cats: set[str] = set()
    for rates in all_seed_cat_rates:
        all_cats.update(rates.keys())

    agg: dict[str, dict[str, Any]] = {}
    for cat in sorted(all_cats):
        lure_rates = []
        correct_rates = []
        for rates in all_seed_cat_rates:
            if cat in rates:
                lure_rates.append(rates[cat]["lure_rate"])
                correct_rates.append(rates[cat]["correct_rate"])
            else:
                lure_rates.append(0.0)
                correct_rates.append(0.0)

        lr_arr = np.array(lure_rates)
        cr_arr = np.array(correct_rates)
        lr_std_pp = float(np.std(lr_arr, ddof=1)) * 100 if len(lr_arr) > 1 else 0.0

        agg[cat] = {
            "lure_rate_mean": float(np.mean(lr_arr)),
            "lure_rate_std": float(np.std(lr_arr, ddof=1)) if len(lr_arr) > 1 else 0.0,
            "lure_rate_std_pp": lr_std_pp,
            "lure_rate_per_seed": {str(s): float(lr) for s, lr in zip(seeds, lure_rates)},
            "correct_rate_mean": float(np.mean(cr_arr)),
            "correct_rate_std": float(np.std(cr_arr, ddof=1)) if len(cr_arr) > 1 else 0.0,
            "correct_rate_per_seed": {str(s): float(cr) for s, cr in zip(seeds, correct_rates)},
            "unstable": lr_std_pp > INSTABILITY_THRESHOLD_PP,
            "n_seeds": len(seeds),
        }

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-seed robustness analysis for behavioral lure rates",
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model ID (e.g. unsloth/Meta-Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--benchmark", default=DEFAULT_BENCHMARK,
        help="Path to benchmark.jsonl",
    )
    parser.add_argument(
        "--seeds", default=DEFAULT_SEEDS,
        help="Comma-separated generation seeds (default: 0,42,123)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Max tokens to generate per item",
    )
    parser.add_argument(
        "--cache-dir", default="/workspace/hf_cache",
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--output", default=None,
        help="Override output path (default: results/robustness/{model}_multiseed.json)",
    )
    parser.add_argument(
        "--n-items", type=int, default=None,
        help="Limit to first N items (for debugging)",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    model_name = args.model.replace("/", "_")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"results/robustness/{model_name}_multiseed.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load benchmark ---
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        print(f"ERROR: benchmark not found at {benchmark_path}", file=sys.stderr)
        return 1

    items: list[dict[str, Any]] = []
    with benchmark_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if args.n_items:
        items = items[:args.n_items]

    n_conflict = sum(1 for it in items if it["conflict"])
    print(f"Benchmark: {len(items)} items ({n_conflict} conflict)")
    print(f"Seeds: {seeds}")
    print(f"Generation: do_sample=True, temperature={TEMPERATURE}, top_p={TOP_P}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Output: {output_path}")
    print()

    # --- Load model (once, reuse across seeds) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {args.model} on {device}...")

    tok_kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if Path(args.cache_dir).exists():
        tok_kwargs["cache_dir"] = args.cache_dir
        model_kwargs["cache_dir"] = args.cache_dir
    if device == "cuda":
        model_kwargs["device_map"] = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB"
          if torch.cuda.is_available() else "Model loaded (CPU).")
    print()

    # --- Run each seed ---
    all_seed_results: list[list[dict[str, Any]]] = []
    all_seed_cat_rates: list[dict[str, dict[str, Any]]] = []
    seed_timings: dict[int, float] = {}

    for seed_idx, seed in enumerate(seeds):
        print(f"{'='*60}")
        print(f"SEED {seed} ({seed_idx+1}/{len(seeds)})")
        print(f"{'='*60}")

        t0 = time.time()
        results = run_single_seed(
            model=model,
            tokenizer=tokenizer,
            items=items,
            seed=seed,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        elapsed = time.time() - t0
        seed_timings[seed] = elapsed

        cat_rates = compute_per_category_lure_rates(results)
        all_seed_results.append(results)
        all_seed_cat_rates.append(cat_rates)

        # Print per-seed summary
        total_lured = sum(1 for r in results if r["conflict"] and r["verdict"] == "lure")
        total_conflict = sum(1 for r in results if r["conflict"])
        overall_lr = total_lured / max(total_conflict, 1)
        print(f"\n  Seed {seed}: overall lure rate = {overall_lr:.1%} "
              f"({total_lured}/{total_conflict}) in {elapsed:.0f}s")
        for cat, st in cat_rates.items():
            print(f"    {cat:20s}  {st['n_lured']}/{st['n_conflict']} lured "
                  f"({st['lure_rate']:.0%})")
        print()

    # --- Aggregate across seeds ---
    print(f"\n{'='*60}")
    print("MULTI-SEED AGGREGATION")
    print(f"{'='*60}")

    agg = aggregate_across_seeds(all_seed_cat_rates, seeds)

    # Print results table
    print(f"\n{'Category':20s} {'Mean LR':>8s} {'Std LR':>8s} {'Std(pp)':>8s} {'Flag':>6s}")
    print("-" * 56)
    n_unstable = 0
    for cat, stats in agg.items():
        flag = " ***" if stats["unstable"] else ""
        if stats["unstable"]:
            n_unstable += 1
        print(f"{cat:20s} {stats['lure_rate_mean']:>7.1%} "
              f"{stats['lure_rate_std']:>8.3f} "
              f"{stats['lure_rate_std_pp']:>7.1f}pp"
              f"{flag}")

    # Overall aggregated lure rate
    overall_per_seed = []
    for results in all_seed_results:
        n_lured = sum(1 for r in results if r["conflict"] and r["verdict"] == "lure")
        n_conf = sum(1 for r in results if r["conflict"])
        overall_per_seed.append(n_lured / max(n_conf, 1))
    overall_arr = np.array(overall_per_seed)
    overall_std_pp = float(np.std(overall_arr, ddof=1)) * 100 if len(overall_arr) > 1 else 0.0

    print(f"\nOverall lure rate: {np.mean(overall_arr):.1%} +/- {np.std(overall_arr, ddof=1):.3f} "
          f"({overall_std_pp:.1f}pp)")
    print(f"  Per seed: {', '.join(f'{s}={lr:.1%}' for s, lr in zip(seeds, overall_per_seed))}")
    print(f"\nUnstable categories (>{INSTABILITY_THRESHOLD_PP}pp std): {n_unstable}")
    if n_unstable > 0:
        unstable_cats = [cat for cat, st in agg.items() if st["unstable"]]
        print(f"  Flagged: {', '.join(unstable_cats)}")

    # --- Save results ---
    output_data = {
        "model": args.model,
        "seeds": seeds,
        "generation_config": {
            "do_sample": True,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": args.max_new_tokens,
        },
        "n_items": len(items),
        "n_conflict": n_conflict,
        "instability_threshold_pp": INSTABILITY_THRESHOLD_PP,
        "overall": {
            "lure_rate_mean": float(np.mean(overall_arr)),
            "lure_rate_std": float(np.std(overall_arr, ddof=1)) if len(overall_arr) > 1 else 0.0,
            "lure_rate_std_pp": overall_std_pp,
            "lure_rate_per_seed": {str(s): float(lr) for s, lr in zip(seeds, overall_per_seed)},
            "unstable": overall_std_pp > INSTABILITY_THRESHOLD_PP,
        },
        "per_category": agg,
        "seed_timings_s": {str(k): round(v, 1) for k, v in seed_timings.items()},
        "per_seed_results": {
            str(seed): results
            for seed, results in zip(seeds, all_seed_results)
        },
    }

    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    total_time = sum(seed_timings.values())
    print(f"Total wall time: {total_time:.0f}s ({total_time/60:.1f} min)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
