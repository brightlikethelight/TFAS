#!/usr/bin/env python3
"""Extract per-head attention entropy metrics at P0 and run statistical analysis.

This script extracts attention-entropy metrics from the S1/S2 benchmark and
performs the full statistical analysis pipeline:

1. Forward pass with ``attn_implementation="eager"`` to get materialized
   attention matrices (Flash Attention fuses the softmax and never returns them).
2. Per-head metric computation at P0 (last prompt token): Shannon entropy,
   normalized entropy, Gini coefficient, max attention, top-5 focus, effective rank.
3. KV-group aggregation (mean over query heads sharing a KV projection).
4. Mann-Whitney U tests (conflict vs control) at both query-head and KV-group
   granularity, with BH-FDR correction across all layer x head tests.
5. Identification of "S2-specialized" heads: q < 0.05 AND |r_rb| >= 0.3.

Memory note:
    For P0-only extraction on prompts of ~50-200 tokens, the full attention
    tensor is ~n_layers x n_heads x seq_len x seq_len x 4 bytes -- well under
    100 MB.  Process one item at a time, compute scalar metrics, discard the
    attention matrices immediately.

GQA note:
    ``output_attentions=True`` returns attention expanded to per-*query*-head
    shape (batch, n_query_heads, seq, seq).  Heads in the same KV group share
    key/value projections and are NOT statistically independent.  We report at
    both per-query-head and per-KV-group granularity (CLAUDE.md requirement).

Usage:
    python scripts/extract_attention.py \\
        --model unsloth/Meta-Llama-3.1-8B-Instruct \\
        --benchmark data/benchmark/benchmark.jsonl \\
        --output results/attention/llama_entropy.json \\
        --cache-dir /workspace/hf_cache

    python scripts/extract_attention.py \\
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
        --benchmark data/benchmark/benchmark.jsonl \\
        --output results/attention/r1_distill_entropy.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
# Inlined so the script is self-contained and runnable without ``pip install -e .``
# Definitions MUST match ``s1s2.utils.stats.{shannon_entropy_bits, gini_coefficient}``.


def _shannon_entropy_bits(probs: np.ndarray) -> float:
    """Shannon entropy in bits for a 1-D probability distribution.

    Uses 1e-12 floor to avoid log(0). Matches the convention in
    s1s2.utils.stats.shannon_entropy_bits.
    """
    safe = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(safe * np.log2(safe)))


def _gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient in [0, 1]. 0 = uniform, 1 = maximally concentrated.

    Standard formula: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
    where x is sorted ascending and i is 1-indexed rank.
    """
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    if cumsum[-1] == 0:
        return 0.0
    return float(
        (2 * np.sum(np.arange(1, n + 1) * sorted_vals)) / (n * cumsum[-1])
        - (n + 1) / n
    )


def compute_head_metrics(attn_row: np.ndarray, seq_len: int) -> dict[str, float]:
    """Compute all attention metrics from a single 1-D attention distribution.

    Parameters
    ----------
    attn_row : shape (seq_len,), the attention weights from one head at one
        query position (already softmax-normalized by the model).
    seq_len : int, the sequence length (for normalized entropy computation).

    Returns
    -------
    dict with keys: entropy, norm_entropy, gini, max_attn, focus_5, effective_rank
    """
    # Defensive: re-normalize in case of numerical drift from bfloat16
    probs = np.clip(attn_row.astype(np.float64), 0.0, None)
    s = probs.sum()
    if s <= 0 or probs.size == 0:
        return {
            "entropy": 0.0,
            "norm_entropy": 0.0,
            "gini": 0.0,
            "max_attn": 0.0,
            "focus_5": 0.0,
            "effective_rank": 1.0,
        }
    probs = probs / s

    entropy_bits = _shannon_entropy_bits(probs)

    # Normalized entropy: controls for sequence length (CLAUDE.md requirement)
    max_entropy = math.log2(seq_len) if seq_len > 1 else 1.0
    norm_entropy = entropy_bits / max_entropy

    gini = _gini_coefficient(probs)
    max_attn = float(np.max(probs))

    k = min(5, probs.size)
    top_k = np.partition(probs, -k)[-k:]
    focus_5 = float(np.sum(top_k))

    effective_rank = float(2.0**entropy_bits)

    return {
        "entropy": entropy_bits,
        "norm_entropy": norm_entropy,
        "gini": gini,
        "max_attn": max_attn,
        "focus_5": focus_5,
        "effective_rank": effective_rank,
    }


METRIC_KEYS: list[str] = [
    "entropy", "norm_entropy", "gini", "max_attn", "focus_5", "effective_rank",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_eager(
    model_id: str, cache_dir: str, device: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model with eager attention -- no Flash Attention.

    ``attn_implementation="eager"`` forces the standard scaled dot-product
    path that materializes and returns the full attention matrix.
    """
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    print(f"Loading model with attn_implementation='eager': {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",  # CRITICAL: need materialized attention
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# KV-group aggregation
# ---------------------------------------------------------------------------


def aggregate_kv_groups(
    per_head_values: list[list[float]],
    n_kv_heads: int,
    n_query_heads: int,
) -> list[list[float]]:
    """Average per-query-head values within each KV group.

    Parameters
    ----------
    per_head_values : list[list[float]], shape (n_layers, n_query_heads)
    n_kv_heads : int, number of KV heads
    n_query_heads : int, number of query heads

    Returns
    -------
    list[list[float]], shape (n_layers, n_kv_heads)
    """
    group_size = n_query_heads // n_kv_heads
    result: list[list[float]] = []
    for layer_vals in per_head_values:
        grouped: list[float] = []
        for kv_idx in range(n_kv_heads):
            start = kv_idx * group_size
            end = start + group_size
            group_mean = float(np.mean(layer_vals[start:end]))
            grouped.append(round(group_mean, 6))
        result.append(grouped)
    return result


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------


def extract_attention_metrics(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    items: list[dict],
    device: str = "cuda",
) -> tuple[int, int, int, list[dict]]:
    """Run prompt-only forward pass and extract P0 attention metrics.

    Returns
    -------
    n_layers : int
    n_heads : int (query heads -- expanded from GQA)
    n_kv_heads : int
    results : list of per-item dicts with metric arrays at both granularities
    """
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    group_size = n_heads // n_kv_heads

    print(
        f"Architecture: {n_layers} layers, {n_heads} query heads, "
        f"{n_kv_heads} KV heads (GQA group size = {group_size})"
    )

    results: list[dict] = []
    t_start = time.time()

    for i, item in enumerate(items):
        # Tokenize prompt only (no generation)
        messages = [{"role": "user", "content": item["prompt"]}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]

        # Forward pass -- prompt only, no generation
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )

        # outputs.attentions: tuple of (batch, n_query_heads, seq_len, seq_len)
        # one tensor per layer. HF expands GQA to query-head granularity.
        attentions = outputs.attentions
        if len(attentions) != n_layers:
            raise RuntimeError(
                f"Expected {n_layers} attention tensors, got {len(attentions)}"
            )
        if attentions[0].shape[1] != n_heads:
            raise RuntimeError(
                f"Expected {n_heads} heads in attention output, got "
                f"{attentions[0].shape[1]}. GQA expansion may not be working."
            )

        # Compute per-query-head metrics at P0 (last prompt token)
        per_layer_metrics: dict[str, list[list[float]]] = {
            k: [] for k in METRIC_KEYS
        }

        for layer_idx in range(n_layers):
            # (1, n_heads, seq_len, seq_len) -> (n_heads, seq_len)
            attn_at_last = (
                attentions[layer_idx][0, :, -1, :].float().cpu().numpy()
            )

            layer_metrics: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
            for head_idx in range(n_heads):
                row = attn_at_last[head_idx]  # (seq_len,)
                m = compute_head_metrics(row, seq_len)
                for k in METRIC_KEYS:
                    layer_metrics[k].append(round(m[k], 6))

            for k in METRIC_KEYS:
                per_layer_metrics[k].append(layer_metrics[k])

        # KV-group aggregation: average across query heads in each group
        kv_group_metrics: dict[str, list[list[float]]] = {}
        for k in METRIC_KEYS:
            kv_group_metrics[k] = aggregate_kv_groups(
                per_layer_metrics[k], n_kv_heads, n_heads,
            )

        results.append({
            "id": item["id"],
            "category": item["category"],
            "conflict": item["conflict"],
            "seq_len": seq_len,
            "metrics": per_layer_metrics,         # (n_layers, n_query_heads)
            "kv_group_metrics": kv_group_metrics,  # (n_layers, n_kv_heads)
        })

        # Free GPU memory from attention tensors immediately
        del outputs, attentions
        if device == "cuda":
            torch.cuda.empty_cache()

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(items) - i - 1) / rate
            mem_gb = (
                torch.cuda.memory_allocated() / 1e9
                if torch.cuda.is_available()
                else 0
            )
            print(
                f"  [{i+1}/{len(items)}] seq_len={seq_len:4d}  "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                f"GPU mem: {mem_gb:.1f} GB)"
            )

    return n_layers, n_heads, n_kv_heads, results


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------


def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from Mann-Whitney U.

    r_rb = 1 - 2U / (n1 * n2), where U is the smaller of the two U values.
    We use the formula: r_rb = 2U / (n1 * n2) - 1, where U is the returned
    statistic (for group 1 > group 2).
    """
    denom = n1 * n2
    if denom == 0:
        return 0.0
    return float(2 * u_stat / denom - 1)


def run_statistical_analysis(
    items: list[dict],
    n_layers: int,
    n_query_heads: int,
    n_kv_heads: int,
) -> dict:
    """Mann-Whitney U tests (conflict vs control) with BH-FDR correction.

    Tests are run per-head for each metric, at both query-head and KV-group
    granularity. BH-FDR correction is applied across all layer x head tests
    within each (metric, granularity) combination.

    Identifies "S2-specialized" heads: significantly higher entropy on conflict
    items (q < 0.05, |r_rb| >= 0.3).
    """
    # Partition items by category and conflict status
    categories = sorted(set(it["category"] for it in items))

    # Build arrays: for each (category, layer, head) -> list of metric values
    # Separate by conflict vs control
    analysis_results: dict = {
        "categories": categories,
        "query_head_tests": {},
        "kv_group_tests": {},
        "s2_specialized_heads": {},
    }

    for granularity, metric_key_prefix, n_heads_g in [
        ("query_head", "metrics", n_query_heads),
        ("kv_group", "kv_group_metrics", n_kv_heads),
    ]:
        test_key = f"{granularity}_tests"

        for metric_name in ["entropy", "norm_entropy", "gini"]:
            # Collect all test results for BH-FDR across layer x head
            all_pvals: list[float] = []
            all_r_rbs: list[float] = []
            all_labels: list[dict] = []  # (layer, head) for each test

            # Also run per-category (for diagnostics, not primary correction)
            per_category_results: dict[str, list[dict]] = {}

            # Global test: all conflict vs all control (primary)
            for layer_idx in range(n_layers):
                for head_idx in range(n_heads_g):
                    conflict_vals: list[float] = []
                    control_vals: list[float] = []

                    for it in items:
                        val = it[metric_key_prefix][metric_name][layer_idx][head_idx]
                        if it["conflict"]:
                            conflict_vals.append(val)
                        else:
                            control_vals.append(val)

                    if len(conflict_vals) < 3 or len(control_vals) < 3:
                        # Not enough data for a meaningful test
                        all_pvals.append(1.0)
                        all_r_rbs.append(0.0)
                    else:
                        u_stat, p_val = sp_stats.mannwhitneyu(
                            conflict_vals,
                            control_vals,
                            alternative="two-sided",
                        )
                        r_rb = _rank_biserial(
                            u_stat, len(conflict_vals), len(control_vals),
                        )
                        all_pvals.append(float(p_val))
                        all_r_rbs.append(r_rb)

                    all_labels.append({
                        "layer": layer_idx,
                        "head": head_idx,
                    })

            # BH-FDR correction across all layer x head tests
            n_tests = len(all_pvals)
            if n_tests > 0 and any(p < 1.0 for p in all_pvals):
                reject, q_vals, _, _ = multipletests(
                    all_pvals, alpha=0.05, method="fdr_bh",
                )
                q_vals = q_vals.tolist()
                reject = reject.tolist()
            else:
                q_vals = [1.0] * n_tests
                reject = [False] * n_tests

            # Package results
            test_results: list[dict] = []
            for idx in range(n_tests):
                test_results.append({
                    "layer": all_labels[idx]["layer"],
                    "head": all_labels[idx]["head"],
                    "p_value": round(all_pvals[idx], 8),
                    "q_value": round(q_vals[idx], 8),
                    "r_rb": round(all_r_rbs[idx], 6),
                    "significant": reject[idx],
                })

            if test_key not in analysis_results:
                analysis_results[test_key] = {}
            analysis_results[test_key][metric_name] = {
                "n_tests": n_tests,
                "n_significant": sum(reject),
                "fdr_method": "BH",
                "alpha": 0.05,
                "tests": test_results,
            }

            # Per-category breakdown (informational -- no separate correction)
            cat_summaries: list[dict] = []
            for cat in categories:
                cat_items = [it for it in items if it["category"] == cat]
                cat_conflict = [it for it in cat_items if it["conflict"]]
                cat_control = [it for it in cat_items if not it["conflict"]]

                if len(cat_conflict) < 3 or len(cat_control) < 3:
                    cat_summaries.append({
                        "category": cat,
                        "n_conflict": len(cat_conflict),
                        "n_control": len(cat_control),
                        "note": "insufficient_n",
                    })
                    continue

                # Aggregate: mean across layers and heads per item
                conflict_means = []
                control_means = []
                for it in cat_conflict:
                    vals = [
                        it[metric_key_prefix][metric_name][l][h]
                        for l in range(n_layers)
                        for h in range(n_heads_g)
                    ]
                    conflict_means.append(float(np.mean(vals)))
                for it in cat_control:
                    vals = [
                        it[metric_key_prefix][metric_name][l][h]
                        for l in range(n_layers)
                        for h in range(n_heads_g)
                    ]
                    control_means.append(float(np.mean(vals)))

                u, p = sp_stats.mannwhitneyu(
                    conflict_means, control_means, alternative="two-sided",
                )
                r = _rank_biserial(u, len(conflict_means), len(control_means))

                cat_summaries.append({
                    "category": cat,
                    "n_conflict": len(cat_conflict),
                    "n_control": len(cat_control),
                    "mean_conflict": round(float(np.mean(conflict_means)), 6),
                    "mean_control": round(float(np.mean(control_means)), 6),
                    "U": float(u),
                    "p_value": round(float(p), 8),
                    "r_rb": round(r, 6),
                })

            analysis_results[test_key][metric_name]["per_category"] = cat_summaries

    # Identify S2-specialized heads (entropy metric, both granularities)
    for granularity, n_heads_g in [
        ("query_head", n_query_heads),
        ("kv_group", n_kv_heads),
    ]:
        test_key = f"{granularity}_tests"
        entropy_tests = analysis_results[test_key].get("entropy", {}).get("tests", [])

        specialized: list[dict] = []
        for t in entropy_tests:
            # Higher entropy on conflict = positive r_rb (conflict > control)
            if t["q_value"] < 0.05 and abs(t["r_rb"]) >= 0.3:
                specialized.append({
                    "layer": t["layer"],
                    "head": t["head"],
                    "q_value": t["q_value"],
                    "r_rb": t["r_rb"],
                    "direction": "conflict_higher" if t["r_rb"] > 0 else "control_higher",
                })

        total_heads = n_layers * n_heads_g
        analysis_results["s2_specialized_heads"][granularity] = {
            "criteria": "q < 0.05 AND |r_rb| >= 0.3 (entropy metric)",
            "n_specialized": len(specialized),
            "n_total": total_heads,
            "proportion": round(len(specialized) / total_heads, 6) if total_heads > 0 else 0,
            "heads": specialized,
        }

    return analysis_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-head P0 attention metrics and run statistical analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", required=True, help="HuggingFace model ID",
    )
    parser.add_argument(
        "--benchmark",
        default="data/benchmark/benchmark.jsonl",
        help="Path to benchmark JSONL (default: data/benchmark/benchmark.jsonl)",
    )
    parser.add_argument(
        "--output", required=True, help="Output JSON path",
    )
    parser.add_argument(
        "--cache-dir",
        default="/workspace/hf_cache",
        help="HuggingFace cache directory (default: /workspace/hf_cache)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device map for model loading (default: auto)",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=None,
        help="Limit number of benchmark items (None = all)",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip statistical analysis (extraction only)",
    )
    args = parser.parse_args()

    # Validate benchmark exists
    bench_path = Path(args.benchmark)
    if not bench_path.exists():
        print(f"ERROR: benchmark not found at {bench_path}", file=sys.stderr)
        sys.exit(1)

    # Load benchmark
    items: list[dict] = []
    with open(bench_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if args.n_items is not None:
        items = items[: args.n_items]
    n_conflict = sum(1 for it in items if it["conflict"])
    n_control = len(items) - n_conflict
    print(f"Benchmark: {len(items)} items ({n_conflict} conflict, {n_control} control)")

    # Load model
    device = args.device
    fwd_device = "cuda" if (device == "auto" and torch.cuda.is_available()) else "cpu"
    model, tokenizer = load_model_eager(args.model, args.cache_dir, device=device)

    if torch.cuda.is_available():
        print(f"GPU memory after model load: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Extract metrics
    t0 = time.time()
    n_layers, n_heads, n_kv_heads, results = extract_attention_metrics(
        model, tokenizer, items, device=fwd_device,
    )
    extract_elapsed = time.time() - t0
    print(f"\nExtraction complete: {extract_elapsed:.0f}s ({extract_elapsed / 60:.1f} min)")

    # Free model from GPU before analysis
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run statistical analysis
    analysis = None
    if not args.skip_analysis:
        print("\nRunning statistical analysis...")
        t1 = time.time()
        analysis = run_statistical_analysis(results, n_layers, n_heads, n_kv_heads)
        analysis_elapsed = time.time() - t1
        print(f"Analysis complete: {analysis_elapsed:.1f}s")

        # Print summary
        for granularity in ["query_head", "kv_group"]:
            spec = analysis["s2_specialized_heads"][granularity]
            print(
                f"  {granularity}: {spec['n_specialized']}/{spec['n_total']} "
                f"S2-specialized heads ({spec['proportion']:.1%})"
            )
            for metric_name in ["entropy", "norm_entropy", "gini"]:
                test_info = analysis[f"{granularity}_tests"][metric_name]
                print(
                    f"    {metric_name}: {test_info['n_significant']}/{test_info['n_tests']} "
                    f"significant after BH-FDR"
                )

    # Assemble output
    total_elapsed = time.time() - t0
    output: dict = {
        "model": args.model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "gqa_group_size": n_heads // n_kv_heads,
        "metric_keys": METRIC_KEYS,
        "extraction_config": {
            "position": "P0",
            "attn_implementation": "eager",
            "torch_dtype": "bfloat16",
            "benchmark": str(bench_path),
            "n_items": len(items),
            "n_conflict": n_conflict,
            "n_control": n_control,
        },
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "extraction_elapsed_s": round(extract_elapsed, 1),
        "total_elapsed_s": round(total_elapsed, 1),
        "items": results,
    }
    if analysis is not None:
        output["analysis"] = analysis

    # Write JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    file_size_kb = out_path.stat().st_size / 1024

    print(f"\n{'=' * 60}")
    print(f"MODEL: {args.model}")
    print(f"Architecture: {n_layers} layers x {n_heads} heads ({n_kv_heads} KV heads)")
    print(f"Items: {len(items)} ({n_conflict} conflict, {n_control} control)")
    print(f"Metrics per head: {METRIC_KEYS}")
    print(f"Output: {out_path} ({file_size_kb:.0f} KB)")
    print(f"Time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
