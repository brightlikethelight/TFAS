#!/usr/bin/env python3
"""Extract per-head attention metrics at P0 (last prompt token) for the full benchmark.

This script complements ``extract_real.py`` (residual streams) by extracting
attention *metrics* -- not raw weights -- for the attention-entropy workstream.

Why a separate script?
    Flash Attention (HuggingFace default) fuses the softmax into the kernel and
    never materializes the attention matrix.  We need ``attn_implementation="eager"``
    to get the raw (n_heads, seq_len, seq_len) softmax output.  That flag changes
    the model's computational graph, so it's cleaner to keep extraction separate.

Memory note (from CLAUDE.md):
    For P0-only extraction on prompts of ~50-200 tokens, the full attention
    tensor is ~n_layers x n_heads x seq_len x seq_len x 4 bytes -- well under
    100 MB.  We materialize it in one shot and compute all metrics from it.
    For generation-time extraction on long sequences, you MUST compute metrics
    incrementally per step (see ``s1s2.extract.hooks``).  This script only does
    P0, so materialization is safe.

GQA note (from CLAUDE.md):
    ``output_attentions=True`` returns attention expanded to per-*query*-head
    shape (batch, n_query_heads, seq, seq), NOT (batch, n_kv_heads, seq, seq).
    So for Llama-3.1-8B (32 query heads, 8 KV heads), you get 32 heads per
    layer, but heads in the same KV group share key/value projections and are
    NOT statistically independent.  Downstream analysis (``core.py``) handles
    this by reporting at both per-query-head and per-KV-group granularity.

Gemma-2 note:
    Odd layers use 4096-token sliding-window attention.  For short prompts
    (< 4096 tokens) this doesn't matter -- the window covers the whole
    sequence.  But downstream analysis must still separate odd/even layers
    (handled in ``core.py``).

Usage:
    python scripts/extract_attention.py \
        --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --output results/attention/llama31_attention.json

    python scripts/extract_attention.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --output results/attention/r1_distill_attention.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
# We inline the metric functions here rather than importing from s1s2.utils.stats
# so the script is self-contained and runnable without ``pip install -e .``.
# The definitions MUST match ``s1s2.attention.core.compute_metrics_from_attention_pattern``
# and ``s1s2.utils.stats.{shannon_entropy_bits, gini_coefficient}``.


def _shannon_entropy_bits(probs: np.ndarray) -> float:
    """Shannon entropy in bits for a 1-D probability distribution."""
    safe = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(safe * np.log2(safe)))


def _gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient in [0, 1]. 0 = uniform, 1 = maximally concentrated."""
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


def compute_head_metrics(attn_row: np.ndarray) -> dict[str, float]:
    """Compute all attention metrics from a single 1-D attention distribution.

    Parameters
    ----------
    attn_row : shape (seq_len,), the attention weights from one head at one
        query position (already softmax-normalized by the model).

    Returns
    -------
    dict with keys: entropy, gini, max_attn, focus_5, effective_rank
    """
    # Defensive: re-normalize in case of numerical drift
    probs = np.clip(attn_row.astype(np.float64), 0.0, None)
    s = probs.sum()
    if s <= 0 or probs.size == 0:
        return {
            "entropy": 0.0,
            "gini": 0.0,
            "max_attn": 0.0,
            "focus_5": 0.0,
            "effective_rank": 1.0,
        }
    probs = probs / s

    entropy_bits = _shannon_entropy_bits(probs)
    gini = _gini_coefficient(probs)
    max_attn = float(np.max(probs))

    k = min(5, probs.size)
    top_k = np.partition(probs, -k)[-k:]
    focus_5 = float(np.sum(top_k))

    effective_rank = float(2.0 ** entropy_bits)

    return {
        "entropy": entropy_bits,
        "gini": gini,
        "max_attn": max_attn,
        "focus_5": focus_5,
        "effective_rank": effective_rank,
    }


METRIC_KEYS: list[str] = ["entropy", "gini", "max_attn", "focus_5", "effective_rank"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_eager(
    model_id: str, cache_dir: str, device: str = "auto"
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model with eager attention (no Flash Attention).

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
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------


def extract_attention_metrics(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    items: list[dict],
    device: str = "cuda",
) -> tuple[int, int, list[dict]]:
    """Run prompt-only forward pass and extract P0 attention metrics.

    Returns
    -------
    n_layers : int
    n_heads : int (query heads -- expanded from GQA)
    results : list of per-item dicts with metric arrays
    """
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    print(
        f"Architecture: {n_layers} layers, {n_heads} query heads, "
        f"{n_kv_heads} KV heads (GQA group size = {n_heads // n_kv_heads})"
    )

    results: list[dict] = []
    t_start = time.time()

    for i, item in enumerate(items):
        # Tokenize prompt only (no generation)
        messages = [{"role": "user", "content": item["prompt"]}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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

        # outputs.attentions is a tuple of (batch, n_query_heads, seq_len, seq_len)
        # one tensor per layer.  HF expands GQA to query-head granularity.
        attentions = outputs.attentions
        if len(attentions) != n_layers:
            raise RuntimeError(
                f"Expected {n_layers} attention tensors, got {len(attentions)}"
            )

        # Verify shape of first layer
        attn_shape = attentions[0].shape
        if attn_shape[1] != n_heads:
            raise RuntimeError(
                f"Expected {n_heads} heads in attention output, got {attn_shape[1]}. "
                f"GQA expansion may not be working as expected."
            )

        # Extract metrics: attention[layer][head][last_pos, :]
        per_layer_metrics: dict[str, list[list[float]]] = {
            k: [] for k in METRIC_KEYS
        }

        for layer_idx in range(n_layers):
            # (batch=1, n_heads, seq_len, seq_len) -> (n_heads, seq_len)
            # Index the last query position (P0 = last prompt token)
            attn_at_last = attentions[layer_idx][0, :, -1, :].float().cpu().numpy()

            layer_metrics: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
            for head_idx in range(n_heads):
                row = attn_at_last[head_idx]  # (seq_len,)
                m = compute_head_metrics(row)
                for k in METRIC_KEYS:
                    layer_metrics[k].append(round(m[k], 6))

            for k in METRIC_KEYS:
                per_layer_metrics[k].append(layer_metrics[k])

        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "conflict": item["conflict"],
                "seq_len": seq_len,
                "metrics": per_layer_metrics,
            }
        )

        # Free GPU memory from attention tensors
        del outputs, attentions
        if device == "cuda":
            torch.cuda.empty_cache()

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(items) - i - 1) / rate
            mem_gb = (
                torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            )
            print(
                f"  [{i+1}/{len(items)}] seq_len={seq_len:4d}  "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                f"GPU mem: {mem_gb:.1f} GB)"
            )

    return n_layers, n_heads, results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-head P0 attention metrics for the S1/S2 benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", required=True, help="HuggingFace model ID"
    )
    parser.add_argument(
        "--benchmark",
        default="data/benchmark/benchmark.jsonl",
        help="Path to benchmark JSONL (default: data/benchmark/benchmark.jsonl)",
    )
    parser.add_argument(
        "--output", required=True, help="Output JSON path"
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
    print(f"Benchmark: {len(items)} items from {bench_path}")

    # Load model
    device = args.device
    # Resolve "auto" to "cuda" for the forward-pass device argument
    fwd_device = "cuda" if (device == "auto" and torch.cuda.is_available()) else "cpu"
    model, tokenizer = load_model_eager(args.model, args.cache_dir, device=device)

    # Report VRAM after loading
    if torch.cuda.is_available():
        print(f"GPU memory after model load: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Extract
    t0 = time.time()
    n_layers, n_heads, results = extract_attention_metrics(
        model, tokenizer, items, device=fwd_device
    )
    elapsed = time.time() - t0

    # Assemble output
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    output = {
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
        },
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(elapsed, 1),
        "items": results,
    }

    # Write JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    file_size_kb = out_path.stat().st_size / 1024
    n_conflict = sum(1 for it in items if it["conflict"])
    n_control = len(items) - n_conflict

    print(f"\n{'=' * 60}")
    print(f"MODEL: {args.model}")
    print(f"Architecture: {n_layers} layers x {n_heads} heads ({n_kv_heads} KV heads)")
    print(f"Items: {len(items)} ({n_conflict} conflict, {n_control} control)")
    print(f"Metrics per head: {METRIC_KEYS}")
    print(f"Output: {out_path} ({file_size_kb:.0f} KB)")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
