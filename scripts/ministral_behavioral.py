#!/usr/bin/env python3
"""Behavioral validation for the Ministral-3-8B matched pair.

Runs both Ministral-3-8B-Instruct-2512 and Ministral-3-8B-Reasoning-2512
on the full 330-item benchmark. This is the cleanest matched comparison in
the study: same base model, same tokenizer, cascade distillation, Apache 2.0.

Ministral uses Mistral's chat template ([INST] / [/INST] tokens) via
tokenizer.apply_chat_template. The Reasoning variant emits thinking traces
inside <think>...</think> blocks (same convention as DeepSeek R1).

Usage (on B200 pod):
    # Instruct (standard)
    python scripts/ministral_behavioral.py \
        --model mistralai/Ministral-3-8B-Instruct-2512 \
        --output results/behavioral/ministral_3b_instruct.json \
        --max-new-tokens 512

    # Reasoning
    python scripts/ministral_behavioral.py \
        --model mistralai/Ministral-3-8B-Reasoning-2512 \
        --output results/behavioral/ministral_3b_reasoning.json \
        --max-new-tokens 4096

    # Run both back-to-back
    python scripts/ministral_behavioral.py --run-pair
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# -- Ministral pair specs -------------------------------------------------
PAIR = {
    "instruct": {
        "hf_id": "mistralai/Ministral-3-8B-Instruct-2512",
        "max_new_tokens": 512,
        "output": "results/behavioral/ministral_3b_instruct.json",
    },
    "reasoning": {
        "hf_id": "mistralai/Ministral-3-8B-Reasoning-2512",
        "max_new_tokens": 4096,
        "output": "results/behavioral/ministral_3b_reasoning.json",
    },
}


def parse_answer(text: str, correct: str, lure: str) -> str:
    """Classify the model's response as correct, lure, or other."""
    if correct and re.search(re.escape(correct), text):
        return "correct"
    if lure and re.search(re.escape(lure), text):
        return "lure"
    return "other"


def split_thinking(text: str) -> tuple[str, str]:
    """Split response into thinking trace and final answer.

    Ministral-Reasoning uses <think>...</think> blocks, same as R1-Distill.
    """
    if "<think>" in text and "</think>" in text:
        ts = text.index("<think>") + len("<think>")
        te = text.index("</think>")
        return text[ts:te].strip(), text[te + len("</think>"):].strip()
    return "", text.strip()


def validate_chat_template(tok: AutoTokenizer, model_name: str) -> None:
    """Verify the tokenizer's chat template works and print the format."""
    test_msg = [{"role": "user", "content": "Hello"}]
    try:
        formatted = tok.apply_chat_template(
            test_msg, tokenize=False, add_generation_prompt=True
        )
        print(f"Chat template OK for {model_name}")
        print(f"  Format sample: {formatted[:120]}...")
    except Exception as e:
        print(f"WARNING: apply_chat_template failed for {model_name}: {e}")
        print("  Falling back to manual [INST] formatting")
        raise


def run_model(
    model_id: str,
    benchmark_path: str,
    n_items: Optional[int],
    max_new_tokens: int,
    cache_dir: str,
    output_path: Optional[str],
    hf_token: Optional[str],
) -> dict:
    """Run behavioral validation for a single model."""
    print(f"\n{'='*60}")
    print(f"Loading {model_id}...")
    print(f"{'='*60}")

    tok_kwargs: dict = {"cache_dir": cache_dir}
    model_kwargs: dict = {
        "cache_dir": cache_dir,
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda",
    }
    if hf_token:
        tok_kwargs["token"] = hf_token
        model_kwargs["token"] = hf_token

    tok = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    # Print model config for verification
    cfg = model.config
    print(f"Architecture: {cfg.architectures}")
    print(f"Layers: {cfg.num_hidden_layers}, Hidden: {cfg.hidden_size}")
    print(f"Heads: {cfg.num_attention_heads}, KV heads: {getattr(cfg, 'num_key_value_heads', 'N/A')}")
    print(f"Vocab: {cfg.vocab_size}")
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Validate chat template
    validate_chat_template(tok, model_id)

    is_reasoning = "reason" in model_id.lower()

    # Load benchmark
    items = []
    with open(benchmark_path) as f:
        for line in f:
            items.append(json.loads(line))
    if n_items:
        items = items[:n_items]
    print(f"Running {len(items)} items (max_new_tokens={max_new_tokens})...")

    results = []
    t_total = time.time()
    for i, item in enumerate(items):
        messages = [{"role": "user", "content": item["prompt"]}]
        input_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(input_text, return_tensors="pt").to("cuda")

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        elapsed = time.time() - t0
        n_gen = out.shape[1] - inputs.input_ids.shape[1]
        response = tok.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False
        )

        thinking, answer = split_thinking(response)
        verdict = parse_answer(
            answer, item["correct_answer"], item.get("lure_answer", "")
        )

        results.append({
            "id": item["id"],
            "category": item["category"],
            "conflict": item["conflict"],
            "correct_answer": item["correct_answer"],
            "lure_answer": item.get("lure_answer", ""),
            "verdict": verdict,
            "n_tokens": n_gen,
            "thinking_len": len(thinking),
            "answer_text": answer[:500],
            "elapsed_s": round(elapsed, 1),
        })

        marker = {
            "correct": "OK",
            "lure": "LURED",
            "other": "X",
        }.get(verdict, "?")
        print(
            f"  [{i+1}/{len(items)}] {item['id'][:40]:40s} {verdict:8s} {marker:6s}"
            f"  ({n_gen} tok, {elapsed:.1f}s)"
            + (f"  think={len(thinking)}" if thinking else "")
        )

    elapsed_total = time.time() - t_total

    # --- Summary ---
    n_conflict = sum(1 for r in results if r["conflict"])
    n_control = len(results) - n_conflict
    n_correct_conflict = sum(
        1 for r in results if r["conflict"] and r["verdict"] == "correct"
    )
    n_lured = sum(
        1 for r in results if r["conflict"] and r["verdict"] == "lure"
    )
    n_correct_control = sum(
        1 for r in results if not r["conflict"] and r["verdict"] == "correct"
    )

    # Per-category breakdown
    categories = sorted(set(r["category"] for r in results))
    cat_stats = {}
    for cat in categories:
        cat_items = [r for r in results if r["category"] == cat]
        cat_conflict = [r for r in cat_items if r["conflict"]]
        cat_lured = sum(1 for r in cat_conflict if r["verdict"] == "lure")
        cat_stats[cat] = {
            "n": len(cat_items),
            "n_conflict": len(cat_conflict),
            "n_lured": cat_lured,
            "lure_rate": cat_lured / max(len(cat_conflict), 1),
        }

    # Reasoning-model specific stats
    thinking_stats = None
    if is_reasoning:
        think_lens = [r["thinking_len"] for r in results if r["thinking_len"] > 0]
        if think_lens:
            thinking_stats = {
                "n_with_thinking": len(think_lens),
                "mean_len": sum(think_lens) / len(think_lens),
                "max_len": max(think_lens),
                "min_len": min(think_lens),
            }

    print(f"\n{'='*60}")
    print(f"MODEL: {model_id}")
    print(f"Items: {len(results)} ({n_conflict} conflict, {n_control} control)")
    print(f"Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"\nConflict:  {n_correct_conflict}/{n_conflict} correct, "
          f"{n_lured}/{n_conflict} lured")
    print(f"Control:   {n_correct_control}/{n_control} correct")
    if n_conflict > 0:
        lure_rate = n_lured / n_conflict
        print(f"\nOVERALL LURE RATE: {lure_rate:.1%}")
        if lure_rate >= 0.30:
            print(">>> GO: lure rate >= 30%")
        else:
            print(">>> CAUTION: lure rate < 30%")

    print(f"\nPer-category:")
    for cat, st in cat_stats.items():
        print(f"  {cat:16s}  {st['n_lured']}/{st['n_conflict']} lured "
              f"({st['lure_rate']:.0%})")

    if thinking_stats:
        print(f"\nThinking traces: {thinking_stats['n_with_thinking']}/{len(results)} "
              f"items, mean={thinking_stats['mean_len']:.0f} chars, "
              f"max={thinking_stats['max_len']}")

    summary = {
        "model": model_id,
        "is_reasoning": is_reasoning,
        "n_items": len(results),
        "n_conflict": n_conflict,
        "n_control": n_control,
        "lure_rate": n_lured / max(n_conflict, 1),
        "correct_conflict": n_correct_conflict,
        "correct_control": n_correct_control,
        "n_lured": n_lured,
        "elapsed_s": round(elapsed_total, 1),
        "per_category": cat_stats,
        "thinking_stats": thinking_stats,
        "results": results,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved to {output_path}")

    # Free GPU memory before loading next model
    del model
    del tok
    torch.cuda.empty_cache()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavioral validation for Ministral-3-8B pair"
    )
    parser.add_argument(
        "--model", default=None,
        help="Single model HF ID (or use --run-pair for both)"
    )
    parser.add_argument("--benchmark", default="data/benchmark/benchmark.jsonl")
    parser.add_argument("--n-items", type=int, default=None, help="Limit items (None=all 330)")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--cache-dir", default="/workspace/hf_cache")
    parser.add_argument("--output", default=None)
    parser.add_argument("--hf-token", default=None, help="HF token (or set HF_TOKEN env var)")
    parser.add_argument(
        "--run-pair", action="store_true",
        help="Run both Instruct and Reasoning back-to-back"
    )
    args = parser.parse_args()

    import os
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    if args.run_pair:
        print("Running Ministral-3-8B matched pair (Instruct + Reasoning)")
        print(f"Benchmark: {args.benchmark}")
        summaries = {}
        for variant, spec in PAIR.items():
            summaries[variant] = run_model(
                model_id=spec["hf_id"],
                benchmark_path=args.benchmark,
                n_items=args.n_items,
                max_new_tokens=args.max_new_tokens or spec["max_new_tokens"],
                cache_dir=args.cache_dir,
                output_path=spec["output"],
                hf_token=hf_token,
            )

        # Comparison summary
        print(f"\n{'='*60}")
        print("MINISTRAL-3-8B PAIR COMPARISON")
        print(f"{'='*60}")
        for variant, s in summaries.items():
            print(f"  {variant:12s}  lure={s['lure_rate']:.1%}  "
                  f"correct_conflict={s['correct_conflict']}/{s['n_conflict']}  "
                  f"correct_control={s['correct_control']}/{s['n_control']}  "
                  f"time={s['elapsed_s']:.0f}s")

        # Per-category delta
        print(f"\nPer-category lure rate delta (Instruct - Reasoning):")
        cats_i = summaries["instruct"]["per_category"]
        cats_r = summaries["reasoning"]["per_category"]
        for cat in sorted(cats_i.keys()):
            lr_i = cats_i[cat]["lure_rate"]
            lr_r = cats_r.get(cat, {}).get("lure_rate", 0)
            delta = lr_i - lr_r
            print(f"  {cat:16s}  Instruct={lr_i:.0%}  Reasoning={lr_r:.0%}  "
                  f"delta={delta:+.0%}")

    elif args.model:
        is_reasoning = "reason" in args.model.lower()
        default_tokens = 4096 if is_reasoning else 512
        run_model(
            model_id=args.model,
            benchmark_path=args.benchmark,
            n_items=args.n_items,
            max_new_tokens=args.max_new_tokens or default_tokens,
            cache_dir=args.cache_dir,
            output_path=args.output,
            hf_token=hf_token,
        )
    else:
        parser.error("Provide --model or --run-pair")


if __name__ == "__main__":
    main()
