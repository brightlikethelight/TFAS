#!/usr/bin/env python3
"""Quick behavioral validation: run one model on N benchmark items, report lure rate.

Usage:
    python scripts/behavioral_validation.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --n-items 20 \
        --max-new-tokens 2048
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_answer(text: str, correct: str, lure: str) -> str:
    """Classify the model's response."""
    # Check for the correct answer
    if re.search(re.escape(correct), text):
        return "correct"
    if lure and re.search(re.escape(lure), text):
        return "lure"
    return "other"


def split_thinking(text: str) -> tuple[str, str]:
    """Split a DeepSeek R1 response into thinking and answer."""
    if "<think>" in text and "</think>" in text:
        ts = text.index("<think>") + len("<think>")
        te = text.index("</think>")
        return text[ts:te].strip(), text[te + len("</think>"):].strip()
    return "", text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--benchmark", default="data/benchmark/benchmark.jsonl")
    parser.add_argument("--n-items", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--cache-dir", default="/workspace/hf_cache")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir,
        dtype=torch.bfloat16, device_map="cuda",
    )
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Load benchmark items
    items = []
    with open(args.benchmark) as f:
        for line in f:
            items.append(json.loads(line))
    items = items[:args.n_items]
    print(f"Running {len(items)} items...")

    results = []
    t_total = time.time()
    for i, item in enumerate(items):
        messages = [{"role": "user", "content": item["prompt"]}]
        input_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(input_text, return_tensors="pt").to("cuda")

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        elapsed = time.time() - t0
        n_gen = out.shape[1] - inputs.input_ids.shape[1]
        response = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)

        thinking, answer = split_thinking(response)
        verdict = parse_answer(answer, item["correct_answer"], item["lure_answer"])

        results.append({
            "id": item["id"],
            "category": item["category"],
            "conflict": item["conflict"],
            "correct_answer": item["correct_answer"],
            "lure_answer": item["lure_answer"],
            "verdict": verdict,
            "n_tokens": n_gen,
            "thinking_len": len(thinking),
            "elapsed_s": round(elapsed, 1),
        })

        marker = "✓" if verdict == "correct" else ("⚠ LURED" if verdict == "lure" else "✗")
        print(f"  [{i+1}/{len(items)}] {item['id'][:40]:40s} {verdict:8s} {marker}  ({n_gen} tok, {elapsed:.1f}s)")

    # Summary
    elapsed_total = time.time() - t_total
    n_conflict = sum(1 for r in results if r["conflict"])
    n_control = len(results) - n_conflict
    n_correct_conflict = sum(1 for r in results if r["conflict"] and r["verdict"] == "correct")
    n_lured = sum(1 for r in results if r["conflict"] and r["verdict"] == "lure")
    n_correct_control = sum(1 for r in results if not r["conflict"] and r["verdict"] == "correct")

    print(f"\n{'='*60}")
    print(f"MODEL: {args.model}")
    print(f"Items: {len(results)} ({n_conflict} conflict, {n_control} control)")
    print(f"Total time: {elapsed_total:.0f}s")
    print(f"\nConflict items:  {n_correct_conflict}/{n_conflict} correct, {n_lured}/{n_conflict} lured")
    print(f"Control items:   {n_correct_control}/{n_control} correct")
    if n_conflict > 0:
        lure_rate = n_lured / n_conflict
        print(f"\nLURE RATE: {lure_rate:.1%}")
        if lure_rate >= 0.30:
            print(">>> GO: lure rate >= 30% — benchmark is effective for this model")
        else:
            print(">>> CAUTION: lure rate < 30% — model may be too good for these items")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"model": args.model, "results": results}, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
