#!/usr/bin/env python3
"""Run behavioral validation on the new benchmark items (sunk cost + natural frequency).

Deploy to B200 pod with the updated benchmark.jsonl:
    scp data/benchmark/benchmark.jsonl root@<pod>:/workspace/s1s2/data/benchmark/
    scp scripts/run_new_items.py root@<pod>:/workspace/s1s2/scripts/
    ssh root@<pod> "cd /workspace/s1s2 && python scripts/run_new_items.py"

Tests two things:
1. Sunk cost fallacy: Is this a new vulnerable category (loss aversion family)?
2. Natural frequency framing: Does Gigerenzer's critique eliminate base rate vulnerability?
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ["HF_HOME"] = "/workspace/hf_cache"

BENCHMARK = "data/benchmark/benchmark.jsonl"
LOG = "/workspace/new_items_log.txt"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def parse_answer(text: str, correct: str, lure: str) -> str:
    """Classify the model's response."""
    text_lower = text.lower().strip()
    correct_lower = correct.lower().strip()
    lure_lower = lure.lower().strip() if lure else ""

    # Check for exact matches first, then broader
    if correct_lower in text_lower:
        return "correct"
    if lure_lower and lure_lower in text_lower:
        return "lure"
    return "other"


def split_thinking(text: str) -> tuple[str, str]:
    """Split thinking trace from answer for reasoning models."""
    if "<think>" in text and "</think>" in text:
        ts = text.index("<think>") + len("<think>")
        te = text.index("</think>")
        return text[ts:te].strip(), text[te + len("</think>"):].strip()
    return "", text.strip()


def run_model(
    model_id: str,
    items: list[dict],
    max_new_tokens: int = 2048,
    label: str = "",
) -> list[dict]:
    """Run a model on a list of benchmark items."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"  Loading {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir="/workspace/hf_cache")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir="/workspace/hf_cache",
        dtype=torch.bfloat16, device_map="cuda",
    )
    log(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    results = []
    for i, item in enumerate(items):
        messages = [{"role": "user", "content": item["prompt"]}]
        if item.get("system_prompt"):
            messages.insert(0, {"role": "system", "content": item["system_prompt"]})

        enc = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        input_ids = (enc.input_ids if hasattr(enc, "input_ids") else enc).to("cuda")
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        response = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        thinking, answer = split_thinking(response)

        verdict = parse_answer(
            answer if answer else response,
            item["correct_answer"],
            item.get("lure_answer", ""),
        )
        results.append({
            "id": item["id"],
            "category": item["category"],
            "subcategory": item.get("subcategory", ""),
            "conflict": item["conflict"],
            "verdict": verdict,
            "response": response[:500],
            "correct_answer": item["correct_answer"],
            "lure_answer": item.get("lure_answer", ""),
        })
        if (i + 1) % 5 == 0:
            log(f"    {i+1}/{len(items)} done")

    del model, tok
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()
    return results


def analyze_results(results: list[dict], label: str) -> None:
    """Print per-category and per-subcategory breakdown."""
    cats = defaultdict(lambda: {"n": 0, "lure": 0, "correct": 0, "other": 0})
    for r in results:
        if r["conflict"]:
            key = f"{r['category']}/{r['subcategory']}" if r.get("subcategory") else r["category"]
            cats[key]["n"] += 1
            cats[key][r["verdict"]] += 1

    log(f"\n  Results for {label}:")
    log(f"  {'Category':<35} {'Lure':>8} {'Correct':>8} {'Other':>8} {'Rate':>8}")
    log(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    total_n, total_lure = 0, 0
    for cat in sorted(cats):
        c = cats[cat]
        rate = c["lure"] / max(c["n"], 1)
        total_n += c["n"]
        total_lure += c["lure"]
        log(f"  {cat:<35} {c['lure']:>8} {c['correct']:>8} {c['other']:>8} {rate:>7.0%}")
    if total_n > 0:
        log(f"  {'TOTAL':<35} {total_lure:>8} {'':>8} {'':>8} {total_lure/total_n:>7.0%}")


def main() -> None:
    log("=" * 70)
    log("NEW BENCHMARK ITEMS: SUNK COST + NATURAL FREQUENCY VALIDATION")
    log("=" * 70)

    # Load items, filter to new categories/subcategories
    with open(BENCHMARK) as f:
        all_items = [json.loads(line) for line in f if line.strip()]

    sunk_cost = [it for it in all_items if it["category"] == "sunk_cost"]
    nat_freq = [it for it in all_items
                if it["category"] == "base_rate" and it.get("subcategory") == "natural_frequency"]
    new_items = sunk_cost + nat_freq

    log(f"Total new items: {len(new_items)} ({len(sunk_cost)} sunk_cost, {len(nat_freq)} natural_freq)")

    models = [
        ("unsloth/Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct", 256),
        ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "R1-Distill-Llama-8B", 2048),
    ]

    all_results = {}
    for model_id, label, max_tokens in models:
        log(f"\n{'='*50}")
        log(f"MODEL: {label}")
        log(f"{'='*50}")

        results = run_model(model_id, new_items, max_new_tokens=max_tokens, label=label)
        analyze_results(results, label)
        all_results[label] = results

        # Save per-model results
        out_path = f"results/behavioral/new_items_{label.lower().replace('-','_').replace('.','')}.json"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"model": label, "model_id": model_id, "results": results}, f, indent=2)
        log(f"  Saved to {out_path}")

    # Summary comparison
    log(f"\n{'='*70}")
    log("SUMMARY COMPARISON")
    log(f"{'='*70}")

    log("\nKey questions:")
    log("  1. Is sunk_cost a NEW vulnerable category? (lure rate > 10%)")
    log("  2. Does natural frequency eliminate base rate vulnerability?")
    log("     Compare: base_rate/representativeness vs base_rate/natural_frequency")

    for label, results in all_results.items():
        sc_conflict = [r for r in results if r["category"] == "sunk_cost" and r["conflict"]]
        nf_conflict = [r for r in results if r["category"] == "base_rate"
                       and r.get("subcategory") == "natural_frequency" and r["conflict"]]
        sc_lure = sum(1 for r in sc_conflict if r["verdict"] == "lure")
        nf_lure = sum(1 for r in nf_conflict if r["verdict"] == "lure")
        log(f"\n  {label}:")
        log(f"    Sunk cost lure rate: {sc_lure}/{len(sc_conflict)} = {sc_lure/max(len(sc_conflict),1):.0%}")
        log(f"    Natural freq base rate lure: {nf_lure}/{len(nf_conflict)} = {nf_lure/max(len(nf_conflict),1):.0%}")

    log(f"\n{'='*70}")
    log("DONE")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
