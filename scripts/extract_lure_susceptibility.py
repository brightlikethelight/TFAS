#!/usr/bin/env python3
"""Extract continuous lure susceptibility scores from a model's P0 logits.

For each benchmark item, computes:

    score = log P(lure_token | prompt) - log P(correct_token | prompt)

at position P0 (last prompt token, before any generation). A positive score means
the model's initial representation favors the lure over the correct answer.

This is a SINGLE forward pass per item (no generation), so it's fast (~2 min for 330 items).

Control items (no lure) get score = NaN.

Usage:
    python scripts/extract_lure_susceptibility.py \
        --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --output results/behavioral/llama31_lure_susceptibility.json

    python scripts/extract_lure_susceptibility.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --output results/behavioral/r1_distill_lure_susceptibility.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_first_answer_token(tokenizer: AutoTokenizer, answer: str) -> int | None:
    """Get the token ID for the first token of an answer string.

    Tries both with and without a leading space, since tokenizers may prepend
    a space depending on context. Returns None if the answer is empty.
    """
    if not answer.strip():
        return None

    # Tokenize without special tokens. Try with a leading space first (more
    # natural for continuation after a prompt), fall back to bare string.
    for variant in [f" {answer.strip()}", answer.strip()]:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            return ids[0]

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract lure susceptibility scores from P0 logits."
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--benchmark", default="data/benchmark/benchmark.jsonl")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--cache-dir", default="/workspace/hf_cache")
    parser.add_argument(
        "--n-items", type=int, default=None, help="Limit items (None=all)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──────────────────────────────────────────────────────
    print(f"Loading {args.model} on {device}...")
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    if device == "cuda":
        print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # ── Load benchmark ──────────────────────────────────────────────────
    items: list[dict] = []
    with open(args.benchmark) as f:
        for line in f:
            items.append(json.loads(line))
    if args.n_items:
        items = items[: args.n_items]
    print(f"Benchmark: {len(items)} items")

    # ── Pre-resolve answer tokens ───────────────────────────────────────
    # Verify tokenization makes sense before burning GPU time.
    skipped_no_correct = 0
    skipped_no_lure = 0
    for item in items:
        correct_tid = get_first_answer_token(tok, item["correct_answer"])
        lure_tid = get_first_answer_token(tok, item.get("lure_answer", ""))
        item["_correct_tid"] = correct_tid
        item["_lure_tid"] = lure_tid
        if correct_tid is None:
            skipped_no_correct += 1
        if lure_tid is None:
            skipped_no_lure += 1

    print(f"Token resolution: {skipped_no_correct} items missing correct token, "
          f"{skipped_no_lure} items missing lure token (expected for controls)")

    # Sanity: show a few resolved tokens.
    for item in items[:3]:
        ct = tok.decode([item["_correct_tid"]]) if item["_correct_tid"] is not None else "N/A"
        lt = tok.decode([item["_lure_tid"]]) if item["_lure_tid"] is not None else "N/A"
        print(f"  {item['id'][:50]:50s} correct='{ct}' lure='{lt}'")

    # ── Extract scores ──────────────────────────────────────────────────
    results: list[dict] = []
    t_total = time.time()

    for i, item in enumerate(items):
        # Build chat-formatted prompt (same as behavioral_validation.py).
        messages = [{"role": "user", "content": item["prompt"]}]
        input_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Logits at the last prompt token (P0). Shape: (vocab_size,)
        logits_p0 = outputs.logits[0, -1, :]
        log_probs = F.log_softmax(logits_p0.float(), dim=-1)

        correct_tid = item["_correct_tid"]
        lure_tid = item["_lure_tid"]

        # Compute individual log-probs.
        log_p_correct = log_probs[correct_tid].item() if correct_tid is not None else None
        log_p_lure = log_probs[lure_tid].item() if lure_tid is not None else None

        # Susceptibility score: positive = model favors lure.
        if log_p_correct is not None and log_p_lure is not None:
            score = log_p_lure - log_p_correct
        else:
            score = None  # Control items or missing tokens -> NaN in JSON

        # Also grab the top-5 predicted tokens for diagnostics.
        top5_vals, top5_ids = torch.topk(log_probs, 5)
        top5 = [
            {"token": tok.decode([tid.item()]), "log_prob": val.item()}
            for tid, val in zip(top5_ids, top5_vals)
        ]

        result = {
            "id": item["id"],
            "category": item["category"],
            "conflict": item["conflict"],
            "difficulty": item.get("difficulty", 2),
            "matched_pair_id": item["matched_pair_id"],
            "correct_answer": item["correct_answer"],
            "lure_answer": item.get("lure_answer", ""),
            "correct_token": tok.decode([correct_tid]) if correct_tid is not None else None,
            "lure_token": tok.decode([lure_tid]) if lure_tid is not None else None,
            "correct_token_id": correct_tid,
            "lure_token_id": lure_tid,
            "log_p_correct": log_p_correct,
            "log_p_lure": log_p_lure,
            "lure_susceptibility_score": score,
            "top5_tokens": top5,
        }
        results.append(result)

        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t_total
            eta = elapsed / (i + 1) * (len(items) - i - 1)
            score_str = f"{score:+.3f}" if score is not None else "  N/A "
            print(
                f"  [{i + 1}/{len(items)}] {item['id'][:45]:45s} "
                f"score={score_str}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
            )

    elapsed_total = time.time() - t_total

    # ── Summary statistics ──────────────────────────────────────────────
    conflict_scores = [
        r["lure_susceptibility_score"]
        for r in results
        if r["conflict"] and r["lure_susceptibility_score"] is not None
    ]
    control_scores = [
        r["lure_susceptibility_score"]
        for r in results
        if not r["conflict"] and r["lure_susceptibility_score"] is not None
    ]

    def _stats(scores: list[float]) -> dict:
        if not scores:
            return {"n": 0}
        s = sorted(scores)
        n = len(s)
        mean = sum(s) / n
        var = sum((x - mean) ** 2 for x in s) / n
        return {
            "n": n,
            "mean": round(mean, 4),
            "std": round(math.sqrt(var), 4),
            "min": round(s[0], 4),
            "median": round(s[n // 2], 4),
            "max": round(s[-1], 4),
            "n_positive": sum(1 for x in s if x > 0),
            "frac_positive": round(sum(1 for x in s if x > 0) / n, 4),
        }

    summary = {
        "model": args.model,
        "n_items": len(items),
        "elapsed_s": round(elapsed_total, 1),
        "conflict_items": _stats(conflict_scores),
        "control_items": _stats(control_scores),
    }

    print(f"\n{'=' * 65}")
    print(f"MODEL: {args.model}")
    print(f"Items: {len(items)}, Time: {elapsed_total:.0f}s ({elapsed_total / 60:.1f} min)")
    print(f"\nConflict items ({summary['conflict_items']['n']}):")
    for k, v in summary["conflict_items"].items():
        print(f"  {k}: {v}")
    print(f"\nControl items ({summary['control_items']['n']}):")
    for k, v in summary["control_items"].items():
        print(f"  {k}: {v}")
    print(f"{'=' * 65}")

    # ── Save ────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "summary": summary,
        "items": results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
