#!/usr/bin/env python3
"""Confidence rating extension: De Neys conflict detection via token probabilities.

De Neys (2012, 2017) showed that humans detect conflicts between heuristic and
normative answers even when they give the biased response -- reduced confidence
on conflict vs control items is a key indicator. This script tests whether LLMs
show the same pattern: lower confidence (higher entropy, lower first-token
probability) on conflict items relative to matched controls.

Key metrics per item:
- first_token_prob: softmax probability of the first generated answer token
- top10_entropy: Shannon entropy (bits) over top-10 token probabilities
- lure_token_prob / correct_token_prob: for conflict items, the probability
  mass on the first token of lure vs correct answers
- confidence_gap: lure_token_prob - correct_token_prob (positive = lure-biased)

Usage:
    python scripts/confidence_paradigm.py \
        --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --benchmark data/benchmark/benchmark.jsonl \
        --output results/confidence/llama_confidence.json \
        --max-new-tokens 256 \
        --cache-dir /workspace/hf_cache
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Parsing helpers -- mirrors extract_real.py / behavioral_validation.py
# ---------------------------------------------------------------------------

def split_thinking(text: str) -> tuple[str, str]:
    """Split a reasoning model response into (thinking, answer)."""
    if "<think>" in text and "</think>" in text:
        ts = text.index("<think>") + len("<think>")
        te = text.index("</think>")
        return text[ts:te].strip(), text[te + len("</think>"):].strip()
    return "", text.strip()


def classify_response(text: str, correct_answer: str, lure_answer: str) -> str:
    """Classify model response as correct / lure / other."""
    if correct_answer and re.search(re.escape(correct_answer), text):
        return "correct"
    if lure_answer and re.search(re.escape(lure_answer), text):
        return "lure"
    return "other"


# ---------------------------------------------------------------------------
# Token-probability lookup for specific answer strings
# ---------------------------------------------------------------------------

def get_answer_first_token_id(
    tokenizer,
    answer_str: str,
) -> int | None:
    """Return the token id of the first subword of ``answer_str``.

    We tokenize with and without a leading space because models may emit the
    answer with or without preceding whitespace. Returns the token id whose
    decoded form best matches the start of the answer, or None if the answer
    string is empty.
    """
    if not answer_str.strip():
        return None
    # Try both forms -- BPE tokenizers often distinguish " 30" from "30"
    candidates: list[int] = []
    for prefix in (answer_str, f" {answer_str}"):
        ids = tokenizer.encode(prefix, add_special_tokens=False)
        if ids:
            candidates.append(ids[0])
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Confidence extraction
# ---------------------------------------------------------------------------

def extract_confidence_metrics(
    first_logits: torch.Tensor,
    generated_token_id: int,
    tokenizer,
    correct_answer: str,
    lure_answer: str,
    is_conflict: bool,
) -> dict:
    """Compute confidence metrics from the logits at the first answer position.

    Parameters
    ----------
    first_logits : shape (vocab_size,)
        Raw logits for the first generated token.
    generated_token_id :
        The token the model actually produced.
    tokenizer :
        For resolving answer strings to token ids.
    correct_answer, lure_answer :
        Canonical answer strings from the benchmark item.
    is_conflict :
        Whether this is a conflict item (controls lure metric computation).

    Returns
    -------
    Dict with first_token_prob, top10_entropy, and (for conflict items)
    lure_token_prob, correct_token_prob, confidence_gap.
    """
    probs = torch.softmax(first_logits.float(), dim=-1)

    # -- First-token probability --
    first_token_prob = probs[generated_token_id].item()

    # -- Top-10 Shannon entropy (bits) --
    top10_vals = torch.topk(probs, min(10, probs.shape[0])).values
    # Re-normalize top-10 so entropy is over this sub-distribution
    top10_norm = top10_vals / top10_vals.sum()
    entropy = -torch.sum(top10_norm * torch.log2(top10_norm + 1e-10)).item()

    metrics: dict = {
        "first_token_prob": round(first_token_prob, 6),
        "top10_entropy": round(entropy, 4),
    }

    # -- Lure vs correct probability (conflict items only) --
    if is_conflict and lure_answer:
        lure_tid = get_answer_first_token_id(tokenizer, lure_answer)
        correct_tid = get_answer_first_token_id(tokenizer, correct_answer)

        lure_prob = probs[lure_tid].item() if lure_tid is not None else None
        correct_prob = probs[correct_tid].item() if correct_tid is not None else None

        metrics["lure_token_prob"] = round(lure_prob, 6) if lure_prob is not None else None
        metrics["correct_token_prob"] = round(correct_prob, 6) if correct_prob is not None else None
        if lure_prob is not None and correct_prob is not None:
            metrics["confidence_gap"] = round(lure_prob - correct_prob, 6)
        else:
            metrics["confidence_gap"] = None
    else:
        metrics["lure_token_prob"] = None
        metrics["correct_token_prob"] = None
        metrics["confidence_gap"] = None

    return metrics


# ---------------------------------------------------------------------------
# Summary statistics & hypothesis tests
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], model_name: str) -> None:
    """Print De Neys confidence analysis summary."""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"CONFIDENCE ANALYSIS — {model_name}")
    print(sep)

    # Group by category x conflict
    by_cat: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"conflict": [], "control": []})
    for r in results:
        key = "conflict" if r["conflict"] else "control"
        by_cat[r["category"]][key].append(r)

    # --- Table 1: Mean confidence by category (conflict vs control) ---
    print(f"\n{'Category':<16} {'Metric':<20} {'Conflict':>12} {'Control':>12} {'Diff':>10}")
    print("-" * 72)
    all_conflict_probs: list[float] = []
    all_control_probs: list[float] = []
    all_conflict_ent: list[float] = []
    all_control_ent: list[float] = []

    categories = sorted(by_cat.keys())
    for cat in categories:
        conf_items = by_cat[cat]["conflict"]
        ctrl_items = by_cat[cat]["control"]
        if not conf_items or not ctrl_items:
            continue

        conf_prob = [r["first_token_prob"] for r in conf_items]
        ctrl_prob = [r["first_token_prob"] for r in ctrl_items]
        conf_ent = [r["top10_entropy"] for r in conf_items]
        ctrl_ent = [r["top10_entropy"] for r in ctrl_items]

        all_conflict_probs.extend(conf_prob)
        all_control_probs.extend(ctrl_prob)
        all_conflict_ent.extend(conf_ent)
        all_control_ent.extend(ctrl_ent)

        m_cp = np.mean(conf_prob)
        m_tp = np.mean(ctrl_prob)
        m_ce = np.mean(conf_ent)
        m_te = np.mean(ctrl_ent)

        print(f"{cat:<16} {'first_token_prob':<20} {m_cp:>12.4f} {m_tp:>12.4f} {m_cp - m_tp:>+10.4f}")
        print(f"{'':<16} {'top10_entropy':<20} {m_ce:>12.4f} {m_te:>12.4f} {m_ce - m_te:>+10.4f}")

    # Overall
    if all_conflict_probs and all_control_probs:
        print("-" * 72)
        mc = np.mean(all_conflict_probs)
        mt = np.mean(all_control_probs)
        print(f"{'OVERALL':<16} {'first_token_prob':<20} {mc:>12.4f} {mt:>12.4f} {mc - mt:>+10.4f}")
        mc_e = np.mean(all_conflict_ent)
        mt_e = np.mean(all_control_ent)
        print(f"{'':<16} {'top10_entropy':<20} {mc_e:>12.4f} {mt_e:>12.4f} {mc_e - mt_e:>+10.4f}")

    # --- Table 2: Confidence for correct vs wrong on conflict items ---
    print(f"\n{'Conflict verdict':<24} {'N':>5} {'Mean prob':>12} {'Mean entropy':>14}")
    print("-" * 60)
    conflict_correct = [r for r in results if r["conflict"] and r["verdict"] == "correct"]
    conflict_lured = [r for r in results if r["conflict"] and r["verdict"] == "lure"]
    conflict_other = [r for r in results if r["conflict"] and r["verdict"] == "other"]

    for label, subset in [
        ("correct (S2 won)", conflict_correct),
        ("lure (S1 won)", conflict_lured),
        ("other", conflict_other),
    ]:
        if not subset:
            print(f"{label:<24} {0:>5} {'—':>12} {'—':>14}")
            continue
        probs = [r["first_token_prob"] for r in subset]
        ents = [r["top10_entropy"] for r in subset]
        print(f"{label:<24} {len(subset):>5} {np.mean(probs):>12.4f} {np.mean(ents):>14.4f}")

    # --- Statistical tests ---
    print(f"\n{'STATISTICAL TESTS':^72}")
    print("-" * 72)

    def _run_mwu(label: str, conflict_vals: list[float], control_vals: list[float]) -> None:
        if len(conflict_vals) < 3 or len(control_vals) < 3:
            print(f"  {label}: insufficient data (n_conflict={len(conflict_vals)}, n_control={len(control_vals)})")
            return
        stat, p = mannwhitneyu(conflict_vals, control_vals, alternative="two-sided")
        n1, n2 = len(conflict_vals), len(control_vals)
        # Rank-biserial correlation as effect size: r = 1 - 2U/(n1*n2)
        r_rb = 1 - (2 * stat) / (n1 * n2)
        direction = "conflict < control" if np.mean(conflict_vals) < np.mean(control_vals) else "conflict > control"
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {label}")
        print(f"    U={stat:.1f}, p={p:.4e}, r_rb={r_rb:+.3f} ({direction}) {sig}")

    # Overall conflict vs control
    _run_mwu("first_token_prob (all)", all_conflict_probs, all_control_probs)
    _run_mwu("top10_entropy (all)", all_conflict_ent, all_control_ent)

    # Per-category
    for cat in categories:
        conf_items = by_cat[cat]["conflict"]
        ctrl_items = by_cat[cat]["control"]
        if not conf_items or not ctrl_items:
            continue
        _run_mwu(
            f"first_token_prob ({cat})",
            [r["first_token_prob"] for r in conf_items],
            [r["first_token_prob"] for r in ctrl_items],
        )

    # De Neys prediction: among WRONG conflict answers, is confidence still lower
    # than matched controls? This is the critical test.
    if conflict_lured:
        # Match lured conflict items to their controls via matched_pair_id
        lured_pair_ids = {r["matched_pair_id"] for r in conflict_lured}
        matched_controls = [
            r for r in results
            if not r["conflict"] and r["matched_pair_id"] in lured_pair_ids
        ]
        if matched_controls:
            print(f"\n  DE NEYS CRITICAL TEST: Lured conflict items vs matched controls")
            _run_mwu(
                "first_token_prob (lured vs matched control)",
                [r["first_token_prob"] for r in conflict_lured],
                [r["first_token_prob"] for r in matched_controls],
            )
            _run_mwu(
                "top10_entropy (lured vs matched control)",
                [r["top10_entropy"] for r in conflict_lured],
                [r["top10_entropy"] for r in matched_controls],
            )

    # Confidence gap summary (conflict items only)
    gaps = [r["confidence_gap"] for r in results if r["confidence_gap"] is not None]
    if gaps:
        print(f"\n  Confidence gap (lure_prob - correct_prob) on conflict items:")
        print(f"    mean={np.mean(gaps):+.4f}, median={np.median(gaps):+.4f}, "
              f"std={np.std(gaps):.4f}, N={len(gaps)}")
        n_favors_lure = sum(1 for g in gaps if g > 0)
        print(f"    {n_favors_lure}/{len(gaps)} items favor lure token ({n_favors_lure / len(gaps):.0%})")

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="De Neys confidence paradigm: token-level confidence on conflict vs control items",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--benchmark", default="data/benchmark/benchmark.jsonl")
    parser.add_argument("--output", required=True, help="Path for per-item JSON output")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--cache-dir", default="/workspace/hf_cache")
    parser.add_argument("--n-items", type=int, default=None, help="Limit items (for smoke testing)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # --- Load model ---
    print(f"Loading {args.model} ({args.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype,
        device_map="cuda",
    )
    model.eval()
    print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # --- Load benchmark ---
    items: list[dict] = []
    with open(args.benchmark) as f:
        for line in f:
            items.append(json.loads(line))
    if args.n_items is not None:
        items = items[:args.n_items]
    print(f"Running {len(items)} items...")

    # --- Process items ---
    results: list[dict] = []
    t_total = time.time()

    for i, item in enumerate(items):
        messages = [{"role": "user", "content": item["prompt"]}]
        if item.get("system_prompt"):
            messages.insert(0, {"role": "system", "content": item["system_prompt"]})

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1]

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        elapsed = time.time() - t0

        generated_ids = outputs.sequences[0][prompt_len:]
        n_gen = len(generated_ids)
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # Parse thinking trace (for reasoning models) before classifying
        _thinking, answer_text = split_thinking(response_text)
        verdict = classify_response(
            answer_text, item["correct_answer"], item["lure_answer"],
        )

        # Confidence metrics from the first generated token's logits
        if n_gen > 0 and len(outputs.scores) > 0:
            first_logits = outputs.scores[0][0]  # (vocab_size,)
            first_token_id = generated_ids[0].item()
            metrics = extract_confidence_metrics(
                first_logits=first_logits,
                generated_token_id=first_token_id,
                tokenizer=tokenizer,
                correct_answer=item["correct_answer"],
                lure_answer=item["lure_answer"],
                is_conflict=item["conflict"],
            )
        else:
            metrics = {
                "first_token_prob": None,
                "top10_entropy": None,
                "lure_token_prob": None,
                "correct_token_prob": None,
                "confidence_gap": None,
            }

        result = {
            "id": item["id"],
            "category": item["category"],
            "subcategory": item["subcategory"],
            "conflict": item["conflict"],
            "matched_pair_id": item["matched_pair_id"],
            "correct_answer": item["correct_answer"],
            "lure_answer": item["lure_answer"],
            "verdict": verdict,
            "n_tokens": n_gen,
            **metrics,
        }
        results.append(result)

        # Progress line
        marker = "Y" if verdict == "correct" else ("L" if verdict == "lure" else "?")
        prob_str = f"p={metrics['first_token_prob']:.3f}" if metrics["first_token_prob"] is not None else "p=N/A"
        ent_str = f"H={metrics['top10_entropy']:.2f}" if metrics["top10_entropy"] is not None else "H=N/A"
        conf_tag = "conflict" if item["conflict"] else "control"
        print(
            f"  [{i + 1}/{len(items)}] {item['id'][:40]:<40s} "
            f"{verdict:<8s} [{marker}] {conf_tag:<8s} {prob_str} {ent_str}  ({n_gen} tok, {elapsed:.1f}s)"
        )

    elapsed_total = time.time() - t_total
    print(f"\nTotal inference time: {elapsed_total:.0f}s ({elapsed_total / max(len(items), 1):.1f}s/item)")

    # --- Save results ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "model": args.model,
        "n_items": len(results),
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {output_path}")

    # --- Summary ---
    model_short = args.model.split("/")[-1] if "/" in args.model else args.model
    # Filter to items that have valid confidence metrics
    valid_results = [r for r in results if r["first_token_prob"] is not None]
    if valid_results:
        print_summary(valid_results, model_short)
    else:
        print("WARNING: No items produced valid confidence metrics.")

    return


if __name__ == "__main__":
    main()
