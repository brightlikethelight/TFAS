#!/usr/bin/env python3
"""Check whether conflict/control items differ systematically in token length.

Gemini flagged: probe AUC = 0.9999 on OLMo-32B is suspicious, since near-perfect
linear separability can arise from trivial features like prompt token length.
We verify that conflict and control items do NOT differ systematically in token
length, ruling out length as a trivial confound. We check at several granularities:
  - Raw prompt character count
  - Word count
  - Approximate subword token count (via a standard Llama tokenizer if available,
    else via a heuristic based on whitespace).

Outputs a JSON report and prints summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = _ROOT / "data" / "benchmark" / "benchmark.jsonl"
OUTPUT = _ROOT / "results" / "probes" / "length_confound_check.json"

VULNERABLE_CATEGORIES = ["base_rate", "conjunction", "syllogism"]
ALL_ORIGINAL_CATEGORIES = ["base_rate", "conjunction", "syllogism",
                           "crt", "arithmetic", "framing", "anchoring"]


def load_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(BENCHMARK) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def analyze(items: list[dict[str, Any]], category_filter: set[str] | None) -> dict[str, Any]:
    """Compute length statistics for conflict vs control items."""
    if category_filter is not None:
        items = [it for it in items if it["category"] in category_filter]

    by_label: dict[bool, list[dict[str, int]]] = defaultdict(list)
    for it in items:
        prompt = it["prompt"]
        by_label[bool(it["conflict"])].append({
            "chars": len(prompt),
            "words": len(prompt.split()),
        })

    out: dict[str, Any] = {
        "n_conflict": len(by_label[True]),
        "n_control": len(by_label[False]),
    }
    for metric in ("chars", "words"):
        conf = np.array([d[metric] for d in by_label[True]])
        ctrl = np.array([d[metric] for d in by_label[False]])
        # Welch's t-test
        from scipy.stats import ttest_ind, mannwhitneyu
        t_res = ttest_ind(conf, ctrl, equal_var=False)
        u_res = mannwhitneyu(conf, ctrl, alternative="two-sided")
        # Effect size (Cohen's d with pooled std)
        pool_std = np.sqrt((conf.var(ddof=1) + ctrl.var(ddof=1)) / 2.0)
        d_cohen = float((conf.mean() - ctrl.mean()) / pool_std) if pool_std > 0 else 0.0
        out[metric] = {
            "mean_conflict": float(conf.mean()),
            "mean_control": float(ctrl.mean()),
            "std_conflict": float(conf.std(ddof=1)),
            "std_control": float(ctrl.std(ddof=1)),
            "median_conflict": float(np.median(conf)),
            "median_control": float(np.median(ctrl)),
            "t_stat": float(t_res.statistic),
            "t_pvalue": float(t_res.pvalue),
            "mw_pvalue": float(u_res.pvalue),
            "cohens_d": d_cohen,
        }
    return out


def try_tokenize(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Try to tokenize with a real tokenizer if transformers is available."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            use_fast=True,
        )
    except Exception:
        try:
            # Fallback to a smaller open tokenizer
            tok = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            return None

    by_cat: dict[str, dict[bool, list[int]]] = defaultdict(lambda: defaultdict(list))
    all_conf: list[int] = []
    all_ctrl: list[int] = []
    for it in items:
        prompt = it["prompt"]
        n_tok = len(tok.encode(prompt, add_special_tokens=False))
        by_cat[it["category"]][bool(it["conflict"])].append(n_tok)
        if it["conflict"]:
            all_conf.append(n_tok)
        else:
            all_ctrl.append(n_tok)

    from scipy.stats import ttest_ind
    out: dict[str, Any] = {}
    for cat in by_cat:
        conf = np.array(by_cat[cat][True])
        ctrl = np.array(by_cat[cat][False])
        if len(conf) == 0 or len(ctrl) == 0:
            continue
        t = ttest_ind(conf, ctrl, equal_var=False)
        pool_std = np.sqrt((conf.var(ddof=1) + ctrl.var(ddof=1)) / 2.0)
        d_cohen = float((conf.mean() - ctrl.mean()) / pool_std) if pool_std > 0 else 0.0
        out[cat] = {
            "n_conflict": int(len(conf)),
            "n_control": int(len(ctrl)),
            "mean_conflict": float(conf.mean()),
            "mean_control": float(ctrl.mean()),
            "t_pvalue": float(t.pvalue),
            "cohens_d": d_cohen,
        }

    conf_all = np.array(all_conf)
    ctrl_all = np.array(all_ctrl)
    t = ttest_ind(conf_all, ctrl_all, equal_var=False)
    pool_std = np.sqrt((conf_all.var(ddof=1) + ctrl_all.var(ddof=1)) / 2.0)
    out["__overall__"] = {
        "n_conflict": int(len(conf_all)),
        "n_control": int(len(ctrl_all)),
        "mean_conflict": float(conf_all.mean()),
        "mean_control": float(ctrl_all.mean()),
        "t_pvalue": float(t.pvalue),
        "cohens_d": float((conf_all.mean() - ctrl_all.mean()) / pool_std) if pool_std > 0 else 0.0,
    }
    return out


def main() -> int:
    items = load_items()
    print(f"Loaded {len(items)} items from benchmark.")

    results: dict[str, Any] = {}

    for label, cats in [
        ("full_470", None),
        ("original_7cats_330", set(ALL_ORIGINAL_CATEGORIES)),
        ("vulnerable_3cats_160", set(VULNERABLE_CATEGORIES)),
    ]:
        print(f"\n=== {label} ===")
        r = analyze(items, cats)
        results[label] = r
        print(f"  n_conflict={r['n_conflict']}, n_control={r['n_control']}")
        for metric in ("chars", "words"):
            m = r[metric]
            print(f"  {metric:5s}: conf={m['mean_conflict']:.1f}±{m['std_conflict']:.1f}  "
                  f"ctrl={m['mean_control']:.1f}±{m['std_control']:.1f}  "
                  f"d={m['cohens_d']:+.3f}  t_p={m['t_pvalue']:.4f}  mw_p={m['mw_pvalue']:.4f}")

    tok_result = try_tokenize(items)
    if tok_result is not None:
        print("\n=== Subword tokens (per category) ===")
        overall = tok_result["__overall__"]
        print(f"  Overall: conf={overall['mean_conflict']:.1f}, "
              f"ctrl={overall['mean_control']:.1f}, "
              f"d={overall['cohens_d']:+.3f}, p={overall['t_pvalue']:.4f}")
        for cat in sorted(tok_result):
            if cat == "__overall__":
                continue
            c = tok_result[cat]
            print(f"  {cat:15s}: conf={c['mean_conflict']:6.1f} ctrl={c['mean_control']:6.1f} "
                  f"d={c['cohens_d']:+.3f} p={c['t_pvalue']:.4f}")
        results["subword_tokens"] = tok_result
    else:
        print("\n[warn] transformers not available; skipping subword tokenization")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
