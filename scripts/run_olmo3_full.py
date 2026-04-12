#!/usr/bin/env python3
"""Full OLMo-3-7B pair pipeline: download, behavioral validation, activation
extraction, probing, and cross-model comparison.

Designed to run end-to-end on the B200 pod after overnight2 completes.

    python scripts/run_olmo3_full.py

Models:
    allenai/OLMo-3-7B-Instruct  (standard instruction-tuned, no thinking)
    allenai/OLMo-3-7B-Think     (reasoning variant, uses <|think_start|>/<|think_end|>)

Both are Apache 2.0, ~7B params, ~14GB VRAM in BF16. No auth required.

OLMo-3-7B-Think uses <|think_start|> and <|think_end|> tokens (NOT <think>/<think>
like DeepSeek R1). The tokenizer handles these as special tokens.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure src layout works without pip install -e .
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS = {
    "instruct": {
        "hf_id": "allenai/OLMo-3-7B-Instruct",
        "is_reasoning": False,
        "max_new_tokens": 256,
        "think_tags": None,
    },
    "think": {
        "hf_id": "allenai/OLMo-3-7B-Think",
        "is_reasoning": True,
        "max_new_tokens": 2048,
        "think_tags": ("<|think_start|>", "<|think_end|>"),
    },
}

BENCHMARK_PATH = _ROOT / "data" / "benchmark" / "benchmark.jsonl"
CACHE_DIR = "/workspace/hf_cache"
LOG_PATH = Path("/workspace/olmo3_log.txt")

RESULTS_BEHAVIORAL_DIR = _ROOT / "results" / "behavioral"
ACTIVATIONS_DIR = _ROOT / "data" / "activations"
RESULTS_PROBES_DIR = _ROOT / "results" / "probes"

# Vulnerable categories: the ones where models actually get lured.
# Immune categories show 0% lure and AUC 1.0 at L0-1 (specificity confound).
VULNERABLE_CATEGORIES = {"base_rate", "conjunction", "syllogism"}
IMMUNE_CATEGORIES = {"crt", "arithmetic", "framing", "anchoring"}

# Probe config
N_FOLDS = 5
N_SEEDS = 3
N_BOOTSTRAP = 1000
PROBE_POSITIONS = ["P0", "P2"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_fh = None


def log(msg: str) -> None:
    """Print and log to file simultaneously."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_fh is not None:
        _log_fh.write(line + "\n")
        _log_fh.flush()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_answer(text: str, correct: str, lure: str) -> str:
    """Classify generated answer as correct, lure, or other."""
    if correct and re.search(re.escape(correct), text):
        return "correct"
    if lure and re.search(re.escape(lure), text):
        return "lure"
    return "other"


def split_thinking(
    text: str,
    open_tag: str = "<think>",
    close_tag: str = "</think>",
) -> tuple[str, str]:
    """Split response into (thinking_trace, answer) around thinking tags.

    Supports both DeepSeek-style <think>...</think> and OLMo-style
    <|think_start|>...<|think_end|> tags.
    """
    if open_tag in text and close_tag in text:
        ts = text.index(open_tag) + len(open_tag)
        te = text.index(close_tag)
        return text[ts:te].strip(), text[te + len(close_tag) :].strip()
    return "", text.strip()


def extract_hidden_state_at_position(
    hidden_states: tuple,
    position: int,
    prompt_len: int,
    layer_idx: int,
) -> np.ndarray | None:
    """Extract a single hidden state vector from generate()'s output_hidden_states.

    hidden_states[step][layer] where step 0 = prompt, step k>=1 = generated token k.
    Layer 0 = embedding, layer 1..N = transformer layers.
    """
    hf_layer = layer_idx + 1  # skip embedding layer

    if position < prompt_len:
        step = 0
        pos_in_step = position
    else:
        step = position - prompt_len + 1

    if step >= len(hidden_states):
        return None
    if hf_layer >= len(hidden_states[step]):
        return None

    hs = hidden_states[step][hf_layer]  # (1, seq_len_at_step, hidden_dim)

    if position < prompt_len:
        if pos_in_step >= hs.shape[1]:
            return None
        vec = hs[0, pos_in_step, :]
    else:
        vec = hs[0, -1, :]

    return vec.float().cpu().numpy().astype(np.float16)


# ---------------------------------------------------------------------------
# Stage 1: Download + behavioral validation
# ---------------------------------------------------------------------------


def run_behavioral(
    model_key: str,
    model_cfg: dict[str, Any],
    items: list[dict],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> list[dict]:
    """Run behavioral validation on all benchmark items. Returns per-item results."""
    hf_id = model_cfg["hf_id"]
    max_new_tokens = model_cfg["max_new_tokens"]
    think_tags = model_cfg["think_tags"]
    open_tag = think_tags[0] if think_tags else "<think>"
    close_tag = think_tags[1] if think_tags else "</think>"

    log(f"  Behavioral validation: {len(items)} items, max_new_tokens={max_new_tokens}")

    results = []
    t_total = time.time()
    for i, item in enumerate(items):
        messages = [{"role": "user", "content": item["prompt"]}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

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
        response = tokenizer.decode(
            out[0][inputs.input_ids.shape[1] :], skip_special_tokens=False
        )

        thinking, answer = split_thinking(response, open_tag, close_tag)
        verdict = parse_answer(answer, item["correct_answer"], item.get("lure_answer", ""))

        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "conflict": item["conflict"],
                "correct_answer": item["correct_answer"],
                "lure_answer": item.get("lure_answer", ""),
                "verdict": verdict,
                "n_tokens": n_gen,
                "thinking_len": len(thinking),
                "elapsed_s": round(elapsed, 1),
            }
        )

        if (i + 1) % 20 == 0 or i == 0 or (i + 1) == len(items):
            marker = (
                "OK"
                if verdict == "correct"
                else ("LURED" if verdict == "lure" else "other")
            )
            log(
                f"    [{i+1}/{len(items)}] {item['id'][:40]:40s} {verdict:8s} {marker} "
                f"({n_gen} tok, {elapsed:.1f}s)"
            )

    # Summary
    elapsed_total = time.time() - t_total
    n_conflict = sum(1 for r in results if r["conflict"])
    n_control = len(results) - n_conflict
    n_correct_conflict = sum(1 for r in results if r["conflict"] and r["verdict"] == "correct")
    n_lured = sum(1 for r in results if r["conflict"] and r["verdict"] == "lure")
    n_correct_control = sum(
        1 for r in results if not r["conflict"] and r["verdict"] == "correct"
    )

    log(f"\n  BEHAVIORAL SUMMARY ({model_key}):")
    log(f"    Items: {len(results)} ({n_conflict} conflict, {n_control} control)")
    log(f"    Conflict: {n_correct_conflict}/{n_conflict} correct, {n_lured}/{n_conflict} lured")
    log(f"    Control:  {n_correct_control}/{n_control} correct")
    if n_conflict > 0:
        lure_rate = n_lured / n_conflict
        log(f"    LURE RATE: {lure_rate:.1%}")
    log(f"    Time: {elapsed_total:.0f}s")

    # Per-category breakdown
    categories = sorted(set(r["category"] for r in results))
    log(f"\n  Per-category lure rates:")
    for cat in categories:
        cat_conflict = [r for r in results if r["category"] == cat and r["conflict"]]
        cat_lured = sum(1 for r in cat_conflict if r["verdict"] == "lure")
        n_cat = len(cat_conflict)
        rate = cat_lured / n_cat if n_cat > 0 else 0
        vuln = "VULNERABLE" if cat in VULNERABLE_CATEGORIES else "immune"
        log(f"    {cat:15s}: {cat_lured}/{n_cat} = {rate:.0%}  ({vuln})")

    return results


# ---------------------------------------------------------------------------
# Stage 2: Activation extraction
# ---------------------------------------------------------------------------


def run_extraction(
    model_key: str,
    model_cfg: dict[str, Any],
    items: list[dict],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    output_path: Path,
) -> None:
    """Extract residual stream activations for all layers at P0+P2, write HDF5."""
    hf_id = model_cfg["hf_id"]
    max_new_tokens = model_cfg["max_new_tokens"]
    think_tags = model_cfg["think_tags"]
    open_tag = think_tags[0] if think_tags else "<think>"
    close_tag = think_tags[1] if think_tags else "</think>"
    is_reasoning = model_cfg["is_reasoning"]

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    n_problems = len(items)
    extract_layers = list(range(n_layers))

    position_labels = ["P0", "P2"]
    n_positions = len(position_labels)

    log(f"  Extraction: {n_problems} items, {n_layers} layers, hidden={hidden_dim}")
    log(f"    Heads: {n_heads} Q, {n_kv_heads} KV")
    log(f"    Positions: {position_labels}")

    # Pre-allocate
    residuals = {
        layer: np.zeros((n_problems, n_positions, hidden_dim), dtype=np.float16)
        for layer in extract_layers
    }
    position_token_indices = np.zeros((n_problems, n_positions), dtype=np.int32)
    position_valid = np.zeros((n_problems, n_positions), dtype=bool)

    behavior = {
        "predicted_answer": [],
        "correct": np.zeros(n_problems, dtype=bool),
        "matches_lure": np.zeros(n_problems, dtype=bool),
        "response_category": [],
        "thinking_text": [],
        "answer_text": [],
        "n_gen_tokens": np.zeros(n_problems, dtype=np.int32),
        "prompt_token_count": np.zeros(n_problems, dtype=np.int32),
    }

    t_total = time.time()
    for i, item in enumerate(items):
        messages = [{"role": "user", "content": item["prompt"]}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1]
        behavior["prompt_token_count"][i] = prompt_len

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )

        gen_ids = outputs.sequences[0]
        n_gen = gen_ids.shape[0] - prompt_len
        response = tokenizer.decode(gen_ids[prompt_len:], skip_special_tokens=False)

        # P0: last prompt token
        p0_pos = prompt_len - 1
        position_token_indices[i, 0] = p0_pos
        position_valid[i, 0] = True

        for layer_idx in extract_layers:
            vec = extract_hidden_state_at_position(
                outputs.hidden_states, p0_pos, prompt_len, layer_idx
            )
            if vec is not None:
                residuals[layer_idx][i, 0, :] = vec

        # P2: last generated token
        p2_pos = gen_ids.shape[0] - 1
        position_token_indices[i, 1] = p2_pos
        position_valid[i, 1] = True

        last_step = len(outputs.hidden_states) - 1
        for layer_idx in extract_layers:
            if last_step > 0:
                vec = extract_hidden_state_at_position(
                    outputs.hidden_states, p2_pos, prompt_len, layer_idx
                )
                if vec is not None:
                    residuals[layer_idx][i, 1, :] = vec

        # Parse behavior
        thinking, answer = split_thinking(response, open_tag, close_tag)
        verdict = parse_answer(answer, item["correct_answer"], item.get("lure_answer", ""))

        behavior["predicted_answer"].append(answer[:200])
        behavior["correct"][i] = verdict == "correct"
        behavior["matches_lure"][i] = verdict == "lure"
        behavior["response_category"].append(verdict)
        behavior["thinking_text"].append(thinking[:2000])
        behavior["answer_text"].append(answer[:500])
        behavior["n_gen_tokens"][i] = n_gen

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_total
            eta = elapsed / (i + 1) * (n_problems - i - 1)
            log(
                f"    [{i+1}/{n_problems}] {verdict:8s} {n_gen:4d} tok  "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
            )

    # Write HDF5 using the data contract schema
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hdf5_model_key = hf_id.replace("/", "_")

    with h5py.File(str(output_path), "w") as f:
        # /metadata
        meta = f.create_group("/metadata")
        meta.attrs["schema_version"] = 1
        meta.attrs["benchmark_path"] = str(BENCHMARK_PATH)
        bm_hash = hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest()
        meta.attrs["benchmark_sha256"] = bm_hash
        meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        meta.attrs["git_sha"] = "unknown"
        meta.attrs["seed"] = 0
        meta.attrs["config"] = json.dumps(
            {
                "model": hf_id,
                "model_key": model_key,
                "is_reasoning": is_reasoning,
                "max_new_tokens": max_new_tokens,
                "n_items": n_problems,
                "extract_layers": extract_layers,
            }
        )

        # /problems
        prob = f.create_group("/problems")
        prob.create_dataset(
            "id", data=np.array([it["id"].encode()[:64] for it in items], dtype="S64")
        )
        prob.create_dataset(
            "category",
            data=np.array([it["category"].encode()[:32] for it in items], dtype="S32"),
        )
        prob.create_dataset(
            "conflict", data=np.array([it["conflict"] for it in items], dtype=bool)
        )
        prob.create_dataset(
            "difficulty",
            data=np.array([it.get("difficulty", 2) for it in items], dtype=np.int8),
        )
        prob.create_dataset(
            "prompt_text",
            data=np.array([it["prompt"].encode()[:2048] for it in items], dtype="S2048"),
        )
        prob.create_dataset(
            "correct_answer",
            data=np.array(
                [it["correct_answer"].encode()[:128] for it in items], dtype="S128"
            ),
        )
        prob.create_dataset(
            "lure_answer",
            data=np.array(
                [it.get("lure_answer", "").encode()[:128] for it in items], dtype="S128"
            ),
        )
        prob.create_dataset(
            "matched_pair_id",
            data=np.array(
                [it["matched_pair_id"].encode()[:64] for it in items], dtype="S64"
            ),
        )
        prob.create_dataset("prompt_token_count", data=behavior["prompt_token_count"])

        # /models/{key}
        mgrp = f.create_group(f"/models/{hdf5_model_key}")
        mmeta = mgrp.create_group("metadata")
        mmeta.attrs["hf_model_id"] = hf_id
        mmeta.attrs["n_layers"] = n_layers
        mmeta.attrs["n_heads"] = n_heads
        mmeta.attrs["n_kv_heads"] = n_kv_heads
        mmeta.attrs["hidden_dim"] = hidden_dim
        mmeta.attrs["head_dim"] = hidden_dim // n_heads
        mmeta.attrs["dtype"] = "float16"
        mmeta.attrs["extracted_at"] = datetime.now(timezone.utc).isoformat()
        mmeta.attrs["is_reasoning_model"] = is_reasoning

        # /models/{key}/residual
        resid = mgrp.create_group("residual")
        for layer_idx in extract_layers:
            resid.create_dataset(
                f"layer_{layer_idx:02d}",
                data=residuals[layer_idx],
                compression="gzip",
                compression_opts=4,
            )

        # /models/{key}/position_index
        pos = mgrp.create_group("position_index")
        pos.create_dataset(
            "labels",
            data=np.array([s.encode() for s in position_labels], dtype="S16"),
        )
        pos.create_dataset("token_indices", data=position_token_indices)
        pos.create_dataset("valid", data=position_valid)

        # /models/{key}/behavior
        beh = mgrp.create_group("behavior")
        beh.create_dataset(
            "predicted_answer",
            data=np.array(
                [s.encode()[:128] for s in behavior["predicted_answer"]], dtype="S128"
            ),
        )
        beh.create_dataset("correct", data=behavior["correct"])
        beh.create_dataset("matches_lure", data=behavior["matches_lure"])
        beh.create_dataset(
            "response_category",
            data=np.array(
                [s.encode()[:16] for s in behavior["response_category"]], dtype="S16"
            ),
        )

        # /models/{key}/generations
        gen = mgrp.create_group("generations")
        gen.create_dataset(
            "full_text",
            data=np.array(["".encode()[:8192]] * n_problems, dtype="S8192"),
        )
        gen.create_dataset(
            "thinking_text",
            data=np.array(
                [s.encode()[:8192] for s in behavior["thinking_text"]], dtype="S8192"
            ),
        )
        gen.create_dataset(
            "answer_text",
            data=np.array(
                [s.encode()[:512] for s in behavior["answer_text"]], dtype="S512"
            ),
        )
        gen.create_dataset(
            "thinking_token_count", data=np.zeros(n_problems, dtype=np.int32)
        )
        gen.create_dataset("answer_token_count", data=behavior["n_gen_tokens"])

    elapsed_total = time.time() - t_total
    n_correct = int(behavior["correct"].sum())
    n_lured = int(behavior["matches_lure"].sum())
    n_conflict = sum(1 for it in items if it["conflict"])
    file_size_mb = output_path.stat().st_size / 1e6

    log(f"\n  EXTRACTION SUMMARY ({model_key}):")
    log(f"    Items: {n_problems}, Layers: {len(extract_layers)}, Hidden: {hidden_dim}")
    log(
        f"    Behavioral: {n_correct}/{n_problems} correct, "
        f"{n_lured}/{n_conflict} lured ({n_lured / max(n_conflict, 1):.1%})"
    )
    log(f"    HDF5: {output_path} ({file_size_mb:.1f} MB)")
    log(f"    Time: {elapsed_total:.0f}s ({elapsed_total / 60:.1f} min)")


# ---------------------------------------------------------------------------
# Stage 3: Probing (vulnerable categories, logistic regression)
# ---------------------------------------------------------------------------


def run_probes_on_h5(
    model_key: str,
    h5_path: Path,
    items: list[dict],
) -> dict[str, Any]:
    """Run logistic probes on vulnerable categories at all layers and positions.

    Returns a dict with per-layer AUC results suitable for cross-model comparison.
    """
    # Build the vulnerable-category mask
    categories = np.array([it["category"] for it in items])
    conflict = np.array([it["conflict"] for it in items])
    vulnerable_mask = np.array([c in VULNERABLE_CATEGORIES for c in categories])

    # Target: conflict (1) vs control (0) restricted to vulnerable categories
    y_full = conflict[vulnerable_mask].astype(np.int8)
    stratify_cats = categories[vulnerable_mask]

    n_vuln = vulnerable_mask.sum()
    n_pos = int(y_full.sum())
    n_neg = len(y_full) - n_pos
    log(f"  Probing: {n_vuln} vulnerable items ({n_pos} conflict, {n_neg} control)")

    with h5py.File(str(h5_path), "r") as f:
        model_keys = list(f["/models"].keys())
        if not model_keys:
            log(f"    ERROR: no models in {h5_path}")
            return {}
        hdf5_key = model_keys[0]
        n_layers = int(f[f"/models/{hdf5_key}/metadata"].attrs["n_layers"])
        hidden_dim = int(f[f"/models/{hdf5_key}/metadata"].attrs["hidden_dim"])
        pos_labels_raw = f[f"/models/{hdf5_key}/position_index/labels"][:]
        pos_labels = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in pos_labels_raw]

    log(f"    HDF5 model key: {hdf5_key}")
    log(f"    Layers: {n_layers}, Hidden: {hidden_dim}")
    log(f"    Available positions: {pos_labels}")

    all_results: dict[str, Any] = {
        "model_key": model_key,
        "hdf5_key": hdf5_key,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "n_vulnerable": int(n_vuln),
        "layers": {},
    }

    for position in PROBE_POSITIONS:
        if position not in pos_labels:
            log(f"    Position {position} not available, skipping")
            continue

        pos_idx = pos_labels.index(position)
        log(f"\n  Position: {position}")

        for layer in range(n_layers):
            # Load activations for this layer+position
            with h5py.File(str(h5_path), "r") as f:
                arr = f[f"/models/{hdf5_key}/residual/layer_{layer:02d}"][:]
                X_full = arr[:, pos_idx, :].astype(np.float32)

            # Apply vulnerable mask
            X = X_full[vulnerable_mask]

            # Stratified k-fold with multi-seed logistic regression
            aucs_per_seed: list[float] = []
            for seed in range(N_SEEDS):
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                fold_probas = np.zeros(len(y_full), dtype=np.float64)

                for train_idx, test_idx in skf.split(X, y_full):
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X[train_idx])
                    X_te = scaler.transform(X[test_idx])

                    clf = LogisticRegression(
                        max_iter=1000,
                        solver="lbfgs",
                        C=1.0,
                        random_state=seed,
                    )
                    clf.fit(X_tr, y_full[train_idx])
                    fold_probas[test_idx] = clf.predict_proba(X_te)[:, 1]

                if len(np.unique(y_full)) >= 2:
                    auc = float(roc_auc_score(y_full, fold_probas))
                else:
                    auc = 0.5
                aucs_per_seed.append(auc)

            mean_auc = float(np.mean(aucs_per_seed))
            std_auc = float(np.std(aucs_per_seed, ddof=1)) if N_SEEDS > 1 else 0.0

            layer_key = f"L{layer:02d}_{position}"
            all_results["layers"][layer_key] = {
                "layer": layer,
                "position": position,
                "auc_mean": round(mean_auc, 4),
                "auc_std": round(std_auc, 4),
                "n_samples": int(n_vuln),
            }

            if layer % 4 == 0 or layer == n_layers - 1:
                log(f"    Layer {layer:2d}/{position}: AUC = {mean_auc:.4f} +/- {std_auc:.4f}")

    return all_results


# ---------------------------------------------------------------------------
# Stage 4: Cross-model comparison
# ---------------------------------------------------------------------------


def cross_model_comparison(
    instruct_probes: dict[str, Any],
    think_probes: dict[str, Any],
) -> dict[str, Any]:
    """Compare probe results between instruct and think models."""
    comparison: dict[str, Any] = {
        "instruct_model": instruct_probes.get("model_key", "instruct"),
        "think_model": think_probes.get("model_key", "think"),
        "per_position": {},
    }

    for position in PROBE_POSITIONS:
        instruct_layers = {}
        think_layers = {}

        for key, val in instruct_probes.get("layers", {}).items():
            if val["position"] == position:
                instruct_layers[val["layer"]] = val["auc_mean"]

        for key, val in think_probes.get("layers", {}).items():
            if val["position"] == position:
                think_layers[val["layer"]] = val["auc_mean"]

        if not instruct_layers or not think_layers:
            continue

        common_layers = sorted(set(instruct_layers.keys()) & set(think_layers.keys()))

        # Find peaks
        instruct_peak_layer = max(common_layers, key=lambda l: instruct_layers[l])
        think_peak_layer = max(common_layers, key=lambda l: think_layers[l])

        # Compute per-layer deltas
        deltas = {l: instruct_layers[l] - think_layers[l] for l in common_layers}
        delta_arr = np.array([deltas[l] for l in common_layers])

        comparison["per_position"][position] = {
            "instruct_peak_layer": instruct_peak_layer,
            "instruct_peak_auc": instruct_layers[instruct_peak_layer],
            "think_peak_layer": think_peak_layer,
            "think_peak_auc": think_layers[think_peak_layer],
            "instruct_mean_auc": float(np.mean([instruct_layers[l] for l in common_layers])),
            "think_mean_auc": float(np.mean([think_layers[l] for l in common_layers])),
            "mean_delta": float(np.mean(delta_arr)),
            "max_delta": float(np.max(delta_arr)),
            "max_delta_layer": common_layers[int(np.argmax(delta_arr))],
            "min_delta": float(np.min(delta_arr)),
            "all_positive": bool(np.all(delta_arr > 0)),
            "n_layers_instruct_higher": int(np.sum(delta_arr > 0)),
            "n_layers_think_higher": int(np.sum(delta_arr < 0)),
            "n_layers_total": len(common_layers),
        }

    return comparison


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_one_model(
    model_key: str,
    model_cfg: dict[str, Any],
    items: list[dict],
) -> tuple[list[dict] | None, dict[str, Any] | None]:
    """Download, validate, extract, and probe one model. Returns (behavioral, probes)."""
    hf_id = model_cfg["hf_id"]
    log(f"\n{'='*70}")
    log(f"MODEL: {hf_id} (key={model_key})")
    log(f"{'='*70}")

    # Paths
    behavioral_path = RESULTS_BEHAVIORAL_DIR / f"olmo3_{model_key}_ALL.json"
    h5_path = ACTIVATIONS_DIR / f"olmo3_{model_key}.h5"
    probe_path = RESULTS_PROBES_DIR / f"olmo3_{model_key}_vulnerable.json"

    # --- Load model ---
    log(f"\n  Loading {hf_id}...")
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(hf_id, cache_dir=CACHE_DIR)

    # Ensure pad_token is set to suppress warnings during generate()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)

    log(f"  Loaded in {time.time() - t_load:.0f}s")
    log(f"  Architecture: {n_layers} layers, {hidden_dim} hidden, {n_heads} Q-heads, {n_kv_heads} KV-heads")
    log(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Verify chat template works
    test_messages = [{"role": "user", "content": "Hello"}]
    try:
        test_text = tokenizer.apply_chat_template(
            test_messages, tokenize=False, add_generation_prompt=True
        )
        log(f"  Chat template OK. Sample prefix: {test_text[:100]}...")
    except Exception as e:
        log(f"  WARNING: Chat template failed: {e}")
        log(f"  Will attempt to continue anyway")

    # Check for think tokens in tokenizer (OLMo-3-7B-Think)
    if model_cfg["think_tags"]:
        open_tag, close_tag = model_cfg["think_tags"]
        open_ids = tokenizer.encode(open_tag, add_special_tokens=False)
        close_ids = tokenizer.encode(close_tag, add_special_tokens=False)
        log(f"  Think tokens: '{open_tag}' -> {open_ids}, '{close_tag}' -> {close_ids}")

    # --- Stage 1: Behavioral validation ---
    log(f"\n  --- Stage 1: Behavioral Validation ---")
    behavioral_results = run_behavioral(model_key, model_cfg, items, tokenizer, model)

    RESULTS_BEHAVIORAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(behavioral_path, "w") as f:
        json.dump(
            {"model": hf_id, "model_key": model_key, "results": behavioral_results},
            f,
            indent=2,
        )
    log(f"  Behavioral saved: {behavioral_path}")

    # --- Stage 2: Activation extraction ---
    log(f"\n  --- Stage 2: Activation Extraction ---")
    run_extraction(model_key, model_cfg, items, tokenizer, model, h5_path)

    # --- Free GPU memory before probing ---
    del model
    del tokenizer
    torch.cuda.empty_cache()
    import gc

    gc.collect()
    log(f"  GPU memory freed. VRAM after: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # --- Stage 3: Probing ---
    log(f"\n  --- Stage 3: Probing (vulnerable categories) ---")
    probe_results = run_probes_on_h5(model_key, h5_path, items)

    RESULTS_PROBES_DIR.mkdir(parents=True, exist_ok=True)
    with open(probe_path, "w") as f:
        json.dump(probe_results, f, indent=2)
    log(f"  Probes saved: {probe_path}")

    return behavioral_results, probe_results


def main() -> int:
    global _log_fh

    # Open log file
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _log_fh = open(LOG_PATH, "w")

    log("=" * 70)
    log("OLMo-3-7B FULL PIPELINE")
    log(f"Started: {datetime.now().isoformat()}")
    log(f"Log: {LOG_PATH}")
    log("=" * 70)

    # Load benchmark
    if not BENCHMARK_PATH.exists():
        log(f"FATAL: benchmark not found at {BENCHMARK_PATH}")
        return 1

    items = []
    with open(BENCHMARK_PATH) as f:
        for line in f:
            items.append(json.loads(line))
    log(f"Benchmark: {len(items)} items from {BENCHMARK_PATH}")

    # Verify CUDA
    if not torch.cuda.is_available():
        log("FATAL: CUDA not available")
        return 1
    log(f"CUDA device: {torch.cuda.get_device_name(0)}")
    log(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Process each model independently so failures are isolated
    all_behavioral: dict[str, list[dict] | None] = {}
    all_probes: dict[str, dict[str, Any] | None] = {}

    for model_key, model_cfg in MODELS.items():
        try:
            behavioral, probes = process_one_model(model_key, model_cfg, items)
            all_behavioral[model_key] = behavioral
            all_probes[model_key] = probes
        except Exception as e:
            log(f"\n  FAILED: {model_key} — {e}")
            log(traceback.format_exc())
            all_behavioral[model_key] = None
            all_probes[model_key] = None
            # Clean up GPU in case of partial failure
            torch.cuda.empty_cache()

    # --- Stage 4: Cross-model comparison ---
    log(f"\n{'='*70}")
    log("CROSS-MODEL COMPARISON")
    log(f"{'='*70}")

    if all_probes.get("instruct") and all_probes.get("think"):
        comparison = cross_model_comparison(all_probes["instruct"], all_probes["think"])

        comparison_path = RESULTS_PROBES_DIR / "olmo3_instruct_vs_think_comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        log(f"Comparison saved: {comparison_path}")

        # Print the key findings
        for position, pos_data in comparison.get("per_position", {}).items():
            log(f"\n  Position {position}:")
            log(
                f"    Instruct peak: AUC {pos_data['instruct_peak_auc']:.4f} "
                f"at Layer {pos_data['instruct_peak_layer']}"
            )
            log(
                f"    Think peak:    AUC {pos_data['think_peak_auc']:.4f} "
                f"at Layer {pos_data['think_peak_layer']}"
            )
            log(f"    Instruct mean AUC: {pos_data['instruct_mean_auc']:.4f}")
            log(f"    Think mean AUC:    {pos_data['think_mean_auc']:.4f}")
            log(f"    Mean delta (Instruct - Think): {pos_data['mean_delta']:.4f}")
            log(
                f"    Max delta: {pos_data['max_delta']:.4f} "
                f"at Layer {pos_data['max_delta_layer']}"
            )
            log(
                f"    Instruct higher at {pos_data['n_layers_instruct_higher']}"
                f"/{pos_data['n_layers_total']} layers"
            )
            if pos_data["all_positive"]:
                log(f"    All deltas positive: YES (instruct always more separable)")
            else:
                log(
                    f"    Think higher at {pos_data['n_layers_think_higher']}"
                    f"/{pos_data['n_layers_total']} layers"
                )
    else:
        log("  Cannot compare: one or both models failed.")
        log(f"    instruct: {'OK' if all_probes.get('instruct') else 'FAILED'}")
        log(f"    think: {'OK' if all_probes.get('think') else 'FAILED'}")

    # --- Behavioral comparison ---
    if all_behavioral.get("instruct") and all_behavioral.get("think"):
        log(f"\n  BEHAVIORAL COMPARISON:")
        for label, results in [("Instruct", all_behavioral["instruct"]), ("Think", all_behavioral["think"])]:
            n_conflict = sum(1 for r in results if r["conflict"])
            n_lured = sum(1 for r in results if r["conflict"] and r["verdict"] == "lure")
            rate = n_lured / n_conflict if n_conflict > 0 else 0
            log(f"    {label:10s}: {n_lured}/{n_conflict} lured ({rate:.1%})")

        # Per vulnerable category
        log(f"\n  Per vulnerable category lure rates:")
        for cat in sorted(VULNERABLE_CATEGORIES):
            instruct_conflict = [r for r in all_behavioral["instruct"] if r["category"] == cat and r["conflict"]]
            think_conflict = [r for r in all_behavioral["think"] if r["category"] == cat and r["conflict"]]
            i_lured = sum(1 for r in instruct_conflict if r["verdict"] == "lure")
            t_lured = sum(1 for r in think_conflict if r["verdict"] == "lure")
            i_n = len(instruct_conflict)
            t_n = len(think_conflict)
            log(
                f"    {cat:15s}: Instruct {i_lured}/{i_n} ({i_lured/max(i_n,1):.0%}) "
                f"vs Think {t_lured}/{t_n} ({t_lured/max(t_n,1):.0%})"
            )

    # --- Final summary ---
    log(f"\n{'='*70}")
    log("PIPELINE COMPLETE")
    log(f"Finished: {datetime.now().isoformat()}")
    log(f"{'='*70}")

    log(f"\nOutput files:")
    for model_key in MODELS:
        log(f"  Behavioral: results/behavioral/olmo3_{model_key}_ALL.json")
        log(f"  Activations: data/activations/olmo3_{model_key}.h5")
        log(f"  Probes: results/probes/olmo3_{model_key}_vulnerable.json")
    log(f"  Comparison: results/probes/olmo3_instruct_vs_think_comparison.json")
    log(f"  Log: {LOG_PATH}")

    _log_fh.close()
    _log_fh = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
