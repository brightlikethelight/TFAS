#!/usr/bin/env python3
"""Overnight3 pipeline: high-impact remaining experiments.

Deploy to B200 pod:
    scp scripts/overnight3.py root@pod:/workspace/s1s2/scripts/
    ssh root@pod "cd /workspace/s1s2 && nohup python scripts/overnight3.py &"

Priority order:
    1. Confidence paradigm (De Neys) — Llama + R1-Distill     ~2 hr GPU
    2. Llama + R1 on full 470-item benchmark                   ~2 hr GPU
    3. Cross-model probe transfer (Llama <-> R1-Distill)       ~30 min CPU
    4. SAE on R1-Distill activations (Goodfire L19)            ~30 min GPU
    5. OLMo bootstrap CIs                                      ~1 hr CPU
    6. Qwen on expanded 470-item benchmark                     ~4 hr GPU

Estimated total: ~10 hours. All results saved under results/ and logged
to /workspace/overnight3_log.txt.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Environment — set before any HF/torch imports
# ---------------------------------------------------------------------------
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
BENCHMARK = PROJECT_ROOT / "data" / "benchmark" / "benchmark.jsonl"
LOG_FILE = Path("/workspace/overnight3_log.txt")
STATE_FILE = Path("/workspace/overnight3_state.json")

# Make src/ importable without pip install -e .
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
LLAMA_ID = "unsloth/Meta-Llama-3.1-8B-Instruct"
R1_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
QWEN_ID = "Qwen/Qwen3-8B"
OLMO_INSTRUCT_ID = "allenai/OLMo-3-7B-Instruct"
OLMO_THINK_ID = "allenai/OLMo-3-7B-Think"

CACHE_DIR = "/workspace/hf_cache"

LLAMA_H5 = PROJECT_ROOT / "data" / "activations" / "llama31_8b_instruct.h5"
R1_H5 = PROJECT_ROOT / "data" / "activations" / "r1_distill_llama.h5"
OLMO_INSTRUCT_H5 = PROJECT_ROOT / "data" / "activations" / "olmo3_instruct.h5"
OLMO_THINK_H5 = PROJECT_ROOT / "data" / "activations" / "olmo3_think.h5"

VULNERABLE_CATEGORIES = ["base_rate", "conjunction", "syllogism"]
IMMUNE_CATEGORIES = ["crt", "arithmetic", "framing", "anchoring"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_fh = None


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    global _log_fh
    if _log_fh is None:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _log_fh = open(LOG_FILE, "a")
    _log_fh.write(line + "\n")
    _log_fh.flush()


def banner(msg: str) -> None:
    sep = "=" * 72
    log(sep)
    log(msg)
    log(sep)


# ---------------------------------------------------------------------------
# State management (checkpoint/resume)
# ---------------------------------------------------------------------------
def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"completed": {}, "failed": {}}


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def mark_completed(state: dict[str, Any], job_name: str, elapsed: float) -> None:
    state["completed"][job_name] = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
    }
    state["failed"].pop(job_name, None)
    save_state(state)


def mark_failed(state: dict[str, Any], job_name: str, error: str) -> None:
    state["failed"][job_name] = {
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "error": error[:2000],
    }
    save_state(state)


# ---------------------------------------------------------------------------
# GPU/CUDA helpers
# ---------------------------------------------------------------------------
def flush_cuda() -> None:
    """Aggressively free VRAM between jobs."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1e9
            if allocated > 0.5:
                log(f"  WARNING: {allocated:.1f} GB still allocated after flush")
    except Exception:
        pass


def check_gpu() -> None:
    import torch
    if not torch.cuda.is_available():
        log("FATAL: CUDA not available")
        sys.exit(1)
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        log(f"  GPU {i}: {name} -- {total:.1f} GB VRAM")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def load_benchmark() -> list[dict]:
    items = []
    with open(BENCHMARK) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def split_thinking(text: str) -> tuple[str, str]:
    """Split reasoning model response into (thinking, answer)."""
    if "<think>" in text and "</think>" in text:
        ts = text.index("<think>") + len("<think>")
        te = text.index("</think>")
        return text[ts:te].strip(), text[te + len("</think>"):].strip()
    return "", text.strip()


def parse_answer(text: str, correct: str, lure: str) -> str:
    """Classify response as correct/lure/other.

    Uses the last 200 chars for reasoning models to avoid matching
    keywords in problem restatements within the CoT.
    """
    text = text.replace("\u0120", " ").replace("\u010a", "\n")
    text_lower = text.lower().strip()
    correct_lower = correct.lower().strip()
    lure_lower = lure.lower().strip() if lure else ""

    answer_region = text_lower[-200:] if len(text_lower) > 200 else text_lower

    if correct_lower in answer_region:
        return "correct"
    if lure_lower and lure_lower in answer_region:
        return "lure"
    if correct_lower in text_lower:
        return "correct"
    if lure_lower and lure_lower in text_lower:
        return "lure"
    return "other"


def load_model_and_tokenizer(
    model_id: str,
) -> tuple[Any, Any]:
    """Load a model in bfloat16 on CUDA. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"  Loading {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9
    log(f"  VRAM: {vram:.1f} GB")
    return model, tok


def unload_model(model: Any, tok: Any) -> None:
    """Delete model and tokenizer, flush CUDA."""
    del model, tok
    flush_cuda()


def run_behavioral(
    model: Any,
    tok: Any,
    items: list[dict],
    max_new_tokens: int,
    label: str,
) -> list[dict]:
    """Run behavioral validation on a list of items. Returns per-item results."""
    import torch

    results = []
    for i, item in enumerate(items):
        messages = [{"role": "user", "content": item["prompt"]}]
        if item.get("system_prompt"):
            messages.insert(0, {"role": "system", "content": item["system_prompt"]})

        enc = tok.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        )
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
            "response": response[:4000],
            "correct_answer": item["correct_answer"],
            "lure_answer": item.get("lure_answer", ""),
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(items):
            log(f"    [{label}] {i+1}/{len(items)} done")

    return results


def analyze_results(results: list[dict], label: str) -> dict[str, Any]:
    """Print and return per-category lure rate breakdown."""
    cats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "lure": 0, "correct": 0, "other": 0},
    )
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

    return dict(cats)


def save_behavioral_results(
    results: list[dict],
    model_id: str,
    label: str,
    filename: str,
) -> Path:
    """Save behavioral results JSON and return the path."""
    out_dir = PROJECT_ROOT / "results" / "behavioral"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with open(out_path, "w") as f:
        json.dump({
            "model": label,
            "model_id": model_id,
            "n_items": len(results),
            "results": results,
        }, f, indent=2)
    log(f"  Saved to {out_path}")
    return out_path


# ===================================================================
# JOB 1: Confidence paradigm (De Neys)
# ===================================================================
def job_confidence_paradigm() -> None:
    """Run confidence paradigm on Llama + R1-Distill via the existing script.

    The confidence_paradigm.py script is self-contained and handles model
    loading, inference with output_scores=True, and statistical analysis.
    """
    import subprocess

    models = [
        {
            "model": LLAMA_ID,
            "output": "results/confidence/llama_confidence.json",
            "max_new_tokens": "256",
            "label": "Llama",
        },
        {
            "model": R1_ID,
            "output": "results/confidence/r1_distill_confidence.json",
            "max_new_tokens": "2048",
            "label": "R1-Distill",
        },
    ]

    for m in models:
        log(f"  Confidence paradigm: {m['label']}")
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "confidence_paradigm.py"),
            "--model", m["model"],
            "--benchmark", str(BENCHMARK),
            "--output", m["output"],
            "--max-new-tokens", m["max_new_tokens"],
            "--cache-dir", CACHE_DIR,
        ]
        log(f"  CMD: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT),
            capture_output=True, text=True,
        )
        # Stream output to log
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                log(f"    {line}")
        if result.returncode != 0:
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-20:]:
                    log(f"    STDERR: {line}")
            raise RuntimeError(
                f"confidence_paradigm.py failed for {m['label']} "
                f"(exit code {result.returncode})"
            )

        log(f"  {m['label']} confidence paradigm complete")
        flush_cuda()


# ===================================================================
# JOB 2: Full 470-item behavioral on Llama + R1-Distill
# ===================================================================
def job_full_470_behavioral() -> None:
    """Run Llama and R1-Distill on the full 470-item benchmark.

    Existing results only cover 330 items. The expanded benchmark adds
    sunk_cost, natural_frequency, certainty_effect, and availability items.
    """
    items = load_benchmark()
    log(f"  Loaded {len(items)} benchmark items")

    models = [
        (LLAMA_ID, "Llama-3.1-8B-Instruct", 256, "llama31_8b_470.json"),
        (R1_ID, "R1-Distill-Llama-8B", 2048, "r1_distill_llama_470.json"),
    ]

    for model_id, label, max_tokens, filename in models:
        banner(f"  BEHAVIORAL: {label} on {len(items)} items")
        model, tok = load_model_and_tokenizer(model_id)

        results = run_behavioral(model, tok, items, max_tokens, label)
        analyze_results(results, label)
        save_behavioral_results(results, model_id, label, filename)

        unload_model(model, tok)


# ===================================================================
# JOB 3: Cross-model probe transfer (Llama <-> R1-Distill)
# ===================================================================
def job_cross_model_probe_transfer() -> None:
    """Train probe on Llama, test on R1-Distill (and vice versa).

    Tests whether the S1/S2 linear direction is shared across models that
    share the same base architecture (Llama backbone). This is CPU-only --
    reads from existing HDF5 activation files.
    """
    import h5py
    import numpy as np
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    # Verify activations exist
    for path, name in [(LLAMA_H5, "Llama"), (R1_H5, "R1-Distill")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} activation file not found: {path}\n"
                "Run extraction first."
            )

    def load_p0_and_labels(h5_path: Path) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
        """Load P0 activations for all layers + conflict labels + categories."""
        with h5py.File(str(h5_path), "r") as f:
            # Metadata
            conflict = f["/problems/conflict"][:].astype(np.int64)
            cats_raw = f["/problems/category"][:]
            categories = np.array([
                c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in cats_raw
            ])

            model_key = list(f["/models"].keys())[0]
            n_layers = int(f[f"/models/{model_key}/metadata"].attrs["n_layers"])

            # Position index
            labels_raw = f[f"/models/{model_key}/position_index/labels"][:]
            labels = [
                s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in labels_raw
            ]
            p0_idx = labels.index("P0")

            # Load P0 activations per layer
            acts = {}
            for layer in range(n_layers):
                arr = f[f"/models/{model_key}/residual/layer_{layer:02d}"][:]
                acts[layer] = arr[:, p0_idx, :].astype(np.float32)

        return acts, conflict, categories

    log("  Loading Llama activations...")
    llama_acts, llama_conflict, llama_cats = load_p0_and_labels(LLAMA_H5)
    log(f"  Llama: {len(llama_conflict)} items, {len(llama_acts)} layers")

    log("  Loading R1-Distill activations...")
    r1_acts, r1_conflict, r1_cats = load_p0_and_labels(R1_H5)
    log(f"  R1-Distill: {len(r1_conflict)} items, {len(r1_acts)} layers")

    # Align items by problem order (should be identical since both use same benchmark)
    n_items = min(len(llama_conflict), len(r1_conflict))
    if n_items < len(llama_conflict) or n_items < len(r1_conflict):
        log(f"  WARNING: item count mismatch (Llama={len(llama_conflict)}, R1={len(r1_conflict)}), using first {n_items}")

    # Filter to vulnerable categories for the transfer test
    vuln_mask = np.isin(llama_cats[:n_items], VULNERABLE_CATEGORIES)
    log(f"  Vulnerable items: {vuln_mask.sum()}")

    n_layers = min(len(llama_acts), len(r1_acts))
    transfer_results: dict[str, list[dict]] = {
        "llama_to_r1": [],
        "r1_to_llama": [],
        "llama_self": [],
        "r1_self": [],
    }

    for layer in range(n_layers):
        X_llama = llama_acts[layer][:n_items][vuln_mask]
        X_r1 = r1_acts[layer][:n_items][vuln_mask]
        y = llama_conflict[:n_items][vuln_mask]

        if len(np.unique(y)) < 2 or len(y) < 10:
            log(f"    Layer {layer:2d}: SKIPPED (insufficient data)")
            for key in transfer_results:
                transfer_results[key].append({"layer": layer, "auc": None, "skipped": True})
            continue

        # Standardize each model's activations independently
        scaler_llama = StandardScaler().fit(X_llama)
        scaler_r1 = StandardScaler().fit(X_r1)
        X_llama_s = scaler_llama.transform(X_llama).astype(np.float32)
        X_r1_s = scaler_r1.transform(X_r1).astype(np.float32)

        # Train on Llama, test on R1 (using full dataset -- this is a transfer test, not CV)
        # Use 5-fold CV to train on Llama, get pooled test-fold predictions for self-AUC
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Self-AUC via CV for both models
        for name, X_train_all, X_test_all in [
            ("llama_self", X_llama_s, X_llama_s),
            ("r1_self", X_r1_s, X_r1_s),
        ]:
            pooled_y, pooled_p = [], []
            for train_idx, test_idx in skf.split(X_train_all, y):
                clf = LogisticRegressionCV(
                    Cs=np.logspace(-4, 4, 15), cv=3, penalty="l2",
                    solver="lbfgs", max_iter=5000, class_weight="balanced",
                    scoring="roc_auc", random_state=42, n_jobs=1,
                )
                clf.fit(X_train_all[train_idx], y[train_idx])
                p = clf.predict_proba(X_test_all[test_idx])[:, 1]
                pooled_y.extend(y[test_idx])
                pooled_p.extend(p)
            auc = roc_auc_score(pooled_y, pooled_p)
            transfer_results[name].append({"layer": layer, "auc": round(auc, 4), "skipped": False})

        # Cross-model transfer: train on ALL of source, test on ALL of target
        # (No CV split needed -- different populations)
        for name, X_source, X_target in [
            ("llama_to_r1", X_llama_s, X_r1_s),
            ("r1_to_llama", X_r1_s, X_llama_s),
        ]:
            clf = LogisticRegressionCV(
                Cs=np.logspace(-4, 4, 15), cv=3, penalty="l2",
                solver="lbfgs", max_iter=5000, class_weight="balanced",
                scoring="roc_auc", random_state=42, n_jobs=1,
            )
            clf.fit(X_source, y)
            p_transfer = clf.predict_proba(X_target)[:, 1]
            auc_transfer = roc_auc_score(y, p_transfer)
            transfer_results[name].append({
                "layer": layer,
                "auc": round(auc_transfer, 4),
                "skipped": False,
            })

        llama_self_auc = transfer_results["llama_self"][-1]["auc"]
        r1_self_auc = transfer_results["r1_self"][-1]["auc"]
        l2r_auc = transfer_results["llama_to_r1"][-1]["auc"]
        r2l_auc = transfer_results["r1_to_llama"][-1]["auc"]
        log(f"    Layer {layer:2d}: Llama={llama_self_auc:.3f}  R1={r1_self_auc:.3f}  "
            f"L->R1={l2r_auc:.3f}  R1->L={r2l_auc:.3f}")

    # Summary: find best transfer layers
    log("\n  TRANSFER SUMMARY (vulnerable categories)")
    log(f"  {'Direction':<20} {'Peak AUC':>10} {'Peak Layer':>12}")
    log(f"  {'-'*20} {'-'*10} {'-'*12}")
    for name in transfer_results:
        valid = [e for e in transfer_results[name] if not e.get("skipped")]
        if valid:
            best = max(valid, key=lambda e: e["auc"])
            log(f"  {name:<20} {best['auc']:>10.3f} {best['layer']:>12d}")

    # Save
    out_dir = PROJECT_ROOT / "results" / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_model_transfer_llama_r1.json"
    with open(out_path, "w") as f:
        json.dump({
            "description": "Cross-model probe transfer: Llama <-> R1-Distill",
            "categories": VULNERABLE_CATEGORIES,
            "n_items": int(vuln_mask.sum()),
            "n_layers": n_layers,
            "results": transfer_results,
        }, f, indent=2)
    log(f"  Saved to {out_path}")


# ===================================================================
# JOB 4: SAE on R1-Distill activations
# ===================================================================
def job_sae_r1_distill() -> None:
    """Apply the Goodfire Llama-3.1-8B-Instruct SAE (L19) to R1-Distill activations.

    R1-Distill shares the Llama backbone, so the SAE may reconstruct reasonably.
    First checks reconstruction fidelity, then runs differential feature analysis
    between conflict vs control items, and compares to the existing Llama features.
    """
    import h5py
    import numpy as np
    import torch

    SAE_RELEASE = "Goodfire/Llama-3.1-8B-Instruct-SAE-l19"
    SAE_ID = "."
    LAYER = 19
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if not R1_H5.exists():
        raise FileNotFoundError(f"R1-Distill activations not found: {R1_H5}")

    # Load R1-Distill layer-19 P0 activations
    log("  Loading R1-Distill activations (layer 19, P0)...")
    with h5py.File(str(R1_H5), "r") as f:
        model_key = list(f["/models"].keys())[0]
        labels_raw = f[f"/models/{model_key}/position_index/labels"][:]
        labels = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in labels_raw]
        p0_idx = labels.index("P0")

        resid = f[f"/models/{model_key}/residual/layer_{LAYER:02d}"][:]
        p0 = resid[:, p0_idx, :].astype(np.float32)

        conflict = f["/problems/conflict"][:].astype(bool)
        cats_raw = f["/problems/category"][:]
        categories = np.array([
            c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in cats_raw
        ])

    log(f"  R1-Distill L19 P0: shape={p0.shape}, conflict={conflict.sum()}, control={len(conflict)-conflict.sum()}")

    # Load SAE (reuse strategy from run_sae_goodfire.py)
    log(f"  Loading Goodfire SAE: {SAE_RELEASE}...")
    try:
        import sae_lens
        sae_raw = sae_lens.SAE.from_pretrained(
            release=SAE_RELEASE, sae_id=SAE_ID, device=DEVICE,
        )
        if isinstance(sae_raw, tuple):
            sae_raw = sae_raw[0]
        sae = sae_raw
        n_features = int(sae.cfg.d_sae)
        hidden_dim = int(sae.cfg.d_in)
        log(f"  SAE loaded: n_features={n_features}, hidden_dim={hidden_dim}")
    except Exception as exc:
        log(f"  WARNING: sae_lens load failed ({exc}), trying HuggingFace download...")
        from huggingface_hub import hf_hub_download, list_repo_files
        repo_files = list_repo_files(SAE_RELEASE)
        weight_files = [f for f in repo_files if f.endswith((".safetensors", ".pt", ".pth", ".bin"))]
        if not weight_files:
            raise FileNotFoundError(f"No weight files in {SAE_RELEASE}")
        weight_path = hf_hub_download(repo_id=SAE_RELEASE, filename=weight_files[0])
        if weight_path.endswith(".safetensors"):
            import safetensors.torch
            state = safetensors.torch.load_file(weight_path, device=DEVICE)
        else:
            state = torch.load(weight_path, map_location=DEVICE, weights_only=False)
        # Minimal wrapper -- see run_sae_goodfire.py Strategy 3 for full logic
        raise RuntimeError(
            f"Manual SAE loading not implemented in overnight3. "
            f"Keys found: {list(state.keys())}. Use run_sae_goodfire.py pattern."
        )

    # Reconstruction fidelity check
    log("  Checking reconstruction fidelity on R1-Distill activations...")
    sample_idx = np.random.default_rng(42).choice(len(p0), min(100, len(p0)), replace=False)
    x_sample = torch.tensor(p0[sample_idx], device=DEVICE, dtype=torch.float32)
    with torch.no_grad():
        z_sample = sae.encode(x_sample)
        x_hat = sae.decode(z_sample)
    mse = torch.mean((x_sample - x_hat) ** 2).item()
    cos_sim = torch.nn.functional.cosine_similarity(x_sample, x_hat, dim=-1).mean().item()
    explained_var = 1.0 - mse / torch.var(x_sample).item()
    log(f"  Reconstruction: MSE={mse:.4f}, cos_sim={cos_sim:.4f}, explained_var={explained_var:.4f}")

    if explained_var < 0.5:
        log("  WARNING: Reconstruction fidelity is low (explained_var < 0.5).")
        log("  SAE may not transfer well to R1-Distill. Proceeding with caution.")

    # Encode all R1-Distill activations
    log("  Encoding all R1-Distill activations through SAE...")
    BATCH = 64
    all_z = []
    x_all = torch.tensor(p0, device=DEVICE, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(x_all), BATCH):
            z = sae.encode(x_all[start:start + BATCH])
            all_z.append(z.cpu().numpy())
    z_all = np.concatenate(all_z, axis=0)  # (n_problems, n_features)
    log(f"  SAE features: shape={z_all.shape}")

    # Differential analysis: conflict vs control for vulnerable categories
    from scipy.stats import mannwhitneyu
    vuln_mask = np.isin(categories, VULNERABLE_CATEGORIES)
    z_vuln = z_all[vuln_mask]
    c_vuln = conflict[vuln_mask]

    z_conflict = z_vuln[c_vuln]
    z_control = z_vuln[~c_vuln]
    log(f"  Vulnerable: {len(z_conflict)} conflict, {len(z_control)} control")

    # Per-feature Mann-Whitney U
    results_features = []
    for feat_idx in range(z_all.shape[1]):
        vals_conf = z_conflict[:, feat_idx]
        vals_ctrl = z_control[:, feat_idx]
        # Skip features that are zero everywhere
        if vals_conf.max() == 0 and vals_ctrl.max() == 0:
            continue
        # Skip features with no variance
        if vals_conf.std() == 0 and vals_ctrl.std() == 0:
            continue
        try:
            stat, p = mannwhitneyu(vals_conf, vals_ctrl, alternative="two-sided")
            effect = np.mean(vals_conf) - np.mean(vals_ctrl)
            results_features.append({
                "feature_idx": feat_idx,
                "p_value": float(p),
                "effect_size": float(effect),
                "mean_conflict": float(np.mean(vals_conf)),
                "mean_control": float(np.mean(vals_ctrl)),
                "U_statistic": float(stat),
            })
        except ValueError:
            continue

    log(f"  Tested {len(results_features)} non-trivial features")

    # BH-FDR correction
    if results_features:
        p_vals = np.array([r["p_value"] for r in results_features])
        n_tests = len(p_vals)
        sorted_idx = np.argsort(p_vals)
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(1, n_tests + 1)
        fdr_threshold = 0.05 * ranks / n_tests

        n_significant = 0
        for i, r in enumerate(results_features):
            r["bh_rank"] = int(ranks[i])
            r["bh_threshold"] = float(fdr_threshold[i])
            r["significant"] = bool(r["p_value"] <= fdr_threshold[i])
            if r["significant"]:
                n_significant += 1

        log(f"  BH-FDR significant features: {n_significant}/{len(results_features)}")

        # Top features by effect size
        sig_features = [r for r in results_features if r["significant"]]
        sig_features.sort(key=lambda r: abs(r["effect_size"]), reverse=True)
        log(f"\n  Top 10 significant features (R1-Distill, L19):")
        for r in sig_features[:10]:
            direction = "conflict>control" if r["effect_size"] > 0 else "control>conflict"
            log(f"    Feature {r['feature_idx']:>6d}: effect={r['effect_size']:+.4f} "
                f"p={r['p_value']:.2e} ({direction})")

    # Save results
    out_dir = PROJECT_ROOT / "results" / "sae"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "r1_distill_goodfire_l19.json"

    # Convert numpy types for JSON serialization
    def as_python(obj: Any) -> Any:
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): as_python(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [as_python(x) for x in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(as_python({
            "model": "R1-Distill-Llama-8B",
            "sae": SAE_RELEASE,
            "layer": LAYER,
            "reconstruction": {
                "mse": mse,
                "cosine_similarity": cos_sim,
                "explained_variance": explained_var,
                "n_sample": len(sample_idx),
            },
            "differential_features": {
                "categories": VULNERABLE_CATEGORIES,
                "n_conflict": int(c_vuln.sum()),
                "n_control": int(len(c_vuln) - c_vuln.sum()),
                "n_tested": len(results_features),
                "n_significant": n_significant if results_features else 0,
                "fdr_q": 0.05,
                "features": results_features,
            },
        }), f, indent=2)
    log(f"  Saved to {out_path}")

    # Clean up GPU memory from SAE
    del sae, x_all, z_all
    flush_cuda()


# ===================================================================
# JOB 5: OLMo bootstrap CIs
# ===================================================================
def job_olmo_bootstrap_cis() -> None:
    """Compute bootstrap CIs for OLMo probes via the existing script.

    CPU-only. Reads from OLMo activation HDF5 files.
    """
    import subprocess

    h5_paths = [
        ("olmo3_instruct", str(OLMO_INSTRUCT_H5)),
        ("olmo3_think", str(OLMO_THINK_H5)),
    ]

    for label, h5_path in h5_paths:
        if not Path(h5_path).exists():
            log(f"  WARNING: {h5_path} not found, skipping OLMo bootstrap for {label}")
            continue

        log(f"  Bootstrap CIs for {label}: {h5_path}")
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "compute_bootstrap_cis.py"),
            "--h5-path", h5_path,
            "--output-dir", "results/bootstrap_cis/",
            "--n-bootstrap", "1000",
        ]
        log(f"  CMD: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT),
            capture_output=True, text=True,
        )
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                log(f"    {line}")
        if result.returncode != 0:
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-20:]:
                    log(f"    STDERR: {line}")
            raise RuntimeError(
                f"compute_bootstrap_cis.py failed for {label} "
                f"(exit code {result.returncode})"
            )
        log(f"  {label} bootstrap CIs complete")


# ===================================================================
# JOB 6: Qwen on expanded 470-item benchmark
# ===================================================================
def job_qwen_expanded_behavioral() -> None:
    """Run Qwen3-8B (NO_THINK + THINK) on full 470-item benchmark.

    Existing results only cover 330 items. This tests the new categories
    (sunk_cost, natural_frequency, certainty_effect, availability).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = load_benchmark()
    log(f"  Loaded {len(items)} benchmark items")

    modes = [
        ("nothink", False, 256, "qwen3_8b_no_think_470.json"),
        ("think", True, 2048, "qwen3_8b_think_470.json"),
    ]

    # Load model once, run both modes
    log(f"  Loading {QWEN_ID}...")
    tok = AutoTokenizer.from_pretrained(QWEN_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9
    log(f"  VRAM: {vram:.1f} GB")

    for mode_name, enable_thinking, max_tokens, filename in modes:
        label = f"Qwen3-8B-{mode_name}"
        banner(f"  BEHAVIORAL: {label} on {len(items)} items")

        results = []
        for i, item in enumerate(items):
            messages = [{"role": "user", "content": item["prompt"]}]
            if item.get("system_prompt"):
                messages.insert(0, {"role": "system", "content": item["system_prompt"]})

            # Qwen3 uses enable_thinking kwarg in apply_chat_template
            input_text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            inputs = tok(input_text, return_tensors="pt").to("cuda")

            with torch.no_grad():
                out = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=1.0,
                )
            response = tok.decode(
                out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
            )
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
                "response": response[:4000],
                "correct_answer": item["correct_answer"],
                "lure_answer": item.get("lure_answer", ""),
                "thinking_mode": mode_name,
                "thinking_length": len(thinking),
            })
            if (i + 1) % 10 == 0 or (i + 1) == len(items):
                log(f"    [{label}] {i+1}/{len(items)} done")

        analyze_results(results, label)
        save_behavioral_results(results, QWEN_ID, label, filename)

    # Unload once after both modes
    del model, tok
    flush_cuda()


# ===================================================================
# Job registry
# ===================================================================
JOBS: list[tuple[str, str, callable]] = [
    ("confidence_paradigm", "De Neys confidence paradigm (Llama + R1)", job_confidence_paradigm),
    ("full_470_behavioral", "Full 470-item behavioral (Llama + R1)", job_full_470_behavioral),
    ("cross_model_transfer", "Cross-model probe transfer (Llama <-> R1)", job_cross_model_probe_transfer),
    ("sae_r1_distill", "SAE Goodfire L19 on R1-Distill", job_sae_r1_distill),
    ("olmo_bootstrap_cis", "OLMo bootstrap CIs", job_olmo_bootstrap_cis),
    ("qwen_expanded", "Qwen 470-item behavioral (NO_THINK + THINK)", job_qwen_expanded_behavioral),
]


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    banner("OVERNIGHT3 PIPELINE -- STARTING")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"State file:   {STATE_FILE}")
    log(f"Log file:     {LOG_FILE}")
    log(f"Python:       {sys.executable}")
    log(f"Start time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    check_gpu()

    state = load_state()

    # Determine jobs to run (skip already-completed)
    jobs_to_run: list[tuple[str, str, callable]] = []
    for name, description, fn in JOBS:
        if name in state["completed"]:
            prev = state["completed"][name]
            log(f"  ALREADY DONE: {name} (took {prev['elapsed_seconds']}s)")
        else:
            jobs_to_run.append((name, description, fn))

    if not jobs_to_run:
        banner("NO JOBS TO RUN -- all completed")
        return

    log(f"\nJobs queued ({len(jobs_to_run)}):")
    for name, description, _ in jobs_to_run:
        log(f"  - {name}: {description}")
    log("")

    # Execute
    t_pipeline = time.time()
    results_summary: list[dict[str, Any]] = []

    for i, (name, description, fn) in enumerate(jobs_to_run, 1):
        banner(f"JOB {i}/{len(jobs_to_run)}: {description} [{name}]")
        t_job = time.time()

        try:
            fn()
            elapsed = time.time() - t_job
            mark_completed(state, name, elapsed)
            status = "OK"
            log(f"  Completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
        except Exception as e:
            elapsed = time.time() - t_job
            tb = traceback.format_exc()
            mark_failed(state, name, tb)
            status = "FAILED"
            log(f"  FAILED after {elapsed:.0f}s: {e}")
            log(f"  Traceback:\n{tb}")

        results_summary.append({
            "job": name,
            "description": description,
            "status": status,
            "elapsed_seconds": round(elapsed, 1),
        })

        # Flush VRAM between all jobs
        flush_cuda()

    # Final report
    total_elapsed = time.time() - t_pipeline
    banner("OVERNIGHT3 PIPELINE COMPLETE")
    log(f"Total wall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min, {total_elapsed/3600:.1f} hr)")
    log("")
    log(f"{'Job':<28} {'Status':<10} {'Time':>10}")
    log(f"{'-'*28} {'-'*10} {'-'*10}")
    for r in results_summary:
        t_str = f"{r['elapsed_seconds']:.0f}s"
        log(f"{r['job']:<28} {r['status']:<10} {t_str:>10}")

    n_ok = sum(1 for r in results_summary if r["status"] == "OK")
    n_fail = sum(1 for r in results_summary if r["status"] == "FAILED")
    log(f"\n{n_ok} succeeded, {n_fail} failed out of {len(results_summary)} jobs")

    if n_fail > 0:
        log("\nFailed jobs can be retried by re-running this script.")
        log("Completed jobs will be skipped automatically.")
        sys.exit(1)


if __name__ == "__main__":
    main()
