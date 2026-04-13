#!/usr/bin/env python3
"""SAE cross-model analysis: Goodfire Llama L19 SAE on R1-Distill-Llama activations.

Tests whether an SAE trained on Llama-3.1-8B-Instruct reconstructs
DeepSeek-R1-Distill-Llama-8B activations decently. If EV >= 50%, runs
differential activation analysis and compares to the 41 significant
features found in Llama.

Rationale: R1-Distill-Llama shares the Llama architecture and was
distilled from it. If the SAE transfers, we can ask which of Llama's
S1/S2 features are also differentially active in R1-Distill — evidence
that reasoning distillation preserved (or altered) these circuits.

Usage
-----
    python scripts/run_sae_r1.py

Saves results to results/sae/r1_distill_goodfire_l19/.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

H5_PATH = _ROOT / "data" / "activations" / "r1_distill_llama.h5"
LAYER = 19
SAE_RELEASE = "Goodfire/Llama-3.1-8B-Instruct-SAE-l19"
VULNERABLE_CATEGORIES = ("base_rate", "conjunction", "syllogism")
FDR_Q = 0.05
TOP_K_FALSIFY = 50
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
OUT_DIR = _ROOT / "results" / "sae" / "r1_distill_goodfire_l19"
LLAMA_SIG_CSV = _ROOT / "results" / "sae" / "llama31_goodfire_l19" / "significant_features.csv"
MIN_EV = 0.50  # abort differential analysis below this

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

try:
    from s1s2.utils.logging import get_logger
    log = get_logger("run_sae_r1")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("run_sae_r1")


def _banner(msg: str) -> None:
    sep = "=" * 72
    log.info(sep)
    log.info(msg)
    log.info(sep)


# ---------------------------------------------------------------------------
# 1. Load R1-Distill activations from HDF5
# ---------------------------------------------------------------------------

def load_activations() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load layer-19 P0 activations and problem metadata from R1-Distill HDF5.

    Returns
    -------
    p0_activations : (n_problems, hidden_dim) float32
    conflict : (n_problems,) bool
    categories : (n_problems,) str array
    prompts : list of prompt strings
    """
    import h5py

    _banner(f"Loading R1-Distill activations from {H5_PATH}")
    if not H5_PATH.exists():
        raise FileNotFoundError(f"Activation file not found: {H5_PATH}")

    with h5py.File(str(H5_PATH), "r") as f:
        # Auto-detect model key
        available_models = list(f["/models"].keys())
        if len(available_models) == 0:
            raise KeyError("No models found in HDF5 /models group")
        model_key = available_models[0]
        log.info("Model key from HDF5: %s", model_key)

        # Problem metadata
        conflict = f["/problems/conflict"][:].astype(bool)
        categories_raw = f["/problems/category"][:]
        categories = np.array(
            [c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else str(c)
             for c in categories_raw]
        )
        prompts_raw = f["/problems/prompt_text"][:]
        prompts = [
            p.decode("utf-8") if isinstance(p, (bytes, bytearray)) else str(p)
            for p in prompts_raw
        ]

        # Residual stream at layer 19
        resid_key = f"/models/{model_key}/residual/layer_{LAYER:02d}"
        if resid_key not in f:
            available = list(f[f"/models/{model_key}/residual"].keys())
            raise KeyError(f"Layer key {resid_key} not in HDF5. Available: {available}")
        resid = f[resid_key][:]  # (n_problems, n_positions, hidden_dim)

        # P0 position
        labels = f[f"/models/{model_key}/position_index/labels"][:]
        labels_str = [
            lb.decode("utf-8") if isinstance(lb, (bytes, bytearray)) else str(lb)
            for lb in labels
        ]
        if "P0" not in labels_str:
            raise ValueError(f"P0 not in position labels: {labels_str}")
        p0_idx = labels_str.index("P0")
        p0_activations = resid[:, p0_idx, :].astype(np.float32)

    log.info(
        "Loaded %d problems, hidden_dim=%d",
        p0_activations.shape[0], p0_activations.shape[1],
    )
    log.info("Categories: %s", dict(zip(*np.unique(categories, return_counts=True))))
    log.info("Conflict: %d, Control: %d", conflict.sum(), (~conflict).sum())

    return p0_activations, conflict, categories, prompts


# ---------------------------------------------------------------------------
# 2. Load Goodfire SAE (raw HF weights -- bypasses sae_lens/transformer_lens)
# ---------------------------------------------------------------------------

def load_goodfire_sae():
    """Load Goodfire Llama-3.1-8B-Instruct SAE for layer 19 via raw weights.

    Uses Strategy 3 from run_sae_goodfire.py (direct HF download + manual
    weight loading). This avoids the sae_lens/transformer_lens dependency
    that caused the overnight pipeline failure.
    """
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files

    _banner(f"Loading Goodfire SAE: {SAE_RELEASE} (device={DEVICE})")
    log.info("Using raw HuggingFace weight loading (no sae_lens).")

    repo_files = list_repo_files(SAE_RELEASE)
    weight_candidates = [
        f for f in repo_files
        if f.endswith((".pth", ".pt", ".safetensors", ".bin"))
        and not f.startswith(".")
    ]
    log.info("Weight file candidates: %s", weight_candidates)

    if not weight_candidates:
        raise FileNotFoundError(
            f"No weight files in {SAE_RELEASE}. Repo contains: {repo_files}"
        )

    weight_file = hf_hub_download(repo_id=SAE_RELEASE, filename=weight_candidates[0])
    log.info("Downloaded: %s", weight_file)

    if weight_file.endswith(".safetensors"):
        import safetensors.torch
        state = safetensors.torch.load_file(weight_file, device=DEVICE)
    else:
        state = torch.load(weight_file, map_location=DEVICE, weights_only=False)

    log.info("Weight keys: %s", list(state.keys()))
    for k, v in state.items():
        if hasattr(v, "shape"):
            log.info("  %s: shape=%s dtype=%s", k, v.shape, v.dtype)

    # Parse encoder/decoder weights
    W_enc = W_dec = b_enc = b_dec = None
    for k, v in state.items():
        kl = k.lower()
        if ("enc" in kl and "weight" in kl) or kl == "w_enc":
            W_enc = v
        elif ("dec" in kl and "weight" in kl) or kl == "w_dec":
            W_dec = v
        elif ("enc" in kl and "bias" in kl) or kl == "b_enc":
            b_enc = v
        elif ("dec" in kl and "bias" in kl) or kl == "b_dec":
            b_dec = v

    if W_enc is None or W_dec is None:
        raise KeyError(f"Cannot identify encoder/decoder weights from keys: {list(state.keys())}")

    # nn.Linear convention: weight is (out_features, in_features)
    # encoder: (n_features, hidden_dim) -> we want (hidden_dim, n_features) for x @ W_enc
    if W_enc.shape[0] > W_enc.shape[1]:
        log.info("Transposing encoder: %s", W_enc.shape)
        W_enc = W_enc.T
    hidden_dim = int(W_enc.shape[0])
    n_features = int(W_enc.shape[1])

    # decoder: (hidden_dim, n_features) -> we want (n_features, hidden_dim) for z @ W_dec
    if W_dec.shape[0] < W_dec.shape[1]:
        log.info("Transposing decoder: %s", W_dec.shape)
        W_dec = W_dec.T

    if b_enc is None:
        b_enc = torch.zeros(n_features, device=DEVICE)
    if b_dec is None:
        b_dec = torch.zeros(hidden_dim, device=DEVICE)

    log.info("SAE architecture: hidden_dim=%d, n_features=%d", hidden_dim, n_features)

    class _RawHandle:
        def __init__(self) -> None:
            self.layer = LAYER
            self.hidden_dim = hidden_dim
            self.n_features = n_features
            self._W_enc = W_enc.float().to(DEVICE)
            self._W_dec = W_dec.float().to(DEVICE)
            self._b_enc = b_enc.float().to(DEVICE)
            self._b_dec = b_dec.float().to(DEVICE)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            x_dev = x.float().to(DEVICE)
            z = x_dev @ self._W_enc + self._b_enc
            return torch.relu(z).cpu()

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            z_dev = z.float().to(DEVICE)
            return (z_dev @ self._W_dec + self._b_dec).cpu()

        def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
            return self.decode(self.encode(x))

    handle = _RawHandle()
    log.info("Loaded SAE: n_features=%d, hidden_dim=%d", handle.n_features, handle.hidden_dim)
    return handle


# ---------------------------------------------------------------------------
# 3. Reconstruction fidelity
# ---------------------------------------------------------------------------

def check_fidelity(sae, activations: np.ndarray) -> dict:
    """Compute reconstruction explained variance, MSE, and L0."""
    import torch

    _banner("Reconstruction fidelity check (R1-Distill activations through Llama SAE)")

    rng = np.random.default_rng(0)
    n = min(512, activations.shape[0])
    idx = rng.choice(activations.shape[0], size=n, replace=False)
    sample = activations[idx].astype(np.float32)
    x = torch.from_numpy(sample)

    with torch.no_grad():
        z = sae.encode(x)
        x_hat = sae.decode(z)
        err = (x - x_hat).float()
        mse = float(err.pow(2).mean().item())
        var = float(x.float().var(unbiased=False).item())
        resid_var = float(err.var(unbiased=False).item())
        ev = 1.0 - (resid_var / var) if var > 0 else 0.0
        l0 = float((z.abs() > 1e-6).float().sum(dim=-1).mean().item())

    is_poor = ev < MIN_EV
    log.info("Explained variance: %.4f (threshold: %.2f)", ev, MIN_EV)
    log.info("MSE: %.6f, Var: %.6f", mse, var)
    log.info("Mean L0: %.1f", l0)
    log.info("Poor fit: %s", is_poor)
    if is_poor:
        log.warning(
            "POOR FIT: EV %.3f < %.2f. Differential analysis will be SKIPPED.",
            ev, MIN_EV,
        )

    return {
        "n_samples": int(n),
        "hidden_dim": int(activations.shape[1]),
        "mse": mse,
        "variance": var,
        "explained_variance": ev,
        "mean_l0": l0,
        "is_poor_fit": is_poor,
    }


# ---------------------------------------------------------------------------
# 4. Filter to vulnerable categories and encode
# ---------------------------------------------------------------------------

def filter_and_encode(
    sae,
    activations: np.ndarray,
    conflict: np.ndarray,
    categories: np.ndarray,
    prompts: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Filter to vulnerable categories and encode through SAE."""
    import torch

    _banner(f"Filtering to categories: {VULNERABLE_CATEGORIES}")

    mask = np.isin(categories, VULNERABLE_CATEGORIES)
    n_keep = mask.sum()
    log.info("Keeping %d / %d problems", n_keep, len(categories))
    if n_keep == 0:
        raise ValueError(
            f"No problems match {VULNERABLE_CATEGORIES}. "
            f"Available: {np.unique(categories).tolist()}"
        )

    activations_filtered = activations[mask]
    conflict_filtered = conflict[mask]
    prompts_filtered = [p for p, m in zip(prompts, mask) if m]
    cats_filtered = categories[mask]

    for cat in VULNERABLE_CATEGORIES:
        n_cat = (cats_filtered == cat).sum()
        n_s1 = ((cats_filtered == cat) & conflict_filtered).sum()
        n_s2 = ((cats_filtered == cat) & ~conflict_filtered).sum()
        log.info("  %s: %d total (%d conflict, %d control)", cat, n_cat, n_s1, n_s2)

    _banner("Encoding activations through SAE")
    n = activations_filtered.shape[0]
    batch_size = 128
    chunks = []
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        x = torch.from_numpy(activations_filtered[start:end].astype(np.float32))
        with torch.no_grad():
            z = sae.encode(x)
        chunks.append(z.detach().float().cpu().numpy())
    feature_activations = np.concatenate(chunks, axis=0)

    log.info(
        "Feature activations shape: %s (%.1f MB)",
        feature_activations.shape, feature_activations.nbytes / 1e6,
    )
    frac_zero = (feature_activations == 0).mean()
    log.info("Feature sparsity: %.2f%% zeros", 100.0 * frac_zero)

    return feature_activations, conflict_filtered, prompts_filtered, activations_filtered


# ---------------------------------------------------------------------------
# 5. Differential activation analysis
# ---------------------------------------------------------------------------

def run_differential(
    feature_activations: np.ndarray,
    conflict: np.ndarray,
) -> dict:
    """Mann-Whitney U + BH-FDR on all features."""
    from scipy import stats as sp_stats

    _banner(f"Differential activation analysis (BH-FDR q={FDR_Q})")

    n_problems, n_features = feature_activations.shape
    s1_idx = np.where(conflict)[0]
    s2_idx = np.where(~conflict)[0]
    n_s1, n_s2 = len(s1_idx), len(s2_idx)
    log.info("S1 (conflict): %d, S2 (control): %d", n_s1, n_s2)

    act_s1 = feature_activations[s1_idx]
    act_s2 = feature_activations[s2_idx]
    mean_s1 = act_s1.mean(axis=0)
    mean_s2 = act_s2.mean(axis=0)
    eps = 1e-6
    log_fc = np.log2((mean_s1 + eps) / (mean_s2 + eps))

    p_values = np.ones(n_features, dtype=np.float64)
    u_stats = np.zeros(n_features, dtype=np.float64)
    effect_sizes = np.zeros(n_features, dtype=np.float64)

    col_max = feature_activations.max(axis=0)
    col_min = feature_activations.min(axis=0)
    is_constant = col_max == col_min

    for i in range(n_features):
        if is_constant[i]:
            u_stats[i] = 0.5 * n_s1 * n_s2
            p_values[i] = 1.0
            continue
        try:
            u, p = sp_stats.mannwhitneyu(
                act_s1[:, i], act_s2[:, i], alternative="two-sided"
            )
        except ValueError:
            u, p = 0.5 * n_s1 * n_s2, 1.0
        u_stats[i] = float(u)
        p_values[i] = float(p)
        nm = n_s1 * n_s2
        effect_sizes[i] = float(1.0 - 2.0 * u / nm) if nm > 0 else 0.0

    # BH-FDR
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    inv_rank = np.arange(1, n + 1, dtype=np.float64)
    raw = ranked * n / inv_rank
    raw = np.minimum.accumulate(raw[::-1])[::-1]
    q_values = np.empty_like(raw)
    q_values[order] = np.clip(raw, 0.0, 1.0)
    rejected = q_values <= FDR_Q

    import pandas as pd
    df = pd.DataFrame({
        "feature_id": np.arange(n_features, dtype=np.int64),
        "mean_S1": mean_s1.astype(np.float64),
        "mean_S2": mean_s2.astype(np.float64),
        "log_fc": log_fc.astype(np.float64),
        "effect_size": effect_sizes,
        "u_stat": u_stats,
        "p_value": p_values,
        "q_value": q_values,
        "significant": rejected.astype(bool),
    })

    n_sig = int(df["significant"].sum())
    log.info("Significant features after BH-FDR: %d / %d", n_sig, n_features)

    sig_df = df[df["significant"]].copy()
    sig_df["abs_log_fc"] = sig_df["log_fc"].abs()
    top10 = sig_df.nlargest(10, "abs_log_fc")
    log.info("Top 10 R1 features by |log2 fold change|:")
    for _, row in top10.iterrows():
        log.info(
            "  feature %d: log_fc=%.3f, effect=%.3f, q=%.2e",
            int(row["feature_id"]), row["log_fc"], row["effect_size"], row["q_value"],
        )

    return {"df": df, "n_significant": n_sig, "n_total": n_features}


# ---------------------------------------------------------------------------
# 6. Ma et al. falsification
# ---------------------------------------------------------------------------

def run_falsification(
    sae,
    diff_df,
    activations: np.ndarray,
    feature_activations: np.ndarray,
    prompts: list[str],
) -> tuple[list[dict], int]:
    """Cheap falsification on top candidates."""
    import torch

    _banner(f"Ma et al. falsification (top {TOP_K_FALSIFY} candidates)")

    sig_df = diff_df[diff_df["significant"]].copy()
    if len(sig_df) == 0:
        log.warning("No significant features to falsify.")
        return [], 0

    sig_df["abs_log_fc"] = sig_df["log_fc"].abs()
    candidates = sig_df.nlargest(TOP_K_FALSIFY, "abs_log_fc")["feature_id"].tolist()
    candidates = [int(c) for c in candidates]
    log.info("Falsifying %d candidate features", len(candidates))

    result_dicts: list[dict] = []
    n_spurious = 0
    mean_residual = activations.mean(axis=0)

    for fid in candidates:
        col = feature_activations[:, fid]
        mean_orig = float(col.mean())
        peak_orig = float(col.max())

        top_prob_idx = np.argsort(-col)[:10]
        token_scores: dict[str, float] = {}
        stoplist = {
            "a", "an", "the", "and", "or", "of", "in", "to", "is", "are",
            "was", "were", "it", "this", "that", "for", "on", "at", "by",
            "with", "as", "be", "has", "have", "had", "but", "not", "if",
            ",", ".", "?", "!", ":", ";",
        }
        for p_idx in top_prob_idx:
            toks = prompts[int(p_idx)].split()
            w = float(col[int(p_idx)])
            for t in toks:
                t_clean = t.strip().lower()
                if t_clean not in stoplist and len(t_clean) > 1:
                    token_scores[t_clean] = token_scores.get(t_clean, 0.0) + w
        triggers = sorted(token_scores, key=lambda k: -token_scores[k])[:5]

        has_trigger = np.array(
            [any(t in prompts[i].lower() for t in triggers)
             for i in range(len(prompts))],
            dtype=bool,
        )
        trigger_centroid = (
            activations[has_trigger].mean(axis=0) if has_trigger.sum() > 0
            else mean_residual
        )

        simulated = np.stack(
            [0.5 * mean_residual + 0.5 * trigger_centroid for _ in range(100)]
        )
        rng = np.random.default_rng(fid + 13)
        jitter = rng.standard_normal(simulated.shape).astype(np.float32)
        jitter *= 0.01 * np.linalg.norm(mean_residual) / (np.sqrt(simulated.shape[1]) + 1e-8)
        simulated = simulated + jitter

        with torch.no_grad():
            z_sim = sae.encode(torch.from_numpy(simulated.astype(np.float32)))
        sim_col = z_sim[:, fid].detach().float().cpu().numpy()
        mean_random = float(sim_col.mean())
        peak_random = float(sim_col.max())
        denom = peak_orig if peak_orig > 1e-8 else 1.0
        ratio = mean_random / denom
        is_spurious = ratio >= 0.5

        if is_spurious:
            n_spurious += 1

        result_dicts.append({
            "feature_id": fid,
            "is_spurious": is_spurious,
            "trigger_tokens": triggers,
            "mean_activation_on_original": mean_orig,
            "peak_activation_on_original": peak_orig,
            "mean_activation_on_random": mean_random,
            "peak_activation_on_random": peak_random,
            "falsification_ratio": ratio,
            "mode": "cheap",
        })

    n_survived = len(result_dicts) - n_spurious
    log.info("Falsified: %d / %d flagged as spurious", n_spurious, len(result_dicts))
    log.info("Surviving: %d / %d", n_survived, len(result_dicts))
    return result_dicts, n_survived


# ---------------------------------------------------------------------------
# 7. Cross-model comparison: Llama's 41 features vs R1-Distill
# ---------------------------------------------------------------------------

def compare_to_llama(r1_diff_df, r1_falsification: list[dict]) -> dict:
    """Load Llama's significant features and check which are also significant in R1."""
    import pandas as pd

    _banner("Cross-model comparison: Llama's 41 features vs R1-Distill")

    if not LLAMA_SIG_CSV.exists():
        log.warning("Llama significant_features.csv not found at %s", LLAMA_SIG_CSV)
        return {"error": f"File not found: {LLAMA_SIG_CSV}"}

    llama_sig = pd.read_csv(LLAMA_SIG_CSV)
    llama_ids = set(llama_sig["feature_id"].astype(int).tolist())
    log.info("Llama significant features: %d", len(llama_ids))

    # Check R1 significance for each Llama feature
    r1_sig_ids = set(
        r1_diff_df[r1_diff_df["significant"]]["feature_id"].astype(int).tolist()
    )
    # Build falsification lookup
    fals_map = {r["feature_id"]: r.get("is_spurious", False) for r in r1_falsification}
    # R1 genuine = significant and not spurious
    r1_genuine_ids = {fid for fid in r1_sig_ids if not fals_map.get(fid, False)}

    shared_significant = llama_ids & r1_sig_ids
    shared_genuine = llama_ids & r1_genuine_ids
    llama_only = llama_ids - r1_sig_ids
    r1_only = r1_sig_ids - llama_ids

    log.info("Shared significant (Llama & R1): %d / %d", len(shared_significant), len(llama_ids))
    log.info("Shared genuine (surviving falsification): %d", len(shared_genuine))
    log.info("Llama-only (not significant in R1): %d", len(llama_only))
    log.info("R1-only (not significant in Llama): %d", len(r1_only))

    # Detailed per-feature comparison
    feature_details = []
    for fid in sorted(llama_ids):
        llama_row = llama_sig[llama_sig["feature_id"] == fid].iloc[0]
        r1_row = r1_diff_df[r1_diff_df["feature_id"] == fid]
        if len(r1_row) == 0:
            continue
        r1_row = r1_row.iloc[0]
        feature_details.append({
            "feature_id": int(fid),
            "llama_log_fc": float(llama_row["log_fc"]),
            "llama_effect_size": float(llama_row["effect_size"]),
            "llama_q_value": float(llama_row["q_value"]),
            "r1_log_fc": float(r1_row["log_fc"]),
            "r1_effect_size": float(r1_row["effect_size"]),
            "r1_q_value": float(r1_row["q_value"]),
            "r1_significant": bool(r1_row["significant"]),
            "r1_is_spurious": fals_map.get(int(fid), False),
            "direction_consistent": (
                np.sign(float(llama_row["log_fc"])) == np.sign(float(r1_row["log_fc"]))
            ),
        })

    # Direction consistency among shared significant features
    if shared_significant:
        consistent = sum(
            1 for d in feature_details
            if d["r1_significant"] and d["direction_consistent"]
        )
        inconsistent = len(shared_significant) - consistent
        log.info(
            "Direction consistency (shared sig): %d consistent, %d inconsistent",
            consistent, inconsistent,
        )
    else:
        consistent = inconsistent = 0

    # Print the comparison table
    log.info("\nPer-feature comparison (Llama's 41 features):")
    log.info(
        "%8s  %10s  %10s  %10s  %10s  %6s  %8s  %9s",
        "FeatID", "Llama_lfc", "R1_lfc", "Llama_q", "R1_q",
        "R1_sig", "R1_spur", "Dir_match",
    )
    for d in feature_details:
        log.info(
            "%8d  %10.3f  %10.3f  %10.2e  %10.2e  %6s  %8s  %9s",
            d["feature_id"],
            d["llama_log_fc"],
            d["r1_log_fc"],
            d["llama_q_value"],
            d["r1_q_value"],
            "YES" if d["r1_significant"] else "no",
            "YES" if d["r1_is_spurious"] else "no",
            "YES" if d["direction_consistent"] else "NO",
        )

    return {
        "n_llama_significant": len(llama_ids),
        "n_r1_significant": len(r1_sig_ids),
        "n_r1_genuine": len(r1_genuine_ids),
        "n_shared_significant": len(shared_significant),
        "n_shared_genuine": len(shared_genuine),
        "shared_feature_ids": sorted(shared_significant),
        "shared_genuine_ids": sorted(shared_genuine),
        "llama_only_ids": sorted(llama_only),
        "r1_only_ids": sorted(r1_only),
        "n_direction_consistent": consistent,
        "n_direction_inconsistent": inconsistent,
        "feature_details": feature_details,
    }


# ---------------------------------------------------------------------------
# 8. Volcano plot
# ---------------------------------------------------------------------------

def make_volcano(
    diff_df, falsification_results: list[dict], llama_ids: set[int], out_dir: Path,
) -> None:
    """Volcano plot with Llama's features highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _banner("Generating volcano plot")

    df = diff_df.copy()
    fals_map = {r["feature_id"]: r["is_spurious"] for r in falsification_results}
    df["is_falsified"] = df["feature_id"].map(lambda fid: fals_map.get(int(fid), False))
    df["is_llama_feature"] = df["feature_id"].isin(llama_ids)

    log_fc = df["log_fc"].to_numpy(dtype=np.float64)
    q = df["q_value"].to_numpy(dtype=np.float64)
    neg_log_q = -np.log10(np.clip(q, 1e-300, 1.0))
    sig = df["significant"].to_numpy(dtype=bool)
    is_fals = df["is_falsified"].to_numpy(dtype=bool)
    is_llama = df["is_llama_feature"].to_numpy(dtype=bool)
    genuine = sig & ~is_fals
    spurious = sig & is_fals
    nonsig = ~sig

    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
    ax.scatter(
        log_fc[nonsig], neg_log_q[nonsig],
        s=8, c="#cccccc", alpha=0.5, label=f"n.s. (q>{FDR_Q})",
    )
    if spurious.any():
        ax.scatter(
            log_fc[spurious], neg_log_q[spurious],
            s=24, c="#888888", marker="x", alpha=0.8, label="falsified",
        )
    if genuine.any():
        ax.scatter(
            log_fc[genuine], neg_log_q[genuine],
            s=32, c="#c03030", edgecolors="#400000", linewidths=0.4,
            label=f"R1 genuine (n={int(genuine.sum())})",
        )
    # Highlight Llama's features with rings
    llama_mask = is_llama & sig & ~is_fals
    if llama_mask.any():
        ax.scatter(
            log_fc[llama_mask], neg_log_q[llama_mask],
            s=80, facecolors="none", edgecolors="#0044cc", linewidths=1.5,
            label=f"Llama shared (n={int(llama_mask.sum())})",
        )
    # Llama features that are NOT significant in R1
    llama_nonsig = is_llama & ~sig
    if llama_nonsig.any():
        ax.scatter(
            log_fc[llama_nonsig], neg_log_q[llama_nonsig],
            s=40, facecolors="none", edgecolors="#0044cc", linewidths=0.8,
            linestyle="dashed", alpha=0.5,
            label=f"Llama n.s. in R1 (n={int(llama_nonsig.sum())})",
        )

    import math
    ax.axhline(-math.log10(max(FDR_Q, 1e-300)), ls="--", color="#555555", lw=0.8)
    ax.axvline(0.0, ls="--", color="#555555", lw=0.8)
    ax.set_xlabel("log2 fold change (conflict / control)")
    ax.set_ylabel(r"$-\log_{10}(q)$")
    ax.set_title(
        "R1-Distill-Llama through Goodfire Llama L19 SAE\n"
        f"Vulnerable categories ({', '.join(VULNERABLE_CATEGORIES)})"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "volcano_l19_r1.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    log.info("Volcano plot saved to %s", out_path)


# ---------------------------------------------------------------------------
# 9. Save results
# ---------------------------------------------------------------------------

def save_results(
    out_dir: Path,
    fidelity: dict,
    diff_df,
    n_significant: int,
    falsification_results: list[dict],
    n_surviving: int,
    comparison: dict,
    elapsed: float,
) -> None:
    """Save everything."""
    import pandas as pd

    _banner(f"Saving results to {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    diff_df.to_csv(out_dir / "differential_results.csv", index=False)
    log.info("Saved differential_results.csv (%d rows)", len(diff_df))

    sig_df = diff_df[diff_df["significant"]].copy()
    sig_df.to_csv(out_dir / "significant_features.csv", index=False)
    log.info("Saved significant_features.csv (%d rows)", len(sig_df))

    with open(out_dir / "falsification_results.json", "w") as f:
        json.dump(falsification_results, f, indent=2, default=str)

    with open(out_dir / "llama_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    summary = {
        "config": {
            "h5_path": str(H5_PATH),
            "layer": LAYER,
            "sae_release": SAE_RELEASE,
            "sae_trained_on": "Llama-3.1-8B-Instruct (cross-model transfer to R1-Distill)",
            "vulnerable_categories": list(VULNERABLE_CATEGORIES),
            "fdr_q": FDR_Q,
            "top_k_falsify": TOP_K_FALSIFY,
            "device": DEVICE,
        },
        "fidelity": fidelity,
        "n_features_in_sae": int(diff_df["feature_id"].max() + 1) if len(diff_df) > 0 else 0,
        "n_significant_after_fdr": n_significant,
        "n_falsification_candidates": len(falsification_results),
        "n_spurious": sum(1 for r in falsification_results if r.get("is_spurious", False)),
        "n_surviving_falsification": n_surviving,
        "llama_comparison": {
            "n_llama_significant": comparison.get("n_llama_significant", 0),
            "n_shared_significant": comparison.get("n_shared_significant", 0),
            "n_shared_genuine": comparison.get("n_shared_genuine", 0),
            "n_direction_consistent": comparison.get("n_direction_consistent", 0),
        },
        "elapsed_seconds": elapsed,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved summary.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()

    _banner("Goodfire L19 SAE cross-model analysis: R1-Distill-Llama-8B")
    log.info("Device: %s", DEVICE)
    log.info("Output: %s", OUT_DIR)

    # --- Step 1: Load R1-Distill activations ---
    p0_activations, conflict, categories, prompts = load_activations()

    # --- Step 2: Load SAE ---
    sae = load_goodfire_sae()
    print(f"\n>>> SAE: {sae.n_features} features, hidden_dim={sae.hidden_dim}")
    print(f">>> Activations: hidden_dim={p0_activations.shape[1]}")

    if sae.hidden_dim != p0_activations.shape[1]:
        log.error(
            "DIMENSION MISMATCH: SAE expects %d, activations have %d.",
            sae.hidden_dim, p0_activations.shape[1],
        )
        raise ValueError(
            f"SAE hidden_dim ({sae.hidden_dim}) != activation hidden_dim "
            f"({p0_activations.shape[1]})"
        )

    # --- Step 3: Reconstruction fidelity ---
    fidelity = check_fidelity(sae, p0_activations)
    ev = fidelity["explained_variance"]
    print(f"\n>>> Reconstruction EV: {ev:.4f}")

    if fidelity["is_poor_fit"]:
        # Save partial results and exit early
        print(f"\n>>> POOR FIT: EV={ev:.3f} < {MIN_EV}. Skipping differential analysis.")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        summary = {
            "config": {
                "h5_path": str(H5_PATH),
                "layer": LAYER,
                "sae_release": SAE_RELEASE,
                "sae_trained_on": "Llama-3.1-8B-Instruct (cross-model transfer)",
                "device": DEVICE,
            },
            "fidelity": fidelity,
            "aborted": True,
            "abort_reason": f"EV={ev:.4f} < {MIN_EV} threshold",
            "elapsed_seconds": time.time() - t0,
        }
        with open(OUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        log.info("Partial results saved to %s", OUT_DIR)
        return

    # --- Step 4: Filter + encode ---
    feature_activations, conflict_filt, prompts_filt, acts_filt = filter_and_encode(
        sae, p0_activations, conflict, categories, prompts
    )

    # --- Step 5: Differential analysis ---
    diff_result = run_differential(feature_activations, conflict_filt)
    diff_df = diff_result["df"]
    n_sig = diff_result["n_significant"]
    print(f"\n>>> R1-Distill significant features (q<={FDR_Q}): {n_sig}")

    # --- Step 6: Falsification ---
    falsification_results, n_surviving = run_falsification(
        sae, diff_df, acts_filt, feature_activations, prompts_filt
    )
    print(f"\n>>> Surviving falsification: {n_surviving}")

    # --- Step 7: Cross-model comparison ---
    comparison = compare_to_llama(diff_df, falsification_results)

    # --- Step 8: Volcano plot ---
    import pandas as pd
    llama_ids: set[int] = set()
    if LLAMA_SIG_CSV.exists():
        llama_sig = pd.read_csv(LLAMA_SIG_CSV)
        llama_ids = set(llama_sig["feature_id"].astype(int).tolist())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_volcano(diff_df, falsification_results, llama_ids, OUT_DIR)

    # --- Step 9: Save ---
    elapsed = time.time() - t0
    save_results(
        OUT_DIR, fidelity, diff_df, n_sig,
        falsification_results, n_surviving, comparison, elapsed,
    )

    # --- Final summary ---
    _banner("RESULTS SUMMARY")
    print(f"  SAE features:                           {sae.n_features}")
    print(f"  Reconstruction EV (R1 activations):     {ev:.4f}")
    print(f"  Reconstruction MSE:                     {fidelity['mse']:.6f}")
    print(f"  Mean L0:                                {fidelity['mean_l0']:.1f}")
    print(f"  R1 significant features (q<={FDR_Q}):     {n_sig}")
    n_spurious = sum(1 for r in falsification_results if r.get("is_spurious", False))
    print(f"  Falsification: {n_spurious} spurious, {n_surviving} surviving")
    print()
    n_shared = comparison.get("n_shared_significant", 0)
    n_shared_genuine = comparison.get("n_shared_genuine", 0)
    n_llama = comparison.get("n_llama_significant", 0)
    n_dir = comparison.get("n_direction_consistent", 0)
    print(f"  CROSS-MODEL COMPARISON:")
    print(f"  Llama significant features:             {n_llama}")
    print(f"  Shared significant (Llama & R1):        {n_shared} / {n_llama}")
    print(f"  Shared genuine (post-falsification):    {n_shared_genuine} / {n_llama}")
    print(f"  Direction consistent:                   {n_dir} / {n_shared}")
    if n_shared > 0:
        print(f"  Shared feature IDs: {comparison.get('shared_feature_ids', [])}")
    print(f"\n  Wall time: {elapsed:.1f}s")
    print()
    log.info("All results saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()
