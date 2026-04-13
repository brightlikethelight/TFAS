#!/usr/bin/env python3
"""Focused SAE analysis: Goodfire L19 SAE on Llama-3.1-8B-Instruct activations.

Self-contained script for the B200 pod. Loads activations from HDF5,
downloads the Goodfire SAE for layer 19, runs differential activation
analysis on the 3 vulnerable categories (base_rate, conjunction,
syllogism), applies BH-FDR correction, runs Ma et al. falsification
on top candidates, and produces a volcano plot.

Usage
-----
    python scripts/run_sae_goodfire.py

The script saves everything to results/sae/llama31_goodfire_l19/.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — make src/ importable even without pip install -e .
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

H5_PATH = _ROOT / "data" / "activations" / "llama31_8b_instruct.h5"
MODEL_KEY: str | None = None  # auto-detected from HDF5 at load time
LAYER = 19
SAE_RELEASE = "Goodfire/Llama-3.1-8B-Instruct-SAE-l19"
SAE_ID = "."  # Goodfire convention for single-SAE repos
VULNERABLE_CATEGORIES = ("base_rate", "conjunction", "syllogism")
FDR_Q = 0.05
TOP_K_FALSIFY = 50
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
OUT_DIR = _ROOT / "results" / "sae" / "llama31_goodfire_l19"

# ---------------------------------------------------------------------------
# Logging — use project infrastructure if available, else plain logging
# ---------------------------------------------------------------------------

try:
    from s1s2.utils.logging import get_logger
    log = get_logger("run_sae_goodfire")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("run_sae_goodfire")


def _banner(msg: str) -> None:
    sep = "=" * 72
    log.info(sep)
    log.info(msg)
    log.info(sep)


# ---------------------------------------------------------------------------
# 1. Load activations from HDF5
# ---------------------------------------------------------------------------

def load_activations() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load layer-19 P0 activations and problem metadata.

    Returns
    -------
    p0_activations : (n_problems, hidden_dim) float32
    conflict : (n_problems,) bool
    categories : (n_problems,) str array
    prompts : list of prompt strings
    """
    import h5py

    _banner(f"Loading activations from {H5_PATH}")
    if not H5_PATH.exists():
        raise FileNotFoundError(
            f"Activation file not found: {H5_PATH}\n"
            "Run extraction first or check the path."
        )

    with h5py.File(str(H5_PATH), "r") as f:
        # Auto-detect model key from HDF5 instead of hardcoding
        global MODEL_KEY
        available_models = list(f["/models"].keys())
        if len(available_models) == 0:
            raise KeyError("No models found in HDF5 /models group")
        MODEL_KEY = available_models[0]
        if len(available_models) > 1:
            log.warning(
                "Multiple models in HDF5: %s — using first: %s",
                available_models, MODEL_KEY,
            )
        log.info("Auto-detected model key from HDF5: %s", MODEL_KEY)

        # Problem metadata
        conflict = f["/problems/conflict"][:].astype(bool)
        categories_raw = f["/problems/category"][:]
        categories = np.array(
            [c.decode("utf-8") if isinstance(c, bytes | bytearray) else str(c)
             for c in categories_raw]
        )
        prompts_raw = f["/problems/prompt_text"][:]
        prompts = [
            p.decode("utf-8") if isinstance(p, bytes | bytearray) else str(p)
            for p in prompts_raw
        ]

        # Residual stream at layer 19
        resid_key = f"/models/{MODEL_KEY}/residual/layer_{LAYER:02d}"
        if resid_key not in f:
            raise KeyError(
                f"Layer key {resid_key} not in HDF5. "
                f"Available: {list(f[f'/models/{MODEL_KEY}/residual'].keys())}"
            )
        resid = f[resid_key][:]  # (n_problems, n_positions, hidden_dim)

        # Find P0 position index
        labels = f[f"/models/{MODEL_KEY}/position_index/labels"][:]
        labels_str = [
            lb.decode("utf-8") if isinstance(lb, bytes | bytearray) else str(lb)
            for lb in labels
        ]
        if "P0" not in labels_str:
            raise ValueError(f"P0 not in position labels: {labels_str}")
        p0_idx = labels_str.index("P0")
        p0_activations = resid[:, p0_idx, :].astype(np.float32)

    log.info("Loaded %d problems, hidden_dim=%d", p0_activations.shape[0], p0_activations.shape[1])
    log.info("Categories: %s", dict(zip(*np.unique(categories, return_counts=True), strict=False)))
    log.info("Conflict items: %d, Control items: %d", conflict.sum(), (~conflict).sum())

    return p0_activations, conflict, categories, prompts


# ---------------------------------------------------------------------------
# 2. Load the Goodfire L19 SAE
# ---------------------------------------------------------------------------

def load_goodfire_sae():
    """Load the Goodfire Llama-3.1-8B-Instruct SAE for layer 19.

    Tries the project's _SAELensHandle wrapper first for protocol
    compatibility, falls back to direct sae_lens loading.
    """
    import torch

    _banner(f"Loading Goodfire SAE: {SAE_RELEASE} (device={DEVICE})")

    # Strategy 1: Use the project's _SAELensHandle wrapper directly
    try:
        import sae_lens

        from s1s2.sae.loaders import _SAELensHandle

        log.info("Attempting sae_lens.SAE.from_pretrained(release=%s, sae_id=%s)", SAE_RELEASE, SAE_ID)
        sae_raw = sae_lens.SAE.from_pretrained(
            release=SAE_RELEASE, sae_id=SAE_ID, device=DEVICE
        )
        # from_pretrained may return (sae, cfg, sparsity) tuple or just the SAE
        if isinstance(sae_raw, tuple):
            sae_raw = sae_raw[0]
        handle = _SAELensHandle(sae_raw, layer=LAYER)
        log.info(
            "Loaded via _SAELensHandle: n_features=%d, hidden_dim=%d",
            handle.n_features, handle.hidden_dim,
        )
        return handle
    except Exception as exc:
        log.warning("Strategy 1 (_SAELensHandle) failed: %s", exc)

    # Strategy 2: Direct sae_lens, wrap manually
    try:
        import sae_lens

        log.info("Attempting direct sae_lens load...")
        sae_raw = sae_lens.SAE.from_pretrained(
            release=SAE_RELEASE, sae_id=SAE_ID, device=DEVICE
        )
        if isinstance(sae_raw, tuple):
            sae_raw = sae_raw[0]

        # Build a minimal duck-typed wrapper matching SAEHandle protocol
        class _DirectHandle:
            def __init__(self, sae, layer: int):
                self._sae = sae
                self.layer = layer
                self.hidden_dim = int(sae.cfg.d_in)
                self.n_features = int(sae.cfg.d_sae)

            def _to_device(self, x):
                dev = next(self._sae.parameters()).device
                dtype = next(self._sae.parameters()).dtype
                return x.to(device=dev, dtype=dtype)

            def encode(self, x):
                with torch.no_grad():
                    z = self._sae.encode(self._to_device(x))
                return z.detach().float().cpu()

            def decode(self, z):
                with torch.no_grad():
                    x_hat = self._sae.decode(self._to_device(z))
                return x_hat.detach().float().cpu()

            def reconstruct(self, x):
                return self.decode(self.encode(x))

        handle = _DirectHandle(sae_raw, layer=LAYER)
        log.info(
            "Loaded via direct sae_lens: n_features=%d, hidden_dim=%d",
            handle.n_features, handle.hidden_dim,
        )
        return handle
    except Exception as exc:
        log.warning("Strategy 2 (direct sae_lens) failed: %s", exc)

    # Strategy 3: Raw HuggingFace download + manual weight loading
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        log.info("Attempting raw HuggingFace download of SAE weights...")

        # List repo files and find the weight file dynamically
        repo_files = list_repo_files(SAE_RELEASE)
        weight_candidates = [
            f for f in repo_files
            if f.endswith((".pth", ".pt", ".safetensors", ".bin"))
            and not f.startswith(".")
        ]
        log.info("Weight file candidates in repo: %s", weight_candidates)

        if not weight_candidates:
            raise FileNotFoundError(
                f"No weight files (.pth/.pt/.safetensors/.bin) in {SAE_RELEASE}. "
                f"Repo contains: {repo_files}"
            )

        weight_file = hf_hub_download(repo_id=SAE_RELEASE, filename=weight_candidates[0])
        log.info("Downloaded weight file: %s", weight_file)

        # Load weights
        if weight_file.endswith(".safetensors"):
            import safetensors.torch
            state = safetensors.torch.load_file(weight_file, device=DEVICE)
        else:
            state = torch.load(weight_file, map_location=DEVICE, weights_only=False)

        log.info("Loaded raw weights, keys: %s", list(state.keys()))
        for k, v in state.items():
            if hasattr(v, "shape"):
                log.info("  %s: shape=%s dtype=%s", k, v.shape, v.dtype)

        # Infer dimensions from weight shapes.
        # Goodfire SAEs use nn.Linear naming: encoder_linear.weight, decoder_linear.weight
        # nn.Linear stores weight as (out_features, in_features), so:
        #   encoder_linear.weight: (n_features, hidden_dim)
        #   decoder_linear.weight: (hidden_dim, n_features)
        W_enc = None
        W_dec = None
        b_enc = None
        b_dec = None
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
            log.warning("Could not identify W_enc/W_dec from keys: %s", list(state.keys()))
            raise KeyError("Cannot identify encoder/decoder weights")

        # Handle nn.Linear convention: weight is (out_features, in_features).
        # encoder: out=n_features, in=hidden_dim -> shape (n_features, hidden_dim)
        # We need W_enc as (hidden_dim, n_features) for matmul: x @ W_enc
        if W_enc.shape[0] > W_enc.shape[1]:
            # (n_features, hidden_dim) -> transpose to (hidden_dim, n_features)
            log.info("Transposing encoder weight from nn.Linear convention: %s", W_enc.shape)
            W_enc = W_enc.T
        hidden_dim = int(W_enc.shape[0])
        n_features = int(W_enc.shape[1])

        # decoder: out=hidden_dim, in=n_features -> shape (hidden_dim, n_features)
        # We need W_dec as (n_features, hidden_dim) for matmul: z @ W_dec
        if W_dec.shape[0] < W_dec.shape[1]:
            # (hidden_dim, n_features) -> transpose to (n_features, hidden_dim)
            log.info("Transposing decoder weight from nn.Linear convention: %s", W_dec.shape)
            W_dec = W_dec.T

        if b_enc is None:
            b_enc = torch.zeros(n_features, device=DEVICE)
        if b_dec is None:
            b_dec = torch.zeros(hidden_dim, device=DEVICE)

        log.info(
            "Inferred SAE architecture: hidden_dim=%d, n_features=%d", hidden_dim, n_features
        )

        class _RawHandle:
            def __init__(self):
                self.layer = LAYER
                self.hidden_dim = hidden_dim
                self.n_features = n_features
                self._W_enc = W_enc.float().to(DEVICE)
                self._W_dec = W_dec.float().to(DEVICE)
                self._b_enc = b_enc.float().to(DEVICE)
                self._b_dec = b_dec.float().to(DEVICE)

            def encode(self, x):
                x_dev = x.float().to(DEVICE)
                z = x_dev @ self._W_enc + self._b_enc
                return torch.relu(z).cpu()

            def decode(self, z):
                z_dev = z.float().to(DEVICE)
                return (z_dev @ self._W_dec + self._b_dec).cpu()

            def reconstruct(self, x):
                return self.decode(self.encode(x))

        handle = _RawHandle()
        log.info(
            "Loaded via raw weights: n_features=%d, hidden_dim=%d",
            handle.n_features, handle.hidden_dim,
        )
        return handle
    except Exception as exc:
        log.error("Strategy 3 (raw HF weights) also failed: %s", exc)
        raise RuntimeError(
            f"All SAE loading strategies failed for {SAE_RELEASE}. "
            "Check the HuggingFace repo page for the correct loading API."
        ) from exc


# ---------------------------------------------------------------------------
# 3. Reconstruction fidelity check
# ---------------------------------------------------------------------------

def check_fidelity(sae, activations: np.ndarray) -> dict:
    """Run reconstruction fidelity and return the report as a dict."""
    _banner("Reconstruction fidelity check")

    try:
        from s1s2.sae.loaders import reconstruction_report
        report = reconstruction_report(
            sae, activations, min_explained_variance=0.5, n_samples=512
        )
        log.info("Explained variance: %.4f", report.explained_variance)
        log.info("MSE: %.6f, Var: %.6f", report.mse, report.variance)
        log.info("Mean L0: %.1f", report.mean_l0)
        log.info("Poor fit: %s", report.is_poor_fit)
        return asdict(report)
    except ImportError:
        log.warning("Could not import reconstruction_report; computing manually.")

    import torch

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

    is_poor = ev < 0.5
    log.info("Explained variance: %.4f", ev)
    log.info("MSE: %.6f, Var: %.6f", mse, var)
    log.info("Mean L0: %.1f", l0)
    log.info("Poor fit: %s", is_poor)
    if is_poor:
        log.warning(
            "POOR FIT: explained variance %.3f < 0.5. "
            "Downstream features may be UNTRUSTWORTHY.", ev
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
    """Filter to vulnerable categories, encode through SAE.

    Returns
    -------
    feature_activations : (n_filtered, n_features)
    conflict_filtered : (n_filtered,) bool
    prompts_filtered : list of strings
    activations_filtered : (n_filtered, hidden_dim) -- raw residuals for falsification
    """
    _banner(f"Filtering to categories: {VULNERABLE_CATEGORIES}")

    mask = np.isin(categories, VULNERABLE_CATEGORIES)
    n_keep = mask.sum()
    log.info("Keeping %d / %d problems", n_keep, len(categories))
    if n_keep == 0:
        raise ValueError(
            f"No problems match categories {VULNERABLE_CATEGORIES}. "
            f"Available: {np.unique(categories).tolist()}"
        )

    activations_filtered = activations[mask]
    conflict_filtered = conflict[mask]
    prompts_filtered = [p for p, m in zip(prompts, mask, strict=False) if m]
    cats_filtered = categories[mask]

    for cat in VULNERABLE_CATEGORIES:
        n_cat = (cats_filtered == cat).sum()
        n_s1 = ((cats_filtered == cat) & conflict_filtered).sum()
        n_s2 = ((cats_filtered == cat) & ~conflict_filtered).sum()
        log.info("  %s: %d total (%d conflict, %d control)", cat, n_cat, n_s1, n_s2)

    _banner("Encoding activations through SAE")
    try:
        from s1s2.sae.differential import encode_batched
        feature_activations = encode_batched(sae, activations_filtered, batch_size=128)
    except ImportError:
        log.warning("Could not import encode_batched; encoding manually.")
        import torch
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
        feature_activations.shape,
        feature_activations.nbytes / 1e6,
    )
    # Sparsity stats
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
    """Run Mann-Whitney U + BH-FDR on all features.

    Returns the DifferentialResult.df as a dict for JSON serialization,
    plus summary stats.
    """
    import pandas as pd

    _banner(f"Differential activation analysis (BH-FDR q={FDR_Q:.2f})")

    try:
        from s1s2.sae.differential import differential_activation
        result = differential_activation(
            feature_activations=feature_activations,
            conflict=conflict,
            fdr_q=FDR_Q,
            subset_label="vulnerable_categories",
        )
        df = result.df
    except ImportError:
        log.warning("Could not import differential_activation; running manually.")
        from scipy import stats as sp_stats

        n_problems, n_features = feature_activations.shape
        s1_idx = np.where(conflict)[0]
        s2_idx = np.where(~conflict)[0]
        n_s1, n_s2 = len(s1_idx), len(s2_idx)

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
        try:
            from s1s2.utils.stats import bh_fdr
            rejected, q_values = bh_fdr(p_values, q=FDR_Q)
        except ImportError:
            n = len(p_values)
            order = np.argsort(p_values)
            ranked = p_values[order]
            inv_rank = np.arange(1, n + 1, dtype=np.float64)
            raw = ranked * n / inv_rank
            raw = np.minimum.accumulate(raw[::-1])[::-1]
            q_values = np.empty_like(raw)
            q_values[order] = np.clip(raw, 0.0, 1.0)
            rejected = q_values <= FDR_Q

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
    n_total = len(df)
    log.info("Significant features after BH-FDR: %d / %d", n_sig, n_total)

    # Print top 10 by absolute log fold change
    sig_df = df[df["significant"]].copy()
    sig_df["abs_log_fc"] = sig_df["log_fc"].abs()
    top10 = sig_df.nlargest(10, "abs_log_fc")
    log.info("Top 10 features by |log2 fold change|:")
    for _, row in top10.iterrows():
        log.info(
            "  feature %d: log_fc=%.3f, effect_size=%.3f, q=%.2e",
            int(row["feature_id"]), row["log_fc"], row["effect_size"], row["q_value"],
        )

    return {
        "df": df,
        "n_significant": n_sig,
        "n_total": n_total,
    }


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
    """Run Ma et al. cheap falsification on top candidates.

    Returns (list of FalsificationResult dicts, n_surviving).
    """

    _banner(f"Ma et al. falsification (cheap mode, top {TOP_K_FALSIFY} candidates)")

    sig_df = diff_df[diff_df["significant"]].copy()
    if len(sig_df) == 0:
        log.warning("No significant features to falsify.")
        return [], 0

    sig_df["abs_log_fc"] = sig_df["log_fc"].abs()
    candidates = sig_df.nlargest(TOP_K_FALSIFY, "abs_log_fc")["feature_id"].tolist()
    candidates = [int(c) for c in candidates]
    log.info("Falsifying %d candidate features", len(candidates))

    try:
        from s1s2.sae.falsification import falsify_candidates
        results = falsify_candidates(
            candidate_feature_ids=candidates,
            sae=sae,
            activations=activations,
            feature_activations=feature_activations,
            prompts=prompts,
            tokenizer=None,
            mode="cheap",
            n_random_texts=100,
            n_top_tokens=5,
            threshold=0.5,
            top_k_features=TOP_K_FALSIFY,
            device="cpu",  # cheap mode works on CPU
        )
        result_dicts = [asdict(r) for r in results]
        n_spurious = sum(1 for r in results if r.is_spurious)
    except ImportError:
        log.warning("Could not import falsify_candidates; running manual cheap falsification.")
        import torch

        result_dicts = []
        n_spurious = 0
        mean_residual = activations.mean(axis=0)
        np.random.default_rng(42)

        for fid in candidates:
            col = feature_activations[:, fid]
            mean_orig = float(col.mean())
            peak_orig = float(col.max())

            # Build trigger centroid from top-activating problems
            top_prob_idx = np.argsort(-col)[:10]
            [prompts[int(i)] for i in top_prob_idx]
            # Collect candidate trigger tokens
            token_scores: dict[str, float] = {}
            stoplist = {
                "a", "an", "the", "and", "or", "of", "in", "to", "is", "are",
                "was", "were", "it", "this", "that", "for", "on", "at", "by",
                "with", "as", "be", "has", "have", "had", "but", "not", "if",
                ",", ".", "?", "!", ":", ";",
            }
            for _idx_i, p_idx in enumerate(top_prob_idx):
                toks = prompts[int(p_idx)].split()
                w = float(col[int(p_idx)])
                for t in toks:
                    t_clean = t.strip().lower()
                    if t_clean not in stoplist and len(t_clean) > 1:
                        token_scores[t_clean] = token_scores.get(t_clean, 0.0) + w
            triggers = sorted(token_scores, key=lambda k: -token_scores[k])[:5]

            # Find trigger-containing residuals
            has_trigger = np.array(
                [any(t in prompts[i].lower() for t in triggers)
                 for i in range(len(prompts))],
                dtype=bool,
            )
            if has_trigger.sum() > 0:
                trigger_centroid = activations[has_trigger].mean(axis=0)
            else:
                trigger_centroid = mean_residual

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
                "notes": "manual cheap falsification (no library import)",
            })

    n_survived = len(result_dicts) - n_spurious
    log.info("Falsified: %d / %d flagged as spurious", n_spurious, len(result_dicts))
    log.info("Surviving: %d / %d", n_survived, len(result_dicts))

    return result_dicts, n_survived


# ---------------------------------------------------------------------------
# 7. Volcano plot
# ---------------------------------------------------------------------------

def make_volcano(diff_df, falsification_results: list[dict], out_dir: Path) -> None:
    """Generate volcano plot with falsification overlay."""

    _banner("Generating volcano plot")

    df = diff_df.copy()

    # Merge falsification info
    fals_map = {r["feature_id"]: r["is_spurious"] for r in falsification_results}
    df["is_falsified"] = df["feature_id"].map(lambda fid: fals_map.get(int(fid), False))

    out_path = out_dir / "volcano_l19.png"

    try:
        from s1s2.sae.volcano import plot_volcano
        plot_volcano(
            df=df,
            title=(
                "Llama-3.1-8B-Instruct Goodfire L19 SAE\n"
                f"Vulnerable categories ({', '.join(VULNERABLE_CATEGORIES)})"
            ),
            out_path=str(out_path),
            fdr_q=FDR_Q,
            annotate_top_k=10,
            dpi=200,
        )
    except ImportError:
        log.warning("Could not import plot_volcano; generating manually.")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
        log_fc = df["log_fc"].to_numpy(dtype=np.float64)
        q = df["q_value"].to_numpy(dtype=np.float64)
        neg_log_q = -np.log10(np.clip(q, 1e-300, 1.0))
        sig = df["significant"].to_numpy(dtype=bool)
        is_fals = df["is_falsified"].to_numpy(dtype=bool)
        genuine = sig & ~is_fals
        spurious = sig & is_fals
        nonsig = ~sig

        ax.scatter(log_fc[nonsig], neg_log_q[nonsig], s=8, c="#cccccc", alpha=0.6, label=f"n.s. (q>{FDR_Q})")
        if spurious.any():
            ax.scatter(log_fc[spurious], neg_log_q[spurious], s=24, c="#888888",
                       marker="x", alpha=0.8, label="falsified (Ma et al.)")
        if genuine.any():
            ax.scatter(log_fc[genuine], neg_log_q[genuine], s=32, c="#c03030",
                       edgecolors="#400000", linewidths=0.4,
                       label=f"genuine (n={int(genuine.sum())})")
        import math
        ax.axhline(-math.log10(max(FDR_Q, 1e-300)), ls="--", color="#555555", lw=0.8)
        ax.axvline(0.0, ls="--", color="#555555", lw=0.8)
        ax.set_xlabel("log2 fold change (S1 / S2)")
        ax.set_ylabel(r"$-\log_{10}(q)$")
        ax.set_title(
            "Llama-3.1-8B-Instruct Goodfire L19 SAE\n"
            f"Vulnerable categories ({', '.join(VULNERABLE_CATEGORIES)})"
        )
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)

    log.info("Volcano plot saved to %s", out_path)


# ---------------------------------------------------------------------------
# 8. Save all results
# ---------------------------------------------------------------------------

def save_results(
    out_dir: Path,
    fidelity: dict,
    diff_df,
    n_significant: int,
    falsification_results: list[dict],
    n_surviving: int,
    elapsed: float,
) -> None:
    """Save all results to the output directory."""

    _banner(f"Saving results to {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Differential results CSV
    diff_df.to_csv(out_dir / "differential_results.csv", index=False)
    log.info("Saved differential_results.csv (%d rows)", len(diff_df))

    # Significant features only
    sig_df = diff_df[diff_df["significant"]].copy()
    sig_df.to_csv(out_dir / "significant_features.csv", index=False)
    log.info("Saved significant_features.csv (%d rows)", len(sig_df))

    # Falsification results
    with open(out_dir / "falsification_results.json", "w") as f:
        json.dump(falsification_results, f, indent=2, default=str)
    log.info("Saved falsification_results.json (%d entries)", len(falsification_results))

    # Summary JSON
    summary = {
        "config": {
            "h5_path": str(H5_PATH),
            "model_key": MODEL_KEY,
            "layer": LAYER,
            "sae_release": SAE_RELEASE,
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

    _banner("Goodfire L19 SAE Analysis — Llama-3.1-8B-Instruct")
    log.info("Device: %s", DEVICE)
    log.info("Output: %s", OUT_DIR)

    # Step 1: Load activations
    p0_activations, conflict, categories, prompts = load_activations()

    # Step 2: Load SAE
    sae = load_goodfire_sae()
    print(f"\n>>> Number of features in SAE: {sae.n_features}")
    print(f">>> SAE hidden dim: {sae.hidden_dim}")
    print(f">>> Activation hidden dim: {p0_activations.shape[1]}")

    # Validate dimension match
    if sae.hidden_dim != p0_activations.shape[1]:
        log.error(
            "DIMENSION MISMATCH: SAE expects hidden_dim=%d but activations have %d. "
            "This SAE cannot be applied to these activations.",
            sae.hidden_dim, p0_activations.shape[1],
        )
        raise ValueError(
            f"SAE hidden_dim ({sae.hidden_dim}) != activation hidden_dim "
            f"({p0_activations.shape[1]}). Check that the SAE matches the model."
        )

    # Step 3: Reconstruction fidelity
    fidelity = check_fidelity(sae, p0_activations)
    print(f"\n>>> Reconstruction explained variance: {fidelity['explained_variance']:.4f}")

    # Step 4: Filter + encode
    feature_activations, conflict_filt, prompts_filt, acts_filt = filter_and_encode(
        sae, p0_activations, conflict, categories, prompts
    )

    # Step 5: Differential analysis
    diff_result = run_differential(feature_activations, conflict_filt)
    diff_df = diff_result["df"]
    n_sig = diff_result["n_significant"]
    print(f"\n>>> Differential features (significant after BH-FDR at q={FDR_Q}): {n_sig}")

    # Step 6: Ma et al. falsification
    falsification_results, n_surviving = run_falsification(
        sae, diff_df, acts_filt, feature_activations, prompts_filt
    )
    print(f"\n>>> Surviving Ma et al. falsification: {n_surviving}")

    # Step 7: Volcano plot
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_volcano(diff_df, falsification_results, OUT_DIR)

    # Step 8: Save results
    elapsed = time.time() - t0
    save_results(OUT_DIR, fidelity, diff_df, n_sig, falsification_results, n_surviving, elapsed)

    # Final summary to stdout
    _banner("RESULTS SUMMARY")
    print(f"  Number of features in SAE:              {sae.n_features}")
    print(f"  Reconstruction explained variance:      {fidelity['explained_variance']:.4f}")
    print(f"  Reconstruction MSE:                     {fidelity['mse']:.6f}")
    print(f"  Mean L0 (features active per token):    {fidelity['mean_l0']:.1f}")
    print(f"  Poor fit:                               {fidelity['is_poor_fit']}")
    print(f"  Differential features (q<={FDR_Q}):       {n_sig}")
    print(f"  Falsification candidates tested:        {len(falsification_results)}")
    n_spurious = sum(1 for r in falsification_results if r.get("is_spurious", False))
    print(f"  Flagged as spurious:                    {n_spurious}")
    print(f"  Surviving falsification:                {n_surviving}")
    print(f"  Wall time:                              {elapsed:.1f}s")
    print()

    # Top 10 by |log_fc|
    sig_df = diff_df[diff_df["significant"]].copy()
    if len(sig_df) > 0:
        # Annotate with falsification status
        fals_map = {r["feature_id"]: r.get("is_spurious", False) for r in falsification_results}
        sig_df["is_spurious"] = sig_df["feature_id"].map(lambda fid: fals_map.get(int(fid), False))
        sig_df["abs_log_fc"] = sig_df["log_fc"].abs()
        top10 = sig_df.nlargest(10, "abs_log_fc")
        print("  Top 10 features by |log2 fold change|:")
        print(f"  {'ID':>8}  {'log_fc':>8}  {'effect':>8}  {'q-value':>10}  {'spurious':>8}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")
        for _, row in top10.iterrows():
            spur = "YES" if row.get("is_spurious", False) else "no"
            print(
                f"  {int(row['feature_id']):>8}  {row['log_fc']:>8.3f}  "
                f"{row['effect_size']:>8.3f}  {row['q_value']:>10.2e}  {spur:>8}"
            )
    print()
    log.info("All results saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()
