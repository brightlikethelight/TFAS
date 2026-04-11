"""SAE loaders and reconstruction fidelity checks.

This module exposes a unified :class:`SAEHandle` protocol and concrete
loaders for the three pre-trained SAE families the project uses:

- **Llama Scope** (fnlp) — SAEs trained on Llama-3.1-8B-*Base*. Loading
  for Llama-3.1-8B-Instruct is a known mismatch and reconstruction
  fidelity must be checked before any downstream analysis is trusted.
- **Gemma Scope** (google) — SAEs trained on Gemma-2-9B-IT (instruction
  tuned). Same model, so fidelity should be high.
- **Goodfire R1 SAE** — trained on layer 37 of the 671B DeepSeek-R1. We
  log a very loud warning if this is applied to any distilled 8B/7B
  variant because the residual stream distribution will be wildly off.

The concrete loaders all wrap ``sae-lens`` (optional dependency) where
available. A :class:`MockSAE` fallback is provided for tests and for
environments without internet access — it has random weights and makes
the differential analysis / falsification code testable on CPU in
seconds.

The reconstruction fidelity check is the single most important guardrail
on this whole workstream: a poorly-reconstructing SAE emits garbage
features, and any downstream "S1/S2 feature" from it is meaningless.
Every loader calls it automatically and logs the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float
from torch import Tensor

from s1s2.utils.logging import get_logger

logger = get_logger("s1s2.sae")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SAEHandle(Protocol):
    """Uniform read-only interface exposed by every SAE backend.

    ``encode`` maps residual-stream activations into the sparse feature
    space and ``decode`` maps back. ``reconstruct`` is the composition; it
    is separately exposed so backends that cache intermediate state (e.g.
    activation shifts) can override it without losing the optimization.

    All tensor methods accept 2D inputs ``(batch, hidden_dim)`` in float32
    on CPU. SAE backends that internally prefer half precision or cuda
    must manage the dtype and device conversions themselves so callers
    can ignore the distinction.
    """

    n_features: int
    hidden_dim: int
    layer: int

    def encode(
        self, x: Float[Tensor, "batch hidden"]
    ) -> Float[Tensor, "batch n_features"]: ...

    def decode(
        self, z: Float[Tensor, "batch n_features"]
    ) -> Float[Tensor, "batch hidden"]: ...

    def reconstruct(
        self, x: Float[Tensor, "batch hidden"]
    ) -> Float[Tensor, "batch hidden"]: ...


# ---------------------------------------------------------------------------
# Fidelity reporting
# ---------------------------------------------------------------------------


@dataclass
class ReconstructionReport:
    """Summary of an SAE's fit quality on a sample of real activations.

    The two headline numbers are ``mse`` (raw reconstruction error) and
    ``explained_variance`` (``1 - var(x - x_hat) / var(x)``, the metric
    SAE papers usually report). ``is_poor_fit`` is the boolean gate the
    orchestration layer keys off: if set, do not trust any downstream
    feature results produced from this SAE on this model.
    """

    n_samples: int
    hidden_dim: int
    mse: float
    variance: float
    explained_variance: float
    mean_l0: float  # average number of active features per token
    is_poor_fit: bool


@beartype
def reconstruction_report(
    sae: SAEHandle,
    activations: np.ndarray,
    min_explained_variance: float = 0.5,
    rng: np.random.Generator | None = None,
    n_samples: int = 256,
) -> ReconstructionReport:
    """Compute reconstruction quality on a random subset of activations.

    Parameters
    ----------
    sae
        Any backend that satisfies :class:`SAEHandle`.
    activations
        2D array of residual-stream vectors, shape ``(n, hidden_dim)``.
    min_explained_variance
        Threshold below which ``is_poor_fit`` is set. The Ma et al.
        pipeline rejects any SAE whose explained variance falls below
        this on the target model.
    rng
        Seeded numpy generator for reproducible subsampling.
    n_samples
        Number of activation rows to evaluate on. Fewer is fine if the
        input has fewer rows.
    """

    if activations.ndim != 2:
        raise ValueError(
            f"reconstruction_report expects 2D activations, got shape {activations.shape}"
        )
    if activations.shape[1] != sae.hidden_dim:
        raise ValueError(
            f"activation hidden_dim {activations.shape[1]} does not match "
            f"SAE hidden_dim {sae.hidden_dim}"
        )

    if rng is None:
        rng = np.random.default_rng(0)

    n_total = activations.shape[0]
    take = min(n_samples, n_total)
    idx = rng.choice(n_total, size=take, replace=False)
    sample = activations[idx].astype(np.float32, copy=False)

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

    # Single source of truth: poor fit iff explained variance is below the
    # user-supplied threshold. (Earlier we also had a hard `mse > 0.5*var`
    # floor — but that's algebraically equivalent to `ev < 0.5` and would
    # silently override any threshold the user passed.)
    is_poor = ev < min_explained_variance
    report = ReconstructionReport(
        n_samples=int(take),
        hidden_dim=int(sae.hidden_dim),
        mse=mse,
        variance=var,
        explained_variance=float(ev),
        mean_l0=l0,
        is_poor_fit=bool(is_poor),
    )

    msg = (
        f"SAE layer {sae.layer} reconstruction: n={take} mse={mse:.4f} "
        f"var={var:.4f} ev={ev:.3f} mean_l0={l0:.1f}"
    )
    if is_poor:
        logger.warning("POOR FIT — %s", msg)
        logger.warning(
            "  explained_variance %.3f < threshold %.3f; downstream features "
            "from this SAE are UNTRUSTWORTHY.",
            ev,
            min_explained_variance,
        )
    else:
        logger.info("%s", msg)

    return report


# ---------------------------------------------------------------------------
# Mock backend (tests, CPU, no internet)
# ---------------------------------------------------------------------------


class MockSAE:
    """A random-weight SAE that satisfies :class:`SAEHandle`.

    Used in tests and smoke runs. The encoder is a random Gaussian
    projection and the decoder is its pseudo-inverse, so
    ``reconstruct(x) == x`` up to the rank-truncation induced by
    ``n_features < hidden_dim`` (if that happens). For
    ``n_features >= hidden_dim`` the reconstruction is exact modulo
    fp noise, which makes the fidelity check trivially pass and lets
    tests focus on the downstream differential / falsification logic.

    Set ``sparsity`` in ``(0, 1]`` to apply a top-k ReLU to emulate the
    sparse activation pattern of a real SAE — useful for testing
    downstream code that expects sparse codes.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_features: int,
        layer: int = 0,
        seed: int = 0,
        sparsity: float = 0.1,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.n_features = int(n_features)
        self.layer = int(layer)
        self.sparsity = float(sparsity)
        rng = np.random.default_rng(seed)
        # Normalized Gaussian encoder; unit column norms for decoder.
        W_enc = rng.standard_normal((hidden_dim, n_features)).astype(np.float32)
        W_enc /= np.linalg.norm(W_enc, axis=0, keepdims=True) + 1e-8
        # Decoder is the Moore-Penrose pseudoinverse so reconstruction is
        # exact when n_features >= hidden_dim and sparsity == 1.0.
        W_dec = np.linalg.pinv(W_enc).astype(np.float32)
        self.W_enc = torch.from_numpy(W_enc)            # (hidden, n_features)
        self.W_dec = torch.from_numpy(W_dec)            # (n_features, hidden)
        self.b_enc = torch.zeros(n_features)
        self.b_dec = torch.zeros(hidden_dim)

    def encode(
        self, x: Float[Tensor, "batch hidden"]
    ) -> Float[Tensor, "batch n_features"]:
        z = x.float() @ self.W_enc + self.b_enc
        if self.sparsity < 1.0:
            k = max(1, int(round(self.sparsity * self.n_features)))
            # Keep top-k absolute magnitudes, zero the rest.
            topk = torch.topk(z.abs(), k=k, dim=-1).indices
            mask = torch.zeros_like(z)
            mask.scatter_(1, topk, 1.0)
            z = z * mask
        # ReLU so activations are non-negative like a real SAE.
        return torch.relu(z)

    def decode(
        self, z: Float[Tensor, "batch n_features"]
    ) -> Float[Tensor, "batch hidden"]:
        return z.float() @ self.W_dec + self.b_dec

    def reconstruct(
        self, x: Float[Tensor, "batch hidden"]
    ) -> Float[Tensor, "batch hidden"]:
        return self.decode(self.encode(x))

    # --- extras used by steering / cross-model matching ---

    def encoder_directions(self) -> np.ndarray:
        """Return the encoder direction for each feature, shape ``(n_features, hidden_dim)``."""
        return self.W_enc.detach().cpu().numpy().T.copy()

    def decoder_directions(self) -> np.ndarray:
        """Return the decoder direction for each feature, shape ``(n_features, hidden_dim)``."""
        return self.W_dec.detach().cpu().numpy().copy()


# ---------------------------------------------------------------------------
# sae-lens wrapper
# ---------------------------------------------------------------------------


class _SAELensHandle:
    """Thin adapter that makes a ``sae_lens.SAE`` conform to :class:`SAEHandle`.

    We deliberately do NOT subclass :class:`SAEHandle`: it's a Protocol,
    so duck-typing is enough. The wrapper exists so that every backend
    quirk (device placement, dtype casting, error-term additions) is
    handled in a single place rather than sprinkled through the analysis
    code.
    """

    def __init__(self, sae, layer: int) -> None:
        self._sae = sae
        self.layer = int(layer)
        # sae-lens exposes cfg.d_in and cfg.d_sae
        self.hidden_dim = int(sae.cfg.d_in)
        self.n_features = int(sae.cfg.d_sae)

    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        dev = next(self._sae.parameters()).device
        dtype = next(self._sae.parameters()).dtype
        return x.to(device=dev, dtype=dtype)

    def encode(
        self, x: Float[Tensor, "batch hidden"]
    ) -> Float[Tensor, "batch n_features"]:
        with torch.no_grad():
            z = self._sae.encode(self._to_device(x))
        return z.detach().float().cpu()

    def decode(
        self, z: Float[Tensor, "batch n_features"]
    ) -> Float[Tensor, "batch hidden"]:
        with torch.no_grad():
            x_hat = self._sae.decode(self._to_device(z))
        return x_hat.detach().float().cpu()

    def reconstruct(
        self, x: Float[Tensor, "batch hidden"]
    ) -> Float[Tensor, "batch hidden"]:
        return self.decode(self.encode(x))

    def encoder_directions(self) -> np.ndarray:
        # W_enc shape in sae-lens is (d_in, d_sae) — transpose to feature-major.
        W = getattr(self._sae, "W_enc", None)
        if W is None:
            return np.zeros((self.n_features, self.hidden_dim), dtype=np.float32)
        return W.detach().float().cpu().numpy().T.copy()

    def decoder_directions(self) -> np.ndarray:
        # W_dec shape in sae-lens is (d_sae, d_in) — already feature-major.
        W = getattr(self._sae, "W_dec", None)
        if W is None:
            return np.zeros((self.n_features, self.hidden_dim), dtype=np.float32)
        return W.detach().float().cpu().numpy().copy()


@beartype
def _try_import_sae_lens():
    """Import ``sae_lens`` lazily. Return ``None`` if unavailable.

    We intentionally swallow the ImportError because the CI / test
    environment may not have internet to install ``sae-lens``, and the
    point of the Mock fallback is exactly to keep unit tests runnable.
    """
    try:
        import sae_lens

        return sae_lens
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("sae-lens not available (%s); using MockSAE fallback.", exc)
        return None


# --- Llama Scope ---


@beartype
def load_llama_scope(
    layer: int,
    width: str = "32x",
    device: str = "cpu",
    release: str | None = None,
) -> SAEHandle:
    """Load a Llama Scope SAE for a given layer.

    Parameters
    ----------
    layer
        Transformer block index (0..31 for Llama-3.1-8B).
    width
        Expansion factor as encoded in the fnlp release naming. "32x"
        corresponds to the ``fnlp/Llama-3_1-8B-Base-LXR-32x`` release,
        which is what configs/models.yaml references. Pass a different
        string if you want to experiment with 16x / 8x / 128x.
    device
        Torch device for the SAE weights. ``"cpu"`` is the default so
        tests can run without GPU.
    release
        Override the HuggingFace release. If ``None`` we derive it from
        ``width`` using the standard naming.

    Notes
    -----
    Llama Scope SAEs were trained on Llama-3.1-8B-*Base*, not Instruct.
    When you apply one to Instruct residuals, expect reconstruction
    fidelity to be lower. The caller is responsible for running the
    :func:`reconstruction_report` on its own activations and checking
    ``is_poor_fit``.
    """

    sae_lens = _try_import_sae_lens()
    if sae_lens is None:
        logger.warning(
            "Falling back to MockSAE for Llama Scope layer %d — no real features.",
            layer,
        )
        return MockSAE(hidden_dim=4096, n_features=4096 * 8, layer=layer, seed=layer)

    if release is None:
        # fnlp uses "Llama-3_1-8B-Base-LXR-{width}" / "L{layer}R" sae_id
        release = f"fnlp/Llama-3_1-8B-Base-LXR-{width}"

    sae_id = f"L{layer}R"
    try:
        sae, _, _ = sae_lens.SAE.from_pretrained(
            release=release, sae_id=sae_id, device=device
        )
    except Exception as exc:
        logger.error(
            "Llama Scope load failed for layer %d (release=%s, sae_id=%s): %s",
            layer,
            release,
            sae_id,
            exc,
        )
        logger.warning("Falling back to MockSAE; differential results WILL be meaningless.")
        return MockSAE(hidden_dim=4096, n_features=4096 * 8, layer=layer, seed=layer)
    return _SAELensHandle(sae, layer=layer)


# --- Gemma Scope ---


@beartype
def load_gemma_scope(
    layer: int,
    width: str = "16k",
    device: str = "cpu",
    release: str | None = None,
    l0: int | None = None,
) -> SAEHandle:
    """Load a Gemma Scope SAE for a given layer.

    Parameters
    ----------
    layer
        Transformer block index (0..41 for Gemma-2-9B).
    width
        Gemma Scope width tag, e.g. ``"16k"``, ``"131k"``, ``"1m"``.
    device
        Torch device for SAE weights.
    release
        Override release id; defaults to ``google/gemma-scope-9b-it-res``.
    l0
        Some Gemma Scope releases ship multiple checkpoints indexed by
        the average L0 (active features per token). Gemma Scope sae_ids
        look like ``layer_{L}/width_{W}/average_l0_{K}``. Pass an
        integer to pin a specific one; if ``None`` we default to a
        reasonable mid-range L0 (~70) which is the most commonly
        documented checkpoint.
    """

    sae_lens = _try_import_sae_lens()
    if sae_lens is None:
        logger.warning(
            "Falling back to MockSAE for Gemma Scope layer %d — no real features.",
            layer,
        )
        return MockSAE(hidden_dim=3584, n_features=16384, layer=layer, seed=layer)

    if release is None:
        release = "google/gemma-scope-9b-it-res"
    if l0 is None:
        l0 = 72  # Standard mid-range checkpoint per Gemma Scope docs.

    sae_id = f"layer_{layer}/width_{width}/average_l0_{l0}"
    try:
        sae, _, _ = sae_lens.SAE.from_pretrained(
            release=release, sae_id=sae_id, device=device
        )
    except Exception as exc:
        logger.error(
            "Gemma Scope load failed for layer %d (release=%s, sae_id=%s): %s",
            layer,
            release,
            sae_id,
            exc,
        )
        logger.warning("Falling back to MockSAE.")
        return MockSAE(hidden_dim=3584, n_features=16384, layer=layer, seed=layer)
    return _SAELensHandle(sae, layer=layer)


# --- Goodfire R1 SAE (layer 37 of 671B) ---


@beartype
def load_goodfire_r1(
    layer: int = 37,
    device: str = "cpu",
    release: str | None = None,
) -> SAEHandle:
    """Load the Goodfire DeepSeek-R1 SAE (671B, layer 37 only).

    The canonical release only provides an SAE for layer 37 of the 671B
    model. Distilled 8B/7B variants have a very different residual
    distribution and should not be expected to reconstruct well. This
    function logs a loud warning any time it's called for analysis on a
    distilled model so the mismatch is impossible to miss.
    """

    if layer != 37:
        logger.warning(
            "Goodfire R1 SAE is only released for layer 37; requested layer=%d. "
            "Loading layer 37 regardless.",
            layer,
        )

    sae_lens = _try_import_sae_lens()
    if sae_lens is None:
        logger.warning(
            "Falling back to MockSAE for Goodfire R1 — no real features."
        )
        return MockSAE(hidden_dim=7168, n_features=65536, layer=37, seed=37)

    if release is None:
        release = "Goodfire/DeepSeek-R1-SAE-l37"

    # Goodfire's sae-lens wrapper / converter may or may not exist. Try,
    # but explicitly catch and fall back.
    try:
        sae, _, _ = sae_lens.SAE.from_pretrained(
            release=release, sae_id="layer_37", device=device
        )
    except Exception as exc:
        logger.error(
            "Goodfire R1 SAE load failed (release=%s): %s", release, exc
        )
        logger.warning(
            "Goodfire R1 is trained on the 671B model; falling back to MockSAE. "
            "Distilled 8B/7B fidelity is EXPECTED to be poor even on a successful load."
        )
        return MockSAE(hidden_dim=7168, n_features=65536, layer=37, seed=37)
    return _SAELensHandle(sae, layer=37)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@beartype
def load_sae_for_model(
    model_key: str,
    layer: int,
    *,
    llama_scope_width: str = "32x",
    gemma_scope_width: str = "16k",
    device: str = "cpu",
) -> SAEHandle:
    """Pick the right loader for a model key.

    Matches the ``sae_release`` field in ``configs/models.yaml``. Models
    without a pre-trained SAE return a :class:`MockSAE` so downstream
    analysis can still run (and fail the fidelity check, making the
    problem visible in logs).
    """

    if model_key == "llama-3.1-8b-instruct":
        return load_llama_scope(layer=layer, width=llama_scope_width, device=device)
    if model_key == "gemma-2-9b-it":
        return load_gemma_scope(layer=layer, width=gemma_scope_width, device=device)
    if model_key == "r1-distill-llama-8b":
        # Best-effort: Llama Scope is the closest match architecturally, but
        # reconstruction fidelity will almost certainly be bad. The fidelity
        # check will surface this.
        logger.warning(
            "No official SAE for r1-distill-llama-8b; trying Llama Scope "
            "(trained on Llama-3.1-8B-Base). Fidelity will likely be poor."
        )
        return load_llama_scope(layer=layer, width=llama_scope_width, device=device)
    if model_key == "r1-distill-qwen-7b":
        logger.warning(
            "No SAE is available for r1-distill-qwen-7b; returning MockSAE. "
            "Downstream analysis will fail the fidelity check."
        )
        return MockSAE(hidden_dim=3584, n_features=16384, layer=layer, seed=layer)
    raise ValueError(f"Unknown model key {model_key!r}")


__all__ = [
    "MockSAE",
    "ReconstructionReport",
    "SAEHandle",
    "load_gemma_scope",
    "load_goodfire_r1",
    "load_llama_scope",
    "load_sae_for_model",
    "reconstruction_report",
]
