"""Feature ablation via projection / clamping.

Two complementary causal manipulations:

1. **Projection ablation**: remove the component of the residual stream
   that lies along a given feature direction. If ``d`` is a unit vector,
   this is ``h' = h - (h @ d) * d``. This nulls the feature without
   injecting any new signal — the residual stream norm decreases but no
   new direction is added. The cleanest "is this feature necessary?"
   test.

2. **Feature clamping** (value substitution): run the SAE encoder on the
   residual stream, replace the chosen feature's activation with a fixed
   value (typically 0 for ablation or a percentile-based value for
   amplification), decode, and write the reconstruction back to the
   residual stream. This is what Anthropic call "feature clamping" in
   their monosemanticity paper. Requires a functioning SAE and is more
   expensive than projection ablation because it does an encode/decode
   per forward pass.

Both are exposed as hook context managers so they compose with the
steering hook in :mod:`s1s2.causal.steering`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float
from torch import Tensor

from s1s2.causal.steering import normalize_direction
from s1s2.utils.logging import get_logger

if TYPE_CHECKING:
    # Imported lazily so that ``s1s2.sae.__init__`` side effects don't fire
    # at ``s1s2.causal`` import time. The FeatureClampHook constructor
    # duck-types on the SAEHandle protocol rather than calling isinstance.
    from s1s2.sae.loaders import SAEHandle

logger = get_logger("s1s2.causal")


# ---------------------------------------------------------------------------
# Pure-function projection ablation
# ---------------------------------------------------------------------------


@beartype
def ablate_direction(
    hidden: Float[Tensor, "... hidden"],
    direction: Float[Tensor, "hidden"],
) -> Float[Tensor, "... hidden"]:
    """Project ``direction`` out of ``hidden``.

    Works on any tensor whose last dim is the residual-stream hidden
    dim — typically ``(batch, seq, hidden)`` but also unit-tested on
    ``(batch, hidden)`` and plain ``(hidden,)`` vectors.

    Mathematically, given ``d`` unit-norm::

        h' = h - (h . d) d

    The returned tensor is on the same device / dtype as ``hidden``.
    """
    d = direction.to(device=hidden.device, dtype=hidden.dtype)
    d_norm = d.norm()
    if float(d_norm.item()) == 0.0:
        raise ValueError("cannot ablate along a zero direction")
    d = d / d_norm
    # Dot product along the last axis, broadcasting over any leading axes.
    coeff = (hidden * d).sum(dim=-1, keepdim=True)
    return hidden - coeff * d


# ---------------------------------------------------------------------------
# Projection ablation hook
# ---------------------------------------------------------------------------


class AblationHook:
    """Context-manager hook that projects a direction out of the residual stream.

    The hook mirrors :class:`s1s2.causal.steering.SteeringHook` — same
    register/remove pattern, same transformer-block navigation — but the
    transformation is ``h' = h - (h . d) d`` instead of ``h + alpha * d``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer: int,
        direction: Float[Tensor, "hidden"] | np.ndarray,
    ) -> None:
        self.model = model
        self.layer = int(layer)
        self.direction = normalize_direction(direction)
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        self._call_count = 0

    def _hook(self, module: torch.nn.Module, inputs: Any, output: Any) -> Any:
        self._call_count += 1
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        if not isinstance(hidden, torch.Tensor):
            return output

        new_hidden = ablate_direction(hidden, self.direction)

        if rest is not None:
            return (new_hidden, *rest)
        return new_hidden

    def __enter__(self) -> AblationHook:
        target = self._resolve_layer_module()
        if self._handle is not None:
            self._handle.remove()
        self._handle = target.register_forward_hook(self._hook)
        self._call_count = 0
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @property
    def is_attached(self) -> bool:
        return self._handle is not None

    @property
    def call_count(self) -> int:
        return self._call_count

    def _resolve_layer_module(self) -> torch.nn.Module:
        inner = getattr(self.model, "model", self.model)
        layers = getattr(inner, "layers", None)
        if layers is None:
            raise AttributeError(f"cannot locate transformer blocks on {type(self.model).__name__}")
        n = len(layers)
        if not (0 <= self.layer < n):
            raise IndexError(f"layer {self.layer} out of range [0, {n})")
        return layers[self.layer]


# ---------------------------------------------------------------------------
# SAE feature clamping
# ---------------------------------------------------------------------------


class FeatureClampHook:
    """Clamp a single SAE feature activation to a fixed value.

    The forward hook encodes the residual stream through the SAE, replaces
    the chosen feature's activation with ``clamp_value``, decodes, and
    writes the reconstruction back to the residual stream.

    This is heavier than :class:`AblationHook` — one encode + one decode
    per forward per-layer — but has two virtues:

    1. It manipulates *exactly* one interpretable unit (the SAE feature)
       rather than a raw vector direction, which is the experimentally
       relevant intervention for Goodfire-style feature engineering.
    2. The "clamp to P99 of its usual range" operation is the canonical
       "turn this feature on" test, and can't be reproduced by linear
       steering alone (it targets a specific SAE code level, not a
       residual-stream shift).

    Note that because the SAE reconstruction is never exact, even
    ``clamp_value=None`` (pass through) is technically lossy. The fidelity
    check in :mod:`s1s2.sae.loaders` is therefore a mandatory prerequisite
    before trusting clamp-based results.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer: int,
        sae: SAEHandle,
        feature_id: int,
        clamp_value: float,
    ) -> None:
        self.model = model
        self.layer = int(layer)
        self.sae = sae
        self.feature_id = int(feature_id)
        self.clamp_value = float(clamp_value)
        if not (0 <= self.feature_id < int(sae.n_features)):
            raise IndexError(f"feature_id {feature_id} out of range [0, {sae.n_features})")
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        self._call_count = 0

    def _hook(self, module: torch.nn.Module, inputs: Any, output: Any) -> Any:
        self._call_count += 1
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        if not isinstance(hidden, torch.Tensor):
            return output

        # Flatten batch + seq to a 2D matrix for the SAE, then reshape back.
        orig_shape = hidden.shape
        flat = hidden.reshape(-1, orig_shape[-1]).detach().float().cpu()
        with torch.no_grad():
            z = self.sae.encode(flat)
            z[:, self.feature_id] = self.clamp_value
            recon = self.sae.decode(z)
        new_hidden = recon.to(device=hidden.device, dtype=hidden.dtype).reshape(orig_shape)

        if rest is not None:
            return (new_hidden, *rest)
        return new_hidden

    def __enter__(self) -> FeatureClampHook:
        target = self._resolve_layer_module()
        if self._handle is not None:
            self._handle.remove()
        self._handle = target.register_forward_hook(self._hook)
        self._call_count = 0
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @property
    def is_attached(self) -> bool:
        return self._handle is not None

    @property
    def call_count(self) -> int:
        return self._call_count

    def _resolve_layer_module(self) -> torch.nn.Module:
        inner = getattr(self.model, "model", self.model)
        layers = getattr(inner, "layers", None)
        if layers is None:
            raise AttributeError(f"cannot locate transformer blocks on {type(self.model).__name__}")
        n = len(layers)
        if not (0 <= self.layer < n):
            raise IndexError(f"layer {self.layer} out of range [0, {n})")
        return layers[self.layer]


__all__ = [
    "AblationHook",
    "FeatureClampHook",
    "ablate_direction",
]
