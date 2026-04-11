"""Activation steering via residual-stream hooks.

We take an SAE feature direction ``d`` (unit-normalised) and add ``alpha * d``
to the residual stream at the output of a specific transformer block. With
``alpha > 0`` the model is pushed *towards* using that feature; with ``alpha
< 0`` it is pushed away from it. This is the simplest causal intervention
test available — if the feature genuinely mediates bias-prone behaviour,
monotonic changes in alpha should produce monotonic changes in
conflict-item accuracy.

Design notes
------------
* The hook is implemented as a context manager so the forward hook is
  guaranteed to be removed even if the inner forward pass raises.
* We accept either a pre-normalised direction or an arbitrary vector; the
  hook normalises it at attach-time so we never introduce a magnitude
  confound across experiments.
* The hook mutates the residual stream *in place* by return-value rewrite,
  matching the decoder-layer tuple convention used by HuggingFace Llama /
  Gemma / Qwen forward outputs: they return ``(hidden, ...)`` tuples.
* Position filtering is supported but rarely needed. By default every
  position is steered, which is the operational definition used in the
  causal brief ("add alpha * d to the residual stream").
* The hook is registered on ``model.model.layers[layer]``. We deliberately
  do NOT grab ``model.model.layers[layer].output`` or similar, because
  different HF model classes expose those attributes inconsistently.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float
from torch import Tensor

from s1s2.utils.logging import get_logger

logger = get_logger("s1s2.causal")


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------


@beartype
def normalize_direction(
    direction: Float[Tensor, "hidden"] | np.ndarray,  # noqa: F821, UP037
) -> Float[Tensor, "hidden"]:  # noqa: F821, UP037
    """Return a unit-norm torch tensor copy of ``direction``.

    Accepts numpy arrays or torch tensors. The output is always ``float32``
    on CPU; callers that need a different device/dtype should cast at use
    time so we don't accidentally tie the direction to the wrong device.
    """
    if isinstance(direction, np.ndarray):
        t = torch.from_numpy(direction.astype(np.float32, copy=False))
    else:
        t = direction.detach().to(dtype=torch.float32, device="cpu")
    norm = float(t.norm().item())
    if norm == 0.0:
        raise ValueError("cannot normalize a zero direction")
    return t / norm


@beartype
def random_unit_direction(
    hidden_dim: int, seed: int
) -> Float[Tensor, "hidden"]:  # noqa: F821, UP037
    """Sample a random unit direction for the random-control baseline.

    Uses a dedicated torch :class:`torch.Generator` so the random state
    doesn't leak into global RNGs — the causal workstream needs multiple
    independent random directions per experiment, and we MUST be able to
    seed them reproducibly.
    """
    gen = torch.Generator().manual_seed(int(seed))
    v = torch.randn(int(hidden_dim), generator=gen, dtype=torch.float32)
    return v / v.norm()


# ---------------------------------------------------------------------------
# Steering hook context manager
# ---------------------------------------------------------------------------


class SteeringHook:
    """Attach a ``+alpha * d`` steering hook to one transformer block.

    Usage::

        with SteeringHook(model, layer=16, direction=d, alpha=1.5):
            out = model(**inputs)

    The hook is registered on ``model.model.layers[layer].register_forward_hook``
    and is removed when the context manager exits. Re-entering the same
    ``SteeringHook`` instance re-registers the hook; nesting multiple
    hooks is supported as long as each uses its own instance.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer: int,
        direction: Float[Tensor, "hidden"] | np.ndarray,  # noqa: F821, UP037
        alpha: float,
        *,
        position_filter: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.model = model
        self.layer = int(layer)
        self.alpha = float(alpha)
        self.direction = normalize_direction(direction)
        self.position_filter = position_filter
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        self._call_count = 0

    # -- hook implementation ------------------------------------------------

    def _hook(self, module: torch.nn.Module, inputs: Any, output: Any) -> Any:
        """Forward hook that rewrites the residual stream output."""
        self._call_count += 1
        # HF decoder layers return (hidden_states, *aux) tuples.
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        if not isinstance(hidden, torch.Tensor):
            return output

        delta_dir = self.direction.to(device=hidden.device, dtype=hidden.dtype)
        if self.alpha == 0.0:
            # No-op hook: still go through the filter path so the code path
            # is exercised under alpha=0 tests, but skip the arithmetic.
            return output

        if self.position_filter is None:
            new_hidden = hidden + self.alpha * delta_dir
        else:
            # Broadcast position mask over hidden dim.
            mask = self.position_filter(hidden).to(device=hidden.device, dtype=hidden.dtype)
            while mask.dim() < hidden.dim():
                mask = mask.unsqueeze(-1)
            new_hidden = hidden + self.alpha * delta_dir * mask

        if rest is not None:
            return (new_hidden, *rest)
        return new_hidden

    # -- context manager protocol ------------------------------------------

    def __enter__(self) -> SteeringHook:
        target = self._resolve_layer_module()
        if self._handle is not None:
            # Defensive: a prior hook wasn't cleaned up — detach before re-attach.
            self._handle.remove()
        self._handle = target.register_forward_hook(self._hook)
        self._call_count = 0
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # -- introspection ------------------------------------------------------

    @property
    def is_attached(self) -> bool:
        return self._handle is not None

    @property
    def call_count(self) -> int:
        """Number of times the hook was invoked inside the active context."""
        return self._call_count

    def _resolve_layer_module(self) -> torch.nn.Module:
        """Navigate to the transformer block ``layer``.

        Supports the standard HuggingFace causal-LM layout ``model.model.layers``
        (Llama, Gemma, Qwen2). Falls back to a bare ``layers`` attribute if the
        outer ``.model`` is absent.
        """
        inner = getattr(self.model, "model", self.model)
        layers = getattr(inner, "layers", None)
        if layers is None:
            raise AttributeError(
                f"cannot locate transformer blocks on {type(self.model).__name__}; "
                "expected a `.model.layers` attribute"
            )
        n = len(layers)
        if not (0 <= self.layer < n):
            raise IndexError(f"layer {self.layer} out of range [0, {n})")
        return layers[self.layer]


# ---------------------------------------------------------------------------
# Stacked / multi-layer steering
# ---------------------------------------------------------------------------


class StackedSteeringHook:
    """Attach steering hooks to several layers at once.

    Used for the "steer many layers towards the same feature" sanity check
    where we expect effects to accumulate additively. Internally wraps a
    list of :class:`SteeringHook` instances and enters/exits them in order.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layers: list[int],
        directions: list[Float[Tensor, "hidden"] | np.ndarray],  # noqa: F821, UP037
        alpha: float,
    ) -> None:
        if len(layers) != len(directions):
            raise ValueError(
                f"layers ({len(layers)}) and directions ({len(directions)}) must match"
            )
        self.hooks = [
            SteeringHook(model, layer=layer, direction=d, alpha=alpha)
            for layer, d in zip(layers, directions, strict=True)
        ]

    def __enter__(self) -> StackedSteeringHook:
        for h in self.hooks:
            h.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in reversed(self.hooks):
            h.__exit__(exc_type, exc, tb)


__all__ = [
    "StackedSteeringHook",
    "SteeringHook",
    "normalize_direction",
    "random_unit_direction",
]
