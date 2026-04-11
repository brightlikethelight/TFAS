"""Incremental attention-metric computation.

For long generations (~32K tokens on reasoning models), we MUST NOT
materialize the full (L, H, T, T) attention tensor -- that's ~128 GB per
layer at float32. Instead, we compute per-token scalar metrics at every
generation step from the row of attention emitted for the new token only,
and discard the tensor immediately.

Design
------
Rather than register forward-hooks on individual attention modules (fragile:
module layout varies across architectures and transformers versions), we
drive extraction from the top-level ``model.generate`` call with
``output_attentions=True``. ``GenerationOutput.attentions`` is a tuple
``[n_steps][n_layers]`` whose element at step >= 1 has shape
``(batch=1, n_q_heads, 1, prompt_len + step)`` -- the single new-query row
over all prior keys. We process each row as it arrives and drop the tensor.

For GQA models, HuggingFace expands the attention over query heads (there
are ``n_q_heads`` rows, not ``n_kv_heads``), so we store metrics per query
head and leave KV-group aggregation to the analysis workstream.

The :class:`AttentionMetricsCollector` computes five scalar metrics per
(step, layer, head):

- Shannon entropy (bits)
- Entropy normalized by ``log2(t+1)`` where ``t`` is the key length
- Gini coefficient
- Max attention weight
- Sum of top-5 attention weights
- Effective rank = ``2 ** entropy``

Memory: ``O(n_steps * n_layers * n_q_heads * 6 floats)``. For 32K gen * 32
layers * 32 heads * 6 floats = ~200 MB. Fine.

All computation happens in float32 on CPU regardless of the forward dtype --
the rows are small enough (``<= 32K`` floats each) that the CPU round-trip
is cheap compared to the GPU forward pass.
"""

from __future__ import annotations

import numpy as np
import torch
from beartype import beartype


@beartype
def _row_metrics(row: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Compute the 6 scalar metrics for a single 1-D attention row.

    Parameters
    ----------
    row : np.ndarray of shape (t,)
        Attention weights. Should sum to 1.0 (softmaxed), but we re-normalize
        defensively. Any masked / invalid entries should already be zero.

    Returns
    -------
    (entropy_bits, entropy_normalized, gini, max_attn, focus_5, effective_rank)

    Normalization of entropy uses ``log2(t)`` rather than ``log2(t+1)`` because
    a degenerate point-mass has entropy 0 regardless, and the maximum entropy
    of a uniform distribution over ``t`` items is exactly ``log2(t)``.
    """
    if row.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # Guard against tiny negatives from softmax fp noise
    row = np.clip(row.astype(np.float64, copy=False), 0.0, None)
    s = row.sum()
    if s <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    probs = row / s

    # Shannon entropy in bits
    safe = np.clip(probs, 1e-12, 1.0)
    entropy_bits = float(-np.sum(safe * np.log2(safe)))

    t = probs.size
    max_entropy = float(np.log2(t)) if t > 1 else 1.0
    entropy_normalized = float(entropy_bits / max_entropy) if max_entropy > 0 else 0.0

    # Gini coefficient (inlined to avoid a per-step call overhead).
    sorted_probs = np.sort(probs)
    cum = np.cumsum(sorted_probs)
    denom = float(cum[-1])
    if denom <= 0:
        gini = 0.0
    else:
        idx = np.arange(1, t + 1, dtype=np.float64)
        gini = float((2.0 * np.sum(idx * sorted_probs)) / (t * denom) - (t + 1) / t)
        # Numerical safety: clamp to [0, 1]
        if gini < 0.0:
            gini = 0.0
        elif gini > 1.0:
            gini = 1.0

    max_attn = float(probs.max())
    k = min(5, t)
    # argpartition is O(t) vs a full sort.
    top5_idx = np.argpartition(probs, -k)[-k:]
    focus_5 = float(probs[top5_idx].sum())

    effective_rank = float(2.0 ** entropy_bits)
    return entropy_bits, entropy_normalized, gini, max_attn, focus_5, effective_rank


METRIC_NAMES: tuple[str, ...] = (
    "entropy",
    "entropy_normalized",
    "gini",
    "max_attn",
    "focus_5",
    "effective_rank",
)


class AttentionMetricsCollector:
    """Streaming metric accumulator over generation steps.

    Usage
    -----
    >>> collector = AttentionMetricsCollector(n_layers=32, n_heads=32)
    >>> # For each generation step:
    >>> for step_attentions in gen_output.attentions:
    ...     # step_attentions is a tuple of length n_layers, each
    ...     # (batch=1, n_heads, q=1, k).
    ...     collector.process_step(step_attentions)
    >>> metrics = collector.finalize()   # dict: name -> (n_steps, n_layers, n_heads)
    """

    def __init__(self, n_layers: int, n_heads: int) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        # Use a list of per-step arrays rather than pre-allocating: the generation
        # length is unknown at init time. Each element is (n_layers, n_heads, 6).
        self._steps: list[np.ndarray] = []

    def process_step(self, step_attentions) -> None:
        """Ingest one step's worth of attention tensors.

        ``step_attentions`` is the HuggingFace format: a tuple of length
        ``n_layers``, where element ``l`` has shape
        ``(batch, n_heads, q_len, k_len)``. For the first step (prompt), q_len
        is the full prompt length; for subsequent steps q_len == 1. In either
        case we extract the LAST row -- the query row corresponding to the
        most recently generated (or next-to-be-generated) token.
        """
        if len(step_attentions) != self.n_layers:
            raise ValueError(
                f"expected {self.n_layers} layers of attention, got {len(step_attentions)}"
            )
        step_out = np.empty((self.n_layers, self.n_heads, 6), dtype=np.float32)
        for layer_idx, attn in enumerate(step_attentions):
            # attn: (batch, n_heads, q_len, k_len)
            if attn.dim() != 4:
                raise ValueError(
                    f"layer {layer_idx}: expected 4D attention, got shape {tuple(attn.shape)}"
                )
            if attn.shape[1] != self.n_heads:
                raise ValueError(
                    f"layer {layer_idx}: expected {self.n_heads} query heads, got {attn.shape[1]}"
                )
            # Take the LAST query position of the first batch element.
            # detach + float -> cpu -> numpy. Using float32 ensures metric
            # accuracy even if the forward ran in bf16.
            row_batch = attn[0, :, -1, :].detach().to(torch.float32).cpu().numpy()
            # row_batch: (n_heads, k_len)
            for head_idx in range(self.n_heads):
                step_out[layer_idx, head_idx] = _row_metrics(row_batch[head_idx])
        self._steps.append(step_out)

    def n_steps(self) -> int:
        return len(self._steps)

    def finalize(self) -> dict[str, np.ndarray]:
        """Stack per-step arrays into a dict of (n_steps, n_layers, n_heads) tensors.

        Returns an empty-but-shaped dict if no steps were processed.
        """
        if not self._steps:
            empty = np.zeros((0, self.n_layers, self.n_heads), dtype=np.float32)
            return dict.fromkeys(METRIC_NAMES, empty)
        stacked = np.stack(self._steps, axis=0)  # (n_steps, n_layers, n_heads, 6)
        return {name: stacked[..., i].astype(np.float32, copy=False) for i, name in enumerate(METRIC_NAMES)}


@beartype
def metrics_at_positions(
    step_metrics: dict[str, np.ndarray],
    position_token_indices: np.ndarray,
    position_valid: np.ndarray,
    prompt_len: int,
) -> dict[str, np.ndarray]:
    """Subset per-step metrics to the canonical position rows.

    ``step_metrics`` is the dict returned by :meth:`AttentionMetricsCollector.finalize`;
    each value has shape ``(n_steps, n_layers, n_heads)``.

    We map each canonical position's absolute token index back to the step
    that produced it:

    - Step 0 corresponds to prompt tokens [0, prompt_len). Any position at
      absolute index < prompt_len maps to step 0 (the hook only stored the
      LAST row of the prompt, so the attention metric for any prompt position
      < prompt_len - 1 is formally undefined -- we copy step 0's row as a
      best-effort). P0 (prompt_len - 1) is exactly this row.
    - Step ``i`` (i >= 1) corresponds to the generated token at absolute
      index ``prompt_len + i - 1``.

    If ``position_valid[pos]`` is False, we emit zeros for that slot.
    """
    n_positions = position_token_indices.shape[0]
    n_layers, n_heads = next(iter(step_metrics.values())).shape[1:]
    n_steps = next(iter(step_metrics.values())).shape[0]
    out = {name: np.zeros((n_layers, n_heads, n_positions), dtype=np.float32) for name in METRIC_NAMES}
    for pos_idx in range(n_positions):
        if not position_valid[pos_idx]:
            continue
        abs_tok = int(position_token_indices[pos_idx])
        step_idx = 0 if abs_tok < prompt_len else 1 + (abs_tok - prompt_len)
        if step_idx < 0 or step_idx >= n_steps:
            # Out of range (e.g. Tend beyond generation length): leave zeros and
            # let downstream filter via /position_index/valid.
            continue
        for name in METRIC_NAMES:
            out[name][:, :, pos_idx] = step_metrics[name][step_idx]
    return out
