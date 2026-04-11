"""Capability preservation evaluation under steering.

Any intervention that boosts "reason correctly on conflict problems" is
boring if it also destroys the model's general capabilities. This module
loads tiny MMLU / HellaSwag subsets (JSONL) and scores multiple-choice
accuracy via the standard "log-likelihood of each option, pick the
highest" protocol. We check that a given intervention does not drop
accuracy below an absolute threshold (configurable; default 2 percentage
points).

Schema
------
Each JSONL line is one multiple-choice item::

    {
      "id": "mmlu_001",
      "question": "What is 2+2?",
      "choices": ["3", "4", "5", "6"],
      "correct_index": 1
    }

This is a tiny superset of both the HuggingFace ``mmlu`` and
``hellaswag`` datasets normalised to one schema. You can also synthesise
fixtures inline — the test suite does exactly that.

Scoring protocol
----------------
For each item, we construct a "question + choice" prompt for every
choice, compute the log-probability of the choice tokens (averaged or
summed), and pick the ``argmax``. This is the standard zero-shot MMLU
recipe and is robust to intervention because no generation is required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from beartype import beartype

from s1s2.utils.logging import get_logger

logger = get_logger("s1s2.causal")


# ---------------------------------------------------------------------------
# Data schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CapabilityItem:
    """One multiple-choice item for capability evaluation."""

    id: str
    question: str
    choices: tuple[str, ...]
    correct_index: int

    def __post_init__(self) -> None:
        if len(self.choices) < 2:
            raise ValueError(f"item {self.id}: need >=2 choices, got {len(self.choices)}")
        if not (0 <= self.correct_index < len(self.choices)):
            raise ValueError(
                f"item {self.id}: correct_index {self.correct_index} "
                f"out of range [0, {len(self.choices)})"
            )


@beartype
def load_capability_jsonl(path: str | Path) -> list[CapabilityItem]:
    """Load a JSONL file of capability items."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"capability eval file not found: {p}")
    items: list[CapabilityItem] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno}: malformed JSON: {exc}") from exc
            items.append(
                CapabilityItem(
                    id=str(raw["id"]),
                    question=str(raw["question"]),
                    choices=tuple(str(c) for c in raw["choices"]),
                    correct_index=int(raw["correct_index"]),
                )
            )
    return items


@beartype
def save_capability_jsonl(path: str | Path, items: list[CapabilityItem]) -> None:
    """Write capability items to JSONL. Used by tests to build fixtures."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(
                json.dumps(
                    {
                        "id": it.id,
                        "question": it.question,
                        "choices": list(it.choices),
                        "correct_index": int(it.correct_index),
                    }
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class CapabilityResult:
    """Output of one capability evaluation pass."""

    benchmark: str
    n_items: int
    accuracy: float
    per_item_correct: list[bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "n_items": int(self.n_items),
            "accuracy": float(self.accuracy),
            "per_item_correct": [bool(x) for x in self.per_item_correct],
        }


@beartype
def _format_prompt(question: str, choice: str) -> str:
    """Zero-shot MMLU-style template.

    We deliberately keep this minimal so the test fixtures don't depend
    on model-specific chat templates. Real benchmark runs can use the
    chat template via a custom formatter.
    """
    return f"Question: {question.strip()}\nAnswer: {choice.strip()}"


@beartype
def _score_choice_loglikelihood(
    model: Any,
    tokenizer: Any,
    question: str,
    choice: str,
    *,
    device: str = "cpu",
) -> float:
    """Return the summed log-likelihood of the choice tokens given the question.

    Implementation detail: we build a "question prefix" and a "prefix +
    choice" tokenisation, run the full sequence through the model, and
    sum the log-probs of the choice tokens (the positions beyond the
    prefix length). Summation, not mean, because some choices are longer
    than others and we want a true likelihood ratio.
    """
    prefix = f"Question: {question.strip()}\nAnswer:"
    full = f"{prefix} {choice.strip()}"

    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full, return_tensors="pt").input_ids.to(device)
    n_prefix = int(prefix_ids.shape[1])
    n_full = int(full_ids.shape[1])
    if n_full <= n_prefix:
        # Choice didn't introduce any new tokens (encoding merged it away).
        return 0.0

    with torch.no_grad():
        out = model(full_ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    # logits[:, t, :] predicts token t+1. To score token t (1 <= t < n_full)
    # we look at logits[:, t-1, :].
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    total = 0.0
    for t in range(n_prefix, n_full):
        next_tok = int(full_ids[0, t].item())
        total += float(log_probs[0, t - 1, next_tok].item())
    return total


@beartype
def score_capability(
    model: Any,
    tokenizer: Any,
    items: list[CapabilityItem],
    *,
    benchmark_name: str = "unknown",
    device: str = "cpu",
) -> CapabilityResult:
    """Run the log-likelihood-per-choice protocol on a list of items.

    For each item we pick ``argmax`` over the per-choice log-likelihood
    sum and compare to ``correct_index``.
    """
    per_item: list[bool] = []
    for it in items:
        lls = [
            _score_choice_loglikelihood(model, tokenizer, it.question, c, device=device)
            for c in it.choices
        ]
        predicted = int(np.argmax(lls))
        per_item.append(predicted == int(it.correct_index))
    acc = float(np.mean(per_item)) if per_item else 0.0
    return CapabilityResult(
        benchmark=benchmark_name,
        n_items=len(items),
        accuracy=acc,
        per_item_correct=per_item,
    )


# ---------------------------------------------------------------------------
# Baseline vs intervention comparison
# ---------------------------------------------------------------------------


@dataclass
class CapabilityComparison:
    """Baseline vs intervention comparison on one benchmark."""

    benchmark: str
    baseline_accuracy: float
    intervention_accuracy: float
    delta_pp: float  # percentage points, positive => intervention helps
    exceeded_max_drop: bool
    max_acceptable_drop_pp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "baseline_accuracy": float(self.baseline_accuracy),
            "intervention_accuracy": float(self.intervention_accuracy),
            "delta_pp": float(self.delta_pp),
            "exceeded_max_drop": bool(self.exceeded_max_drop),
            "max_acceptable_drop_pp": float(self.max_acceptable_drop_pp),
        }


@beartype
def compare_capability(
    baseline: CapabilityResult,
    intervention: CapabilityResult,
    *,
    max_acceptable_drop_pp: float = 2.0,
) -> CapabilityComparison:
    """Compare two :class:`CapabilityResult` snapshots on the same benchmark.

    ``delta_pp`` is ``(intervention.accuracy - baseline.accuracy) * 100``.
    ``exceeded_max_drop`` is True iff the intervention dropped accuracy
    by more than ``max_acceptable_drop_pp`` percentage points.
    """
    if baseline.benchmark != intervention.benchmark:
        logger.warning(
            "compare_capability: benchmark name mismatch (%s vs %s)",
            baseline.benchmark,
            intervention.benchmark,
        )
    delta = (intervention.accuracy - baseline.accuracy) * 100.0
    return CapabilityComparison(
        benchmark=baseline.benchmark,
        baseline_accuracy=float(baseline.accuracy),
        intervention_accuracy=float(intervention.accuracy),
        delta_pp=float(delta),
        exceeded_max_drop=bool(-delta > float(max_acceptable_drop_pp)),
        max_acceptable_drop_pp=float(max_acceptable_drop_pp),
    )


__all__ = [
    "CapabilityComparison",
    "CapabilityItem",
    "CapabilityResult",
    "compare_capability",
    "load_capability_jsonl",
    "save_capability_jsonl",
    "score_capability",
]
