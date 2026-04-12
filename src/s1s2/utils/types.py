"""Common type aliases.

Centralizes string identifiers, position labels, and category names so a
typo doesn't propagate silently across workstreams.
"""

from __future__ import annotations

from typing import Final, Literal

# Position labels (see docs/data_contract.md)
PositionLabel = Literal["P0", "P2", "T0", "T25", "T50", "T75", "Tend", "Tswitch"]

ALL_POSITIONS: Final[tuple[PositionLabel, ...]] = (
    "P0",
    "P2",
    "T0",
    "T25",
    "T50",
    "T75",
    "Tend",
    "Tswitch",
)
STANDARD_POSITIONS: Final[tuple[PositionLabel, ...]] = ("P0", "P2")
REASONING_POSITIONS: Final[tuple[PositionLabel, ...]] = (
    "P0",
    "P2",
    "T0",
    "T25",
    "T50",
    "T75",
    "Tend",
    "Tswitch",
)

# Task categories
TaskCategory = Literal[
    "crt",
    "base_rate",
    "syllogism",
    "anchoring",
    "framing",
    "conjunction",
    "arithmetic",
    "sunk_cost",
]

ALL_CATEGORIES: Final[tuple[TaskCategory, ...]] = (
    "crt",
    "base_rate",
    "syllogism",
    "anchoring",
    "framing",
    "conjunction",
    "arithmetic",
    "sunk_cost",
)

# Behavioral response classification
ResponseCategory = Literal["correct", "lure", "other_wrong", "refusal"]

# Probe target labels
ProbeTarget = Literal[
    "task_type",        # conflict (S1) vs no-conflict (S2)
    "correctness",      # will model answer correctly?
    "bias_susceptible", # will model give the lure answer specifically?
    "processing_mode",  # matched-pair contrast
]

# Model keys (must match configs/models.yaml)
ModelKey = Literal[
    "llama-3.1-8b-instruct",
    "gemma-2-9b-it",
    "r1-distill-llama-8b",
    "r1-distill-qwen-7b",
]

ALL_MODELS: Final[tuple[ModelKey, ...]] = (
    "llama-3.1-8b-instruct",
    "gemma-2-9b-it",
    "r1-distill-llama-8b",
    "r1-distill-qwen-7b",
)
