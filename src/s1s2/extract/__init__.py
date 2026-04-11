"""Activation extraction pipeline.

Loads HuggingFace models, runs inference on benchmark problems, extracts
residual stream activations, computes incremental attention metrics + token
surprises, and writes everything to a single HDF5 file that conforms to
``docs/data_contract.md``.

Public API
----------
- :class:`ActivationWriter`: the HDF5 writer. Do not write to
  ``data/activations/`` from any other module.
- :func:`validate_file`: check schema conformance of an HDF5 file on disk.
- :class:`SingleModelExtractor`: the per-model extraction driver.
- :class:`ModelSpec`, :class:`GenerationConfig`, :class:`ExtractionConfig`,
  :class:`ProblemMetadata`, :class:`RunMetadata`: typed dataclasses consumed
  by the CLI.
- :func:`score_response`: behavioral classifier for a single generation.
- :func:`find_thinking_span`, :func:`split_thinking_answer`,
  :func:`compute_positions`: thinking-trace parsing helpers.
- :class:`AttentionMetricsCollector`: streaming attention metric accumulator.
"""

from s1s2.extract.core import (
    ExtractionConfig,
    GenerationConfig,
    ModelSpec,
    PerProblemOutputs,
    SingleModelExtractor,
    build_problem_metadata_from_items,
)
from s1s2.extract.hooks import (
    METRIC_NAMES,
    AttentionMetricsCollector,
    metrics_at_positions,
)
from s1s2.extract.parsing import (
    PositionInfo,
    ThinkingSpan,
    build_token_char_spans,
    compute_positions,
    find_thinking_span,
    split_thinking_answer,
)
from s1s2.extract.scoring import (
    ScoringResult,
    score_response,
    score_response_detailed,
)
from s1s2.extract.writer import (
    SCHEMA_VERSION,
    ActivationWriter,
    ProblemMetadata,
    RunMetadata,
    validate_file,
)

__all__ = [
    "METRIC_NAMES",
    "SCHEMA_VERSION",
    "ActivationWriter",
    "AttentionMetricsCollector",
    "ExtractionConfig",
    "GenerationConfig",
    "ModelSpec",
    "PerProblemOutputs",
    "PositionInfo",
    "ProblemMetadata",
    "RunMetadata",
    "ScoringResult",
    "SingleModelExtractor",
    "ThinkingSpan",
    "build_problem_metadata_from_items",
    "build_token_char_spans",
    "compute_positions",
    "find_thinking_span",
    "metrics_at_positions",
    "score_response",
    "score_response_detailed",
    "split_thinking_answer",
    "split_thinking_answer",
    "validate_file",
]
