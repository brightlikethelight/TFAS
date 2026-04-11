"""SAE feature analysis pipeline.

Loads pre-trained SAEs (Llama Scope, Gemma Scope, Goodfire R1), computes
differential feature activation between conflict and no-conflict items,
applies the Ma et al. (2026) falsification framework, and exports volcano
plots and ranked feature lists.

Public entry points
-------------------
* :class:`SAEAnalysisRunner` — orchestration across ``(model, layer)`` cells.
* :class:`SAERunnerConfig` — knobs passed to the runner.
* :func:`run_sae_analysis` — functional wrapper around the runner.
* :func:`run_sae_from_hydra` — Hydra-aware driver used by the CLI.

Primitives
----------
* :func:`load_sae_for_model`, :class:`MockSAE` — SAE loaders.
* :func:`reconstruction_report` — fidelity gate.
* :func:`encode_batched`, :func:`differential_activation`,
  :func:`matched_pair_subset` — differential test primitives.
* :func:`ma_et_al_falsification`, :func:`falsify_candidates`,
  :func:`find_trigger_tokens` — Ma et al. (2026) spurious filter.
* :func:`plot_volcano` — the canonical volcano plot.
"""

from __future__ import annotations

from s1s2.sae.core import (
    DEFAULT_LAYERS_FOR_MODEL,
    DEFAULT_MODEL_HDF5_KEYS,
    DEFAULT_MODEL_SAE_RELEASES,
    SAEAnalysisRunner,
    SAERunnerConfig,
    run_sae_analysis,
)
from s1s2.sae.differential import (
    DifferentialResult,
    differential_activation,
    encode_batched,
    matched_pair_subset,
)
from s1s2.sae.falsification import (
    RANDOM_TEXTS,
    FalsificationResult,
    falsify_candidates,
    find_trigger_tokens,
    get_random_texts,
    ma_et_al_falsification,
)
from s1s2.sae.loaders import (
    MockSAE,
    ReconstructionReport,
    SAEHandle,
    load_gemma_scope,
    load_goodfire_r1,
    load_llama_scope,
    load_sae_for_model,
    reconstruction_report,
)
from s1s2.sae.volcano import plot_volcano

__all__ = [  # noqa: RUF022 - grouped by subsystem, not alphabetised
    # Orchestration
    "SAEAnalysisRunner",
    "SAERunnerConfig",
    "run_sae_analysis",
    "DEFAULT_MODEL_HDF5_KEYS",
    "DEFAULT_MODEL_SAE_RELEASES",
    "DEFAULT_LAYERS_FOR_MODEL",
    # Loaders
    "SAEHandle",
    "MockSAE",
    "ReconstructionReport",
    "reconstruction_report",
    "load_llama_scope",
    "load_gemma_scope",
    "load_goodfire_r1",
    "load_sae_for_model",
    # Differential
    "DifferentialResult",
    "encode_batched",
    "differential_activation",
    "matched_pair_subset",
    # Falsification
    "RANDOM_TEXTS",
    "get_random_texts",
    "FalsificationResult",
    "find_trigger_tokens",
    "ma_et_al_falsification",
    "falsify_candidates",
    # Plotting
    "plot_volcano",
]


def __getattr__(name: str):
    """Lazily import the Hydra-dependent CLI entry points.

    Hydra must not be imported at package-import time because ``pytest``
    collects ``s1s2.sae`` during test discovery and we want tests that
    don't touch Hydra to keep working even if the Hydra config hasn't
    been authored or if ``hydra-core`` changes its import side effects.
    """
    if name in ("run_sae_from_hydra", "runner_config_from_hydra", "main"):
        from s1s2.sae import cli as _cli

        return getattr(_cli, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
