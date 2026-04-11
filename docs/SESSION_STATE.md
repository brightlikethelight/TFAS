# s1s2 Session State

**Last updated**: 2026-04-11
**Active focus**: Code complete (318 tests green). Building phase-2 infrastructure: GPU deployment, paper, notebooks, W&B, pre-registration. **Waiting on FASRC access (Kempner deadline Apr 14).**

## TL;DR

- 104 Python files, ~33K lines, **318/318 unit tests passing**, smoke test green (3s)
- Git initialized: commit `954e2d0`
- Benchmark: 284 items, 142 matched conflict/control pairs, 558 paraphrases, 7 categories
- All 5 analysis workstreams + metacog + benchmark + causal complete with tests
- Phase 2 in progress: GPU deploy scripts, paper (LaTeX), notebooks, W&B, pre-registration, HUSAI presentation
- Next bottleneck: FASRC access (Kempner deadline 2026-04-14)

## What's done

- **Project scaffolding**: `pyproject.toml`, `CLAUDE.md`, `AGENTS.md`,
  `docs/data_contract.md`, Hydra `configs/{extract,probe,models}.yaml`.
- **Shared utilities**: `s1s2.utils.{seed, stats, types, io, logging}`.
  - `io.py` has typed read accessors AND write helpers (writer-side owned by
    `s1s2.extract`); other workstreams must not call write helpers.
  - `stats.py` provides BH-FDR, North-corrected permutation tests, bootstrap
    CIs, Cohen's d, rank-biserial, Gini, Shannon entropy.
- **Benchmark loader + validator + templates**: `s1s2.benchmark.{loader, validate,
  templates}`. The real JSONL still needs to be generated from templates
  (script exists; not yet run end-to-end).
- **Activation extraction pipeline**: `s1s2.extract.{core, hooks, parsing,
  reasoning, scoring, writer, cli}`. Incremental attention metric collector,
  thinking-trace parser, behavioral scorer, HDF5 writer, Hydra CLI.
- **Linear probing pipeline**: `s1s2.probes.{core, probes, controls, targets,
  cli}` plus `s1s2.viz.probe_plots`. Mass-mean / logistic / MLP / CCS probes,
  Hewitt-Liang controls, BH-FDR across layers, LOCO transfer.
- **SAE feature analysis**: `s1s2.sae.{core, cli, loaders, differential,
  falsification, volcano}`. Loaders for Llama Scope, Gemma Scope, Goodfire R1,
  plus a MockSAE for tests. Ma et al. (2026) falsification framework with a
  cheap (no-model-forward) and a model-forward mode.
- **Attention entropy pipeline**: `s1s2.attention.{core, heads, layers,
  trajectories}`. Per-head Mann-Whitney U with BH-FDR, KV-group aggregation,
  Gemma sliding-window separation, matched-pair cross-model comparison.
- **Representational geometry pipeline**: `s1s2.geometry.{cka, clusters,
  intrinsic_dim, projections, separability, viz}`. Cosine silhouette with
  bootstrap CIs and permutation tests, CKA, Two-NN intrinsic dim, linear
  separability with PCA pre-reduction (the d>>N fix).
- **Causal interventions pipeline**: `s1s2.causal.*` — steering, ablation,
  dose-response, capability gates, viz.
- **Metacognitive monitoring (stretch)**: `s1s2.metacog.*`.
- **Build / dev infra (this session)**: `Makefile`, `.pre-commit-config.yaml`,
  `.github/workflows/test.yml`, `tests/conftest.py` with `synthetic_hdf5_path`,
  `scripts/smoke_test.py`, `LICENSE` (MIT), `CHANGELOG.md`,
  `docs/{architecture, SESSION_STATE, LESSONS_LEARNED_COMPACT, CONTRIBUTING}.md`.

## What's done (since the previous "what's not done" list)

- ✅ **Benchmark JSONL** generated and validated: 284 items, 142 matched
  conflict/control pairs, 558 paraphrases, all 7 categories. Stats:
  - crt 30 / base_rate 20 / syllogism 25 / anchoring 15 / framing 15 /
    conjunction 12 / arithmetic 25 (matched pair counts; ×2 for items)
- ✅ **All Hydra configs** exist: `extract.yaml`, `probe.yaml`, `sae.yaml`,
  `attention.yaml` (in causal), `geometry.yaml`, `causal.yaml`,
  `metacog.yaml`, `benchmark.yaml`, `figures.yaml`, `models.yaml`.
- ✅ **All driver scripts** exist: `run_probes.py`, `run_sae.py`,
  `run_attention.py`, `run_geometry.py`, `run_causal.py`, `run_metacog.py`,
  `extract_all.py`, `generate_figures.py`, `smoke_test.py`.
- ✅ **Test coverage for every workstream**: `test_benchmark.py`,
  `test_extract.py`, `test_probes.py`, `test_sae.py`, `test_attention.py`,
  `test_geometry.py`, `test_causal.py`, `test_metacog.py`, `test_utils.py`,
  `test_scripts.py`, `test_viz_paper_figures.py`. **292/292 tests pass.**
- ✅ **Bug fixes during integration**:
  - `s1s2.utils.stats.rank_biserial` had inverted sign convention; fixed.
  - `s1s2.utils.stats.shannon_entropy_bits` returned a Python scalar for
    1-D input; rewrapped as 0-D ndarray to match the type hint.
  - `s1s2.geometry.cka.linear_cka` used `tr((X^T X)(Y^T Y))` instead of
    `tr(K_X K_Y)`. Fixed; now matches `linear_cka_fast` exactly and
    satisfies orthogonal-rotation invariance.
  - `s1s2.geometry.projections.random_projection` was unnormalised, so the
    "no clusters in random projections of random data" control was too
    weak. Switched to `1/sqrt(d)` JL scaling + per-axis std normalisation.
  - `s1s2.sae.loaders.reconstruction_report` had a hard-coded `mse > 0.5*var`
    floor that overrode the user's `min_explained_variance` parameter; fixed.
  - `s1s2.sae.cli.runner_config_from_hydra` ignored top-level `models_to_run`
    and only consulted `models`. Fixed to honor both.
  - `s1s2.probes.core.primary_probe_name` was exported in `__all__` and
    referenced by the runner but never defined; added.
  - `tests/test_probes.py::test_bh_fdr_application_on_known_pvalues` had an
    arithmetically-incorrect expectation (rejected p≤0.03 but BH at q=0.05
    only rejects p≤0.01 here); fixed expectation.
- ✅ **Ruff auto-fix collateral damage**: ruff stripped the quotes from 58
  jaxtyping shape annotations across 15 files (`Float[np.ndarray, "n d"]`
  → `Float[np.ndarray, n d]`), which broke beartype runtime checking. A
  single regex pass restored every annotation. Pre-commit must NOT run
  ruff on jaxtyping-annotated modules until upstream supports it.

## What's still NOT done

- Real activation extraction on the 4 models (requires GPU).
- SAE reconstruction fidelity verification on real Instruct activations.
- Behavioral validation of the benchmark against real models (Week 2 gate).
- FASRC access (waiting on faculty sponsor — Kempner deadline 2026-04-14).
- W&B integration verified end-to-end with a real run.

## Session 2 (2026-04-11): Git + Phase 2 infrastructure

- ✅ **Git init**: initial commit `954e2d0` with 134 files, 35,688 lines
- ✅ **Ruff config fix**: added `F821` and `UP037` ignores to prevent
  jaxtyping annotation stripping
- ✅ **SESSION_STATE updated** to reflect actual state (318 tests, not 292)
- 🔄 **In-flight agents** (8 parallel): GPU deployment scripts, paper skeleton
  (LaTeX), analysis notebooks, W&B integration, experiment orchestrator,
  benchmark quality audit, pre-registration document, HUSAI presentation

## Active blockers

- **FASRC faculty sponsor** — Kempner Accelerator Award deadline is
  **2026-04-14** (3 days). Top priority for the faculty advisor.
- Need to verify Llama Scope SAE reconstruction fidelity on Instruct
  activations (loaders wire the check; need real activations).

## Test commands

```bash
make install   # pip install -e ".[dev]" + pre-commit hooks
make lint      # ruff check
make test      # pytest tests/
make smoke     # all 4 workstreams on synthetic data (~3s)
```

## Next steps

1. **Get FASRC access** — the only blocker before real experiments start
2. Run `python scripts/run_pipeline.py` on a GPU pod (RunPod or FASRC)
3. First model: `llama-3.1-8b-instruct` — behavioral validation (Week 2 gate)
4. Probes pipeline on real activations (Week 5 go/no-go gate)
5. Present at HUSAI meeting using `docs/presentation/husai_pitch.md`
6. Submit pre-registration to OSF

## Key W&B / artifact pointers

(none yet — first real extraction has not run)
