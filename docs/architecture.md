# s1s2 Architecture

This document describes how the s1s2 codebase fits together. For the experimental
framing and motivation see `README.md`. For the file format every workstream
agrees on, see `docs/data_contract.md` — that is the single source of truth
for activations and is referenced from this document.

## 1. Project overview

s1s2 is a research codebase for *"Mechanistic Signatures of Dual-Process
Cognition in LLMs"* (HUSAI semester project). We run a battery of cognitive
bias tasks through standard and reasoning models, cache their internal
activations, and probe those activations with five complementary methods to
test whether reasoning training amplifies a graded "deliberation intensity"
dimension.

We deliberately treat S1/S2 as a *continuous* operational distinction, not a
binary cognitive claim. See `CLAUDE.md` for the framing rules.

## 2. Pipeline schematic

```
                          ┌──────────────────────┐
                          │   benchmark/         │
                          │   (JSONL templates,  │
                          │   loader, validator) │
                          └──────────┬───────────┘
                                     │ BenchmarkItem[]
                                     ▼
                          ┌──────────────────────┐
                          │   extract/           │
                          │   HF model + hooks   │
                          │   thinking-trace     │
                          │   parsing, scoring,  │
                          │   incremental attn   │
                          │   metrics, surprises │
                          └──────────┬───────────┘
                                     │ writes
                                     ▼
                       ┌─────────────────────────────┐
                       │  data/activations/{run}.h5  │
                       │  schema_version=1           │
                       │  (see docs/data_contract.md)│
                       └─────────────┬───────────────┘
                                     │ read-only
        ┌──────────────┬─────────────┼─────────────┬──────────────┐
        ▼              ▼             ▼             ▼              ▼
  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ probes/  │  │  sae/      │  │attention/│  │geometry/ │  │ metacog/ │
  │ linear / │  │  Llama,    │  │ Mann-W   │  │ silhou., │  │ surprise │
  │ MLP /    │  │  Gemma,    │  │ U + BH,  │  │ CKA,     │  │ corr,    │
  │ CCS,     │  │  Goodfire, │  │ KV-group │  │ separab. │  │ trajec.  │
  │ H&L      │  │  Mock SAE; │  │ aggreg., │  │ w/ d>>N  │  │          │
  │ controls │  │  Ma et al  │  │ Gemma    │  │ fix      │  │          │
  │          │  │  falsific. │  │ window   │  │          │  │          │
  └────┬─────┘  └─────┬──────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
       │              │               │              │              │
       └──────────────┴───────────────┼──────────────┴──────────────┘
                                      │ derived results JSON
                                      ▼
                          ┌──────────────────────┐
                          │   causal/            │
                          │   feature steering,  │
                          │   ablation,          │
                          │   dose-response,     │
                          │   capability gates   │
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │   viz/ + figures/    │
                          │   publication plots  │
                          └──────────────────────┘
```

The cache is **write-once, read-many**. Only `s1s2.extract.*` writes to it; the
five analysis workstreams plus `causal/` and `metacog/` are strict consumers.
`docs/data_contract.md` is the contract that decouples them.

## 3. Module layout

```
src/s1s2/
├── benchmark/      Loader, validator, templates for the cognitive bias benchmark
├── extract/        Activation extraction pipeline (writer, hooks, scoring, parsing)
├── probes/         Linear / MLP / CCS probes with Hewitt-Liang controls
├── sae/            SAE feature analysis with Ma et al. falsification
├── attention/      Per-head differential analysis with GQA + sliding-window aware
├── geometry/       Silhouette / CKA / separability with d>>N mitigation
├── causal/         Steering, ablation, dose-response, capability checks
├── metacog/        Surprise correlation, difficulty detector, trajectory probing
├── viz/            Plotting helpers (read from results/)
└── utils/          Shared: io, seed, stats, types, logging
```

Each workstream follows a consistent internal layout (see `CLAUDE.md`):

```
<workstream>/
├── __init__.py     Public API surface
├── core.py         Main analysis logic — pure functions, no globals
├── stats.py        Workstream-specific statistical helpers (when needed)
├── data.py         IO helpers (when needed beyond utils.io)
├── viz.py          Plotting (gitignored output)
└── cli.py          Hydra-decorated entry point
```

The split keeps `core.py` independently unit-testable: tests pass plain numpy
arrays into pure functions and never need a Hydra context.

## 4. Module dependency graph

Allowed dependencies are strictly downstream. Cycles are forbidden.

```
benchmark   ── (no dependencies on other workstreams)
utils       ── (foundation; everything imports from here)
extract     ── benchmark, utils
probes      ── utils
sae         ── utils
attention   ── utils
geometry    ── utils
metacog     ── utils
causal      ── extract, sae, probes, utils
viz         ── (reads results/ directly; no upstream dependency)
```

In particular: probes / sae / attention / geometry never import from each
other. They communicate exclusively through the HDF5 cache and the
`results/` directory.

## 5. Data contract

The HDF5 schema is documented in [docs/data_contract.md](data_contract.md).

Highlights:

* One HDF5 file per benchmark run: `data/activations/{run_name}.h5`.
* Schema version is currently **1**. Bumping requires a migration in
  `s1s2.utils.io.migrate()` and a CHANGELOG entry.
* Per-problem metadata in `/problems/...` is shared across models.
* Per-model groups under `/models/{hf_id_with_underscores}/` carry residual
  streams, attention metrics, token surprises, generations, and behavior.
* Position labels (`P0`, `P2`, `T0`, `T25`, `T50`, `T75`, `Tend`, `Tswitch`)
  are uniform across reasoning and non-reasoning models — non-applicable
  positions are simply marked `valid=False` to keep ragged datasets out.
* Read accessors live in `s1s2.utils.io`. Writer helpers live there too,
  but only `s1s2.extract.*` is allowed to call them.

## 6. Statistical contract

Every analysis workstream is held to the same rules (`CLAUDE.md` is the
source of truth; this is a summary):

| Concern                            | Rule                                                        |
|------------------------------------|-------------------------------------------------------------|
| Multiple comparisons (exploratory) | Benjamini-Hochberg FDR at q=0.05                            |
| Multiple comparisons (confirmatory)| Bonferroni                                                  |
| Effect sizes                       | Always reported alongside p-values (Cohen's d, r_rb, AUC)   |
| Confidence intervals               | Bootstrap percentile, 1000 resamples                        |
| Permutation tests                  | 1000-10000 shuffles, North et al. +1 correction             |
| Cross-validation                   | 5-fold stratified by (target, category)                     |
| Probe expressiveness control       | Hewitt-Liang random-label baseline; selectivity is the metric|
| Multi-seed                         | Minimum 3 seeds for any stochastic procedure                |

Implementations live in `s1s2.utils.stats`. Workstreams must use them.

## 7. Reproducibility story

* **Seeds**: every script calls `s1s2.utils.seed.set_global_seed(seed)`,
  which seeds Python's `random`, NumPy, PyTorch CPU+CUDA, and (optionally)
  enables deterministic cuBLAS GEMMs via `CUBLAS_WORKSPACE_CONFIG`.
* **Configs**: every experiment is driven by Hydra YAML in `configs/`. The
  full config is serialized into HDF5 (`/metadata/config`) and into every
  result JSON, so the input that produced any artifact is always
  reconstructable.
* **Git SHA**: written into `/metadata/git_sha` and into every result file.
* **W&B**: every run logs to W&B (`project=s1s2`, group by workstream, tags
  by model). The run ID is also written into the result JSON for traceability.
* **Checkpoints**: long extraction jobs checkpoint after each model, so a
  pod termination loses at most one model's worth of progress.
* **Synthetic smoke test**: `scripts/smoke_test.py` runs the full pipeline on
  CPU in <60 s using a synthetic HDF5 cache. CI runs it on every commit.

## 8. Where to extend

### Adding a new model

1. Add an entry to `configs/models.yaml`. Required fields: `hf_id`, `hdf5_key`
   (HF id with `/` -> `_`), `family`, `n_layers`, `n_heads`, `n_kv_heads`,
   `hidden_dim`, `head_dim`, `is_reasoning`, optional `sae_release` and
   `sliding_window`.
2. The extraction pipeline picks up the new model automatically. Run
   `python scripts/extract_all.py models_to_extract=[your-model]`.
3. Update the `models_to_probe` list in `configs/probe.yaml` (and the
   matching list in any analysis configs you want it to run on).
4. Sanity check with `python scripts/smoke_test.py` against your synthetic
   data path; the analysis modules will pick up the new model from the
   cache.

### Adding a new probe target

1. Add the target string to `s1s2.utils.types.ProbeTarget`.
2. Implement `_target_<name>` in `s1s2.probes.targets` returning a
   `TargetData` dataclass. Decide what `mask`, `stratify_key`, `group_id`,
   and `category` should mean for your target.
3. Add the new target to `ALL_TARGETS`.
4. Add a unit test in `tests/test_probes.py` that builds the target from a
   synthetic HDF5 cache and checks `td.y.shape[0] == td.mask.sum()`.

### Adding a new analysis workstream

1. Create a new directory under `src/s1s2/<workstream>/` following the
   `core.py / stats.py / viz.py / cli.py / __init__.py` layout.
2. Read activations through `s1s2.utils.io` accessors only. Never invent
   your own HDF5 keys.
3. Write derived results to `results/<workstream>/...`. Never modify the
   HDF5 cache.
4. Add a runner to `scripts/smoke_test.py` and a `make <workstream>` target.
5. Document the analysis in `README.md` and link from this file's pipeline
   schematic.

### Changing the data contract

1. Bump `SCHEMA_VERSION` in `s1s2.utils.io`.
2. Add migration logic to `s1s2.utils.io.migrate(...)`.
3. Update `docs/data_contract.md` with the new schema.
4. Add a CHANGELOG entry under "Data contract".
5. Re-extract any caches (or run the migration helper).
6. Notify other workstream owners — the data contract is shared by
   convention, not technically enforced.

## 9. Smoke test as the canonical "is it working" check

`scripts/smoke_test.py` is the single command for "did I break the cross-
workstream wiring?". It:

* Builds a 20-problem / 4-layer synthetic HDF5 cache via the same writer
  helpers production extraction uses.
* Runs each workstream against it (probes, SAE w/ MockSAE + Ma et al.
  falsification, attention with synthetic per-head metrics, geometry).
* Wraps each runner in try/except so a single broken workstream doesn't
  abort the rest.
* Prints a summary table and exits non-zero if anything failed.

CI runs it on every commit. Run it locally with `make smoke`.

## 10. Result file conventions

Every workstream writes its derived results to `results/<workstream>/`.
File naming follows `{model}_{target}_layer{NN}_{position}.json` (or a
nested variant for older code paths). The contents are always:

* The summary statistics (point estimate, CI, p-value).
* The full per-fold breakdown (for probes) or per-head dataframe (for
  attention).
* The Hydra config that produced them.
* The git SHA and W&B run ID.

This is sufficient to reproduce a figure from a result file with no other
state — `viz/` reads results files directly.
