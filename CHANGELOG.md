# Changelog

All notable changes to the s1s2 project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to semantic versioning for its public API (the `s1s2.*` package). Schema bumps
to the HDF5 activation cache use the `schema_version` field documented in
`docs/data_contract.md`.

When you change anything that touches another workstream — the data contract,
a public function signature, or a result-file format — bump the relevant
section here so other contributors can find it.

## [Unreleased]

### Added
- Initial scaffold: `pyproject.toml`, package skeleton, `CLAUDE.md`/`AGENTS.md`
  conventions, Hydra configs (`configs/{extract,probe,models}.yaml`).
- Data contract (`docs/data_contract.md`) at `schema_version=1` defining the
  HDF5 activation cache used by every workstream.
- Shared utilities: `s1s2.utils.{io, seed, stats, types, logging}`.
- Benchmark loader, validator, and templates (`s1s2.benchmark.*`).
- Activation extraction pipeline (`s1s2.extract.*`): hooks-based, incremental
  attention metrics, thinking-trace parsing, behavioral scoring, HDF5 writer.
- Linear probing pipeline (`s1s2.probes.*`): mass-mean / logistic / MLP / CCS
  probes with Hewitt-Liang controls, BH-FDR across layers, LOCO transfer.
- SAE feature analysis (`s1s2.sae.*`): loaders (Llama Scope, Gemma Scope,
  Goodfire R1, MockSAE), differential analysis with BH-FDR, Ma et al. (2026)
  falsification framework, volcano plots.
- Attention entropy pipeline (`s1s2.attention.*`): per-head Mann-Whitney U
  with BH-FDR, KV-group aggregation for GQA, Gemma sliding-window separation,
  matched-pair cross-model comparison.
- Representational geometry pipeline (`s1s2.geometry.*`): cosine silhouette
  with bootstrap CIs and permutation tests, CKA across model pairs, Two-NN
  intrinsic dimensionality, linear separability with PCA pre-reduction.
- Causal interventions pipeline (`s1s2.causal.*`): SAE feature steering,
  ablation, dose-response curves, capability-preservation checks.
- Metacognitive monitoring stretch goal (`s1s2.metacog.*`).
- Documentation: `docs/architecture.md`, `docs/SESSION_STATE.md`,
  `docs/LESSONS_LEARNED_COMPACT.md`, `docs/CONTRIBUTING.md`.
- Build infrastructure: `Makefile`, `.pre-commit-config.yaml`,
  `.github/workflows/test.yml`.
- Shared test fixtures (`tests/conftest.py`) including `synthetic_hdf5_path`,
  `synthetic_benchmark_items`, and a seeded `rng`.
- End-to-end smoke test (`scripts/smoke_test.py`) that builds a synthetic
  HDF5 cache and runs every workstream against it on CPU in under a minute.

### Changed
- (none yet)

### Removed
- (none yet)

### Data contract
- `schema_version=1` is the only supported version.

### Migration notes
- (n/a — first release)
