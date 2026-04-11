# Contributing to s1s2

Short, opinionated guide. The longer rationale lives in `CLAUDE.md` and
`docs/architecture.md`.

## Ground rules

1. **Data contract is the source of truth**. Read
   [`docs/data_contract.md`](data_contract.md) before adding code that
   touches activations. If you need to extend the schema, bump
   `SCHEMA_VERSION`, write a migration in `s1s2.utils.io`, update the
   contract doc, and add a CHANGELOG entry under "Data contract".
2. **One workstream owns one subdirectory** under `src/s1s2/`. Workstreams
   communicate through the HDF5 cache and the `results/` directory — never
   by importing each other's `core.py`.
3. **Public APIs are exposed via `__init__.py`**. If a function is not in
   `__all__` of its package's `__init__.py`, callers outside that package
   should not be using it.
4. **Never write to the activation cache** outside `s1s2.extract`. Other
   workstreams are read-only consumers.
5. **Never commit secrets**. `.env`, API keys, and credentials are
   gitignored — the pre-commit hook double-checks this.

## Coding standards

- **Python 3.11+** with type hints on all public functions.
- **`@beartype`** decorator on every public function exposed in `__init__.py`.
- **`jaxtyping`** for tensor shape annotations:
  `Float[Tensor, "batch seq hidden"]`. Use string annotations consistent
  with the data contract dimensions.
- **`ruff`** for linting (config in `pyproject.toml`), **`black`** for
  formatting (line length 100). Run `make format` before committing.
- **Docstrings explain WHY, not WHAT**. Skip "this function returns x"
  comments; do explain "we use cosine because Euclidean concentrates in
  high-dim".
- **Config-driven via Hydra**. No hardcoded paths or model IDs in code.

## Statistical standards (NON-NEGOTIABLE)

These are enforced by code review, not lint. They are not optional.

- **Multiple comparisons (exploratory)**: Benjamini-Hochberg FDR at q=0.05.
- **Multiple comparisons (confirmatory)**: Bonferroni.
- **Permutation tests**: 1000-10000 shuffles, North et al. +1 correction
  (`p = (n_extreme + 1) / (n_perms + 1)`).
- **Bootstrap CIs**: 1000 resamples, percentile method.
- **Cross-validation**: 5-fold stratified by both target label AND task
  category (use `s1s2.probes.core.make_stratify_key`).
- **Effect sizes** must be reported alongside every p-value (Cohen's d for
  parametric tests, rank-biserial for non-parametric, AUC for classification).
- **Multi-seed**: minimum 3 seeds for any stochastic procedure. Report
  mean ± std.
- **Hewitt-Liang control task is mandatory for probes**. Selectivity
  (real AUC − random-label AUC) is the meaningful metric, not raw AUC.
  See `s1s2.probes.controls`.
- **Ma et al. (2026) falsification is mandatory for SAE features**. For
  every "S1/S2 feature" candidate, run
  `s1s2.sae.falsification.ma_et_al_falsification` and report the
  spuriousness rate. See `docs/LESSONS_LEARNED_COMPACT.md` for the rationale.

## Workflow

```bash
# Setup (once)
make install              # pip install -e ".[dev]" + pre-commit hooks

# During development
make format               # black + ruff --fix
make lint                 # ruff check
make test                 # pytest
make smoke                # end-to-end smoke test on synthetic data

# Before committing
make lint && make test && make smoke
```

If `make smoke` fails, fix the underlying issue before committing. The
smoke test is the canonical "is everything wired correctly?" check.

## Adding a new workstream

See `docs/architecture.md` § "Where to extend → Adding a new analysis
workstream". Briefly:

1. Create `src/s1s2/<workstream>/` with `core.py`, `__init__.py`, and
   (when relevant) `stats.py`, `viz.py`, `cli.py`.
2. Read activations through `s1s2.utils.io` only — never invent your own
   HDF5 keys.
3. Write derived results to `results/<workstream>/...` as JSON or Parquet.
4. Add a runner to `scripts/smoke_test.py` — it should be wrapped in
   try/except so other workstreams still report a status.
5. Add a `make <workstream>` target.
6. Add unit tests in `tests/test_<workstream>.py` mirroring the pattern in
   `tests/test_probes.py`.
7. Update `docs/architecture.md` (pipeline schematic + dependency graph)
   and `CHANGELOG.md`.

## Adding a new probe target

1. Add the literal to `s1s2.utils.types.ProbeTarget`.
2. Implement `_target_<name>` in `s1s2.probes.targets` returning a
   `TargetData` dataclass with `y`, `mask`, `stratify_key`, `group_id`,
   and `category`.
3. Add the new target to `ALL_TARGETS`.
4. Add a unit test in `tests/test_probes.py` that builds the target on a
   synthetic HDF5 cache and asserts the shapes.

## Adding a new model

1. Add an entry to `configs/models.yaml` with `hf_id`, `hdf5_key`, `family`,
   architecture dimensions, `is_reasoning`, optional `sae_release` and
   `sliding_window`.
2. Run `python scripts/extract_all.py models_to_extract=[your-model]`.
3. Update `models_to_probe` in any analysis configs that should pick it up.

## Commit hygiene

- Concise commit messages, focus on **why** rather than **what**.
- Commit frequently — always one `git reset` away from the last good state.
- Never force-push to `main` without explicit permission.
- Pre-commit hooks (`make install` sets these up) will catch most lint /
  formatting issues. Don't bypass them with `--no-verify` — fix the
  underlying problem.

## Code review

For load-bearing changes (reward functions, statistical tests, the data
contract), run a Codex CLI review before merging:

```bash
codex exec review --uncommitted
```

Different model = different blind spots. Catches bugs that self-review
misses. See the global `~/.claude/CLAUDE.md` for the writer-reviewer pattern.
