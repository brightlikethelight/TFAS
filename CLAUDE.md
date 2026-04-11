# Project Conventions for the s1s2 Codebase

This file is loaded into context for all Claude Code sessions in this repo. It documents conventions, the data contract, and gotchas. Read it before editing.

## Project at a Glance

This is a research codebase for **"Mechanistic Signatures of System 1 vs System 2 Processing in LLMs"** — a HUSAI semester project. We probe the question: do LLMs have a graded "deliberation intensity" dimension internally, and does reasoning training amplify it?

Five analysis workstreams operate over a shared activation cache (HDF5):

1. **Linear probing** (`src/s1s2/probes/`)
2. **SAE feature analysis** (`src/s1s2/sae/`)
3. **Attention entropy** (`src/s1s2/attention/`)
4. **Representational geometry** (`src/s1s2/geometry/`)
5. **Causal interventions** (`src/s1s2/causal/`)

Plus a benchmark builder (`src/s1s2/benchmark/`), an extraction pipeline (`src/s1s2/extract/`), and a metacognitive monitoring stretch goal (`src/s1s2/metacog/`).

## Critical Framing Rules

- **Never** call this "LLMs have System 1 and System 2." Use "S1-like / S2-like processing signatures" or "deliberation-intensity gradient."
- The dual-process framing is **operational**, not a cognitive claim. Frame mechanistic findings, not anthropomorphism.
- Treat S1/S2 as a **continuous** dimension, not a binary. Even Evans & Stanovich abandoned the binary.

## Coding Standards

- **Python 3.11+** with type hints on all public functions.
- **`beartype`** decorator on public API functions exposed in `__init__.py`.
- **`jaxtyping`** for tensor shape annotations: `Float[Tensor, "batch seq hidden"]`.
- **`ruff`** for linting, **`black`** for formatting (line length 100).
- Docstrings explain WHY, not WHAT. Skip obvious comments.
- Config-driven via Hydra; no hardcoded paths in code.
- Never log secrets. Use `.env` (gitignored) for API keys.

## Module Structure (every src/s1s2/<workstream>/ dir)

```
<workstream>/
├── __init__.py        # Public API exports
├── core.py            # Main analysis logic (functions, no globals)
├── data.py            # IO: load activations from HDF5, write results
├── stats.py           # Statistical tests (BH-FDR, permutation, bootstrap)
├── viz.py             # Plotting functions
└── cli.py             # Hydra-decorated CLI entry point
```

Tests live in `tests/test_<workstream>.py` (mirroring the module name).

## Data Contract — READ `docs/data_contract.md`

All workstreams read activations from a single HDF5 file format. The schema is in `docs/data_contract.md`. **Do not invent your own format.** If the contract needs extending, update the spec doc and notify other workstream owners via the CHANGELOG.

## Statistical Standards (NON-NEGOTIABLE)

- **Multiple comparisons**: Benjamini-Hochberg FDR at q=0.05 across all tests within a workstream. Bonferroni for hypothesis-confirmatory tests only.
- **Effect sizes**: always report alongside p-values. Cohen's d, rank-biserial correlation, or AUC depending on the test.
- **Confidence intervals**: bootstrap 1000 resamples for accuracy/AUC metrics.
- **Permutation tests**: 1000-10000 shuffles for null distributions. The North et al. (2002) +1 correction: `(n_extreme + 1) / (n_perms + 1)`.
- **Cross-validation**: 5-fold stratified by task category AND target label. Nested CV for hyperparameter selection.
- **Hewitt & Liang control task**: every probe must report selectivity (real - random-label accuracy). Selectivity < 5pp = signal is probe expressiveness, not representation.
- **Multi-seed**: 3+ seeds for everything stochastic. Report mean ± std.

## Critical Confound Controls

1. **Task difficulty confound**: every conflict (S1) item has a no-conflict control matched on surface form and difficulty. Run analyses on the matched subset.
2. **Sequence length confound** (attention entropy): always normalize by `log2(t)` AND report Gini coefficient (which is scale-invariant).
3. **d >> N pitfall** (geometry): with 4096 dims and ~500 points, random classes are linearly separable. PCA to 50-100 dims before SVM.
4. **Probe expressiveness**: Hewitt & Liang controls. Mandatory.
5. **SAE feature spuriousness**: Ma et al. (2026) falsification. Inject top tokens into random text — if feature still activates, it's spurious. Mandatory.
6. **Memorization** (benchmark): novel structural isomorphs only. Classic CRT items are baselines for measuring contamination, not primary stimuli.

## Models We Use (do not hardcode IDs in scripts)

| Model key | HuggingFace ID | Layers | Q heads | KV heads | Hidden | Reasoning |
|-----------|----------------|--------|---------|----------|--------|-----------|
| llama-3.1-8b-instruct | `meta-llama/Llama-3.1-8B-Instruct` | 32 | 32 | 8 | 4096 | no |
| gemma-2-9b-it | `google/gemma-2-9b-it` | 42 | 16 | 8 | 3584 | no |
| r1-distill-llama-8b | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 32 | 32 | 8 | 4096 | yes |
| r1-distill-qwen-7b | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 28 | 28 | 4 | 3584 | yes |

HDF5 model keys are the HuggingFace ID with `/` → `_`.

## Pre-Trained SAEs

- **Llama Scope** (`fnlp/Llama-3_1-8B-Base-LXR-32x`): SAEs for **Llama-3.1-8B-Base** (not Instruct). Reconstruction fidelity may degrade on Instruct activations — verify before trusting downstream.
- **Gemma Scope** (`google/gemma-scope-9b-it-res`): SAEs for Gemma-2-9b-it at multiple layers, widths 16K to 1M.
- **Goodfire R1 SAE** (`Goodfire/DeepSeek-R1-SAE-l37`): SAE on layer 37 of DeepSeek-R1 (671B). May not transfer to R1-Distill — verify.
- **For R1-Distill models**: pre-trained SAEs likely don't exist. Either train custom SAEs via SAELens, or report distillation comparison via probes only.

## Gotchas

### Attention extraction
- **Never materialize the full attention matrix** for long sequences. At 32K tokens × 32 heads × 32 layers × float32, that's ~128 GB per layer. Compute entropy/Gini **incrementally** per generation step inside the hook.
- HuggingFace's `output_attentions=True` returns expanded query-head attention even for GQA. Verify shape.
- **Gemma-2 sliding window**: odd layers use 4096-token window. Analyze odd vs even layers separately. Never pool them.

### GQA non-independence
- Llama: 32 query heads × 8 KV heads (groups of 4). Heads in the same KV group share key/value projections — they are NOT statistically independent.
- Report at TWO granularities: (1) per query-head, (2) per KV-group. KV-group is the more conservative.

### Reasoning model thinking traces
- R1-Distill emits `<think>...</think>` blocks. Parse before computing answer-position metrics.
- Use `max_new_tokens=4096` minimum for reasoning models. Generation can be 32K tokens.
- The "performative reasoning" trap: a model might "decide" at T0 and emit decorative reasoning. Compare probe accuracy at T0 vs Tend to detect.
- "Aha moments" (`wait`, `actually`) are mostly decorative — only ~2.3% causally influence the final answer. Use TrueThinking Score to filter.

### SAE pitfalls
- **Reconstruction fidelity check first**: load the SAE, reconstruct on a sample of activations, compute MSE / explained variance. If reconstruction loss is high, the SAE doesn't fit your model — abort before trusting downstream.
- **Ma et al. falsification**: For every "S1/S2-specific feature," inject the top-3 activating tokens into 100 random non-cognitive-bias texts. If the feature still activates, it's a token-level artifact. Report falsification rate.
- **Token-trivial features**: many "reasoning features" fire on tokens like "Let" or "First". Always inspect top-activating tokens.

### Probing pitfalls
- **Hewitt & Liang control tasks**: train probes on random labels. Selectivity (real - control) is the meaningful number, not raw accuracy.
- **Cross-domain transfer**: train on 6 categories, test on 7th. If accuracy drops to chance, the probe learned domain features, not S1/S2 features.

### Geometry pitfalls
- **Random projection baseline**: if your UMAP shows clean clusters, project to 2D via random Gaussian matrices 100 times — if clusters appear in random projections too, structure is genuine and UMAP is unnecessary; if they don't, UMAP is hallucinating.
- **d >> N**: Cover's theorem says random classes are linearly separable when dim >> samples. PCA to 50-100 components first.

## Reproducibility

- Always seed RNGs: `torch.manual_seed`, `np.random.seed`, `random.seed`. Use `s1s2.utils.seed.set_global_seed()`.
- Save Hydra configs alongside results for every run.
- W&B run ID written to results JSON for traceability.
- For long extraction jobs: checkpoint after each model so partial progress is saved.

## Where to Save Results

- **Activations**: `data/activations/{run_name}.h5` (gitignored)
- **Per-workstream results**: `results/{workstream}/{model}_{target}_{layer}.json`
- **Figures**: `figures/{figure_name}.{pdf,png}` (gitignored, but reproducible from results)
- **W&B**: project `s1s2`, group by workstream, tags by model
