# Lessons Learned (Compact)

Accumulates critical lessons future sessions should know. Keep entries 1-3
sentences each. If a lesson grows beyond a few sentences, link out to a
longer note rather than expanding it here.

## Methodology

- **Graded, not binary**. S1/S2 must be framed as a graded "deliberation
  intensity" dimension, not a binary. Even Evans & Stanovich abandoned the
  binary; Coda-Forno et al. (2025) found overlapping subspaces.
- **Difficulty controls are non-negotiable**. Every S1 (conflict) item needs
  a no-conflict matched-difficulty control. Difficulty confound is the #1
  reviewer criticism on every cognitive bias / LLM paper.
- **Behavior first, mechanism second**. The Week 2 gate is behavioral
  validation: models must show >30% lure responses on conflict items before
  any mechanistic claim is meaningful. If they don't, the benchmark is
  measuring noise.

## Tooling

- **Never call HuggingFace with `output_attentions=True` and store the
  result**. At 32K tokens × 32 heads × 32 layers × float32 that's ~128 GB
  per layer. Compute attention metrics incrementally inside the hook
  (`s1s2.extract.hooks.AttentionMetricsCollector` does this).
- **TransformerLens `HookedTransformer` + `hook_pattern`** is the cleanest
  way to capture attention patterns when you do need them, but only for
  short prompts.
- **HDF5 + h5py is fine** for the activation cache despite the bf16 gap —
  store as float16 (lossy) or float32 (2x storage, lossless). The data
  contract documents this.
- **Hydra configs**: serialize the resolved config into the HDF5
  `/metadata/config` and into every result JSON. Otherwise you cannot
  reproduce a number you wrote down.

## Models

- **Llama-3.1-8B uses GQA**: 32 query heads × 8 KV heads. Heads in the same
  KV group are NOT statistically independent (they share key/value
  projections). Report at TWO granularities: per query-head (exploratory)
  and per KV-group (confirmatory, more conservative).
- **Gemma-2 alternates attention kernels**: odd layers use a 4096-token
  sliding window, even layers are global. Analyze them separately. Pooling
  is a category error.
- **R1-Distill reasoning models** generate up to 32K tokens in
  `<think>...</think>`. Need `max_new_tokens >= 4096` minimum (we use 8192
  by default). Parse the thinking span before computing answer-position
  metrics.

## SAEs

- **Llama Scope** is trained on Llama-3.1-8B-**Base**, NOT Instruct. Verify
  reconstruction fidelity (>=0.5 explained variance) before trusting any
  feature analysis. The loader logs a warning but does NOT abort — that's
  the caller's job.
- **Goodfire R1 SAE** is trained on the 671B model, not the 8B distill.
  Likely does not transfer. Always check the reconstruction report.
- **Ma et al. (2026): 45-90% of claimed SAE "reasoning features" are
  spurious**. Always run the falsification framework: inject the feature's
  top-activating tokens into random non-cognitive-bias text. If the feature
  still activates, it is a token-level artifact, not a processing-mode
  feature. `s1s2.sae.falsification.ma_et_al_falsification` implements this.

## Statistical

- **Hewitt & Liang control task is NON-NEGOTIABLE for probes**. Selectivity
  = `real_AUC - random_label_AUC` is the meaningful number, not raw AUC.
  Probes with selectivity < 5pp are reporting probe expressiveness, not
  representation content.
- **Benjamini-Hochberg FDR** for exploratory multiple comparisons (q=0.05).
  **Bonferroni** for confirmatory pre-registered tests.
- **Permutation tests use the North et al. (2002) +1 correction**:
  `p = (n_extreme + 1) / (n_perms + 1)`. This prevents zero p-values and
  is statistically correct.
- **Bootstrap**: 1000 resamples for CI estimation; use the percentile method.
  Use the *paired* bootstrap when comparing two probes / two models on the
  same problems — breaking the pairing inflates the CI.
- **Cross-validation**: 5-fold stratified by the cantor pairing of (target
  label, task category). Plain target-stratification leaks category balance.

## Reasoning models

- **"Aha moments" are mostly decorative**. Only ~2.3% of `wait` / `actually`
  / `but` tokens causally influence the final answer ("Can Aha Moments Be
  Fake?", 2025). Use the TrueThinking Score to filter before claiming an
  aha moment "matters".
- **Performative reasoning trap**: a model can "decide" the answer at T0
  (start of thinking) and emit decorative reasoning that does not change
  the answer. Compare probe accuracy at T0 vs Tend to detect this — if T0
  already separates well, the thinking trace is performative.
- **Length confound**: reasoning models naturally write longer outputs. Any
  metric over the trace must be normalized by token count or report the
  scale-invariant version (Gini for attention, etc.).

## Geometry

- **Cover's theorem trap (d >> N)**: with d=4096 and N~500, random classes
  are linearly separable. So "linear SVM achieves 100%" is uninformative.
  PCA to 50-100 components before SVM, and report the *PCA-space accuracy*
  as the primary number. `s1s2.geometry.separability` does this.
- **Random projection baseline for UMAP**: project to 2D via random Gaussian
  matrices 100 times. If clusters appear in random projections too, the
  structure is genuine and UMAP is unnecessary; if they don't, UMAP is
  hallucinating. Required for any visual claim.
- **Cosine silhouette over Euclidean** in high dim: pairwise Euclidean
  distances concentrate (everything looks equidistant). Cosine is
  scale-invariant per vector and empirically cleaner for transformer
  residuals.

## Reproducibility

- **Always serialize seed + git_sha + Hydra config + W&B run id** alongside
  every result file. Otherwise you can write down a number and never
  reproduce it.
- **Smoke test as a gate**: `scripts/smoke_test.py` runs every workstream
  on synthetic data in <60 s. CI runs it on every commit. If the smoke test
  fails, do not commit.
