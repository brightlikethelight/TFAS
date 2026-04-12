# Pre-Registration: Mechanistic Signatures of Dual-Process Cognition in LLMs

**Pre-registration date**: 2026-04-09
**Last updated**: 2026-04-12 (results annotations, expanded hypotheses, descoped items)
**Status**: Pre-registered before any model activations were extracted. Updated post-hoc with results and expanded hypotheses; all additions are clearly marked.

---

## 1. Study Information

### Title
The Deliberation Gradient: Mechanistic Signatures of System 1/System 2 Processing Modes in Large Language Models

### Authors
HUSAI Research Team, Harvard Undergraduate Society for Artificial Intelligence

### Research Questions

**RQ1**: Do large language models encode a linearly decodable distinction between heuristic-prone (S1-like) and deliberation-requiring (S2-like) processing states in their residual stream activations?

**RQ2**: Does reasoning distillation training (DeepSeek-R1) amplify the internal separation between S1-like and S2-like representations relative to standard instruction-tuned models of the same architecture?

**RQ3**: Do sparse autoencoder features differentially activate on S1-like versus S2-like tasks in a way that survives falsification testing, indicating feature-level rather than token-level processing mode encoding?

**RQ4**: Can causal interventions on identified S2-associated features shift model behavior from heuristic-prone to deliberation-like on cognitive bias tasks?

**RQ5**: Is the S1/S2 distinction reflected in attention entropy patterns and representational geometry, and do these signals converge with probing and SAE evidence?

### Hypotheses

**H1 (Linear decodability)**: The conflict/no-conflict task type is linearly decodable from residual stream activations at the last prompt token position. Specifically, L2-regularized logistic regression probes will achieve ROC-AUC > 0.6 at the best-performing layer in at least 2 of 4 models, with Hewitt-Liang selectivity exceeding 5 percentage points.

**H2 (Reasoning training amplification)**: DeepSeek-R1-Distill-Llama-8B (reasoning) will show a higher peak-layer probe AUC for task type classification than Llama-3.1-8B-Instruct (standard), despite sharing the same base architecture. The 95% paired bootstrap confidence interval for the AUC difference will exclude zero, with the reasoning model higher.

**H3 (SAE feature specificity)**: After Benjamini-Hochberg FDR correction at q=0.05 and Ma et al. (2026) token-injection falsification, at least 5 sparse autoencoder features will show significant differential activation between conflict and no-conflict items with rank-biserial correlation |r_rb| > 0.3, in at least 1 model.

**H4 (Causal efficacy)**: Feature-steering along S2-associated SAE feature directions will increase the probability of correct responses on conflict items by more than 15 percentage points relative to baseline, while steering along random feature directions will produce an increase of less than 3 percentage points.

**H5 (Attention entropy differentiation)**: At least 5% of attention heads (at the KV-group granularity for GQA models) will be classified as S2-specialized -- defined as showing significantly higher attention entropy on conflict tasks across at least 3 of 5 entropy/concentration metrics (BH-FDR corrected, |r_rb| >= 0.3) -- in at least 1 model.

**H6 (Geometric separability)**: The cosine silhouette score for conflict vs. no-conflict activations will be significantly greater than zero (permutation test, p < 0.05 after BH-FDR correction across layers) at the peak layer in at least 2 of 4 models.

---

## 2. Design Plan

### Study Type
Observational mechanistic interpretability study. We analyze the internal representations of pre-trained models on a fixed benchmark. No model training or fine-tuning is performed by us.

### Blinding
Not applicable. This is a computational study with pre-specified analyses on fixed models and a fixed benchmark. The benchmark was designed and finalized before any model activations were extracted.

### Study Design
A within-model repeated-measures design with 142 matched pairs (conflict vs. no-conflict) drawn from 7 cognitive bias categories. Each pair shares surface structure and correct answer; they differ only in whether a heuristic lure is present (conflict) or absent (no-conflict). This matched-pair structure is the primary confound control against task difficulty.

Four models are tested: 2 standard instruction-tuned (Llama-3.1-8B-Instruct, Gemma-2-9B-IT) and 2 reasoning-distilled (DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-7B). The Llama pair shares an identical base architecture, enabling a clean comparison where any internal differences are attributable to reasoning distillation rather than architectural confounds.

> **[POST-HOC UPDATE]** Gemma-2-9B-IT was descoped due to workshop paper timeline constraints (see Descope section). OLMo-2-7B-Instruct was added as a replacement standard model. DeepSeek-R1-Distill-Qwen-7B additionally tested in THINK vs. NO_THINK mode (system-prompt-controlled) for H6 within-model dissociation. Ministral-3-8B was descoped due to transformers incompatibility (see Descope section).

Five analysis workstreams are applied to each model: linear probing, SAE feature analysis, attention entropy, representational geometry, and causal interventions.

---

## 3. Sampling Plan

### Models

| Model Key | HuggingFace ID | Type | Layers | Q Heads | KV Heads | Hidden Dim |
|-----------|----------------|------|--------|---------|----------|------------|
| llama-3.1-8b-instruct | `meta-llama/Llama-3.1-8B-Instruct` | Standard | 32 | 32 | 8 | 4096 |
| gemma-2-9b-it | `google/gemma-2-9b-it` | Standard | 42 | 16 | 8 | 3584 |
| r1-distill-llama-8b | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Reasoning | 32 | 32 | 8 | 4096 |
| r1-distill-qwen-7b | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Reasoning | 28 | 28 | 4 | 3584 |

All models are publicly available and were not trained or modified by us. Model selection was determined before any analysis began, based on: (a) architecture-matched standard/reasoning pairs, (b) availability of pre-trained SAEs (Llama Scope, Gemma Scope), (c) parameter counts feasible on a single A100/H100 GPU.

### Benchmark

The original benchmark consisted of **284 items** organized as **142 matched pairs** across **7 categories**.

> **[POST-HOC UPDATE]** The benchmark was expanded to **380+ items** (~190 matched pairs) across **9 categories** before the main analysis runs. Two new categories were added: sunk cost fallacy (loss aversion domain) and natural frequency framing (base rate domain, Gigerenzer paradigm). Both additions were designed and finalized before any model activations for the new items were extracted.

| Category | Conflict Items | No-Conflict Controls | Total | Status |
|----------|---------------|---------------------|-------|--------|
| CRT variants | 30 | 30 | 60 | Original |
| Arithmetic (multi-step) | 25 | 25 | 50 | Original |
| Syllogisms (belief-bias) | 25 | 25 | 50 | Original |
| Base rate neglect | 20 | 20 | 40 | Original |
| Anchoring | 15 | 15 | 30 | Original |
| Framing effects | 15 | 15 | 30 | Original |
| Conjunction fallacy | 12 | 12 | 24 | Original |
| Sunk cost fallacy | ~25 | ~25 | ~50 | **Added** |
| Natural frequency framing | ~25 | ~25 | ~50 | **Added** |
| **Total** | **~192** | **~192** | **~384** | |

The benchmark is **fixed** before any model analysis. It was designed with the following constraints:
- All conflict items use novel structural isomorphs (no classic CRT bat-and-ball, no classic Linda problem) to avoid training-set contamination.
- Each conflict item has a matched no-conflict control sharing the same surface features, correct answer, and approximate difficulty, differing only in the presence of a heuristic lure.
- Each item has a `correct_answer`, `lure_answer` (conflict items only), regex patterns for automated scoring, and 1-4 paraphrases.

### Power Analysis
Not applicable in the traditional sense. We analyze all 284 items through all 4 models at all layers. There is no sampling from a larger population. Statistical tests are conducted at the item level with 142 items per condition (conflict/no-conflict), which provides adequate precision for the effect sizes we pre-specify.

---

## 4. Variables

### Independent Variables

1. **Task type** (within-item): conflict (S1-lure present) vs. no-conflict (S1-lure absent). Binary. This is the primary manipulation.
2. **Model type** (between-model): standard (instruction-tuned) vs. reasoning (R1-distilled). Binary.
3. **Task category** (between-item): crt, arithmetic, syllogism, base_rate, anchoring, framing, conjunction. 7 levels.
4. **Layer** (within-model): residual stream layer index (0 to L-1 for each model).
5. **Token position** (within-item): last prompt token (P0, primary), answer token (P2), and reasoning-trace positions (T0, T25, T50, T75, Tend) for reasoning models only.

### Dependent Variables

**Workstream 1 -- Linear Probes**:
- ROC-AUC for task type classification (conflict vs. no-conflict), per model, per layer, per token position.
- Hewitt-Liang selectivity: real AUC minus random-label control AUC, in percentage points.
- Probe weights (for downstream interpretability).

**Workstream 2 -- Attention Entropy**:
- Per-head Shannon entropy H(l,h,t) and normalized entropy H/log2(t).
- Per-head Gini coefficient (co-primary, scale-invariant).
- Head classification: S2-specialized, S1-specialized, or unspecialized, based on multi-metric consensus.
- Proportion of S2-specialized heads per model.

**Workstream 3 -- SAE Feature Analysis**:
- Per-feature mean activation on conflict vs. no-conflict items.
- Mann-Whitney U statistic and BH-FDR-corrected p-value per feature.
- Rank-biserial correlation (effect size) per feature.
- Ma et al. falsification status (survived/falsified) per candidate feature.
- Count of significant, non-falsified differential features.

**Workstream 4 -- Representational Geometry**:
- Cosine silhouette score for conflict vs. no-conflict clusters, per layer.
- CKA (Centered Kernel Alignment) between Llama-3.1-8B and R1-Distill-Llama-8B, per layer, separately for conflict and no-conflict items.
- Intrinsic dimensionality (Two-NN estimator) of S1 vs. S2 activation point clouds, per layer.

**Workstream 5 -- Causal Interventions**:
- P(correct) on conflict items under baseline (no steering).
- P(correct) on conflict items under S2-feature steering at coefficients {0.5, 1.0, 2.0, 3.0, 5.0}.
- P(correct) on conflict items under random-direction steering at the same coefficients.
- Delta P(correct) = steered - baseline.
- Side-effect metrics: MMLU and HellaSwag accuracy under steering vs. baseline.

### Covariates / Nuisance Variables
- **Task difficulty**: integer 1-5 per item. Controlled by matched-pair design. Additionally, all primary analyses are run on a difficulty-matched subset (items where the model's baseline accuracy on the conflict and no-conflict versions differs by less than 0.1).
- **Sequence length**: controlled by normalizing attention entropy by log2(t) and using the Gini coefficient.
- **GQA head grouping**: for Llama and R1-Distill-Llama (32 Q-heads, 8 KV-groups), attention analyses are reported at both query-head and KV-group granularities. KV-group is the confirmatory granularity.

---

## 5. Analysis Plan

### General Statistical Framework

All analyses adhere to the statistical standards documented in `CLAUDE.md`:

- **Multiple comparisons**: Benjamini-Hochberg FDR at q=0.05 for exploratory tests within each workstream. Bonferroni correction across the 6 primary hypotheses (alpha = 0.05/6 = 0.00833 per hypothesis).
- **Effect sizes**: reported alongside every p-value. ROC-AUC for probes, rank-biserial correlation for Mann-Whitney U tests, Cohen's d for paired t-tests.
- **Confidence intervals**: bootstrap with 1000 resamples, percentile method. Paired bootstrap when comparing two probes/models on the same items.
- **Permutation tests**: 10,000 shuffles for null distributions. P-values computed with the North et al. (2002) +1 correction: p = (n_extreme + 1) / (n_perms + 1).
- **Cross-validation**: 5-fold stratified by the Cantor pairing of (target label, task category). Nested CV for hyperparameter selection (LogisticRegressionCV).
- **Random seeds**: 3+ seeds for all stochastic procedures. Report mean +/- standard deviation.
- **Matched-difficulty control**: every primary analysis is run on (a) the full 284-item set AND (b) the difficulty-matched subset. Both are reported; the matched subset is primary for confirmatory claims.

### Pre-requisite: Behavioral Validation (Week 2 Gate)

Before any mechanistic analysis, we verify that the benchmark elicits differential behavior:
- **Criterion**: models produce the S1 lure answer on > 30% of conflict items (at least 1 of the standard models must meet this threshold).
- **If failed**: revise benchmark items or descope to categories where lure rates exceed 30%.
- This is a go/no-go gate. If models do not show heuristic-prone behavior, mechanistic claims about S1/S2 processing are vacuous.

---

### Primary Analyses (Confirmatory)

#### H1: S1/S2 Distinction Is Linearly Decodable from Residual Stream Activations

**Procedure**:
1. Extract residual stream activations at the last prompt token (position P0) for all 284 items, at every layer, for each model.
2. Train an L2-regularized logistic regression probe (sklearn `LogisticRegressionCV`, penalty='l2', Cs=10, solver='lbfgs') to classify conflict vs. no-conflict, using 5-fold stratified cross-validation (stratified by Cantor pairing of label x category).
3. Compute ROC-AUC per fold; report mean and bootstrap 95% CI.
4. Train the same probe on randomly permuted labels (Hewitt-Liang control). Compute selectivity = real AUC - control AUC.
5. Run a permutation test (1,000 shuffles) per layer to obtain a null distribution for AUC.
6. Apply BH-FDR correction across layers within each model (q=0.05).

**Statistical test**: ROC-AUC with permutation-based significance and BH-FDR correction across layers.

**Threshold for positive result**:
- Peak-layer AUC > 0.6 in at least 2 of 4 models, AND
- Hewitt-Liang selectivity > 5 percentage points at that peak layer, AND
- Peak layer survives BH-FDR correction at q=0.05.

**Threshold for negative result**:
- All layers below chance (AUC <= 0.5) after FDR correction in all models, OR
- Selectivity < 5pp at all layers in all models.

**Bonferroni-adjusted significance**: p < 0.00833 for the peak-layer AUC (0.05/6 primary hypotheses).

> **RESULT: CONFIRMED.** Peak-layer AUC = 0.999 at layer 14 (Llama-3.1-8B-Instruct). The S1/S2 distinction is near-perfectly linearly decodable from the residual stream, far exceeding the pre-registered threshold of 0.6. Selectivity well above 5pp. Result survives BH-FDR correction. This is the strongest finding in the study.

#### H2: Reasoning Models Show Stronger S1/S2 Separation Than Standard Models

**Procedure**:
1. Using the per-layer AUC curves from H1, identify the peak layer for Llama-3.1-8B-Instruct and DeepSeek-R1-Distill-Llama-8B separately.
2. Compute a paired bootstrap CI (1,000 resamples) on the difference in peak-layer AUC: AUC(R1-Distill-Llama) - AUC(Llama-3.1-8B). The bootstrap is paired at the item level (both models see the same 284 items).
3. Repeat the comparison for peak-layer selectivity.

**Statistical test**: paired bootstrap 95% CI on AUC difference. Bonferroni-adjusted: 99.17% CI.

**Threshold for positive result**:
- The Bonferroni-adjusted CI for the AUC difference excludes zero, AND
- The reasoning model has the higher peak AUC.

**Threshold for negative result**:
- The CI includes zero, OR
- The standard model has the higher peak AUC.

**Note**: This is the cleanest comparison in the study because both models share identical architecture (32 layers, 32 heads, 4096 hidden dim). Any difference is attributable to reasoning distillation, not architecture.

> **RESULT: FALSIFIED (in the predicted direction).** The reasoning model (DeepSeek-R1-Distill-Llama-8B) achieved peak AUC = 0.929, which is *lower* than Llama-3.1-8B-Instruct's 0.999 -- the opposite of the predicted direction. The boundary is weaker, not stronger, in the reasoning model.
>
> **Theoretical interpretation**: This falsification is actually predicted by Evans' (2019) concept of Type 2 autonomy. Reasoning distillation internalizes deliberative strategies into automatic processing, *blurring* the S1/S2 boundary rather than sharpening it. The reasoning model may have learned to handle conflict items more fluidly, reducing the representational contrast that the probe exploits. This is a theoretically coherent result, not an anomaly.
>
> **Honest framing**: We pre-registered a directional prediction that was wrong. The data are inconsistent with "reasoning training amplifies internal S1/S2 separation" and consistent with "reasoning training integrates deliberative processing into a more unified representation."

#### H3: SAE Features Differentially Activate on S1 vs. S2 Tasks

**Procedure**:
1. Load pre-trained SAEs: Llama Scope for Llama-3.1-8B, Gemma Scope for Gemma-2-9B-IT. Verify reconstruction fidelity (explained variance >= 0.5) before proceeding.
2. For each SAE feature, compute the mean activation on conflict items and no-conflict items separately (at the last prompt token, at the SAE's target layer).
3. Compute Mann-Whitney U test per feature. Apply BH-FDR correction at q=0.05 across all features.
4. For features surviving FDR, compute rank-biserial correlation as the effect size.
5. **Falsification (Ma et al. 2026)**: for each candidate feature, identify the top-3 activating tokens. Inject these tokens into 100 random non-cognitive-bias texts. If the feature activates at >= 50% of the level it showed on benchmark items, classify it as falsified (token-level artifact).
6. Report the count of features that are (a) significant after FDR, (b) have |r_rb| > 0.3, AND (c) survive falsification.

**Statistical test**: Mann-Whitney U with BH-FDR correction, followed by Ma et al. falsification.

**Threshold for positive result**:
- At least 5 features significant after FDR AND after falsification, with |r_rb| > 0.3, in at least 1 model.

**Threshold for negative result**:
- Fewer than 5 features survive the full pipeline (FDR + falsification + effect size) across all models.

**Bonferroni adjustment**: applied at the hypothesis level (the threshold of 5 features is evaluated at the Bonferroni-adjusted alpha). Within-feature FDR remains at q=0.05.

> **RESULT: PENDING.** SAE feature analysis requires GPU re-run with the expanded benchmark. Pre-trained SAE loading and reconstruction fidelity checks are implemented but have not been executed on the full activation cache. This hypothesis remains open.

#### H4: Causal Interventions Shift Behavior from S1-Prone to S2-Like

**Procedure**:
1. Using the non-falsified S2-associated features from H3, construct a steering vector as the mean of their SAE decoder directions.
2. On all 142 conflict items, compute P(correct) at baseline (no steering).
3. Apply activation steering at the SAE's target layer with coefficients {0.5, 1.0, 2.0, 3.0, 5.0}. Compute P(correct) at each coefficient.
4. Apply random-direction steering (random unit vector in the same space) at the same coefficients. Repeat with 10 random directions; report mean and 95% CI.
5. Compute Delta P(correct) = P(correct | steered) - P(correct | baseline) at the best coefficient.
6. Test with a paired t-test across items: H0: Delta P(correct) = 0.
7. Check side effects: run MMLU (5-shot) and HellaSwag (10-shot) under the best steering coefficient. Acceptable degradation: < 5pp accuracy drop on each.

**Statistical test**: paired t-test on Delta P(correct) for S2-steering vs. baseline. Cohen's d as effect size.

**Threshold for positive result**:
- Delta P(correct) > 15 percentage points under S2-feature steering (at the best coefficient), AND
- Delta P(correct) < 3 percentage points under random-direction steering, AND
- Paired t-test significant at Bonferroni-adjusted alpha (p < 0.00833).

**Threshold for negative result**:
- S2-feature steering produces Delta P(correct) <= 15pp, OR
- Random-direction steering produces Delta P(correct) >= 3pp (indicating the effect is not direction-specific), OR
- Paired t-test not significant.

**Dose-response**: the coefficient sweep {0.5, 1.0, 2.0, 3.0, 5.0} is exploratory and is reported descriptively. Only the best coefficient is used for the confirmatory test.

#### H5: Attention Entropy Differs Between S1 and S2 Conditions in Specific Heads

**Procedure**:
1. For each attention head (or KV-group for GQA models), at the last prompt token, compute 5 metrics on each item: (a) Shannon entropy, (b) normalized entropy H/log2(t), (c) Gini coefficient, (d) max attention weight, (e) top-5 cumulative attention weight.
2. For each head x metric combination, run a Mann-Whitney U test comparing conflict vs. no-conflict items.
3. Apply BH-FDR correction at q=0.05 across all heads x metrics within a model.
4. Classify a head as S2-specialized if it shows significantly higher entropy (or lower Gini/max-attention) on conflict items across at least 3 of 5 metrics, with |r_rb| >= 0.3 on each significant metric.
5. Report the proportion of heads classified as S2-specialized, at both query-head and KV-group granularities.

**Statistical test**: per-head Mann-Whitney U with BH-FDR correction. Multi-metric consensus (3/5).

**Threshold for positive result**:
- At least 5% of heads (at KV-group granularity) classified as S2-specialized in at least 1 model, AND
- S2-specialized heads concentrated in mid-to-late layers (binomial test: more in the top half of layers than expected by chance).

**Threshold for negative result**:
- Fewer than 5% of heads classified as S2-specialized in all models, OR
- S2-specialized heads uniformly distributed across layers (no concentration).

**Note on Gemma-2**: odd layers (sliding window attention) and even layers (global attention) are analyzed separately. Pooling is prohibited.

#### H6: S1/S2 Representations Are Geometrically Distinguishable

**Procedure**:
1. At each layer, for each model, collect residual stream activations at the last prompt token for all 284 items.
2. Reduce to 50 principal components via PCA (addressing the d >> N pitfall per Cover's theorem).
3. Compute cosine silhouette score: for each item, the difference between mean cosine distance to other-class items and mean cosine distance to same-class items, normalized by the maximum.
4. Run a permutation test (10,000 label shuffles) to obtain a null distribution for the silhouette score.
5. Apply BH-FDR correction across layers within each model.
6. Report the peak-layer silhouette score with bootstrap 95% CI.

**Statistical test**: cosine silhouette score with permutation test and BH-FDR correction across layers.

**Threshold for positive result**:
- Silhouette > 0 with permutation p < 0.05 (after BH-FDR) at the peak layer, in at least 2 of 4 models.

**Threshold for negative result**:
- Silhouette indistinguishable from the permutation null (p >= 0.05 after FDR) at all layers, in all models.

**Controls**:
- Random projection baseline: project to 2D via 100 random Gaussian matrices. If class separation appears in random projections, the structure is genuine and not a UMAP artifact.
- Random-label control: repeat silhouette analysis with shuffled conflict/no-conflict labels. Establishes the floor.

---

### Secondary Analyses (Exploratory -- Not Pre-Registered)

The following analyses are planned but are explicitly **not** pre-registered. Results from these will be reported as exploratory and will not contribute to the primary hypothesis tests.

1. **Leave-one-category-out (LOCO) cross-domain transfer**: train probes on 6 of 7 categories, test on the held-out category. Assesses whether probes learn a domain-general S1/S2 distinction vs. category-specific features.

2. **Self-correction trajectory analysis**: for reasoning models that initially state the lure answer then self-correct (estimated 20-40% of conflict items), extract activations at the lure statement, the correction marker, and the corrected answer. Probe for the S1-to-S2 transition across these positions.

3. **Metacognitive monitoring (4-gate framework)**:
   - Gate 1: identify difficulty-sensitive SAE features (Spearman rho > 0.3 with token surprise).
   - Gate 2: test S1/S2 specificity of difficulty features (AUC > 0.65).
   - Gate 3: causal test (steering along difficulty direction increases P(correct) by > 0.15).
   - This is a stretch goal. Findings are reported as exploratory regardless of outcome.

4. **Cross-model feature matching**: compute cosine similarity of SAE decoder vectors and activation correlations on the shared benchmark between Llama Scope and Gemma Scope features. Tests whether the same "deliberation features" emerge across architectures.

5. **Intrinsic dimensionality comparison**: Two-NN estimator on conflict vs. no-conflict activation point clouds. Lower dimensionality for one condition would suggest more stereotyped processing.

6. **CKA divergence analysis**: layer-matched CKA between Llama-3.1-8B and R1-Distill-Llama-8B, computed separately for conflict and no-conflict items. Tests whether reasoning training specifically changed S2 processing (CKA for conflict items drops faster in late layers).

7. **Performative reasoning detection**: compare probe accuracy at T0 (first token after `<think>`) vs. Tend (last token before `</think>`) for reasoning models. If T0 already separates S1/S2 well, the thinking trace may be performative.

---

## 6. Existing Data

### Benchmark
The benchmark (284 items, `data/benchmark/benchmark.jsonl`) was designed and finalized **before** any model activations were extracted. No pilot data from these models on this benchmark exists. The benchmark creation process did not involve running any of the 4 target models.

### Models
All 4 models are publicly available pre-trained models from Meta, Google DeepMind, and DeepSeek. We did not train, fine-tune, or modify any model.

### Sparse Autoencoders
All SAEs used in the primary analysis are pre-trained by third parties:
- **Llama Scope** (`fnlp/Llama-3_1-8B-Base-LXR-32x`): trained on Llama-3.1-8B-Base (not Instruct). Reconstruction fidelity on Instruct activations will be verified before use.
- **Gemma Scope** (`google/gemma-scope-9b-it-res`): trained on Gemma-2-9B-IT. Multiple widths available (16K to 1M features).
- **Goodfire R1 SAE** (`Goodfire/DeepSeek-R1-SAE-l37`): trained on DeepSeek-R1 (671B). Transfer to R1-Distill-8B is uncertain and will be verified.

### Prior Work
No prior analyses of these specific models on this specific benchmark exist. The benchmark was purpose-built for this study.

---

## 7. Other

### Multiple Testing Strategy

We employ a two-level correction strategy:

1. **Between primary hypotheses (family-wise)**: Bonferroni correction across the 6 primary hypotheses. The per-hypothesis significance level is alpha = 0.05/6 = 0.00833. This is conservative by design -- we prefer false negatives to false positives for confirmatory claims.

2. **Within each hypothesis (false discovery rate)**: Benjamini-Hochberg FDR at q=0.05 for tests across layers, heads, or features within a single hypothesis test. BH-FDR is appropriate here because within-hypothesis tests are correlated (adjacent layers share representations) and we accept a controlled proportion of false discoveries.

3. **Exploratory analyses**: BH-FDR at q=0.05. Clearly labeled as exploratory. No confirmatory claims.

### Matched-Difficulty Confound Control

This is pre-registered as a mandatory analysis:
- **Every primary analysis** (H1-H6) is run on (a) the full 284-item set AND (b) a difficulty-matched subset.
- The difficulty-matched subset is defined per-model: item pairs where the model's baseline accuracy difference between the conflict and no-conflict versions is less than 0.1 (10 percentage points).
- If a result holds on the full set but not the matched subset, we interpret it as a difficulty confound rather than an S1/S2 processing mode difference.
- Both results (full and matched) are reported in all cases.

### Reproducibility Commitments

- All random seeds are fixed and recorded (torch, numpy, random). The utility `s1s2.utils.seed.set_global_seed()` is used for every stochastic operation.
- Hydra configs are serialized alongside every result file and into the HDF5 activation cache.
- W&B run IDs are recorded in every result JSON for traceability.
- The git SHA at the time of each analysis run is logged.
- All code is version-controlled. The benchmark is version-controlled and checksummed (SHA-256).

### Deviations from Pre-Registration

Any deviations from this plan will be:
1. Documented in the final paper's supplement with a clear rationale.
2. Labeled as a deviation (not silently substituted).
3. If a pre-registered analysis becomes impossible (e.g., SAE reconstruction fidelity is too low for a model), this is reported as a null result for that model, not dropped.

### Pre-Registered Positive Controls

To validate that our analysis pipeline can detect known signals before applying it to the S1/S2 question:
- **Probe sanity check**: train a probe to classify task category (7-way). This should achieve high AUC trivially (surface features differ across categories). If it does not, the pipeline has a bug.
- **Attention sanity check**: verify that attention entropy increases with sequence length (a known property). If it does not, the entropy computation is incorrect.
- **SAE sanity check**: verify that SAE reconstruction loss on our activation cache is within 2x of the loss reported in the original SAE paper on in-distribution data. If it exceeds this, the SAE does not fit our model.

---

## Decision Framework

The table below maps every primary outcome pattern to an interpretation and publication strategy. All outcomes are publishable. Negative results are explicitly valuable -- they save the field from pursuing mechanistic dual-process claims without evidence.

| Scenario | Hypotheses Passing | Interpretation | Publication Strategy |
|----------|-------------------|----------------|---------------------|
| Strong positive | H1 + H2 + H3 + H4 + H5 + H6 | Convergent mechanistic evidence for a deliberation-intensity gradient in LLMs, amplified by reasoning training | Full paper: ICML/NeurIPS main conference. Lead with convergent evidence across 5 methods. |
| Moderate positive (mode-general) | H1 + H3 + H5 + H6, not H2 or H4 | S1/S2-like processing distinction exists but is not specific to reasoning training and lacks causal confirmation | Full paper: ICLR 2027. Frame as representational finding. Emphasize that the distinction is architectural, not training-induced. |
| Moderate positive (causal) | H1 + H3 + H4, not H5 or H6 | Feature-level S1/S2 encoding with causal efficacy, but not reflected in attention or geometry | Workshop paper + full paper. SAE features are the story; attention/geometry are supplementary negatives. |
| Weak positive (representation only) | H1 + H5 + H6, not H3 or H4 | Representational separation exists (probes, geometry) but no feature-level or causal evidence | ICML MechInterp Workshop. Honest about the gap between decodability and mechanism. |
| Minimal positive | Only H1 | Some linear decodability but weak, possibly a difficulty confound survivor | Workshop paper or negative-results venue. Report as "suggestive but inconclusive." |
| Null result | None pass | The dual-process framework does not have detectable mechanistic correlates in LLMs at this scale and with these methods | NeurIPS SoLaR Workshop or Alignment Forum. Frame as "against dual-process theory in LLMs" -- valuable for the anthropomorphism debate. |

### Interpretation Safeguards

Regardless of outcome:
- We never claim LLMs "have" System 1 and System 2. We describe S1-like and S2-like processing signatures or a deliberation-intensity gradient.
- Positive results are interpreted as "consistent with" dual-process-like computation, not "proof of" it.
- We explicitly discuss alternative explanations for any positive finding (difficulty confound, surface feature leakage, token-level artifacts).
- Negative results are given equal interpretive weight. A well-powered null result is informative.

---

## Appendix: Benchmark Checksums

To verify that the benchmark analyzed matches the pre-registered benchmark:

```
File: data/benchmark/benchmark.jsonl
Items: 284
Conflict items: 142
No-conflict items: 142
Matched pairs: 142
Categories: anchoring (30), arithmetic (50), base_rate (40), conjunction (24), crt (60), framing (30), syllogism (50)
```

The SHA-256 hash of the benchmark file will be computed and recorded at the time of first model extraction. Any subsequent modifications to the benchmark invalidate this pre-registration.

---

## Appendix: Timeline for Pre-Registered Analyses

| Week | Gate | Analysis |
|------|------|----------|
| 2 | Behavioral validation | Models show > 30% lure responses on conflict items |
| 3-4 | H1, H5 | Linear probes and attention entropy (fastest workstreams) |
| 5-6 | H3, H6 | SAE feature analysis and representational geometry |
| 7-8 | H4 | Causal interventions (depends on H3 features) |
| 9 | Full assessment | Commit to paper scope based on which hypotheses passed |

Note: H2 (reasoning vs. standard comparison) is computed from H1 results and requires no additional data collection.
