# SAE Differential Feature Analysis: Llama-3.1-8B-Instruct, Goodfire L19

**Layer**: 19 (of 32)
**SAE**: `Goodfire/Llama-3.1-8B-Instruct-SAE-l19` (65,536 features)
**Dataset**: 470 items, 80 conflict (S1) vs 80 matched controls (S2) from vulnerable categories (base_rate, conjunction, syllogism)
**Test**: Mann-Whitney U per feature, BH-FDR q < 0.05
**Falsification**: Ma et al. (2026) cheap mode -- inject top-5 activating tokens into random text

## Summary Statistics

| Metric | Value |
|--------|-------|
| SAE explained variance | 74.0% |
| Mean L0 (active features per input) | 47.9 |
| Features tested | 65,536 |
| Significant after BH-FDR | 41 (0.063%) |
| Spurious (falsification) | 0 |
| Surviving features | **41 (100%)** |

The SAE reconstruction fidelity is adequate (74% EV). The Goodfire L19 SAE is trained on the Instruct model, so there is no base/instruct mismatch -- a meaningful advantage over the LlamaScope SAEs (trained on base).

## Feature Direction: S1-Preferring vs S2-Preferring

Of the 41 significant features:

| Direction | Count | Description |
|-----------|-------|-------------|
| **S1-preferring** (conflict > control) | **24** (58.5%) | Higher activation on conflict items |
| **S2-preferring** (control > conflict) | **17** (41.5%) | Higher activation on control items |

The split is roughly 3:2, indicating that the S1/S2 distinction in this layer is not one-directional. The model represents both "conflict-present" and "conflict-absent" as active computations, not just the presence/absence of a single signal.

### Activation Pattern Subtypes

| Subtype | S1-pref | S2-pref | Total |
|---------|---------|---------|-------|
| Exclusive (fires on one condition only, zero on the other) | 18 | 3 | 21 |
| Graded (fires on both, differs in magnitude) | 6 | 14 | 20 |

This asymmetry is notable:
- **S1-preferring features are mostly binary** (18 of 24 fire exclusively on conflict items). These look like "conflict detectors" -- they activate only when the S1 lure is present.
- **S2-preferring features are mostly graded** (14 of 17 fire on both conditions but more strongly on control items). These look like "base-rate / statistical reasoning" features that are partially suppressed in the presence of a conflict.

This is consistent with a model where conflict detection is a discrete, sparse signal (a few features switch on), while deliberative reasoning is a graded modulation of many features.

## Top 10 Features by Effect Size

Effect size is rank-biserial correlation |r_rb| from the Mann-Whitney U test. The sign convention in the CSV is: **negative effect_size = S1 > S2, positive = S2 > S1** (note: this is inverted from the docstring in `differential.py` line 49, which is a bug in the docstring).

| Rank | Feature | Direction | |r_rb| | q-value | mean_S1 | mean_S2 | Activation ratio |
|------|---------|-----------|--------|---------|---------|---------|------------------|
| 1 | 19622 | S2 > S1 | 0.653 | 5.93e-08 | 0.033 | 0.086 | 0.38x |
| 2 | 39019 | S1 > S2 | 0.548 | 1.12e-06 | 0.135 | 0.025 | 5.53x |
| 3 | 5402 | S2 > S1 | 0.547 | 1.44e-05 | 0.195 | 0.299 | 0.65x |
| 4 | 30789 | S2 > S1 | 0.543 | 4.03e-06 | 0.049 | 0.237 | 0.20x |
| 5 | 1593 | S2 > S1 | 0.468 | 3.72e-07 | 0.000 | 0.034 | ~0x |
| 6 | 34045 | S1 > S2 | 0.438 | 8.81e-07 | 0.052 | 0.000 | S1-exclusive |
| 7 | 30094 | S1 > S2 | 0.425 | 1.15e-06 | 0.202 | 0.000 | S1-exclusive |
| 8 | 37305 | S2 > S1 | 0.423 | 3.35e-06 | 0.002 | 0.153 | 0.01x |
| 9 | 51172 | S1 > S2 | 0.400 | 3.35e-06 | 0.047 | 0.000 | S1-exclusive |
| 10 | 57338 | S1 > S2 | 0.400 | 3.35e-06 | 0.100 | 0.000 | S1-exclusive |

The largest effect size is feature 19622 (|r_rb| = 0.653, a "large" effect by conventional standards). Four of the top 10 are S1-exclusive features (zero activation on control items), suggesting highly specific conflict detectors.

Feature 30094 stands out: it is S1-exclusive with the highest mean activation among exclusive features (0.202), suggesting it captures a strong, specific conflict signal.

## Falsification Results

All 41 features survived the Ma et al. (2026) falsification test (0% spurious rate).

| Category | Count | Notes |
|----------|-------|-------|
| Falsification ratio = 0.0 | 38 | No activation on random text with same tokens |
| Falsification ratio > 0 but not spurious | 3 | Partial activation but below threshold |
| Spurious | 0 | -- |

The three features with nonzero falsification ratios:

| Feature | Ratio | Mean (original) | Mean (random) | Notes |
|---------|-------|-----------------|---------------|-------|
| 5402 | 0.476 | 0.247 | 0.246 | Near-identical activation -- borderline |
| 19622 | 0.295 | 0.060 | 0.059 | Near-identical -- borderline |
| 30789 | 0.218 | 0.143 | 0.121 | Some context-dependence |

**Concern**: Features 5402 and 19622 have falsification ratios of 0.48 and 0.29, meaning they activate nearly as much on random text with the same tokens. These are the #1 and #3 features by effect size. The cheap falsification mode (no model forward pass, just token overlap check) may be too lenient here. These features likely respond to surface tokens ("respondent", "professional", "more", "Answer") rather than to semantic conflict structure. A full-forward falsification (running random texts through the model) would likely flag them as spurious.

**Recommendation**: Re-run falsification in full mode for at least the top 10 features. Report results with and without features 5402 and 19622.

## Trigger Token Analysis

The top activating tokens across all 41 features reveal a concerning pattern:

| Token | Frequency (of 41 features) |
|-------|---------------------------|
| "professional" | 34 (83%) |
| "respondent" | 30 (73%) |
| "only" | 20 (49%) |
| "One" | 11 (27%) |
| "1000" | 11 (27%) |
| "single" | 11 (27%) |

These are not cognitive-bias-specific tokens -- they are **survey/scenario framing tokens** that appear in the benchmark items. "professional", "respondent", "1000" are part of the base-rate neglect scenario scaffolding. This is a dataset-content confound:

- S1-preferring features trigger on tokens like "only", "single", "One" -- which appear in conflict item phrasing ("only 1 in 1000")
- S2-preferring features trigger on tokens like "random", "no", "information" -- which appear in control item phrasing ("randomly selected", "no further information")

This does NOT mean the features are necessarily spurious (they passed falsification -- these tokens alone don't activate them in random contexts). But it does mean these features may encode **scenario structure** rather than a general S1/S2 processing mode. They may be "base-rate problem detectors" rather than "System 1 conflict detectors."

## Interpretation: What Do These Features Represent?

Three plausible interpretations, ordered from most to least conservative:

### 1. Task-Structure Detectors (conservative)
The 41 features encode the structure of cognitive bias scenarios (survey framing, probability language, conjunctive statements). The S1/S2 differential reflects different surface forms between conflict and control items, not different processing modes. The features would not generalize to other S1/S2 tasks (e.g., Stroop, anchoring).

**Evidence for**: trigger tokens are scenario-specific; 18 S1-exclusive features look like "has the word 'only' in a specific context" detectors.

### 2. Bias-Specific Processing Features (moderate)
The features encode processing states specific to base-rate, conjunction, and syllogistic reasoning. The S1-exclusive features fire when the model detects a conflict between heuristic and normative reasoning within these specific bias types. This is a meaningful finding but narrower than "S1 vs S2."

**Evidence for**: features passed falsification (tokens alone don't trigger them -- context matters); the asymmetric pattern (discrete S1 detectors vs graded S2 modulation) is consistent with cognitive accounts of dual-process conflict detection.

### 3. General Deliberation-Intensity Features (ambitious)
The features partially capture a general S1/S2 gradient that would transfer across bias types. The graded S2-preferring features (14 features that fire on both conditions but more on control) may reflect a "careful reasoning" mode that gets partially suppressed by the S1 lure.

**Evidence for**: graded features like 19622 and 5402 show continuous modulation, not binary switching; this is what a genuine processing-mode signal would look like.

**Current evidence supports interpretation #2.** To distinguish #2 from #3, we need cross-category transfer tests (train SAE-based classifier on base_rate+conjunction, test on syllogism -- if features transfer, it's more general).

## Comparison to CogBias (Macmillan et al.)

CogBias (Macmillan et al., 2024) found that cognitive biases are linearly separable in LLM representation space using probes. Our SAE analysis adds mechanistic specificity:

| Aspect | CogBias (probes) | Our SAE analysis |
|--------|------------------|------------------|
| Method | Linear probes on residual stream | SAE feature decomposition |
| Granularity | "There exists a linear direction" | "Here are 41 specific features" |
| Interpretability | Direction in 4096-D space (opaque) | Individual features with activation patterns |
| Falsification | Not performed | Ma et al. (100% survival) |
| Activation pattern | Binary separability | Both discrete detectors and graded modulators |
| Transfer | Tested across categories | Not yet tested (L19 only) |

The key advance is going from "the representation space contains the information" (which is almost trivially true in high dimensions -- Cover's theorem) to "these specific features carry the signal." However, we currently cannot claim these features are more interpretable than a probe direction until we:
1. Label features with semantic descriptions (e.g., via Neuronpedia or max-activating examples)
2. Show they generalize across bias types
3. Show they are causally relevant (activation patching)

## Limitations

1. **Single layer (L19)**: We only have a Goodfire SAE for layer 19. The S1/S2 signal likely evolves across layers -- earlier layers may detect surface conflict cues, middle layers may compute normative reasoning, later layers may form the final answer. Without multi-layer SAE coverage, we cannot characterize this trajectory.

2. **Single model**: Only Llama-3.1-8B-Instruct. No comparison to reasoning models (R1-Distill), base models, or other architectures (Gemma).

3. **Cheap falsification mode**: The Ma et al. test ran in "cheap" mode (no model forward pass). Features 5402 and 19622 have high falsification ratios (0.48, 0.29) and may be token-level artifacts. Full-forward falsification is needed.

4. **No semantic labels**: We report feature IDs but not what they "mean." Without max-activating dataset examples or Neuronpedia dashboards, the features are opaque numbers.

5. **Vulnerable categories only**: The analysis restricts to base_rate, conjunction, and syllogism. These share structural elements (probability scenarios, surveys). The 41 features may not generalize to other bias types (anchoring, framing, sunk cost).

6. **No causal evidence**: Differential activation is correlational. We have not shown that ablating these features changes model behavior on conflict items. Without activation patching, we cannot claim these features are mechanistically relevant vs. epiphenomenal.

7. **Effect size convention bug**: The docstring in `differential.py` line 49 states "positive => S1 > S2" but the code computes the opposite (positive r_rb = S2 > S1 given scipy's mannwhitneyu convention). This should be fixed to avoid confusion in future analyses.

## Next Steps

1. **Full-forward falsification** on top 10 features (especially 5402, 19622)
2. **Cross-category transfer**: train on 2 bias types, test on the 3rd
3. **Multi-layer SAE**: if Goodfire releases more layers, or train custom SAEs via SAELens at layers 5, 12, 19, 26
4. **Feature labeling**: run max-activating examples through Neuronpedia or manual inspection
5. **Activation patching**: zero-ablate top S1-preferring features, measure shift in model output on conflict items
6. **Fix docstring bug** in `differential.py` line 49
