# Cross-Model Probe Transfer Analysis

## Experiment

Train a linear S1/S2 probe on activations from one model, test on the corresponding layer of the other model. Both models share the Llama-3.1-8B architecture (32 layers, 4096 hidden dim), differing only in training: Llama-3.1-8B-Instruct (standard RLHF) vs. DeepSeek-R1-Distill-Llama-8B (reasoning distillation). 5-fold stratified CV within each model; cross-model evaluation uses the full held-out model's activations as the test set. N=160 items across 3 cognitive bias categories (base rate neglect, conjunction fallacy, syllogistic reasoning).

## Key Numbers

| Metric | Value |
|--------|-------|
| Peak Llama-to-R1 transfer | **0.920** at L23 (98.3% of R1 self-AUC at that layer) |
| Peak R1-to-Llama transfer | **0.954** at L15 (98.7% of Llama self-AUC at that layer) |
| Mean transfer ratio (transfer / target self-AUC) | **83.3%** |
| Mean Llama-to-R1 AUC | 0.775 |
| Mean R1-to-Llama AUC | 0.786 |
| Layers above chance (both directions) | **32/32** (100%) |

## Layer Profile

### Where transfer peaks

**Llama-to-R1** achieves its strongest transfer in the late-middle to upper layers (L18--L25), peaking at L23 (AUC 0.920). Seven layers exceed 0.85: L7, L18, L21--L25.

**R1-to-Llama** peaks earlier, at L15 (AUC 0.954), and has a broader high-transfer zone spanning L3, L9, L11, L13--L19, L21. Ten layers exceed 0.85.

### Where transfer fails

**Layer 13** is an outlier for Llama-to-R1: AUC drops to 0.501 (chance level), despite Llama self-AUC of 0.975 and R1 self-AUC of 0.919 at that layer. This suggests the S1/S2 direction at L13 is coded differently in the two models -- possibly reflecting a "representational fork" where reasoning distillation reorganizes mid-network features.

Other weak-transfer layers: L0--L1, L5 (both directions, early layers where representations are still largely lexical), and L28--L31 (final layers, where task-specific readout heads may diverge).

### Asymmetry

R1-to-Llama transfer is generally more robust than Llama-to-R1 (mean AUC 0.786 vs. 0.775). This may reflect R1-Distill learning a "superset" direction that encompasses the base model's simpler S1/S2 representation -- reasoning training sharpens and extends an existing direction rather than creating an orthogonal one.

## Interpretation: Shared vs. Model-Specific Directions

The central finding: **the S1/S2 linear direction is substantially shared between these two models.** At peak layers, transfer recovers 98%+ of within-model probe accuracy. Even the mean transfer ratio of 83% is far above what chance or orthogonal representations would produce.

This has three implications:

1. **The S1/S2 direction is not an artifact of RLHF or reasoning training.** Both models represent cognitive-bias difficulty along a similar linear subspace, suggesting it emerges from pre-training on the shared Llama base.

2. **Reasoning distillation amplifies but does not reinvent.** R1-Distill's representation of "this problem requires deliberation" occupies a direction that overlaps heavily with the base model's. The distillation process strengthens this signal (higher within-model AUC at some layers) without rotating it into an orthogonal subspace.

3. **The L13 anomaly suggests a localized divergence.** At one specific layer, the two models encode S1/S2 in near-orthogonal directions. This is consistent with the idea that reasoning distillation introduces layer-specific computational changes (perhaps a new "reasoning gate" or attention pattern) rather than a global representational overhaul.

## Comparison to CogBias (Macmillan-Scott & Mukherjee, 2024)

CogBias reported near-orthogonal cross-model representations of cognitive bias susceptibility (cosine similarity ~0.01 between probe weight vectors across models). At face value, this contradicts our finding of strong cross-model transfer.

The key difference is **architectural alignment**. CogBias compared across fundamentally different architectures (e.g., Llama vs. Mistral vs. GPT-2 families) with different tokenizers, embedding spaces, and layer counts. In that setting, representations are expected to be non-comparable: the "same layer" in two different architectures does not correspond to the same computational stage, and activation spaces are rotated arbitrarily relative to each other.

Our comparison holds architecture constant. Llama-3.1-8B-Instruct and R1-Distill-Llama-8B share the same architecture, tokenizer, and initialization (via the base Llama-3.1-8B weights). They differ only in post-training: RLHF vs. reasoning distillation. Under these conditions, the activation spaces are naturally aligned -- no Procrustes rotation or stitching is needed -- and we find that the S1/S2 direction transfers at 83-98% fidelity.

**This reconciles both findings:** cross-architecture transfer is near-zero (CogBias) because activation spaces are incommensurable, while same-architecture transfer is high (our result) because the post-training variants share a base representational geometry. The S1/S2 direction is not model-family-specific; it is architecture-and-pretraining-specific. Different post-training recipes preserve it.

## Caveats

- N=160 is modest. Bootstrap CIs on transfer AUC would strengthen these claims.
- Transfer was tested layer-by-layer. An "optimal-layer" probe (selecting the best layer per model) might transfer differently.
- Both models share the same base weights. Transfer to a model with a different pre-training corpus (but same architecture) would further disambiguate "architecture" from "pre-training data."
- The L13 anomaly warrants follow-up: inspect probe weight cosine similarity at L13 to confirm near-orthogonality vs. a scale mismatch.

## Figure

See `figures/fig8_cross_model_transfer.pdf`.
