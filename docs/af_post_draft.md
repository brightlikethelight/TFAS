# Do LLMs Have a Deliberation Mode? 84% Bias Susceptibility Drops to 0% with Reasoning Training

**Bright Liu, Harvard Undergraduate AI Safety Research (HUSAI)**

*Preliminary results from an ongoing project. Posting for priority and community feedback.*

---

Llama-3.1-8B-Instruct falls for the base rate neglect fallacy 84% of the time. DeepSeek-R1-Distill-Llama-8B, which shares the exact same architecture and base weights, falls for it 4% of the time. On conjunction fallacy items the drop is from 55% to 0%; on belief-bias syllogisms, 52% to 0%.

Same weights. Same architecture. Different training. What changed?

We built a benchmark to measure cognitive-bias susceptibility in LLMs and combined it with linear probing of internal representations to start answering that question. This post reports preliminary behavioral and mechanistic results on 8B-parameter models.

## Background: dual-process theory as an operational lens

Dual-process theory (Kahneman, Evans & Stanovich) distinguishes fast/intuitive "System 1" processing from slow/deliberative "System 2" processing. We are not claiming LLMs literally have these systems. We use the framework operationally: cognitive bias tasks that reliably trip up human System 1 reasoning serve as a useful testbed for asking whether reasoning-trained models process information differently at the representational level.

The question that matters for safety: can we detect, from internal activations alone, whether a model is engaged in something more like reflexive pattern-matching versus something more like deliberative reasoning? And does reasoning training (chain-of-thought distillation) change this internal signature?

## The benchmark: 330 matched conflict/control pairs

We constructed a benchmark of 330 items across seven categories: CRT variants, base rate neglect, syllogistic reasoning, anchoring, framing, conjunction fallacy, and arithmetic. Each item is a matched pair: a conflict version (where an intuitive-but-wrong lure competes with the correct answer) and a control version (structurally identical, but the intuitive response and the correct response agree). Conflict items are operationally "S1-like" stimuli; control items are operationally "S2-like" stimuli.

All items are novel structural isomorphs, not copies of classic problems. Classic CRT items (bat-and-ball, lily pad, widgets-and-machines) appear only as baselines for measuring contamination effects. The benchmark is config-driven and fully reproducible from a fixed seed.

Total counts by category: 30 CRT pairs, 20 base rate pairs, 25 syllogism pairs, 15 anchoring pairs, 15 framing pairs, 12 conjunction pairs, 25 arithmetic pairs.

## Behavioral results

We scored model outputs on whether they produce the correct answer (for conflict items, this means resisting the lure).

**Bias susceptibility rate by model and category** (percentage of conflict items where the model produces the lure answer):

| Category | Llama-3.1-8B-Instruct | R1-Distill-Llama-8B | Qwen-3-8B (no thinking) |
|---|---|---|---|
| Base rate neglect | 84% | 4% | 56% |
| Conjunction fallacy | 55% | 0% | 95% |
| Syllogism bias | 52% | 0% | 0% |
| CRT | 0% | 0% | -- |
| Arithmetic | 0% | 0% | -- |
| Framing | 0% | 0% | -- |
| Anchoring | 0% | 0% | -- |

The headline result is the Llama/R1-Distill comparison. These two models share the Llama-3.1-8B architecture and base weights. R1-Distill-Llama-8B was fine-tuned with reasoning-trace distillation from DeepSeek-R1. The bias susceptibility drops are large: 80 percentage points on base rate neglect, 55pp on conjunction fallacy, 52pp on syllogism bias.

The four categories showing 0% susceptibility across all models (CRT, arithmetic, framing, anchoring) likely reflect benchmark difficulty rather than universal bias resistance -- see Caveats.

The Qwen-3-8B column is notable: 95% conjunction fallacy susceptibility is the highest we have observed, and 0% syllogism bias is the lowest for any non-reasoning model. This model was tested without its thinking mode enabled.

## First mechanistic result: linear probes on internal representations

We extracted residual stream activations at every layer for all benchmark items and trained linear probes (logistic regression, 5-fold stratified CV, Hewitt & Liang control tasks) to classify conflict vs. control items from internal representations alone.

The question: can the model's internal state distinguish items that require deliberation from items that don't?

**Peak probe performance (AUC, logistic regression):**

| Model | Peak Layer | AUC | Bootstrap 95% CI |
|---|---|---|---|
| Llama-3.1-8B-Instruct | 14 | 0.999 | [0.996, 1.000] |
| R1-Distill-Llama-8B | 14 | 0.929 | -- |

Both models peak at layer 14 out of 32. The conflict/control distinction is linearly decodable from internal representations in both models, but the separation is substantially sharper in the non-reasoning model.

This is the opposite of what a naive "reasoning models are better at distinguishing hard from easy" hypothesis would predict. The reasoning-trained model has a *less* crisp internal boundary between conflict and control items.

## Interpretation: S2-by-default

We interpret this pattern as follows. Llama-3.1-8B-Instruct maintains a sharp internal distinction between items that trigger an intuitive-pattern-matching mode and items that don't -- and then often *fails* on the items it correctly identifies as tricky. R1-Distill-Llama-8B has a blurred distinction because reasoning training has made it process everything more uniformly through a deliberative pathway. It doesn't need to distinguish S1 from S2 items because it applies S2-like processing by default.

The 0.07 AUC gap (0.999 vs. 0.929) captures this: the non-reasoning model internally "knows" which items are conflict items with near-perfect fidelity but can't act on that knowledge. The reasoning model has partially lost the distinction because it no longer needs it.

## Within-model comparison: Qwen-3-8B (preliminary)

Qwen-3-8B offers a natural within-model test because it can be run with or without its thinking mode. We have behavioral results (95% conjunction fallacy susceptibility without thinking) but mechanistic results are still in progress. If the same "blurring" pattern appears when comparing thinking-on vs. thinking-off activations within a single model, it would strengthen the S2-by-default hypothesis considerably. We will report this when the analysis is complete.

## What this means for safety

If reasoning-trained models process all inputs through an S2-like pathway, this has several implications:

1. **Monitoring deliberation**: The probe results suggest that linear probes on middle-layer activations can detect whether a model is in a "deliberation-like" state. This could serve as a runtime monitor for whether a model is actually reasoning about a problem or pattern-matching through it.

2. **Reasoning fidelity**: The gap between internal detection and behavioral performance in Llama-3.1-8B-Instruct (it "knows" items are tricky but still fails) suggests that detecting the need for deliberation and actually executing deliberation are separable capabilities. Reasoning training appears to address the execution side.

3. **Performative reasoning risk**: If future reasoning models emit chain-of-thought traces without internal deliberation (performative reasoning), probes like these could detect the discrepancy. We plan to investigate this directly.

## What's coming next

- **SAE feature analysis**: Identifying specific sparse autoencoder features that activate differentially on conflict vs. control items, with Ma et al. (2026) falsification to filter token-level artifacts.
- **Causal interventions**: The probe results show correlation, not causation. We plan activation patching and steering experiments to test whether the representations identified by probes causally influence model outputs.
- **More models**: Extending to Gemma-2-9B-IT and DeepSeek-R1-Distill-Qwen-7B to test generality across architectures.
- **Qwen-3-8B thinking toggle**: Completing the within-model mechanistic comparison.

## Caveats and limitations

**The 0% categories may reflect benchmark difficulty, not bias resistance.** Four of our seven categories show 0% susceptibility across all tested models. This likely means our items in those categories are too easy (or that instruction-tuned models have already been trained to resist those specific bias patterns), not that these models are immune to all cognitive biases. We are revising these categories.

**Probe decodability does not establish a causal role.** A linear probe achieving 0.999 AUC shows that the conflict/control distinction is linearly represented in layer 14 -- it does not show that the model uses this representation to determine its behavior. Activation patching experiments are needed to establish causality. These are planned but not yet complete.

**The 0.07 AUC gap may not be meaningfully large.** The difference between 0.999 and 0.929 AUC is real and replicable (permutation test p < 0.001 for both), but whether this gap is practically meaningful for the S2-by-default interpretation is debatable. Both models still represent the distinction well above chance. The interpretation depends on the gap being qualitatively important, which we have not formally established.

**We have only tested 8B-parameter models.** All results are from 8B-scale models. The behavioral and mechanistic patterns may not hold at larger scales, where models may have qualitatively different internal organization. We cannot make claims about frontier models from this data.

**The reasoning model comparison is confounded.** R1-Distill-Llama-8B differs from Llama-3.1-8B-Instruct not only in reasoning training but also in the specifics of fine-tuning data, optimization, and any other differences in the training pipeline. We cannot attribute the behavioral or mechanistic differences solely to reasoning-trace distillation.

**Benchmark novelty is not guaranteed.** While we used novel structural isomorphs, we cannot rule out that training data contained similar patterns. The classic CRT items serve as contamination baselines, but this is an imperfect control.

## Acknowledgments

This work is part of a semester project at Harvard Undergraduate Studies in AI Safety (HUSAI). Thanks to the HUSAI faculty and peer reviewers for feedback on the experimental design.

Code and benchmark will be released upon completion of the full analysis.

---

*Feedback welcome, especially on: (1) the S2-by-default interpretation -- are there alternative explanations for the probe AUC gap? (2) Additional confounds we should control for. (3) Whether the 0% categories suggest the benchmark needs harder items or is working as intended.*
