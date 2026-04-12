# Training Changes How LLMs Represent Cognitive Biases; Inference-Time Thinking Does Not

**Bright Liu, Harvard Undergraduate AI Safety Research (HUSAI)**

---

Llama-3.1-8B-Instruct falls for base rate neglect 84% of the time. DeepSeek-R1-Distill-Llama-8B -- same architecture, same base weights, different training -- falls for it 4% of the time. We trained linear probes on internal representations to understand what changed. The headline finding: **reasoning training reshapes how models internally represent bias-susceptible inputs, but inference-time chain-of-thought does not.**

Qwen-3-8B with thinking enabled vs. disabled produces nearly identical internal representations (probe AUC 0.971 at L34 in both modes) despite a behavioral gap of 21% vs. 7% lure rate. The model "thinks harder" in its output without changing its internal geometry. By contrast, the Llama/R1-Distill pair, which differs only in training, shows a persistent representational gap (AUC 0.999 vs. 0.929, same layer). Training rewrites representations; inference-time reasoning routes around them.

This dissociation matters for safety. If you want to monitor whether a model is "really reasoning" from its internals, you need to know what actually changes those internals -- and what doesn't.

## The benchmark

We built 330 matched conflict/control item pairs across seven cognitive bias categories (CRT, base rate neglect, syllogistic reasoning, anchoring, framing, conjunction fallacy, arithmetic). Conflict items pit an intuitive-but-wrong lure against the correct answer; control items are structurally identical but the intuitive and correct answers agree. All items are novel structural isomorphs, not copies of classic problems.

## Behavioral results

Bias susceptibility rate (% of conflict items where the model produces the lure answer):

| Category | Llama-3.1-8B | R1-Distill-Llama-8B | Qwen-3-8B (no think) | Qwen-3-8B (think) |
|---|---|---|---|---|
| Base rate neglect | 84% | 4% | 56% | 4% |
| Conjunction fallacy | 55% | 0% | 95% | 55% |
| Syllogism bias | 52% | 0% | 0% | -- |
| CRT | 0% | 0% | -- | -- |
| Arithmetic | 0% | 0% | -- | -- |
| Framing | 0% | 0% | -- | -- |
| Anchoring | 0% | 0% | -- | -- |
| **Overall lure rate** | **27.3%** | **2.4%** | **21%** | **7%** |

Three categories discriminate (base rate, conjunction, syllogism); four show floor effects across all models, likely reflecting benchmark difficulty rather than universal bias immunity.

The Llama-to-R1-Distill comparison is the cleanest: same architecture, same base weights, 80pp drop on base rate neglect, 55pp on conjunction, 52pp on syllogisms. The Qwen think/no-think comparison is the cleanest within-model test: identical weights, 14pp overall reduction. Thinking helps on base rate neglect (56% to 4%) but conjunction stays stubbornly high (95% to 55%).

## Probing internal representations

We extracted residual stream activations at every layer for all 330 items and trained linear probes (logistic regression, 5-fold stratified CV, Hewitt & Liang control tasks) to classify conflict vs. control items. The question: does the model's internal state distinguish inputs that require deliberation from those that don't?

**Peak probe performance on vulnerable categories (base rate + conjunction + syllogism):**

| Model | Peak Layer | AUC |
|---|---|---|
| Llama-3.1-8B-Instruct | L14 | 0.999 |
| R1-Distill-Llama-8B | L14 | 0.929 |
| Qwen-3-8B (no think) | L34 | 0.971 |
| Qwen-3-8B (think) | L34 | 0.971 |

The Llama/R1-Distill pair peaks at the same layer (14/32) but with a persistent gap: the non-reasoning model maintains a sharper internal boundary between conflict and control items. The reasoning model has blurred this boundary -- consistent with processing everything more uniformly through a deliberative pathway ("S2-by-default").

## The headline finding: training vs. inference dissociation

The Qwen probe result is the most novel finding. **Think and no-think modes produce identical probe curves** -- peak AUC 0.971 at L34 in both conditions. Same weights, same representations, despite a 14pp behavioral gap. Inference-time chain-of-thought changes what the model *outputs* without changing what it *represents* in the residual stream.

Compare this with the Llama/R1-Distill pair, where reasoning *training* produces a measurable representational shift (0.999 to 0.929 at the same layer). The dissociation is clean:

- **Training** (Llama vs. R1-Distill): changes representations. AUC gap = 0.07.
- **Inference** (Qwen think vs. no-think): does not change representations. AUC gap = 0.00.

This suggests reasoning training rewrites how the model encodes bias-susceptible inputs at the representation level, while inference-time thinking operates downstream -- likely in the generation/decoding process -- without altering the residual stream geometry that probes read.

## Cross-prediction resolves the specificity confound

A concern: maybe probes just detect the presence of lure text (a surface feature), not anything about processing mode. We tested this with cross-model transfer. A probe trained on Llama representations was applied to R1-Distill representations.

**Llama-to-R1-Distill transfer AUC: 0.378.**

This is *below chance* (0.5), meaning the Llama probe actively anti-predicts on R1-Distill activations. The two models don't just differ in degree -- they represent the same conflict items in geometrically different directions. If the probe were merely detecting lure text, it would transfer positively. The negative transfer confirms the probe captures model-specific processing signatures, not surface features of the input.

## Transfer matrix and lure susceptibility scores

We computed a full transfer matrix across bias categories. The key finding: **base rate and conjunction categories share representations** (bidirectional transfer AUC = 0.993), while other category pairs show weaker transfer. This suggests these two bias types engage overlapping internal mechanisms -- consistent with both involving probabilistic reasoning under salient but misleading cues.

We also extracted continuous lure susceptibility scores (how much the model's internal state favors the lure answer vs. the correct answer):

| Model | Mean lure susceptibility | Interpretation |
|---|---|---|
| Llama-3.1-8B-Instruct | +0.42 | Representations favor the lure |
| R1-Distill-Llama-8B | -0.33 | Representations favor the correct answer |

The sign flip is striking. Llama's residual stream, on average, points toward the lure; R1-Distill's points away from it. Reasoning training doesn't just blur the S1/S2 boundary -- it flips the default direction of the representation from lure-favoring to correct-favoring.

## What this means for safety

**1. Monitoring inference-time "reasoning" from internals may not work the way you hope.** If thinking tokens don't change residual stream representations, a probe-based monitor would not detect whether a model is using its chain-of-thought or ignoring it. The internal state looks the same either way. This is directly relevant to detecting performative reasoning (models that emit CoT without actually conditioning on it).

**2. Training-time interventions have deeper representational effects than inference-time ones.** If you want models that "really reason" at the representational level, that appears to require training -- not prompting. This is a point in favor of reasoning distillation and against the assumption that sufficiently long chain-of-thought prompting produces the same internal changes.

**3. Cross-model probe transfer as a specificity tool.** Negative transfer (AUC < 0.5) between a base model and its reasoning-trained variant is a clean signal that probes capture model-specific processing, not input artifacts. This technique may generalize to other interpretability contexts where probe specificity is in question.

## Caveats

**We have only tested 8B-parameter models.** The training-vs-inference dissociation may not hold at larger scales where models have qualitatively different internal organization. We cannot make claims about frontier models from this data.

**The 0.07 AUC gap is real but modest.** Both models represent the conflict/control distinction well above chance (0.929 vs. 0.999). The S2-by-default interpretation depends on this gap being qualitatively meaningful, which we haven't formally established beyond permutation tests (p < 0.001).

**Probe decodability is not causality.** High AUC shows the distinction is linearly represented -- not that the model uses this representation to determine behavior. Activation patching experiments are planned but not yet complete.

**The reasoning model comparison is confounded.** R1-Distill differs from Llama not only in reasoning training but in fine-tuning data, optimization details, and other pipeline differences. The Qwen within-model comparison partially controls for this, but is imperfect (thinking mode may differ from no-think in ways beyond reasoning).

**Four benchmark categories show floor effects.** CRT, arithmetic, framing, and anchoring produce 0% lure rates across all models. These items may be too easy, or instruction-tuned models may have been trained to resist these specific patterns.

## What's next

- **Causal interventions**: Activation patching to test whether probed representations causally influence outputs.
- **SAE feature analysis**: Sparse autoencoder features on Llama L19 with Ma et al. falsification to filter token-level artifacts.
- **Scale**: Extending to larger models to test whether the training/inference dissociation holds.

---

Code and benchmark will be released upon completion of the full analysis.

*Feedback welcome, especially on: (1) alternative explanations for why inference-time thinking doesn't change residual stream representations, (2) whether the negative cross-model transfer (AUC 0.378) is as strong evidence against the surface-feature confound as we think, (3) additional controls for the training-vs-inference dissociation.*
