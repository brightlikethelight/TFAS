# Same Architecture, Different Minds: How Reasoning Training Reorganizes Cognitive Bias Processing in LLMs

**Bright Liu, Harvard Undergraduate AI Safety Research (HUSAI)**

---

Llama-3.1-8B-Instruct falls for base rate neglect **84% of the time**. DeepSeek-R1-Distill-Llama-8B -- same architecture, same parameter count, different training -- falls for it **4%** of the time. We wanted to know what changed inside.

So we ran five workstreams of mechanistic analysis across six models and three architecture families: linear probes, cross-category transfer, training-vs-inference dissociation, sparse autoencoder feature decomposition, and attention head specialization. The evidence converges on a single story.

The result we did not expect: **the model that resists biases has *less* distinct internal encoding of bias-susceptible inputs, not more.** The model that succumbs to biases *detects* the conflict perfectly in its representations -- it just fails to act on the detection. Reasoning training does not add a deliberation module. It makes deliberation the default processing mode, erasing the need for a conflict alarm that was never reliably acted on in the first place.

## 1. The benchmark

We constructed 330 matched conflict/control item pairs across seven cognitive bias categories: base rate neglect, conjunction fallacy, syllogistic reasoning (belief-bias), CRT variants, multi-step arithmetic, framing, and anchoring. Every conflict item pits an intuitive-but-wrong lure answer against the correct answer. Every control item is structurally identical but the intuitive and correct answers agree. All items are novel structural isomorphs -- not copies of classic textbook problems -- to avoid contamination from training data.

## 2. Six findings

### Finding 1: Sharp, domain-specific behavioral boundaries

| Category | Llama-3.1-8B | R1-Distill-Llama-8B | OLMo-7B-Instruct | OLMo-7B-Think | Qwen-3-8B (no think) | Qwen-3-8B (think) |
|---|---|---|---|---|---|---|
| Base rate neglect | 84% | 4% | 14.9% | 0.9% | 56% | 4% |
| Conjunction fallacy | 55% | 0% | -- | -- | 95% | 55% |
| Syllogism bias | 52% | 0% | -- | -- | 0% | -- |
| CRT | 0% | 0% | -- | -- | 0% | 0% |
| Arithmetic | 0% | 0% | -- | -- | 0% | 0% |
| Framing | 0% | 0% | -- | -- | 0% | 0% |
| Anchoring | 0% | 0% | -- | -- | 0% | 0% |
| **Overall** | **27.3%** | **2.4%** | -- | -- | **21%** | **7%** |

Three categories (base rate, conjunction, syllogism) discriminate between models. Four categories show floor effects everywhere. These models resist CRT lures, arithmetic tricks, framing manipulations, and anchoring effects -- while simultaneously falling for elementary probability errors at near-ceiling rates. Whatever "fast thinking" these models do, it is specific to probabilistic estimation, not a general heuristic mode. The immune categories serve double duty as built-in negative controls for the mechanistic analyses below.

The Llama-to-R1-Distill comparison is the cleanest: identical architecture, identical base weights, differing only in reasoning distillation. Base rate: `84% -> 4%`. Conjunction: `55% -> 0%`. Syllogism: `52% -> 0%`. The Qwen think/no-think comparison is the cleanest within-model test: identical weights, `21% -> 7%` overall. But conjunction remains stubbornly high even with explicit chain-of-thought (`95% -> 55%`), suggesting deliberation helps unevenly across bias types.

### Finding 2: Probes reveal detection without resolution

We extracted residual stream activations at every layer for all 330 items and trained L2-regularized logistic regression probes (5-fold stratified CV, Hewitt & Liang random-label controls) to classify conflict vs. control items.

**Peak probe AUC on vulnerable categories:**

| Model | Peak Layer | AUC |
|---|---|---|
| Llama-3.1-8B-Instruct | L14 | **0.999** |
| R1-Distill-Llama-8B | L14 | **0.929** |
| OLMo-7B-Instruct | -- | **0.998** |
| OLMo-7B-Think | -- | **0.993** |
| Qwen-3-8B (no think) | L34 | **0.971** |
| Qwen-3-8B (think) | L34 | **0.971** |

We pre-registered the prediction that reasoning models would show *stronger* internal separation -- a sharper boundary between "this needs deliberation" and "this is routine." The data say the opposite.

Llama achieves near-perfect internal separation (AUC `0.999`) while falling for biases 27% of the time. R1-Distill achieves *reduced* separation (AUC `0.929`) while resisting them. Both peak at the same layer (14 of 32). The locus does not move; the sharpness decreases. The OLMo pair shows the same pattern: `0.998 -> 0.993`.

The interpretation: **"S2-by-default" processing.** The standard model maintains a crisp internal flag -- "this input is dangerous, engage deliberation" -- but often fails to act on it. The reasoning-trained model has partially lost this flag because it applies deliberation-like computation to *everything*. It does not need to distinguish dangerous inputs from safe ones because its default processing already incorporates the extra effort.

In short: Llama *knows* it should think harder. It just does not.

### Finding 3: The probe captures processing mode, not surface features

The most obvious confound: maybe probes just detect surface features of lure text. We tested this by applying a probe trained on Llama's vulnerable-category activations to Llama's *immune*-category activations (where the model shows 0% lure rate but the input still contains lure text).

**Transfer AUC: `0.378`** -- below chance.

The probe learned a direction specific to how Llama processes problems it is *vulnerable to*. That direction does not transfer to problems it handles correctly, even though those problems also contain textual lures. If the probe had merely learned "lure text is present," transfer would be positive. The negative transfer confirms the probe captures a processing-mode-specific signal, not an input artifact.

We also computed pairwise cross-category transfer. **Base rate and conjunction categories transfer at AUC `0.993`** (bidirectional). A probe trained purely on base rate neglect items near-perfectly classifies conjunction fallacy items. These are superficially different tasks -- ignoring prior probabilities vs. judging compound event likelihood -- but both require calibrated probabilistic reasoning that competes with salient narrative content. The shared representation suggests a single underlying vulnerability circuit, not independent failure modes. Transfer to immune categories is near zero.

### Finding 4: Training and inference change different things

Qwen-3-8B offers a within-model comparison no cross-model study can: the same weights, run with and without explicit chain-of-thought.

**Think and no-think modes produce identical probe curves.** Peak AUC `0.971` at L34 in both conditions. Same weights, same internal geometry, despite a 14 percentage-point behavioral gap.

Now compare with the Llama/R1-Distill pair, where different training produces both different behavior *and* different probe separability:

| Manipulation | AUC gap | Behavioral gap |
|---|---|---|
| **Training** (Llama vs. R1-Distill) | 0.070 | 24.9 pp |
| **Inference** (Qwen think vs. no-think) | 0.000 | 14.0 pp |

Training changes the residual stream representation. Inference-time chain-of-thought changes the output without touching the representation. CoT operates downstream -- in the generation/decoding process -- while leaving the residual stream geometry untouched. The model's initial "read" of the problem is set by the weights, full stop.

We extracted continuous lure susceptibility scores measuring how much each model's internal state favors the lure vs. the correct answer at the initial prompt representation:

| Model | Mean lure susceptibility |
|---|---|
| Llama-3.1-8B-Instruct | **+0.422** |
| R1-Distill-Llama-8B | **-0.326** |

Llama's residual stream, on average, *points toward* the lure. R1-Distill's *points away from it*. Reasoning training does not just add a correction step downstream -- it flips the model's initial disposition.

### Finding 5: SAE features survive falsification

We ran differential activation analysis using the Goodfire sparse autoencoder (layer 19 of Llama-3.1-8B-Instruct, 131K features). After Benjamini-Hochberg correction at q=0.05, **41 features** show significantly different activation between conflict and control items.

We then applied the Ma et al. (2026) falsification protocol: for each significant feature, inject its top-activating tokens into 100 random non-cognitive-bias texts. If the feature still activates, it is a token-level artifact, not a processing-mode feature. Ma et al. found that 45-90% of claimed "reasoning features" in prior work were spurious by this test.

**All 41 features survived falsification. Zero were spurious.**

This is unusually clean. Our interpretation: because the benchmark items are structurally matched (conflict and control items share surface form, differing only in whether the intuitive and correct answers agree), the SAE features that discriminate between conditions cannot be explained by superficial text features. The matched-pair design of the benchmark constrains the SAE analysis in the same way it constrains the probes. The features are responding to the processing-mode difference, not the input difference, because the input difference has been controlled away.

### Finding 6: Reasoning models have 2x more specialized attention heads

We computed attention entropy across all heads and layers, then tested for systematic differences between conflict and control items using per-head Mann-Whitney U tests with Benjamini-Hochberg correction.

R1-Distill has **5.6%** of attention heads showing significant entropy differences between conflict and control items (S2-specialized heads). Llama has **2.9%**. The reasoning-trained model does not merely change the residual stream -- it recruits roughly twice as many attention heads into conflict-sensitive processing.

This complements the probe results. The probes show that reasoning training *blurs* the conflict/control boundary in the residual stream (AUC drops). The attention analysis shows where the slack is taken up: more heads attend differently to conflict items, distributing the deliberation computation more broadly rather than concentrating it in a single, sharp direction.

## 3. The OLMo replication

The first four findings were established on Llama and Qwen -- two architecture families from the same ecosystem (Meta and Alibaba, both based on dense transformer variants). The obvious question: is this a property of these specific model families, or something more general?

OLMo (Allen AI) provides the strongest available test. It is a fully open-source model with a known, documented training pipeline -- independently developed architecture, independent training data, independent instruction tuning. The OLMo-7B-Instruct/Think pair gives us our third architecture family.

**Behavioral replication:** OLMo-7B-Instruct shows a 14.9% lure rate on base rate neglect. OLMo-7B-Think shows 0.9%. The same pattern: instruct model susceptible, reasoning model resistant.

**Probe replication:** OLMo-7B-Instruct achieves AUC `0.998`. OLMo-7B-Think achieves AUC `0.993`. The gap is smaller than Llama's (`0.005` vs. `0.070`), but the direction is the same: reasoning training reduces probe separability. The instruct model detects the conflict better internally while failing to act on it more often.

With three independent architecture families all showing the same pattern -- behavioral improvement, representational blurring, detection without resolution -- this is no longer a one-model curiosity. It is a consistent consequence of reasoning training.

## 4. The surprises

Two results broke our expectations in ways that sharpen the story.

### Natural frequency reversal

Gigerenzer's (1995) ecological rationality thesis predicts that natural frequency formats ("3 out of 100") should reduce base rate neglect compared to probability formats ("3%") because natural frequencies preserve the nested-set structure humans evolved to process. This is a robust finding in human experiments.

We reformulated our base rate items in natural frequency format and tested Llama. The prediction: lower lure rates.

**The result: 100% lure rate.** Every single natural frequency conflict item elicited the wrong answer, compared to 84% for the standard probability format. Natural frequencies made Llama *worse*, not better.

This is a clean reversal of the human pattern. LLMs did not evolve in environments where "3 out of 100" carried privileged statistical meaning. For Llama, the natural frequency format appears to strengthen the narrative framing (the concrete "3 out of 100 engineers are described as...") rather than activating frequency-based reasoning. The ecological rationality thesis does not transfer to systems without the relevant evolutionary history. (R1-Distill natural frequency results require re-collection due to a scoring pipeline bug we documented separately; Llama results are valid.)

### OLMo loss aversion vulnerability

When we extended the benchmark to include loss aversion items (sunk cost framing), most models showed 0% lure rates -- they correctly recognized sunk costs as irrelevant. OLMo was the exception: **33% lure rate** on loss aversion items.

This is a category where Llama, R1-Distill, and Qwen all show floor effects. The vulnerability is architecture- or training-data-specific to OLMo, not a general property of instruct-tuned models. It demonstrates that different model families have different vulnerability profiles, and that testing on a single architecture gives an incomplete picture of the failure surface.

## 5. Safety implications

**Monitoring inference-time reasoning from internals is harder than it looks.** If thinking tokens do not change residual stream representations, a probe-based monitor cannot distinguish a model that genuinely conditions on its chain-of-thought from one that ignores it. The internal state looks identical either way. This is directly relevant to detecting performative reasoning -- models that emit CoT as decoration without it influencing the answer. Our result does not prove performative reasoning occurs, but it shows that the tool people might reach for to detect it (residual stream probes) would not catch it.

**Training-time interventions go deeper than inference-time ones.** If you want models that reason at the representational level, that requires training, not prompting. This is a concrete data point in the debate between reasoning distillation and inference-time scaling: they produce qualitatively different internal changes, even when the behavioral improvements look similar on aggregate metrics. A model whose default representation already points away from the lure (R1-Distill, susceptibility `-0.326`) is in a fundamentally different state than a model whose representation points toward the lure but whose CoT sometimes overrides it.

**Domain-specific vulnerabilities are invisible to aggregate benchmarks.** A model that aces CRT problems and arithmetic reasoning can simultaneously fail at base rate estimation 84% of the time. OLMo can resist every bias category except loss aversion. Safety evaluations that treat "reasoning ability" as a monolithic capability will miss these sharp, architecture-specific failure surfaces.

**SAE-based monitoring looks promising.** The 41 features that survive Ma et al. falsification provide interpretable, feature-level handles on conflict processing. Unlike linear probes (which operate in the full residual stream), SAE features decompose the representation into individually inspectable units. With zero spurious features in our analysis, the signal-to-noise ratio is high enough for practical monitoring. The matched-pair benchmark design appears to be doing significant work here -- it constrains which features can appear significant to ones that genuinely track processing mode.

**Runtime monitoring is feasible, with caveats.** The high probe AUC (`0.998-0.999` in instruct models, `0.971-0.993` in reasoning models) at identified layers suggests lightweight linear probes could flag when a model is in a heuristic-prone state *before* it generates an answer. The attention specialization finding (5.6% vs. 2.9% of heads) provides a complementary signal. But the training-vs-inference dissociation means these monitors track the model's *default disposition*, not its *runtime reasoning*. They can tell you the model is predisposed to fail; they cannot tell you whether the CoT will save it.

## 6. What is next

**Causal interventions.** All mechanistic results so far are correlational. The 41 SAE features and the probe-identified directions are candidates for activation patching: if clamping these features changes the model's answer on conflict items, we have causal evidence that the representations drive behavior, not merely reflect it. This is the single most important remaining experiment.

**Scale.** All results come from 7-8B parameter models. Whether the S2-by-default pattern holds at 70B+ is an open question. The OLMo replication shows generality across architectures; scale is the remaining axis.

**Cross-architecture SAE.** The Goodfire SAE analysis covers Llama layer 19. Extending to OLMo and Qwen (likely requiring custom-trained SAEs via SAELens) would test whether the same interpretable features recur across architectures, or whether each model family implements conflict processing through different feature sets.

**Clean training ablation.** R1-Distill differs from Llama in the full fine-tuning pipeline, not just reasoning distillation. OLMo's open training pipeline makes a cleaner ablation possible: same data, same recipe, with and without reasoning traces. This would isolate the contribution of reasoning training to the representational changes we observe.

## The punchline

Six models. Three architecture families. Five workstreams of evidence. The same story each time.

Standard instruct-tuned models maintain a near-perfect internal alarm for inputs that require careful reasoning. That alarm does not reliably trigger careful reasoning. Reasoning-trained models blur this alarm -- not because they have become worse at detecting conflict, but because they no longer need to detect it. Their default processing mode already incorporates the deliberation that the alarm was supposed to trigger.

Reasoning training does not add System 2. It makes System 2 the default.

---

Code and benchmark: [github.com/brightliu/s1s2](https://github.com/brightliu/s1s2) (release upon publication).

*Feedback welcome. In particular: (1) alternative explanations for why inference-time thinking does not change residual stream representations, (2) whether the clean SAE falsification result (0/41 spurious) is too good and suggests a methodological blind spot in the Ma et al. protocol as applied to matched-pair designs, (3) how to interpret the OLMo loss aversion vulnerability -- is this a training data artifact or something deeper about the architecture, and (4) predictions for how the pattern changes at 70B+ scale.*
