# Same Architecture, Different Minds: How Reasoning Training Reorganizes Cognitive Bias Processing in LLMs

**Bright Liu, Harvard Undergraduate AI Safety Research (HUSAI)**

---

Llama-3.1-8B-Instruct falls for base rate neglect **84% of the time**. DeepSeek-R1-Distill-Llama-8B -- same architecture, same parameter count, different training -- falls for it **4% of the time**. We wanted to know what changed inside. So we trained linear probes on residual stream activations across 330 matched cognitive bias items and four model configurations.

The result we did not expect: **the model that resists biases has *less* distinct internal encoding of bias-susceptible inputs, not more**. And the model that succumbs to biases *detects* the conflict perfectly in its internal representations -- it just fails to act on the detection.

If that sounds like a familiar story from human dual-process psychology, it should. But in LLMs, the mechanism turns out to be different in a way that matters for alignment.

## The benchmark

We constructed 330 matched conflict/control item pairs across seven cognitive bias categories: base rate neglect, conjunction fallacy, syllogistic reasoning (belief-bias), CRT variants, multi-step arithmetic, framing, and anchoring. Every conflict item pits an intuitive-but-wrong lure answer against the correct answer. Every control item is structurally identical but the intuitive and correct answers agree. All items are novel structural isomorphs -- not copies of classic textbook problems -- to avoid contamination from training data.

## Behavioral results: sharp categorical boundaries

| Category | Llama-3.1-8B | R1-Distill-Llama-8B | Qwen-3-8B (no think) | Qwen-3-8B (think) |
|---|---|---|---|---|
| Base rate neglect | 84% | 4% | 56% | 4% |
| Conjunction fallacy | 55% | 0% | 95% | 55% |
| Syllogism bias | 52% | 0% | 0% | -- |
| CRT | 0% | 0% | 0% | 0% |
| Arithmetic | 0% | 0% | 0% | 0% |
| Framing | 0% | 0% | 0% | 0% |
| Anchoring | 0% | 0% | 0% | 0% |
| **Overall** | **27.3%** | **2.4%** | **21%** | **7%** |

Two patterns jump out.

**The vulnerability is domain-specific, not general.** Three categories (base rate, conjunction, syllogism) discriminate between models. Four categories show floor effects everywhere. These models resist CRT lures, arithmetic tricks, framing manipulations, and anchoring effects -- while simultaneously falling for elementary probability errors at near-ceiling rates. Whatever "fast thinking" these models do, it is specific to probabilistic estimation, not a general heuristic mode. The immune categories serve double duty as built-in negative controls for the mechanistic analyses below.

**Reasoning training produces enormous effects.** The Llama-to-R1-Distill comparison is the cleanest in the study: identical architecture, identical base weights, differing only in reasoning distillation. Base rate: `84% -> 4%`. Conjunction: `55% -> 0%`. Syllogism: `52% -> 0%`. The Qwen think/no-think comparison is the cleanest within-model test: identical weights, `21% -> 7%` overall. But note that conjunction remains stubbornly high even with explicit chain-of-thought (`95% -> 55%`), suggesting deliberation helps unevenly across bias types.

## Probing the residual stream

We extracted residual stream activations at every layer for all 330 items and trained L2-regularized logistic regression probes (5-fold stratified CV, Hewitt & Liang random-label controls) to classify conflict vs. control items. The question: does the model internally distinguish inputs that require deliberation from those that do not?

**Peak probe AUC on the three vulnerable categories:**

| Model | Peak Layer | AUC |
|---|---|---|
| Llama-3.1-8B-Instruct | L14 | **0.999** |
| R1-Distill-Llama-8B | L14 | **0.929** |
| Qwen-3-8B (no think) | L34 | **0.971** |
| Qwen-3-8B (think) | L34 | **0.971** |

## The puzzle

Here is the finding that surprised us. We pre-registered the prediction that reasoning models would show *stronger* internal S1/S2 separation -- a sharper boundary between "this needs deliberation" and "this is routine." The data say the opposite.

Llama, the model that falls for biases 27% of the time, achieves near-perfect internal separation (AUC `0.999`). R1-Distill, the model that resists biases, achieves *reduced* separation (AUC `0.929`). Both peak at the same layer (14 of 32). The locus does not move; the sharpness decreases.

The interpretation that reconciles the behavioral improvement with the representational blurring is what we call **"S2-by-default" processing**. The standard model maintains a crisp internal flag: "this input is dangerous, engage deliberation." But it often fails to act on that flag -- it detects the conflict and produces the lure answer anyway. The reasoning-trained model has partially lost this flag because it applies deliberation-like computation to *everything*. It does not need to distinguish dangerous inputs from safe ones because its default processing already incorporates the extra effort.

In short: Llama *knows* it should think harder. It just does not.

## Four converging lines of evidence

### 1. Cross-model transfer: the probe is genuine

The most obvious confound: maybe probes just detect surface features of lure text, not anything about processing mode. We tested this by applying a probe trained on Llama's vulnerable-category activations to Llama's *immune*-category activations (where the model shows 0% lure rate but the input still contains lure text).

**Transfer AUC: `0.378`** -- below chance.

This is the key specificity result. The probe learned a direction specific to how Llama processes problems it is *vulnerable to*. That direction does not transfer to problems it handles correctly, even though those problems also contain textual lures. If the probe had merely learned "lure text is present," transfer would be positive. The negative transfer confirms the probe captures a processing-mode-specific signal, not an input artifact.

### 2. Transfer matrix: shared representations across bias types

We computed pairwise cross-category transfer. The standout result: **base rate and conjunction categories transfer at AUC `0.993`** (bidirectional). A probe trained purely on base rate neglect items near-perfectly classifies conjunction fallacy items, and vice versa.

These are superficially different tasks -- ignoring prior probabilities vs. judging compound event likelihood -- but both require calibrated probabilistic reasoning that competes with salient narrative content. The shared representation suggests a single underlying vulnerability circuit, not independent failure modes. Transfer to immune categories is near zero, confirming the specificity of this shared representation.

### 3. Lure susceptibility scores: a sign flip

We extracted continuous scores measuring how much each model's internal state favors the lure vs. the correct answer at the initial prompt representation:

| Model | Mean lure susceptibility |
|---|---|
| Llama-3.1-8B-Instruct | **+0.422** |
| R1-Distill-Llama-8B | **-0.326** |

Llama's residual stream, on average, *points toward* the lure. R1-Distill's *points away from it*. Reasoning training does not just add a correction step downstream -- it flips the model's initial disposition. The representation itself encodes a different default.

### 4. The training vs. inference dissociation

This is the most novel finding. Qwen-3-8B offers a within-model comparison no cross-model study can: the same weights, run with and without explicit chain-of-thought.

**Think and no-think modes produce identical probe curves.** Peak AUC `0.971` at L34 in both conditions. Same weights, same internal geometry, despite a 14 percentage-point behavioral gap (`21% -> 7%` overall lure rate).

Now compare with the Llama/R1-Distill pair, where different training produces both different behavior *and* different probe separability (`0.999 -> 0.929`):

| Manipulation | AUC gap | Behavioral gap |
|---|---|---|
| **Training** (Llama vs. R1-Distill) | 0.070 | 24.9 pp |
| **Inference** (Qwen think vs. no-think) | 0.000 | 14.0 pp |

The dissociation is clean. Training changes the residual stream representation. Inference-time chain-of-thought changes the output without touching the representation. CoT operates downstream -- in the generation/decoding process -- while leaving the residual stream geometry untouched. The model's initial "read" of the problem is set by the weights, full stop.

## The De Neys connection

This pattern has a direct analogue in human cognitive psychology. Wim De Neys' conflict detection research (2012, 2014) showed that humans *detect* the conflict between heuristic and normative responses even when they give the heuristic answer. The evidence: lower confidence ratings and slower response times on conflict items, even among participants who produce the wrong answer. The system detects the problem; it just fails to override the heuristic response.

Our Llama results look strikingly similar. The model achieves AUC `0.999` on conflict/control separation -- near-perfect *detection* -- while producing the lure answer 84% of the time on base rate items. Detection without resolution. The signal is present in the residual stream; it is not acted upon.

But the mechanism of resolution differs from what De Neys describes in humans. In humans, conflict detection triggers effortful Type 2 processing that can override the heuristic. In LLMs, the resolution comes not from a runtime override but from training-time reorganization of the representation itself (the lure susceptibility sign flip from `+0.422` to `-0.326`). And inference-time "thinking" -- the closest analogue to human effortful processing -- changes behavior without changing the representation. The functional architecture is different even where the behavioral pattern is similar.

## What this means for safety

**Monitoring inference-time reasoning from internals is harder than it looks.** If thinking tokens do not change residual stream representations, a probe-based monitor cannot distinguish a model that is genuinely conditioning on its chain-of-thought from one that is ignoring it. The internal state looks identical either way. This is directly relevant to detecting performative reasoning -- models that emit CoT as decoration without it influencing the answer. Our result does not prove performative reasoning occurs, but it shows that the tool people might reach for to detect it (residual stream probes) would not catch it.

**Training-time interventions go deeper than inference-time ones.** If you want models that "really reason" at the representational level, that requires training, not prompting. This is a concrete data point in the debate between reasoning distillation and inference-time scaling: they produce qualitatively different internal changes, even when the behavioral improvements look similar on aggregate metrics. For safety-critical deployments, the distinction matters. A model whose default representation already points away from the lure (R1-Distill, susceptibility `-0.326`) is in a fundamentally different state than a model whose representation points toward the lure but whose CoT sometimes overrides it.

**Domain-specific vulnerabilities are invisible to aggregate benchmarks.** A model that aces CRT problems and arithmetic reasoning can simultaneously fail at base rate estimation 84% of the time. Safety evaluations that treat "reasoning ability" as a monolithic capability will miss these sharp, domain-specific failure modes.

**Runtime monitoring may still be feasible.** The high probe AUC (`0.999` in Llama, `0.971` in Qwen) at identified layers suggests lightweight linear probes could flag when a model is in a heuristic-prone state *before* it generates an answer. This is preliminary, but the lure susceptibility score provides a continuous measure of risk. If causal experiments confirm these representations drive behavior, this becomes a practical tool for deployment-time oversight.

## Limitations

We are honest about what we do not know.

**Scale.** All results come from 8B-parameter models. The internal organization of 70B+ models may differ qualitatively. We make no claims beyond the tested scale.

**No causal evidence.** All mechanistic results are correlational. Probes show that the S1/S2 distinction is linearly decodable; they do not show that the model *uses* this direction. Activation patching experiments are planned but not complete. Until then, the mechanistic story is observational.

**The dead salmon concern.** Linear probes in `d=4096` with `N~140` items can find spurious signal (cf. the fMRI dead salmon study, Bennett et al. 2009). We control for this via Hewitt & Liang random-label baselines and cross-category transfer (which kills the all-categories probe), but the concern is real and readers should weigh it.

**Training confound.** R1-Distill differs from Llama in the full fine-tuning pipeline, not just reasoning distillation. The Qwen within-model comparison partially mitigates this, but is an inference-time test rather than a training-time one. A clean training ablation (same pipeline, with and without reasoning traces) would be stronger evidence.

**Floor effects.** Four of seven categories show 0% lure rates everywhere. These may be too easy, or instruction-tuned models may have specifically learned to resist these patterns. Either way, they limit the generality of our claims to probabilistic reasoning domains.

## What is next

- **SAE feature analysis**: Sparse autoencoder features on Llama residual stream to identify interpretable features that distinguish conflict/control processing, with Ma et al. (2026) falsification to filter token-level artifacts.
- **Causal interventions**: Activation patching along probe-identified directions to test whether these representations causally influence outputs.
- **OLMo replication**: Open-weight models with known training pipelines to control for the training confound.
- **Scale**: Extending to larger models to test whether the training/inference dissociation holds.

---

Code and benchmark will be released upon completion of the full analysis.

*Feedback welcome. In particular: (1) alternative explanations for why inference-time thinking does not change residual stream representations, (2) whether the negative cross-model transfer (AUC `0.378`) is as strong evidence against the surface-feature confound as we claim, and (3) how seriously to take the d >> N concern given our controls.*
