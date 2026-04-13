# Same Architecture, Different Minds: How Reasoning Training Reorganizes Cognitive Bias Processing in LLMs

**Bright Liu, Harvard Undergraduate AI Safety Research (HUSAI)**

---

Llama-3.1-8B-Instruct falls for base rate neglect **84% of the time**. DeepSeek-R1-Distill-Llama-8B -- same architecture, same weights before fine-tuning -- falls for it **4%** of the time. We wanted to know what changed inside.

Five workstreams of mechanistic analysis across six models and three architecture families converge on a single story: **the model that resists biases has *less* distinct internal encoding of bias-susceptible inputs, not more.** The model that succumbs to biases *detects* the conflict perfectly -- it just fails to act on the detection. Reasoning training does not add a deliberation module. It makes deliberation the default processing mode.

## 1. The benchmark

470 matched conflict/control item pairs across eleven cognitive bias categories. Every conflict item pits an intuitive-but-wrong lure against the correct answer; every control item is structurally identical but the intuitive and correct answers agree. All items are novel structural isomorphs to avoid training data contamination.

## 2. Six findings

### Finding 1: Sharp vulnerability boundaries -- but they depend on how you decode

| Category | Llama-3.1-8B (greedy) | Llama-3.1-8B (sampled, 3-seed mean) | R1-Distill-Llama-8B (greedy) |
|---|---|---|---|
| Base rate neglect | 84% | 81.0% +/- 3.3pp | 4% |
| Framing | 0% | **53.3% +/- 2.9pp** | 0% |
| Syllogism bias | 52% | 41.3% +/- 9.2pp | 0% |
| CRT | 0% | **35.6% +/- 5.1pp** | 0% |
| Conjunction | 55% | 3.3% +/- 2.9pp | 0% |
| Arithmetic | 0% | 4.0% +/- 0.0pp | 0% |
| Anchoring | 0% | 0.0% | 0% |
| Loss aversion | 0% | 0.0% | 0% |
| **Overall** | **27.3%** | **27.5% +/- 1.3pp** | **2.4%** |

The overall lure rate is stable across decoding strategies (27.3% greedy vs. 27.5% +/- 1.3pp sampled), but the *category profiles shift dramatically*. Framing: 0% greedy to 53% sampled. CRT: 0% to 36%. Conjunction: 55% to 3%. Only anchoring and loss aversion are truly immune across all conditions. Greedy-only benchmarking can dramatically mischaracterize which failure modes a model has.

The Llama-to-R1-Distill comparison is the cleanest test: identical architecture, identical base weights, differing only in reasoning distillation. Base rate: `84% -> 4%`. Conjunction: `55% -> 0%`. Syllogism: `52% -> 0%`.

| Model | Overall | Base rate | Conjunction | Syllogism |
|---|---|---|---|---|
| OLMo-7B-Instruct | 14.9% | 14.9% | -- | -- |
| OLMo-7B-Think | 0.9% | 0.9% | -- | -- |
| Qwen-3-8B (no think) | 21% | 56% | 95% | 0% |
| Qwen-3-8B (think) | 7% | 4% | 55% | -- |

The Qwen think/no-think comparison is the cleanest within-model test: identical weights, `21% -> 7%` overall. But conjunction remains stubbornly high even with explicit chain-of-thought (`95% -> 55%`), suggesting deliberation helps unevenly across bias types.

### Finding 2: Probes reveal detection without resolution

L2-regularized logistic regression probes (5-fold stratified CV, Hewitt & Liang random-label controls) on residual stream activations at every layer, classifying conflict vs. control.

**Peak probe AUC on vulnerable categories (bootstrap 95% CIs, 1000 resamples):**

| Model | Peak Layer | AUC | 95% CI |
|---|---|---|---|
| Llama-3.1-8B-Instruct | L16 | **0.974** | [0.952, 0.992] |
| R1-Distill-Llama-8B | L31 | **0.930** | [0.894, 0.960] |
| OLMo-7B-Instruct | L24 | **0.996** | [0.988, 1.000] |
| OLMo-7B-Think | L22 | **0.962** | [0.934, 0.982] |
| Qwen-3-8B (no think) | L34 | **0.971** | -- |
| Qwen-3-8B (think) | L34 | **0.971** | -- |

We pre-registered the prediction that reasoning models would show *stronger* internal separation -- a sharper boundary between "this needs deliberation" and "this is routine." The data say the opposite.

Llama achieves near-perfect internal separation (AUC 0.974, CI [0.952, 0.992]) while falling for biases 27% of the time. R1-Distill achieves *reduced* separation (AUC 0.930, CI [0.894, 0.960]) while resisting them. The CIs do not overlap. Peak layer shifts from L16 to L31 -- reasoning training relocates peak processing-mode encoding deeper. The OLMo pair shows the same pattern: 0.996 [0.988, 1.000] to 0.962 [0.934, 0.982], again with non-overlapping CIs.

The interpretation: **"S2-by-default" processing.** The standard model maintains a crisp internal flag -- "this input is dangerous, engage deliberation" -- but often fails to act on it. The reasoning-trained model has partially lost this flag because it applies deliberation-like computation to *everything*. It does not need to distinguish dangerous inputs from safe ones because its default processing already incorporates the extra effort.

In short: Llama *knows* it should think harder. It just does not.

### Finding 3: The probe captures processing mode, not surface features

The obvious confound: maybe probes just detect surface features of lure text. We applied a probe trained on Llama's vulnerable-category activations to its *immune*-category activations (0% lure rate, but lure text still present). **Transfer AUC: 0.378** -- below chance. If the probe had learned "lure text is present," transfer would be positive. The negative transfer confirms it captures a processing-mode signal, not an input artifact.

Pairwise cross-category transfer sharpens this. **Base rate and conjunction transfer at AUC 0.993** (bidirectional) -- a probe trained on base rate neglect near-perfectly classifies conjunction fallacy items. Both require calibrated probabilistic reasoning competing with salient narrative content, suggesting a single underlying vulnerability circuit. Transfer to syllogism is weaker (0.594-0.627 inbound), consistent with a distinct reasoning failure (belief-logic conflict rather than probability estimation). Transfer to immune categories is near zero.

### Finding 4: Training and inference change different things

Qwen-3-8B offers a within-model comparison no cross-model study can: the same weights, run with and without explicit chain-of-thought.

**Think and no-think modes produce identical probe curves.** Peak AUC 0.971 at L34 in both conditions. Same weights, same internal geometry, despite a 14 percentage-point behavioral gap.

Now compare with the Llama/R1-Distill pair, where different training produces both different behavior *and* different probe separability:

| Manipulation | AUC gap | Behavioral gap |
|---|---|---|
| **Training** (Llama vs. R1-Distill) | 0.044 | 24.9 pp |
| **Inference** (Qwen think vs. no-think) | 0.000 | 14.0 pp |

Training changes the residual stream representation. Inference-time chain-of-thought changes the output without touching the representation. CoT operates downstream -- in the generation/decoding process -- while leaving the residual stream geometry untouched. The model's initial "read" of the problem is set by the weights, full stop.

We extracted continuous lure susceptibility scores measuring how much each model's internal state favors the lure vs. the correct answer at the initial prompt representation:

| Model | Mean lure susceptibility |
|---|---|
| Llama-3.1-8B-Instruct | **+0.422** |
| R1-Distill-Llama-8B | **-0.326** |

Llama's residual stream, on average, *points toward* the lure. R1-Distill's *points away from it*. Reasoning training does not just add a correction step downstream -- it flips the model's initial disposition.

### Finding 5: SAE features survive falsification -- but do not transfer cross-model

Differential activation analysis using the Goodfire SAE (layer 19 of Llama, 131K features). After Benjamini-Hochberg correction at q=0.05, **41 features** show significantly different activation between conflict and control items, explaining **74% of variance**.

We applied the Ma et al. (2026) falsification protocol: inject each feature's top-activating tokens into 100 random non-cognitive-bias texts. If the feature still activates, it is a token-level artifact. Ma et al. found 45-90% of claimed "reasoning features" in prior work were spurious by this test. **All 41 of our features survived. Zero were spurious.** The matched-pair design does significant work here -- features that discriminate between conditions cannot be explained by surface text because that has been controlled away.

However, **the Llama SAE does not transfer to R1-Distill**. Explained variance drops from 74% to 25%. Reasoning training reorganizes the representation enough that the same features are near-uninformative cross-model. SAE-based monitoring trained on one model variant may not generalize even within the same architecture family.

### Finding 6: Reasoning models have 2x more specialized attention heads

Per-head Mann-Whitney U tests (Benjamini-Hochberg corrected) on attention entropy between conflict and control items: R1-Distill has **5.6%** of heads showing significant entropy differences vs. Llama's **2.9%**. The reasoning-trained model recruits roughly twice as many attention heads into conflict-sensitive processing.

This complements the probe results. Reasoning training *blurs* the conflict/control boundary in the residual stream (AUC drops), but the slack is taken up by more heads attending differently to conflict items -- distributing deliberation computation more broadly rather than concentrating it in a single sharp direction.

## 3. The OLMo replication

OLMo (Allen AI) -- fully open-source, independently developed, independent training data -- provides the strongest available out-of-distribution test.

**Behavioral:** OLMo-7B-Instruct: 14.9% lure rate. OLMo-7B-Think: 0.9%. **Probe:** Instruct AUC 0.996 [0.988, 1.000], Think 0.962 [0.934, 0.982]. The gap (0.034, non-overlapping CIs) is smaller than Llama's (0.044) but statistically robust and directionally consistent.

Three independent architecture families, same pattern: behavioral improvement, representational blurring, detection without resolution. Not a one-model curiosity.

## 4. The surprises

### Natural frequency reversal

Gigerenzer's (1995) ecological rationality thesis predicts that natural frequency formats ("3 out of 100") should reduce base rate neglect compared to probability formats ("3%"). Robust finding in human experiments. We reformulated our base rate items in natural frequency format.

**Llama: 100% lure rate** -- every item wrong, vs. 84% for probability format. Natural frequencies made Llama *worse*. **R1-Distill: 50% lure rate** (corrected from 40% after a scoring bug fix), up from 4% on the standard format. Reasoning distillation's resistance partially collapses when the problem is reframed in the format designed to *help* human reasoners.

The natural frequency format appears to strengthen narrative framing rather than activating frequency-based reasoning. The ecological rationality thesis does not transfer to systems without the relevant evolutionary history.

### OLMo loss aversion vulnerability

Most models show 0% lure rates on loss aversion items. OLMo: **33%**. The vulnerability is OLMo-specific, demonstrating that different model families have different vulnerability profiles. Single-architecture testing misses these.

## 5. Limitations

We want to be explicit about what this work does not establish.

**All mechanistic results are correlational.** The probes, SAE features, and attention patterns are associations between representations and behavior. We have not run activation patching or feature steering to confirm causal relationships. The 41 SAE features and probe-identified directions are *candidates* for causal intervention, not confirmed mechanisms.

**Scale is untested.** All results come from 7-8B parameter models. Whether the S2-by-default pattern holds at 70B+ is an open question.

**Decoding dependence weakens behavioral claims.** The category vulnerability profiles shift substantially between greedy and sampled decoding (e.g., framing: 0% greedy vs. 53% sampled). We report both, but claims about which categories are "immune" are conditioned on decoding strategy. The probe results, which measure representations rather than generation, are not affected by this issue.

**SAE coverage is partial.** The Goodfire SAE analysis covers only Llama layer 19. We lack custom SAEs for OLMo and Qwen, and the Llama SAE does not transfer to R1-Distill. We cannot yet say whether the same interpretable features recur across architectures.

**The R1-Distill comparison is not a clean ablation.** R1-Distill differs from Llama in the full fine-tuning pipeline, not just reasoning distillation. OLMo's open training pipeline would enable a cleaner ablation (same data, same recipe, with and without reasoning traces), but we have not run it.

## 6. Safety implications

**Monitoring inference-time reasoning from internals is harder than it looks.** If thinking tokens do not change residual stream representations, a probe-based monitor cannot distinguish a model that genuinely conditions on its chain-of-thought from one that ignores it. The internal state looks identical either way. This is directly relevant to detecting performative reasoning -- models that emit CoT as decoration without it influencing the answer. Our result does not prove performative reasoning occurs, but it shows that the tool people might reach for to detect it (residual stream probes) would not catch it.

**Training-time interventions go deeper than inference-time ones.** If you want models that reason at the representational level, that requires training, not prompting. A model whose default representation already points away from the lure (R1-Distill, susceptibility -0.326) is in a fundamentally different state than a model whose representation points toward the lure but whose CoT sometimes overrides it. This is a concrete data point in the reasoning distillation vs. inference-time scaling debate.

**Domain-specific vulnerabilities are invisible to aggregate benchmarks.** A model that aces CRT problems under greedy decoding can simultaneously fail at base rate estimation 84% of the time, and fail at CRT 36% of the time under sampled decoding. OLMo can resist every bias category except loss aversion. Safety evaluations that treat "reasoning ability" as a monolithic capability will miss these sharp, context-dependent failure surfaces.

**SAE-based monitoring looks promising but fragile.** The 41 features that survive Ma et al. falsification provide interpretable, feature-level handles on conflict processing with 74% explained variance. But the failure to transfer from Llama to R1-Distill (74% to 25% EV) means monitoring tools built on one model's SAE decomposition may not generalize, even within the same architecture family. This is a practical obstacle for deployment-time monitoring.

## 7. What is next

**Causal interventions** are the single most important remaining experiment. The 41 SAE features and probe-identified directions are candidates for activation patching: if clamping these features changes the model's answer on conflict items, we have causal evidence that the representations drive behavior, not merely reflect it.

**Scale.** Whether the S2-by-default pattern holds at 70B+ is the remaining axis of generality after the OLMo replication established cross-architecture generality.

**Cross-architecture SAE.** Training custom SAEs for OLMo and Qwen via SAELens would test whether the same interpretable features recur across architectures or whether each model family implements conflict processing through architecture-specific feature sets.

**Clean training ablation.** OLMo's open training pipeline makes a controlled ablation possible: same data, same recipe, with and without reasoning traces, to isolate the contribution of reasoning training to the representational changes.

## The punchline

Six models. Three architecture families. Five workstreams of evidence. The same story each time.

Standard instruct-tuned models maintain a near-perfect internal alarm for inputs that require careful reasoning. That alarm does not reliably trigger careful reasoning. Reasoning-trained models blur this alarm -- not because they have become worse at detecting conflict, but because they no longer need to detect it. Their default processing mode already incorporates the deliberation that the alarm was supposed to trigger.

Reasoning training does not add System 2. It makes System 2 the default.

---

Code and benchmark: [github.com/brightliu/s1s2](https://github.com/brightliu/s1s2) (release upon publication).

## Feedback welcome

This post accompanies a paper submission; we are actively looking for ways to strengthen the work before the full version. In particular:

1. **Are we overclaiming anywhere?** The correlational-to-causal gap is the one we are most aware of. Are there others we are missing?
2. **What experiments would strengthen the weak points?** We have identified causal interventions, scale, and cross-architecture SAE as priorities. Are there others that would be more decisive?
3. **Alternative explanations.** Why might inference-time thinking fail to change residual stream representations? Is there a more parsimonious account of the probe AUC drop than "S2-by-default"?
4. **The SAE falsification result (0/41 spurious) is unusually clean.** Is the Ma et al. protocol less stringent when applied to matched-pair designs, and if so, what additional falsification test would we need?
5. **Interested in collaboration?** We are planning a full-length paper with causal interventions, scale experiments, and cross-architecture SAE analysis. If you have relevant expertise or compute, we would welcome the conversation.

*Correspondence: bliu [at] college.harvard.edu*
