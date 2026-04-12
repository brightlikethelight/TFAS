# Presentation Outline: ICML MechInterp Workshop Talk

**Title**: Same Architecture, Different Minds: How Reasoning Training Reorganizes Cognitive Bias Processing
**Presenter**: Bright Liu, Harvard HUSAI
**Format**: 12 minutes presentation + 5 minutes Q&A
**Target**: ICML 2026 Mechanistic Interpretability Workshop

---

## Slide 1: Title Slide

### On slide
- Title: **Same Architecture, Different Minds: How Reasoning Training Reorganizes Cognitive Bias Processing**
- Bright Liu
- Harvard Undergraduate Society for Artificial Intelligence (HUSAI)
- ICML 2026 MechInterp Workshop
- Small logos: Harvard, HUSAI

### Speaker notes (~15 sec)
"I'm Bright Liu from Harvard HUSAI. Today I'll show you that reasoning training doesn't just make LLMs produce better answers -- it fundamentally reorganizes how they represent problems internally. And I'll show you a clean dissociation: training changes representations, but inference-time chain-of-thought does not."

---

## Slide 2: The Question

### On slide
- Heading: **What changes inside when LLMs learn to reason?**
- Three bullet points:
  - Reasoning-trained LLMs resist cognitive biases that trip up standard models
  - Is this a genuine internal reorganization, or surface-level output mimicry?
  - **Natural experiment**: Llama-3.1-8B-Instruct vs. DeepSeek-R1-Distill-Llama-8B -- identical architecture, identical parameter count, different training
- Small architecture diagram: same box (Llama 8B, 32 layers, 4096 hidden) with two arrows diverging to "Standard instruction tuning" and "Reasoning distillation"

### Speaker notes (~30 sec)
"The motivation is simple. We know reasoning-trained models do better on tasks that require careful thinking. But mechanistic interpretability lets us ask a sharper question: does the model's internal representation of the problem actually change, or does it just learn to produce better-looking reasoning traces? We have a clean natural experiment. Llama-3.1-8B-Instruct and R1-Distill-Llama-8B share the exact same architecture and parameter count. They differ only in training. Any representational difference we find is attributable to reasoning distillation, not architecture."

### Anticipated questions
- **Q: Why not compare base vs. instruct-tuned?** A: The instruct-to-reasoning-distill comparison isolates reasoning training specifically. Base-to-instruct conflates instruction following with reasoning ability.
- **Q: Why 8B scale?** A: Practical -- single GPU for activation extraction, pre-trained SAEs available. We acknowledge scale as a limitation.

---

## Slide 3: Benchmark Design

### On slide
- Heading: **Cognitive Bias Benchmark: De Neys Paradigm for LLMs**
- Key stats in a callout box: **330 matched pairs, 7 categories, 4 heuristic families**
- Example item pair (abbreviated):
  - **Conflict**: "In a city where 95% of taxis are Green and 5% are Blue, a witness (80% reliable) says the taxi was Blue. What color was it most likely?" [Lure: Blue. Correct: Green]
  - **Control**: Same structure, witness reliability and base rates agree [No lure]
- Categories listed in two groups:
  - Vulnerable: base rate neglect, conjunction fallacy, syllogistic reasoning
  - Immune (negative controls): CRT, arithmetic, framing, anchoring
- Note: "All novel isomorphs -- no classic textbook items in the test set"

### Speaker notes (~30 sec)
"We built a benchmark using De Neys' matched-pair paradigm from cognitive psychology. Every conflict item has a matched control where the intuitive and correct answers agree. This controls for task difficulty and surface features. We have seven categories spanning four heuristic families. Critically, all items are novel structural isomorphs -- not copies of classic problems you'd find in training data. The classic items exist only as contamination baselines. This turned out to matter: three categories discriminate between models; four show floor effects and serve as built-in negative controls for the mechanistic analyses."

### Anticipated questions
- **Q: How do you ensure no contamination?** A: Novel isomorphs with randomized surface features. We include classic items as baselines -- if a model only gets classic items right, that flags memorization.
- **Q: Why only 330 items?** A: Each pair requires careful structural matching. 330 is sufficient for probe analyses (tight CIs) but we acknowledge limits for per-category breakdowns.

---

## Slide 4: Behavioral Results (KEY SLIDE)

### On slide
- Heading: **Category-Specific Vulnerability, Not General Deficit**
- Main table (clean, high-contrast):

| Model | Overall | Base rate | Conjunction | Syllogism |
|-------|---------|-----------|-------------|-----------|
| Llama-3.1-8B-Instruct | **27.3%** | 84% | 55% | 52% |
| R1-Distill-Llama-8B | **2.4%** | 4% | 0% | 0% |
| Qwen 3-8B (no think) | **21%** | 56% | 95% | 0% |
| Qwen 3-8B (think) | **7%** | 4% | 55% | 0% |

- Footnote: "4 additional categories (CRT, arithmetic, framing, anchoring) show 0% lure rates across all models"
- Two callout annotations:
  - Arrow on Llama row: "84% base rate -- near ceiling heuristic capture"
  - Arrow on R1-Distill row: "Same architecture, 80pp drop"
- Small inset: Qwen think/no-think delta highlighted -- "Same weights, 14pp behavioral gap"

### Speaker notes (~45 sec)
"Here are the behavioral results. Three things to notice. First, vulnerability is sharply categorical, not graded. Llama falls for base rate neglect 84% of the time -- near ceiling -- while scoring 0% on CRT and arithmetic lures. This isn't a general 'fast thinking' deficit. It's specific to probabilistic estimation tasks. Second, the Llama-to-R1-Distill comparison. Same architecture, 80 percentage-point drop on base rate neglect, conjunction goes to zero, syllogism goes to zero. Reasoning distillation crushes these specific vulnerabilities. Third, the Qwen within-model comparison. Same weights, but enabling chain-of-thought drops the overall rate from 21% to 7%. Notably, conjunction stays stubbornly high at 55% even with explicit thinking. Deliberation helps, but it doesn't close the gap that training closes. That asymmetry will matter for the mechanistic story."

### Anticipated questions
- **Q: Why does conjunction resist thinking in Qwen?** A: Likely requires a qualitatively different kind of probabilistic reasoning (calibrating joint probabilities) that explicit CoT doesn't reliably elicit. Base rate problems have a more algorithmic solution path.
- **Q: Is the 0% on CRT/arithmetic meaningful or a benchmark limitation?** A: Both, honestly. These LLMs at 8B scale have strong arithmetic. The floor effect means these categories can't discriminate -- but they serve as negative controls.

---

## Slide 5: The Probe

### On slide
- Heading: **Linear Probes on the Residual Stream**
- **Figure 1a** (the paper's key figure): Layer-wise probe AUC curves
  - Llama curve (blue): rises to 0.999 at layer 14, stays high
  - R1-Distill curve (orange): rises to 0.929 at layer 14, diverges from Llama
  - Hewitt-Liang random-label control (gray dashed): flat at ~0.5
  - X-axis: Layer (0-32), Y-axis: AUC
  - Both curves peak at L14 -- annotated
- Summary stats below figure:

| Model | Peak AUC | Peak Layer |
|-------|----------|------------|
| Llama | 0.999 | 14 |
| R1-Distill | 0.929 | 14 |
| Qwen (no think) | 0.971 | 34 |

- Method note: "L2-regularized logistic regression, 5-fold stratified CV, Hewitt-Liang selectivity controls"

### Speaker notes (~40 sec)
"We extracted residual stream activations at every layer for all 330 items and trained linear probes to classify conflict versus control. This figure is the mechanistic core of the paper. Both models peak at layer 14 -- the locus doesn't move. But Llama achieves 0.999 AUC while R1-Distill achieves 0.929. The gray line is the Hewitt-Liang random-label control, confirming the signal isn't just probe expressiveness. Qwen falls in between at 0.971, peaking later at layer 34 of its deeper architecture. The critical point: these are linear probes on the *same* inputs. Both models see identical prompts. The AUC gap means reasoning training changed the geometry of how these inputs are represented."

### Anticipated questions
- **Q: Why linear probes and not nonlinear?** A: Linear probes test whether information is linearly accessible -- the standard in mech interp. Nonlinear probes can extract information that the model itself may not use. We want to know what's in the residual stream's linear readout space.
- **Q: Is 0.929 still very high -- is the gap meaningful?** A: Good question. At this AUC range, the gap corresponds to the tail of the distribution -- the hardest-to-classify items. But the gap is robust across CV folds and aligns with behavioral changes. The direction matters more than the magnitude.

---

## Slide 6: The Puzzle

### On slide
- Heading: **The Model That Resists Biases Encodes Them Less Distinctly**
- Visual: Two-panel schematic
  - Left ("Llama -- Standard"): Two well-separated clusters labeled "Conflict" and "Control" with a clean decision boundary. Caption: "AUC 0.999 -- sharp S1/S2 boundary"
  - Right ("R1-Distill -- Reasoning"): Two overlapping clusters, blurred boundary. Caption: "AUC 0.929 -- blurred S1/S2 boundary"
- Below: "Pre-registered prediction: reasoning models would show STRONGER separation. Data showed the OPPOSITE."
- Key interpretive frame in a box: **"S2-by-default"**: The reasoning model applies deliberation-like computation to everything. It doesn't need to flag which items require extra effort -- its default processing already incorporates that effort.

### Speaker notes (~40 sec)
"This is the result we didn't expect. We pre-registered the hypothesis that reasoning models would show *stronger* internal separation between conflict and control items -- a sharper S1/S2 boundary. The data show the opposite. The model that *resists* biases maintains a *less* distinct internal boundary. The interpretation that fits both the behavioral improvement and the representational blurring is what we call 'S2-by-default' processing. Llama maintains a crisp distinction between 'this needs deliberation' and 'this doesn't' -- and then often fails to act on it. R1-Distill has partially lost this distinction because it processes everything through a more deliberative pathway. It doesn't need to flag items as requiring extra effort because its default mode already *is* effortful. This connects to Evans' concept of Type 2 autonomy in dual-process theory -- reasoning that has become automatic through practice."

### Anticipated questions
- **Q: Could the AUC drop just mean R1-Distill learned a nonlinear boundary?** A: Possible but unlikely. The geometry analysis shows overlapping distributions (silhouette 0.059), not complex nonlinear structure. And the Hewitt-Liang selectivity is high for both.
- **Q: Is "S2-by-default" falsifiable?** A: Yes -- if causal interventions can steer R1-Distill *into* a heuristic mode, that would mean S1 processing is suppressed, not absent. This is planned future work.

---

## Slide 7: Specificity -- Is the Probe Real?

### On slide
- Heading: **Cross-Prediction Resolves the Specificity Confound**
- **Figure 1b** (the paper's confound-resolution figure): Bar chart showing four conditions:
  - Train vulnerable / Test vulnerable: AUC 0.999 (Llama)
  - Train vulnerable / Test immune: AUC **0.378** (below chance!)
  - Train immune / Test immune: AUC 1.000
  - Train immune / Test vulnerable: low
- Highlight box on 0.378: "Below chance = the probe direction for vulnerable categories is ORTHOGONAL to lure-text detection"
- Additional finding: "Base rate <-> conjunction transfer AUC = 0.993. These bias types share internal representations."
- R1-Distill layer-dissociation note: "Early layers (L4-8): 0.878 transfer. Late layers (L31): 0.385. Reasoning training reorganizes layer-wise information flow."

### Speaker notes (~35 sec)
"The obvious objection: maybe the probe just detects the presence of lure text in the prompt, not anything about processing mode. We tested this by cross-predicting -- training probes on vulnerable categories and testing on immune categories, where models show 0% lure rates but lure text is still present. The transfer AUC is 0.378 -- *below chance*. The probe direction learned from bias-susceptible processing is orthogonal to the surface-feature direction. This resolves the confound. As a bonus, the transfer matrix reveals that base rate neglect and conjunction fallacy share representations -- bidirectional transfer AUC of 0.993. These are superficially different tasks but engage the same internal mechanism. For R1-Distill, early layers show high transfer, late layers show low transfer -- reasoning training reorganizes *where* in the network processing-specific information lives."

### Anticipated questions
- **Q: Below-chance transfer means anti-correlation. What does that mean mechanistically?** A: The direction that separates conflict/control on vulnerable items actively *reverses* on immune items. Vulnerable-category conflict items and immune-category control items land on the same side. This implies the probe captures processing intensity, not task structure.
- **Q: Isn't 0.378 within noise of 0.5?** A: Across 5 CV folds, the confidence interval excludes 0.5 from above. The below-chance finding is robust.

---

## Slide 8: Training vs. Inference Dissociation (KEY SLIDE)

### On slide
- Heading: **Training Changes Representations; Thinking Changes Behavior**
- Central figure: Qwen think vs. no-think probe overlay
  - Two curves (solid green = no-think, dashed green = think) that are nearly identical
  - Both peak at AUC 0.971 at layer 34
  - Annotated: "AUC gap = 0.000"
- Comparison table:

| Comparison | Type | Behavioral gap | Representational gap |
|------------|------|---------------|---------------------|
| Llama vs. R1-Distill | Different training | 24.9pp | 0.070 AUC |
| Qwen think vs. no-think | Different inference | 14pp | 0.000 AUC |

- Bottom punchline in bold: **"Same weights + thinking = same representations, different outputs. Different weights from reasoning training = different representations AND different outputs."**

### Speaker notes (~45 sec)
"This is the most novel finding. Qwen-3-8B with and without chain-of-thought produces *identical* probe curves -- peak AUC 0.971 at layer 34 in both conditions. Same weights, same internal geometry, despite a 14 percentage-point behavioral gap. The model 'thinks harder' in its output without changing its residual stream representation. Now compare that with the Llama/R1-Distill pair, where different training produces both a behavioral gap AND a representational gap. The dissociation is clean. Training rewrites how the model encodes bias-susceptible inputs at the representation level. Inference-time thinking operates downstream -- likely in the generation and decoding process -- without altering the residual stream geometry that our probes measure. This matters for safety: if you want to monitor whether a model is 'really reasoning' from its internals, you need to know what actually changes those internals. Chain-of-thought doesn't. Training does."

### Anticipated questions
- **Q: You're probing at the last prompt token before generation. Maybe thinking changes later-token representations?** A: Good point. Our probes measure the model's initial encoding of the problem. CoT's effect may be entirely in the generation phase, after the representation is fixed. That's actually the claim -- the initial read is set by weights, CoT overrides it downstream.
- **Q: Doesn't this contradict work showing CoT changes internal states?** A: Work by Lanham et al. and others examines mid-generation representations. We examine the pre-generation representation. Both can be true: CoT doesn't change the *initial* encoding but may change representations during generation.

---

## Slide 9: The Theoretical Story

### On slide
- Heading: **From Evans' Type 2 Autonomy to S2-by-Default Processing**
- Three-panel conceptual diagram:
  1. **Standard model (Llama)**: Input -> S1 pathway (fast, default) vs. S2 pathway (slow, rare). Lure susceptibility: **+0.42** (representations favor the lure)
  2. **Reasoning model (R1-Distill)**: Input -> S2 pathway (default for everything). S1/S2 boundary blurred. Lure susceptibility: **-0.33** (representations favor correct answer)
  3. **Standard + CoT (Qwen think)**: Input -> S1 encoding (unchanged) -> CoT overrides at output. Lure susceptibility: **+0.42 internal, corrected externally**
- Evans quote (small): "Type 2 processing that has become autonomous through practice or expertise"
- Key phrase: **"R1-Distill doesn't add System 2. It makes System 2 the default."**

### Speaker notes (~35 sec)
"Putting it together. The standard model maintains a sharp internal distinction between items requiring deliberation and items that don't -- and then often fails to act on that distinction. Its initial representations actively favor the lure, with a mean lure susceptibility of positive 0.42. The reasoning model has blurred this distinction because it applies deliberative computation to everything. Its initial representations favor the *correct* answer -- negative 0.33. That sign flip is not a small correction. It's a fundamental change in the model's initial disposition. This maps onto Evans' concept of Type 2 autonomy in dual-process theory: reasoning that has become automatic through practice. R1-Distill doesn't add System 2 on top of System 1 -- it makes System 2 the default processing mode."

### Anticipated questions
- **Q: Is mapping dual-process theory onto LLMs justified?** A: We use it as an operational framework, not a cognitive claim. We don't claim LLMs 'have' System 1 and System 2. We claim there are graded, linearly decodable signatures that correlate with heuristic-prone versus deliberation-requiring conditions, and reasoning training modulates them.
- **Q: Could R1-Distill just have memorized the correct answers?** A: The items are novel isomorphs not in any training set. And the representational geometry change (AUC gap, lure susceptibility sign flip) is a deeper signal than output accuracy alone.

---

## Slide 10: Implications for AI Safety

### On slide
- Heading: **Monitoring, Evaluation, and the Limits of Chain-of-Thought**
- Three implications as numbered blocks:
  1. **Runtime monitoring is feasible**: Linear probes on middle-layer activations detect "heuristic mode" with AUC > 0.97. Lightweight enough for deployment-time oversight. *Flag outputs with high lure susceptibility for human review.*
  2. **Domain-specific vulnerabilities are invisible to generic benchmarks**: A model scoring perfectly on CRT and arithmetic can fail at base rate estimation 84% of the time. Safety evaluations must probe specific failure modes, not aggregate competence.
  3. **CoT is not a substitute for training**: Qwen's conjunction fallacy rate stays at 55% even with explicit thinking, vs. R1-Distill's 0%. For safety-critical applications, trust calibration cannot rely on chain-of-thought alone.
- Rigor notes (small text):
  - "Dead Salmon concern addressed via Hewitt-Liang controls + cross-prediction"
  - "Probe results are correlational -- causal interventions pending"

### Speaker notes (~30 sec)
"Three implications for safety. First, monitoring is feasible. The probe AUC is high enough that a lightweight linear classifier on middle-layer activations could flag when a model is in a heuristic-prone state. Second, domain-specific vulnerabilities are invisible to aggregate benchmarks. You can't test 'reasoning ability' as a monolith -- you have to probe specific failure modes. Third, and most practically: chain-of-thought is not a substitute for reasoning training. If you need reliable probabilistic reasoning, prompting the model to think harder is not equivalent to training it to reason. The representations are different."

### Anticipated questions
- **Q: How would you deploy the probe in practice?** A: Extract activations at a fixed layer (L14 for Llama-family models), run a pre-trained logistic regression classifier, flag inputs where P(conflict) > threshold. Adds milliseconds of latency. The challenge is calibrating the threshold and validating on deployment distribution.
- **Q: Isn't 0.929 still very high for monitoring? The gap is small.** A: For monitoring, what matters is absolute decodability, not the gap. The gap tells us about training effects; the absolute AUC tells us about monitoring feasibility. Even R1-Distill's 0.929 is operationally useful.

---

## Slide 11: Limitations and Future Work

### On slide
- Heading: **What We Don't Know Yet**
- Limitations (honest, brief):
  - No causal evidence -- all mechanistic results are correlational
  - 8B scale only -- unknown if findings hold at 70B+
  - Small N on vulnerable subset (~140 items across 3 categories)
  - Training confound: R1-Distill differs from Llama in full fine-tuning pipeline, not just reasoning traces
- Future work as three concrete next steps:
  1. **SAE feature decomposition**: Which specific features are reorganized by reasoning training? (Goodfire L19 SAE ready)
  2. **Causal interventions**: Can we steer standard models to resist biases by intervening on identified features?
  3. **Scale**: Replicate at 70B+ to test generality

### Speaker notes (~25 sec)
"Honest limitations. All mechanistic results are correlational. We show that information is present in the representation; we don't yet show the model *uses* that information. Our SAE and causal intervention pipelines are built but waiting on GPU time. We also acknowledge scale -- all results are from 8B models. The internal organization of 70B+ models may differ qualitatively. And the R1-Distill training confound: we can't isolate reasoning-trace training from other fine-tuning differences. The Qwen within-model comparison partially mitigates this, but it tests inference-time effects, not training-time effects."

### Anticipated questions
- **Q: When will causal results be available?** A: We have the full pipeline built and validated on synthetic data. We're waiting on GPU allocation for the activation extraction runs. Optimistically, within 2-3 months.
- **Q: Why not train your own reasoning model to control the training variable?** A: Cost and compute constraints. But the Qwen think/no-think comparison gives us the cleanest within-model test, and R1-Distill vs. Llama gives us the cross-training test. Together they triangulate.

---

## Slide 12: Summary

### On slide
- Heading: **Four Contributions**
- Numbered list:
  1. **Benchmark**: 330 matched conflict/control items revealing sharp, category-specific bias vulnerability in standard LLMs
  2. **Representational evidence**: Reasoning training reduces S1/S2 linear separability (AUC 0.999 -> 0.929) -- the opposite of our pre-registered prediction, consistent with "S2-by-default" processing
  3. **Training vs. inference dissociation**: Training changes representations (AUC gap = 0.07); inference-time CoT does not (AUC gap = 0.00)
  4. **Specificity**: Cross-prediction AUC of 0.378 (below chance) confirms probe captures processing mode, not surface features
- **The punchline** (large, centered, final line):

  > "Reasoning training doesn't add System 2 -- it makes System 2 the default."

- Contact / links: paper, code repo, email

### Speaker notes (~20 sec)
"To summarize. We built a benchmark, found category-specific vulnerability, and showed three mechanistic results. Reasoning training blurs the S1/S2 boundary rather than sharpening it -- consistent with making deliberation the default mode. Training changes representations while inference-time thinking does not. And cross-prediction confirms the probe captures processing mode, not surface features. The punchline: reasoning training doesn't add System 2. It makes System 2 the default. Thank you. I'm happy to take questions."

---

## Timing Budget

| Slide | Topic | Time | Cumulative |
|-------|-------|------|------------|
| 1 | Title | 0:15 | 0:15 |
| 2 | The Question | 0:30 | 0:45 |
| 3 | Benchmark Design | 0:30 | 1:15 |
| 4 | Behavioral Results | 0:45 | 2:00 |
| 5 | The Probe | 0:40 | 2:40 |
| 6 | The Puzzle | 0:40 | 3:20 |
| 7 | Specificity | 0:35 | 3:55 |
| 8 | Training vs. Inference | 0:45 | 4:40 |
| 9 | Theoretical Story | 0:35 | 5:15 |
| 10 | Safety Implications | 0:30 | 5:45 |
| 11 | Limitations / Future | 0:25 | 6:10 |
| 12 | Summary | 0:20 | 6:30 |

**Total: ~6:30** -- well under the 12-minute limit, leaving buffer for pauses, audience reactions, and the inevitable slide where you spend longer than planned (likely Slide 4 or 8). Target delivery pace: ~7-8 minutes, leaving 4-5 minutes of slack before hitting the 12-minute wall.

---

## Q&A Preparation: Top 10 Anticipated Questions

### 1. "Your probe might just detect surface features of the input text."
**Prepared answer**: Cross-prediction AUC of 0.378 (below chance) when transferring from vulnerable to immune categories. If the probe detected surface features, transfer would be high -- immune items contain lure text too. The below-chance result means the probe direction is orthogonal to surface features. Additionally, the inter-model delta (both models see identical inputs, but differ in AUC by 0.07) cannot be explained by surface features.

### 2. "How do you rule out the dead salmon problem / probe expressiveness?"
**Prepared answer**: Hewitt-Liang random-label controls at every layer. Selectivity (real AUC minus random-label AUC) exceeds 5 percentage points at peak layers. The random-label probe sits at ~0.5 AUC, confirming the signal is in the representation, not the probe capacity.

### 3. "Why linear probes? Nonlinear probes might reveal more."
**Prepared answer**: Linear probes test whether information is in the linear readout space -- the space downstream layers can access via linear transformations. Nonlinear probes can extract information the model itself may not use. For mechanistic claims, linear probes are the appropriate tool. That said, the modest silhouette scores (0.079) confirm the signal lives in a narrow linear direction, not broad geometric structure.

### 4. "Isn't mapping dual-process theory onto LLMs unjustified anthropomorphism?"
**Prepared answer**: We use the dual-process framework operationally, not as a cognitive claim. We never claim LLMs "have" System 1 and System 2. We claim: (a) there exist graded, linearly decodable signatures correlated with heuristic-prone vs. deliberation-requiring conditions, and (b) reasoning training modulates these signatures. The S1/S2 vocabulary is a useful shorthand for the audience, but the claims are purely about representational geometry.

### 5. "Could R1-Distill's lower AUC just mean it learned a nonlinear decision boundary?"
**Prepared answer**: The geometry analysis addresses this. Cosine silhouette scores are 0.059 for R1-Distill (vs. 0.079 for Llama) -- overlapping distributions in both cases, not complex nonlinear structure that a linear probe would miss. The signal genuinely is less distinct in R1-Distill, not differently shaped.

### 6. "Your Qwen result shows identical AUC. Isn't that just because you probe at the last prompt token before generation starts?"
**Prepared answer**: Exactly right, and that's the point. We probe the model's initial encoding of the problem -- what's in the weights before any generation-time computation. CoT's effect operates in the generation phase, downstream of this representation. The dissociation shows that the initial read is set by weights (training), while CoT overrides it at output (inference). Both claims are interesting.

### 7. "What about the conjunction fallacy staying at 55% in Qwen-think? Doesn't that undermine the reasoning story?"
**Prepared answer**: It actually strengthens it. It shows that explicit deliberation is not a universal fix. Conjunction fallacy may require a qualitatively different kind of probabilistic calibration that CoT doesn't reliably elicit, while base rate problems have more algorithmic solution paths. This category-specificity is itself a finding about the structure of LLM reasoning.

### 8. "8B models are small. Does this generalize?"
**Prepared answer**: Unknown, and we say so. The internal organization of 70B+ models may differ qualitatively. Our claim is empirical at this scale. The methodology (matched-pair benchmark + cross-model probing + within-model thinking toggle) generalizes regardless of the specific results.

### 9. "You mention monitoring feasibility, but how practical is extracting middle-layer activations at inference time?"
**Prepared answer**: For real-time monitoring, you need a forward hook at a single layer -- negligible compute overhead. The probe itself is logistic regression, sub-millisecond inference. The engineering challenge is deployment integration, not computation. The harder question is calibrating the threshold on deployment-distribution inputs, which we haven't tested.

### 10. "The lure susceptibility sign flip (+0.42 to -0.33) -- could this be an artifact of the probe fitting to different distributions?"
**Prepared answer**: The probe is trained on matched pairs within each model separately. The sign flip means R1-Distill's initial representation of conflict items is geometrically closer to the "control" side of the decision boundary. This is a property of the representation, not the probe fitting procedure. Both probes are fit with the same regularization and CV scheme.

---

## Presentation Style Notes

- **Do not read slides.** Slides are visual anchors; speaker notes are the content.
- **Key slides to linger on**: Slides 4 (behavioral), 6 (the puzzle), and 8 (dissociation). These carry the narrative arc: phenomenon, surprise, resolution.
- **Transition from 6 to 7**: "So the puzzle is real. But is the probe signal real? Let's address the most obvious objection..." -- creates narrative tension and resolves it.
- **Transition from 8 to 9**: "Now we have all the pieces. Let's put the story together." -- signals the synthesis.
- **Slide 12 should feel like a mic drop.** Deliver the punchline with a pause before it. Let the audience read it. Then say "Thank you" and stop.
- **Framing language throughout**: Use "S1-like / S2-like processing signatures" not "System 1 and System 2." Use "operational framework" not "theory of LLM cognition." The MechInterp audience will respect precision.
