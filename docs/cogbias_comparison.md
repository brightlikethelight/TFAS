# Detailed Comparison: s1s2 vs. CogBias (Huang et al., 2026)

**CogBias reference**: Huang et al. (April 2026). "CogBias: Probing and Steering Cognitive Biases in LLMs." arXiv:2604.01366.
**Our project**: "Mechanistic Signatures of S1/S2 Processing in LLMs" (ICML MechInterp Workshop submission).
**Date**: 2026-04-12.

---

## Executive Summary

CogBias and our project are the two concurrent papers probing cognitive bias representations inside LLMs. They are complementary, not competing. CogBias asks: *can we detect and steer biases away?* We ask: *what changes in the representation when models learn to reason?* CogBias demonstrates the applied intervention. We provide the mechanistic characterization that explains when and why such interventions should (or should not) work.

---

## 1. What CogBias Does That We Do Not

### 1.1 Activation steering with measured bias reduction

CogBias extracts bias directions from middle-layer activations and steers models at inference time, achieving 26-32% bias reduction across their four families. This is a concrete intervention result that we do not attempt. Our causal intervention pipeline (activation steering, feature ablation) is designed but marked as future work pending GPU time. We report representational characterization only.

**Implication for positioning**: We should explicitly cite CogBias's steering results as validating the practical utility of the directions we characterize. Conversely, our finding that reasoning training compresses the S1/S2 boundary (AUC 0.974 to 0.930) predicts that steering should be less effective on reasoning-trained models -- a testable prediction CogBias does not address.

### 1.2 Broader bias taxonomy (4 families, including Social and Response)

CogBias covers four families: Judgment, Information Processing, Social, and Response biases. Their datasets include BBQ (social bias) and BiasMonkey (response formatting bias). We restrict ourselves to cognitive biases from the Kahneman/Tversky heuristics-and-biases tradition: probabilistic reasoning errors (base rate neglect, conjunction fallacy), logical reasoning errors (syllogistic belief bias), and decision biases (sunk cost, framing, anchoring). We do not probe social biases (stereotyping, demographic prejudice) or response biases (positional, verbosity).

This is a deliberate scope choice, not a limitation. Our research question -- *does reasoning training change how models represent heuristic-prone inputs?* -- requires matched conflict/control pairs where "conflict" has a precise normative definition (the heuristic answer deviates from the Bayesian/logical answer). Social biases lack clean normative baselines in this sense. CogBias's broader taxonomy is appropriate for their question (can we steer arbitrary biases?); our narrower taxonomy is appropriate for ours (what happens mechanistically at the S1/S2 boundary?).

### 1.3 Larger probing dataset

CogBias uses Malberg30k (30,000 items) for their primary probing. We use 470 items across 11 categories. Their scale advantage gives better statistical power for per-category analyses. Our matched-pair design (every conflict item has a structurally identical control) and novel-isomorph construction (no copied textbook problems) give better confound control. These are different trade-offs for different goals.

### 1.4 Multiple external benchmark datasets

CogBias evaluates across four existing benchmarks (Malberg30k, CoBBLEr, BBQ, BiasMonkey), demonstrating generalization across dataset constructions. We use a single custom benchmark. Their multi-dataset design strengthens external validity claims. Our single-benchmark design, with its matched-pair structure and cross-category transfer tests, strengthens internal validity claims.

---

## 2. What We Do That CogBias Does Not (Novel Contributions)

### 2.1 Reasoning model comparison on matched architecture

CogBias tests Llama and Qwen in standard instruction-tuned configurations only. They do not include any reasoning-distilled model. Our Llama-3.1-8B-Instruct vs. DeepSeek-R1-Distill-Llama-8B comparison is the cleanest available test of how reasoning training changes bias representations: identical architecture, identical parameter count, differing only in reasoning distillation. This comparison produces the headline finding -- reasoning training *reduces* probe separability (AUC 0.974 to 0.930, non-overlapping bootstrap CIs) while dramatically improving behavioral outcomes (27.3% to 2.4% lure rate) -- which CogBias cannot address because they lack reasoning model baselines.

This is our single largest differentiation. CogBias characterizes where biases live in standard models. We characterize how that landscape changes when models are trained to reason.

### 2.2 Within-model thinking toggle (same weights, different inference)

Qwen-3-8B run with and without explicit chain-of-thought reasoning (THINK vs. NO_THINK) provides a within-model control that eliminates all confounds from architecture and weight differences. The result -- identical probe AUC (0.971 at L34) despite different behavioral lure rates (21% vs. 7%) -- establishes the dissociation between training effects and inference effects on bias representations. CogBias has no analogous within-model inference comparison.

### 2.3 Cross-prediction specificity test

We train probes on vulnerable categories (where models show high lure rates) and test on immune categories (where models show 0% lure rates but inputs still contain textual lures). The transfer AUC of 0.378 (below chance) for Llama at L14 resolves the most serious confound in representational probing of cognitive biases: that probes might detect lure text features rather than processing-mode features. CogBias reports high probe accuracy (93.9-99.8%) but does not test whether their probes survive this kind of specificity control.

### 2.4 Continuous lure susceptibility scoring

We extract graded P0 scores measuring how much a model's residual stream state at the last prompt token favors the lure versus the correct answer. Llama: +0.422 (actively favors lure). R1-Distill: -0.326 (actively favors correct). The sign flip and the graded distribution show that reasoning training does not just add a downstream correction -- it changes the model's initial disposition. CogBias treats bias as binary (present/absent); we measure it as a continuous internal signal.

### 2.5 Cross-category transfer matrix

We compute pairwise probe transfer between all bias categories. The standout result: base rate neglect and conjunction fallacy transfer at AUC 0.993 bidirectionally, while transfer to immune categories is near zero. This reveals that superficially different probabilistic biases share a single internal "vulnerability circuit." CogBias reports per-family probe accuracy but does not compute within-taxonomy transfer to identify shared representations.

### 2.6 Bootstrap confidence intervals with Hewitt-Liang controls

All our probe AUCs carry bootstrap 95% CIs (1000 resamples, percentile method). Llama: 0.974 [0.952, 0.992]. R1-Distill: 0.930 [0.894, 0.960]. Non-overlapping CIs establish statistical significance for the inter-model difference. Every probe also reports selectivity against a Hewitt-Liang random-label control baseline, preventing over-interpretation of high raw AUC in high-dimensional representation spaces. CogBias reports accuracy ranges but does not provide confidence intervals or random-label controls.

### 2.7 Natural frequency framing test (Gigerenzer falsification)

We reformulated base rate items using natural frequencies ("3 out of 100" instead of "3%") following Gigerenzer and Hoffrage (1995). Results:
- Llama: 16% correct (probability format) to 100% correct (natural frequency). Replicates Gigerenzer.
- R1-Distill: 96% correct (probability) to 40% correct (natural frequency). **Reverses** Gigerenzer.

The reasoning model performs *worse* when given the format that helps humans and the standard model. This demonstrates that reasoning distillation installs format-specific competence, not general probabilistic reasoning. CogBias does not test format transfer for any of their bias categories.

### 2.8 Cross-architecture replication (Llama + OLMo)

We replicate the core vulnerability pattern across three independent model families:
- **Llama-3.1-8B-Instruct**: 84% base rate, 55% conjunction, peak probe AUC 0.974.
- **Qwen-3-8B**: 56% base rate, 95% conjunction, probe AUC 0.971.
- **OLMo-3-7B-Instruct**: 45.7% base rate, 50% conjunction, peak probe AUC 0.998.

CogBias tests Llama and Qwen but reports that cross-model transfer fails (49.7% accuracy, cosine similarity 0.01). We observe the same failure of direct transfer, but by adding OLMo we show the *behavioral pattern* (base rate and conjunction as universal vulnerability categories, CRT and arithmetic as universal immune categories) generalizes even when the representational direction does not. This reframes the cross-model transfer failure as a direction alignment problem rather than evidence that different models encode biases differently.

### 2.9 Attention entropy analysis

We analyze per-head attention entropy across conflict and control conditions to determine whether S1/S2 signatures appear in attention patterns or are confined to the residual stream. Preliminary results indicate 2x more S2-specialized attention heads in the reasoning model. CogBias works exclusively with residual-stream probing and does not analyze attention patterns.

### 2.10 De Neys theoretical grounding

We situate our findings within De Neys's (2012, 2023) "logical intuition" framework and the Stanovich (2018) tripartite model, providing theoretical predictions for each result. Our pre-registration derives hypotheses from this literature. CogBias frames their work in terms of cognitive bias taxonomies (social psychology literature) rather than dual-process theory from cognitive psychology. The theoretical grounding is complementary: they connect to the bias literature, we connect to the reasoning literature.

---

## 3. Complementary Findings

### 3.1 Both find biases are linearly separable in middle layers

CogBias reports 93.9-99.8% probe accuracy with middle layers (around L40 in their models) as optimal. We report AUC 0.974 with L16 as optimal for Llama (32 layers), 0.971 at L34 for Qwen (36 layers), and 0.998 at L16-24 for OLMo (32 layers). Normalizing for architecture depth, both papers converge on middle-to-late layers as the locus of bias-relevant information. This mutual replication across independent benchmarks, probing methods, and research groups strengthens the claim that cognitive bias representations are a real phenomenon, not an artifact of any single experimental setup.

### 3.2 Both find cross-model transfer of bias directions is weak

CogBias: cross-model transfer accuracy 49.7%, cosine similarity between bias directions 0.01. Our finding: cross-architecture replication of the behavioral pattern holds, but representational directions are model-specific. The bias direction learned from Llama does not transfer to Qwen or OLMo, even though all three models exhibit the same category-specific vulnerability profile.

### 3.3 Our reasoning model finding explains CogBias's cross-model transfer failure

CogBias observes a puzzling dissociation: behavioral correlation between Llama and Qwen bias profiles is r=0.621 (moderate agreement on which biases are strong), but representational cosine similarity is 0.01 (the directions are orthogonal). They note this as an open question.

Our results offer a mechanistic explanation. We find that even within the same architecture family, reasoning training compresses and relocates the bias-relevant direction (peak layer shifts from L16 to L31, AUC drops from 0.974 to 0.930). If reasoning training on a *matched* architecture already reorganizes the representational geometry this substantially, it is unsurprising that *different* architectures with *different* training pipelines would end up with orthogonal directions -- even if the behavioral endpoints partially converge. The behavioral correlation reflects shared task demands (base rate neglect is hard for all standard models). The representational orthogonality reflects different solutions to those demands, shaped by different training data and optimization trajectories.

This reframes CogBias's negative cross-model result from "cross-model steering will not work" to "cross-model steering requires alignment of learned solutions, which reasoning training disrupts further." Their steering results within a single model remain valid; the between-model failure is architecturally predicted.

---

## 4. Head-to-Head on Methodology

| Dimension | CogBias | s1s2 (ours) |
|-----------|---------|-------------|
| **Bias scope** | 4 families (Judgment, Info Processing, Social, Response) | 4 heuristic families (Representativeness, Availability, Anchoring, Framing), 11 categories |
| **Datasets** | 4 external (Malberg30k, CoBBLEr, BBQ, BiasMonkey) | 1 custom (470 items, matched pairs, novel isomorphs) |
| **Models** | Llama, Qwen (standard only) | 6 configs: Llama Instruct, R1-Distill, Qwen NO_THINK, Qwen THINK, OLMo Instruct, OLMo Think |
| **Reasoning models** | None | R1-Distill-Llama-8B, Qwen THINK, OLMo Think |
| **Probing method** | Linear probes on residual stream | Linear probes on residual stream + bootstrap CIs + Hewitt-Liang controls |
| **Causal intervention** | Activation steering (26-32% reduction) | Planned (future work) |
| **Specificity control** | Not reported | Cross-prediction test (AUC 0.378) |
| **Transfer analysis** | Cross-model (fails at 49.7%) | Cross-category (base rate/conjunction at 0.993) + cross-model |
| **Confidence intervals** | Not reported | Bootstrap 95% CIs, non-overlapping for key comparison |
| **Format robustness** | Not tested | Natural frequency framing (Gigerenzer test) |
| **Theoretical framing** | Cognitive bias taxonomy (social psych) | Dual-process theory (De Neys, Stanovich) |

---

## 5. Why Our Paper Should Be Accepted Alongside CogBias

### 5.1 Different research questions

CogBias asks: "Can we detect and steer cognitive biases in LLMs?" This is an applied question about intervention feasibility. We ask: "What changes in the internal representation when models learn to reason about bias-prone problems?" This is a mechanistic question about training effects. Both are legitimate workshop contributions. Neither subsumes the other.

### 5.2 Complementary methods

CogBias demonstrates that activation steering can reduce bias by 26-32%. We demonstrate that reasoning training reduces the very representational signal that steering operates on (AUC 0.974 to 0.930). Together, these findings suggest a two-pronged approach: use CogBias-style steering on standard models, and use reasoning training to shift the default processing mode. The natural follow-up -- testing whether CogBias's steering vectors can push standard models toward the reasoning model's representational geometry -- requires both papers to exist.

### 5.3 We address their open questions

CogBias's most puzzling result is the dissociation between behavioral correlation (r=0.621) and representational orthogonality (cosine 0.01) across models. Our matched-architecture comparison and cross-architecture replication provide a framework for understanding this: training-dependent reorganization of bias representations is the norm, not the exception. Same behavioral pattern, different internal solutions. This directly resolves their open question with mechanistic evidence.

### 5.4 They validate our representational claims

CogBias's replication of middle-layer linear separability for cognitive biases, using completely independent benchmarks and probing pipelines, strengthens confidence in our probing results. When two groups independently find the same phenomenon with different methods, the phenomenon is real. Reviewers evaluating our paper can point to CogBias for external replication; reviewers evaluating CogBias can point to us for mechanistic depth.

### 5.5 Together: the full pipeline from characterization to intervention

The combination covers the full path:
1. **Characterize** where biases live (both papers).
2. **Understand** how reasoning training changes that representation (our paper).
3. **Intervene** via activation steering on standard models (CogBias).
4. **Predict** when steering will and will not work (our paper: steering should be less effective on reasoning models because the target direction is compressed).

Neither paper alone covers all four steps. Accepting both gives the workshop audience the complete mechanistic-to-applied arc.

---

## 6. What We Should Cite and How

**In Related Work**: CogBias as the closest concurrent work on probing cognitive bias representations. Note their steering results as the applied complement to our characterization. Acknowledge their broader taxonomy.

**In Discussion**: Use CogBias's cross-model transfer failure (49.7%, cosine 0.01) as independent evidence for our claim that reasoning training reorganizes bias representations. Our matched-architecture comparison provides the mechanistic explanation for what they observed between architectures.

**In Future Work**: Cite CogBias's steering results as motivation for our planned causal interventions. Frame the natural prediction: steering vectors extracted from standard models should be less effective on reasoning-trained models, because the target direction is compressed (AUC 0.974 to 0.930) and relocated (L16 to L31).

---

## 7. Risks and Honest Assessment

### Where CogBias is stronger
- **Causal evidence**: They have steering results. We have representational characterization only. A reviewer who values intervention over observation will prefer their contribution.
- **Scale**: 30k items vs. 470 items. Their statistical power on per-bias comparisons is substantially higher.
- **Breadth**: Four bias families vs. our focus on probabilistic and logical reasoning biases. If a reviewer wants comprehensive coverage, CogBias delivers.

### Where we are stronger
- **Reasoning model comparison**: This is a clean differentiator. No other paper (including CogBias) probes bias representations in reasoning-distilled models. The reasoning model story (S2-by-default, training/inference dissociation, format-specific competence) is novel.
- **Confound control**: Cross-prediction specificity (AUC 0.378), Hewitt-Liang controls, bootstrap CIs, matched-pair benchmark design. Our statistical rigor is higher per claim.
- **Theoretical depth**: We derive predictions from dual-process theory and test them. CogBias is more empirical/engineering-oriented.

### Where neither paper is strong
- **Scale of models**: Both test 7-8B parameter models. Neither addresses whether these findings hold at frontier scale (70B+). This is a shared limitation that reviewers may raise for both papers.
- **Causal mechanism**: CogBias shows steering works but not why. We show representational changes but not whether they are causally upstream of behavior. Full causal evidence (ablation studies, interchange interventions) remains future work for both.
