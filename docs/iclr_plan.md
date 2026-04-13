# ICLR 2027 Full Paper Plan

**Paper**: Representational Signatures of Deliberation-Intensity Gradients in LLMs
**Target venue**: ICLR 2027 (submission deadline ~October 2026)
**Workshop paper**: ICML 2026 MechInterp Workshop (submitted May 2026)
**Last updated**: 2026-04-12

---

## Workshop-to-full-paper gap analysis

The workshop paper establishes:
- Category-specific vulnerability profiles across 3 architecture families (Llama, Qwen, OLMo)
- Probe AUC gap between standard and reasoning-trained models (0.974 vs 0.930, non-overlapping CIs)
- Training-vs-inference dissociation (Qwen THINK/NO_THINK identical at P0)
- Cross-prediction specificity (0.378 AUC, below chance)
- Cross-category transfer matrix (base_rate <-> conjunction at 0.993)

The workshop paper is vulnerable to five criticisms the full paper must close:

1. **No causal evidence.** Everything is correlational. The word "mechanistic" is aspirational without steering/ablation data.
2. **Scale is 7-8B only.** Reviewers at ICLR will demand evidence above 8B.
3. **Qwen dissociation is trivially expected at P0.** Within-CoT probing is needed to make this claim non-obvious.
4. **Natural frequency N=10 with a known scoring bug.** The Gigerenzer claim is not publishable at current sample size.
5. **No human comparison.** The dual-process framing invites the question "how do these compare to humans?"

---

## Workstream 1: Scale (highest priority)

### Objective
Test whether the AUC gap between standard and reasoning-trained models holds, shrinks, or grows at 4x scale.

### Models
- **OLMo 3-32B Instruct** (standard, ~32B params)
- **OLMo 3-32B Think** (reasoning-trained, matched architecture)

OLMo 3 is the cleanest pair available at this scale: same architecture, same base weights, same organization (AI2), only training recipe differs. Llama-3.1-70B-Instruct vs R1-Distill-Llama-70B would be a stronger comparison but R1-Distill-70B does not exist -- DeepSeek only released 8B and 1.5B distillations.

### Experiments

| Experiment | Items | Compute | Hardware | Time est. |
|------------|-------|---------|----------|-----------|
| Behavioral (470 items, greedy) | 470 x 2 models | ~$15 | 1x A100-80GB | 4-6h |
| Activation extraction (470 items, 64 layers) | 470 x 2 models | ~$20 | 1x A100-80GB | 6-8h |
| Probes + bootstrap CIs | CPU post-processing | ~$0 | Local | 1h |
| Multi-seed behavioral (3 seeds, sampled) | 470 x 2 x 3 | ~$15 | 1x A100-80GB | 8-12h |

**Total estimated cost: ~$50 on A100-80GB.**

OLMo-3-32B fits in bf16 on a single A100-80GB (32B params x 2 bytes = 64 GB, within 80 GB). No quantization needed. If VRAM is tight during extraction (activations + model), use gradient checkpointing or extract one layer at a time.

### Key questions
- Does the AUC gap replicate at 32B? (Predicted: yes, gap may narrow because larger models are generally better calibrated.)
- Does the peak layer shift replicate? (At 7B: L16 -> L31. At 32B with 64 layers, expect analogous shift.)
- Does the behavioral vulnerability profile change? (At 7B-OLMo: base_rate 45.7%, conjunction 50%, syllogism 0%. At 32B, syllogism may become vulnerable -- larger models are more susceptible to belief bias in some studies.)

### Dependencies
- None. Can start immediately.

### Deliverable
Table 3 in the paper: "Scale analysis" showing behavioral and probe results at 7B and 32B for the OLMo family, with bootstrap CIs on the AUC difference.

---

## Workstream 2: Causal interventions

### Objective
Move from correlation to causation. Steer along the S1/S2 direction identified by probes. If amplifying S2 features reduces lure rate and ablating them increases it, the representation is functionally relevant.

### Approach A: Probe-direction steering (simpler, do first)

Use the linear probe weight vector at the peak layer as the steering direction. This requires no SAE.

| Experiment | Method | Models | Items |
|------------|--------|--------|-------|
| S2-amplification | Add alpha * w_probe to residual stream at peak layer | Llama, R1-Distill | 470 |
| S2-ablation | Subtract alpha * w_probe from residual stream at peak layer | Llama, R1-Distill | 470 |
| Dose-response | Sweep alpha in [-3, -2, -1, 0, 1, 2, 3] | Llama | 470 |
| Random-direction control | Steer along 10 random unit vectors, same alpha range | Llama | 470 |

**Target effect**: CogBias achieved 26-32% bias reduction via activation steering. We should match or exceed this.

**Success criterion**: S2-amplification reduces Llama lure rate by >=15pp on vulnerable categories (pre-registered as H4). S2-ablation increases it. Random control moves it <3pp.

**Compute**: ~$20 on A100-80GB (generation with hooks, 7 alpha values x 470 items x 2 directions + controls).

### Approach B: SAE-based feature steering (stronger, do second)

The Goodfire L19 SAE for Llama has 41 surviving features but two problems: (a) trigger tokens are scenario-scaffolding, (b) top features have borderline falsification ratios. Before using SAE features for steering:

1. **Re-run Ma et al. falsification in full-forward mode** for the top 10 features (not the cheap token-overlap check). If features 5402 and 19622 fail, exclude them.
2. **Train a custom SAE for R1-Distill.** The Goodfire SAE does not transfer (EV=25%). Use SAELens on R1-Distill-Llama-8B at L31 (peak layer). Training a 65K-feature SAE on ~10K activations from our benchmark + a 1M-token general corpus should take ~2h on a single GPU.
3. **Feature-level steering**: Clamp the top 5 S2-preferring features to their 95th-percentile activation and regenerate. Separately, zero-ablate the top 5 S1-preferring features. Measure lure rate change.

**Compute**: ~$30 (SAE training + feature steering runs).

### Approach C: Interchange interventions (strongest, stretch goal)

Patch activations from control-item forward passes into conflict-item forward passes at the peak layer. If the model switches from lure to correct answer, the representation at that layer is causally upstream of the decision. This is the gold standard (Geiger et al., 2021).

**Compute**: ~$10 (paired forward passes with hooks).

### Dependencies
- Approach A: needs probe weight vectors (already computed for Llama/R1-Distill)
- Approach B: needs custom R1-Distill SAE (must train)
- Approach C: needs paired activation data (already extracted)
- All approaches use the existing `src/s1s2/causal/` module (steering.py, ablation.py already written)

### Deliverable
Figure 3 in the paper: dose-response curve showing lure rate vs. steering magnitude for Llama on vulnerable categories, with random-direction controls. Table 4: lure rate reduction from S2-amplification across models.

---

## Workstream 3: Within-CoT probing (Qwen)

### Objective
Resolve the "P0 is trivially identical" critique. Probe at positions within Qwen's thinking trace to test whether the representation shifts during chain-of-thought.

### Method

For each Qwen THINK response:
1. Identify the `<think>...</think>` block boundaries.
2. Extract residual stream activations at 5 positions: T0 (first token after `<think>`), T25, T50, T75, Tend (last token before `</think>`).
3. Train probes at each position and report AUC.

### Key predictions
- **If AUC at Tend equals AUC at T0 (both 0.971)**: CoT does not reshape the S1/S2 representation at all. The behavioral improvement comes from downstream read-out, not representation change. This is a strong and interesting negative result.
- **If AUC at Tend is lower than at T0**: CoT actively blurs the S1/S2 boundary during reasoning, approaching the R1-Distill pattern. This would mean inference-time CoT produces a weaker version of the same effect as reasoning training.
- **If AUC at Tend is higher than at T0**: CoT sharpens the distinction. The model "figures out" which items are conflict items during reasoning.

### Complications
- Thinking trace lengths vary widely (100-4000 tokens). T25/T50/T75 are relative positions, not absolute.
- Probe training at non-P0 positions requires rethinking the label. The label is still conflict/control (item-level), but the position is now within generated text. Need to verify that the generated text at T50 is not itself encoding the answer (which would be a confound -- the probe detects the answer, not the processing mode).
- **Mitigation**: Extract the token at each position and check whether it's answer-relevant. Report probe AUC stratified by whether the model has "committed" to an answer or not (heuristic: has the correct/lure answer string appeared in the trace before this position).

### Compute
- Qwen THINK activations at 5 positions: ~$15 on A100-80GB (need to re-extract with position tracking, 330 items).
- Probe training: CPU, <1h.

### Dependencies
- Qwen THINK behavioral results (done, 330 items)
- Needs re-extraction with multi-position hooks (new script)

### Deliverable
Figure 4 in the paper: probe AUC vs. position within CoT (T0 through Tend), with the Llama and R1-Distill P0 AUCs as horizontal reference lines. Shows whether inference-time reasoning converges toward the training-time effect.

---

## Workstream 4: Natural frequency at scale

### Objective
Replace the N=10 pilot with a properly powered study. Test the Gigerenzer reversal claim across all models with enough items for significance.

### Method

1. **Expand the natural frequency item set to N=50** (50 conflict + 50 control = 100 items). Each is a base-rate problem with a natural-frequency reformulation. Use GPT-4 for item generation, then manually verify each item's logical structure and expected Bayesian answer.
2. **Fix the scoring bug**: The R1-Distill Unicode BPE issue (`run_new_items.py` line 108) must be fixed before any new data collection. Use `tokenizer.decode(..., clean_up_tokenization_spaces=True)` and remove the 500-character truncation.
3. **Run all 6 models**: Llama, R1-Distill, Qwen NO_THINK, Qwen THINK, OLMo Instruct, OLMo Think.
4. **Statistical test**: Fisher exact test for each model comparing probability-format vs. natural-frequency lure rates. Report exact binomial CIs. At N=50, power analysis: to detect a 30pp difference (84% -> 100% or 84% -> 54%) at alpha=0.05, power is >0.95 with N=50 per condition.

### Key questions
- Does the Llama 100% lure rate on natural frequency replicate at N=50? (If it drops to, say, 70%, the Gigerenzer reversal claim weakens but is still directionally interesting.)
- Does R1-Distill actually show the reversal, or was the 40% entirely a scoring artifact? (If R1-Distill stays at ~4% on natural frequency, the reversal disappears and the story simplifies: reasoning training helps on both formats.)
- Do reasoning-trained models (R1-Distill, Qwen THINK, OLMo Think) uniformly resist natural frequency framing, or is it model-specific?

### Compute
- Benchmark expansion: ~4h human time for item design and verification.
- Behavioral evaluation: ~$10 on A100-80GB (100 items x 6 models).

### Dependencies
- Fix the scoring bug in `run_new_items.py` (or write a new clean script)
- Generate and validate 40 new natural frequency items (have 10 already)

### Deliverable
Table 5 in the paper: Natural frequency vs. probability format lure rates across all models, with Fisher exact p-values and exact binomial CIs. Addresses Gigerenzer/Tooby hypothesis directly.

---

## Workstream 5: Multi-seed robustness on all models

### Objective
Complete the 3-seed x 470-item behavioral matrix for all models. Currently only Llama is done; R1-Distill is running.

### Status

| Model | Multi-seed status | Location |
|-------|------------------|----------|
| Llama-3.1-8B-Instruct | DONE (3 seeds) | `results/robustness/unsloth_*_multiseed.json` |
| R1-Distill-Llama-8B | RUNNING | `results/robustness/deepseek-ai_*_multiseed.json` |
| Qwen 3-8B NO_THINK | TODO | -- |
| Qwen 3-8B THINK | TODO | -- |
| OLMo-3-7B Instruct | TODO | -- |
| OLMo-3-7B Think | TODO | -- |

### Method
- Seeds: 0, 42, 123 (same as Llama)
- Temperature: 0.7, top_p: 0.95 (same as Llama)
- Max new tokens: 4096 for reasoning models, 512 for standard
- Script: `scripts/multiseed_behavioral.py`

### Compute
- ~$15 per model (470 items x 3 seeds x sampled decoding)
- 4 remaining models: ~$60 total
- Qwen THINK will be slowest (long thinking traces)

### Key question
Does the Llama finding (category profiles shift dramatically between greedy and sampled) replicate across all models? If R1-Distill also shows profile shifts, the greedy-vs-sampled distinction is a general phenomenon, not Llama-specific.

### Deliverable
Supplementary Table S1: Full 6-model x 3-seed x 11-category matrix with per-category means and standard deviations. Main text reports overall stability (whether overall lure rates change) and highlights any category-level surprises.

---

## Workstream 6: Additional bias categories

### Objective
Strengthen the vulnerability profile analysis with more data on categories that showed interesting model-specific patterns.

### Priority targets

| Category | Current finding | What to test | Models |
|----------|----------------|-------------|--------|
| Loss aversion | 33% in OLMo, untested in Llama/R1-Distill | Run existing 15 items on Llama, R1-Distill, Qwen | All 6 |
| Certainty effect | 0% greedy, 7% sampled in Llama | Redesign items -- current ones may be too easy | All 6 |
| Availability heuristic | 9% sampled, unstable (10pp std) in Llama | Add 10 more items for stability | All 6 |
| Gambler's fallacy | Not yet included | Design 20 new items | All 6 |

### Certainty effect redesign
The current certainty effect items produce 0% lure rate across all models in greedy and near-0% in sampled. This suggests the items are not sufficiently tempting. Redesign with:
- Kahneman & Tversky's original Allais paradox structure
- High-stakes scenarios (medical outcomes, not monetary gambles)
- Verified by pilot testing on GPT-4 (should produce >20% lure rate to be usable)

### Compute
- Behavioral: ~$5 per batch of new items (minimal)
- Main cost is item design time (~8h human time for gambler's fallacy + certainty redesign)

### Deliverable
Expanded Table 1 in the paper: vulnerability profiles across 13+ categories instead of 11.

---

## Workstream 7: Human comparison study

### Objective
Run the same benchmark on human participants for a direct LLM-human comparison. This is the single most impactful addition for an ICLR audience.

### Method

1. **Platform**: Prolific (pre-screened for English fluency, 18-65 age range, US/UK)
2. **N**: 100 participants (power analysis: at N=100 with 470 binary items, we can detect a 10pp difference in lure rate between humans and any model at power >0.95)
3. **Items**: Same 470-item benchmark, reformatted for survey presentation (Qualtrics or Gorilla). Each participant sees all 470 items in randomized order, split across 2 sessions to avoid fatigue.
4. **Attention checks**: 5 catch trials (obvious answers) per session. Exclude participants who fail >1 catch trial.
5. **Payment**: ~$15/participant for ~60 min total. Total cost: ~$1,700 (participants + Prolific fees).
6. **IRB**: Harvard IRB exemption for minimal-risk survey research. File the protocol by June 2026, expect approval within 2-4 weeks.

### Key questions
- Which categories are humans more/less susceptible to than LLMs? (Prediction: humans will show higher base rate neglect but lower conjunction fallacy, based on Kahneman & Tversky norms.)
- Does the natural frequency format help humans but not LLMs? (Direct test of Gigerenzer hypothesis in matched design.)
- Can we plot human lure rates against LLM lure rates per category and compute a correlation? (High correlation = similar vulnerability profile. Low correlation = LLM biases are not "human-like.")

### Dependencies
- IRB approval (2-4 week lead time)
- Item reformatting for survey platform (1 week)
- Data collection (1-2 weeks on Prolific)
- Analysis (1 week)
- Total pipeline: ~6-8 weeks from start

### Deliverable
Figure 5 in the paper: human vs. LLM lure rates by category (scatter plot with identity line). This will be the most-cited figure from the paper if the correlation is strong. Table 6: per-category human lure rates with 95% CIs, compared to each model.

---

## Compute budget

| Workstream | GPU cost | Human time | Calendar time |
|------------|----------|------------|---------------|
| 1. Scale (OLMo-32B) | $50 | 2 days | 1 week |
| 2. Causal interventions | $60 | 3 days | 2 weeks |
| 3. Within-CoT probing | $15 | 1 day | 1 week |
| 4. Natural frequency | $10 | 2 days | 1 week |
| 5. Multi-seed robustness | $60 | 1 day | 1 week |
| 6. Additional categories | $5 | 3 days | 1 week |
| 7. Human study | $1,700 | 2 weeks | 6-8 weeks |
| **Total** | **~$1,900** | **~4 weeks FTE** | **~14 weeks** |

GPU costs assume A100-80GB at RunPod spot pricing (~$1.50/hr). Human study cost is the dominant expense. Total GPU cost is under $200 -- this is a cheap paper computationally.

---

## Timeline

### Phase 1: May 2026 (post-workshop submission)
- Submit workshop paper (May 8)
- Fix scoring bug in natural frequency pipeline
- Complete R1-Distill multi-seed (may already be done by then)
- File IRB for human study
- Start item design for natural frequency expansion and certainty effect redesign

### Phase 2: June 2026 (core experiments)
- **Week 1-2**: OLMo-32B scale experiments (Workstream 1). This is the highest priority because it's the most likely reviewer objection.
- **Week 2-3**: Causal interventions Approach A -- probe-direction steering on Llama and R1-Distill (Workstream 2A).
- **Week 3-4**: Multi-seed behavioral for remaining 4 models (Workstream 5). Can run in parallel with causal.
- **Week 4**: Within-CoT probing on Qwen (Workstream 3). Requires new extraction script.

### Phase 3: July 2026 (deepening + human study)
- **Week 1-2**: Causal interventions Approach B -- train custom R1-Distill SAE, feature-level steering (Workstream 2B).
- **Week 2-3**: Natural frequency at scale (Workstream 4). Run once item set is finalized.
- **Week 3-4**: Human study data collection (Workstream 7, assuming IRB approved by early July).
- **Ongoing**: Additional bias categories as time permits (Workstream 6).

### Phase 4: August 2026 (analysis + writing)
- **Week 1**: Finish human study analysis.
- **Week 2-3**: Paper writing. Target structure below.
- **Week 4**: Internal review, advisor feedback.

### Phase 5: September 2026 (polish + submit)
- **Week 1-2**: Address advisor comments, finalize figures.
- **Week 3**: Final proofread, number verification (learned from workshop paper -- check every number against source data).
- **Week 4**: Submit to ICLR 2027.

---

## Paper structure (target ~20 pages including appendix)

### Main text (~10 pages)

1. **Introduction** (1.5p): Dual-process theory applied to LLMs, what's known (CogBias, Hagendorff), what's missing (reasoning model comparison, causal evidence, scale, human comparison).
2. **Related work** (1p): CogBias (concurrent, complementary), Coda-Forno et al. (overlapping subspaces), Lampinen et al. (review), Zhang et al. (R1 self-verification).
3. **Benchmark** (1p): 470+ items, matched-pair design, novel isomorphs, category taxonomy.
4. **Behavioral results** (1.5p): Category-specific vulnerability, reasoning training effect, multi-seed robustness, natural frequency.
5. **Representational analysis** (2p): Probes with bootstrap CIs, cross-prediction, transfer matrix, within-CoT probing, scale results.
6. **Causal interventions** (1.5p): Probe-direction steering, SAE feature steering, dose-response, interchange interventions.
7. **Human comparison** (1p): Direct LLM-human comparison on matched items.
8. **Discussion** (1.5p): S2-by-default interpretation, memorization alternative, implications for safety monitoring.

### Appendix (~10 pages)
- Full multi-seed behavioral matrix (all models, all seeds, all categories)
- Per-layer probe AUC curves for all models
- SAE feature analysis details
- Attention entropy analysis
- Geometry (silhouette, CKA) details
- Human study materials and demographics
- Natural frequency item set
- Additional bias category items

---

## Risk register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| OLMo-32B AUC gap does not replicate | 30% | High | If gap narrows to non-significance, reframe as "scale effect" -- still informative. Test a third pair if budget allows. |
| Causal steering has no effect | 20% | High | Probe direction may not be causally upstream. Approach C (interchange) is the fallback. If all three approaches fail, report the negative result -- it constrains what the probe measures. |
| IRB delayed past July | 40% | Medium | Start Prolific recruitment as soon as approved. If delayed past August, run a smaller pilot (N=30) and present as preliminary. |
| Human lure rates don't match any model | 25% | Low | This is actually interesting -- "LLM biases are not human-like" is a publishable finding in itself. |
| R1-Distill NF reversal was entirely a scoring bug | 60% | Medium | If R1-Distill shows ~4% lure on NF format (no reversal), drop the reversal claim. The Llama finding (NF doesn't help) is still worth reporting at N=50. |
| Qwen within-CoT probing shows trivial result (AUC flat) | 50% | Medium | Flat AUC across CoT is itself publishable as a negative result (CoT doesn't reshape representations). Report it honestly. |
| Custom R1-Distill SAE has poor reconstruction | 30% | Medium | If EV <60%, SAE-based steering will be unreliable. Fall back to probe-direction steering (Approach A) which doesn't need an SAE. |

---

## Critical path

The minimum viable ICLR paper requires workstreams 1, 2A, and 3. Everything else strengthens the paper but is not strictly necessary.

```
                    May              June             July             August          September
                    |                |                |                |               |
Workshop submit ----+
                    |
Fix NF scoring -----+
                    |
IRB filing ---------+----[approval]--+
                                     |
OLMo-32B ---------[==WS1==]         |
                         |           |
Probe steering ----------[==WS2A==] |
                              |      |
Multi-seed ------[========WS5=======]|
                                     |
CoT probing ----------[==WS3==]     |
                                     |
SAE + feature steer ----[===WS2B===]|
                                     |
NF at scale --------[===WS4===]     |
                                     |
Human study ---------[======WS7=====]
                                          |
Paper writing -------------------------[=====WRITE=====]
                                                        |
Polish + submit -----------------------------------[==SUBMIT==]
```

### Minimum viable submission (if time-constrained)
1. Scale (OLMo-32B probe results)
2. Causal (probe-direction steering on Llama)
3. Within-CoT probing (Qwen)
4. Paper + workshop results

This yields a paper with behavioral + representational + causal + temporal evidence across 3 architectures at 2 scales. The human study and expanded NF are high-value additions but not blocking.

---

## Open questions for advisor

1. **Author list**: Who from HUSAI should be on the full paper? Anyone external for the human study?
2. **OLMo-32B or Llama-3.1-70B**: OLMo is a cleaner pair (same org, explicit Instruct/Think split). Llama-70B would be higher profile but there's no matched reasoning model. Preference?
3. **Human study scope**: N=100 on full 470-item benchmark, or N=200 on a 100-item subset? Former is more complete; latter has higher per-item power.
4. **SAE training budget**: Should we invest in training SAEs for all models, or focus on Llama (where Goodfire exists) and R1-Distill (where we need a custom one)?
5. **Causal intervention ambition**: Probe-direction steering (simple, likely to work) vs. interchange interventions (gold standard, harder to get right). Both, or focus on one?
