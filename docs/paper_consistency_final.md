# Paper Consistency Check: NeurIPS vs. Workshop

**Date**: 2026-04-12
**Status**: Pre-submission final check

## Verdict: 4 issues found (2 real inconsistencies, 2 framing mismatches)

---

## 1. Shared Numbers -- Match Status

### Behavioral rates
| Claim | Workshop | NeurIPS | Match? |
|-------|----------|---------|--------|
| Llama lure rate | 27.3% | 27.3% | YES |
| R1-Distill lure rate | 2.4% | 2.4% | YES |
| OLMo Instruct lure | 14.9% | 14.9% | YES |
| OLMo Think lure | 0.9% | 0.9% | YES |
| Qwen no-think lure | 21% | 21% | YES |
| Qwen think lure | 7.3% | 7.3% | YES |
| Per-category table values | Identical 7-row tables in both | Identical | YES |

### Probe AUCs
| Claim | Workshop | NeurIPS | Match? |
|-------|----------|---------|--------|
| Llama peak AUC | 0.974 [0.952, 0.992] | 0.974 [0.952, 0.992] | YES |
| R1-Distill peak AUC | 0.930 [0.894, 0.960] | 0.930 [0.894, 0.960] | YES |
| Llama peak layer | L16 | L16 | YES |
| R1-Distill peak layer | L31 | L31 | YES |
| Qwen AUC (both modes) | 0.971 at L34 | 0.971 at L34 | YES |
| OLMo Instruct AUC | 0.996 [0.988, 1.000] at L24 | 0.996 [0.988, 1.000] at L24 | YES |
| OLMo Think AUC | 0.962 [0.934, 0.982] at L22 | 0.962 [0.934, 0.982] at L22 | YES |
| 30/32 layers Instruct > Think | YES (workshop) | YES (NeurIPS) | YES |
| Hewitt-Liang control ceiling | <= 0.66 | <= 0.66 | YES |
| Selectivity > 34pp | YES | YES | YES |
| Cross-prediction AUC (immune) | 0.378 | 0.378 | YES |
| Within-category AUC | 0.987 | 0.987 | YES |
| Cross-model Llama->R1 | 0.920 | 0.920 | YES |
| Cross-model R1->Llama | 0.954 | 0.954 | YES |
| Llama control AUC | 0.63 | 0.63 | YES |
| R1-Distill control AUC | 0.55 | 0.55 | YES |

### Causal steering
| Claim | Workshop | NeurIPS | Match? |
|-------|----------|---------|--------|
| Total swing | 37.6pp | 37.6pp | YES |
| alpha=+5 lure rate | 31.2% | 31.2% | YES |
| alpha=-5 lure rate | 68.8% | 68.8% | YES |
| Baseline (unsteered) | 52.5% | 52.5% | YES |
| S2 direction reduction | 21.3pp | 21.3pp | YES |
| R1-Distill swing | 7.5pp | 7.5pp | YES |
| R1-Distill range | 10.0% to 2.5% | 10.0% to 2.5% | YES |
| Random controls range | 52%-58% | 52%-58% | YES |

### Within-CoT probing
| Claim | Workshop | NeurIPS | Match? |
|-------|----------|---------|--------|
| T0 AUC | 0.973 | 0.973 | YES |
| T25 AUC | 0.791 | 0.791 | YES |
| T50 AUC | 0.772 | 0.772 | YES |
| T75 AUC | 0.754 | 0.754 | YES |
| Tend AUC | 0.971 | 0.971 | YES |
| P0 AUC | 0.938 | 0.938 | YES |
| P2 AUC | 0.500 | 0.500 | YES |

### SAE and attention
| Claim | Workshop | NeurIPS | Match? |
|-------|----------|---------|--------|
| SAE features surviving | 41 | 41 | YES |
| SAE EV | 74% | 74% | YES |
| SAE L0 sparsity | 48 (workshop) vs 47.9 (NeurIPS) | **MISMATCH** | **NO** |
| S2-specialized heads R1 | 57 | 57 | YES |
| S2-specialized heads Llama | 30 | 30 | YES |

### Scale analysis (NeurIPS only)
| Claim | NeurIPS | Workshop mentions? |
|-------|---------|-------------------|
| OLMo-3.1-32B-Instruct lure | 19.6% | NO (not in scope) |
| OLMo-3.1-32B-Think lure | 0.4% | NO |
| 32B probe AUC | 0.9999 at L20 | NO |

### Conflict detection
| Claim | Workshop | NeurIPS | Match? |
|-------|----------|---------|--------|
| First-token prob gap | 4.2pp (0.751 vs 0.793) | 4.2pp (0.751 vs 0.793) | YES |
| Entropy 16% higher | 0.948 vs 0.817 | 0.948 vs 0.817 | YES |

### Lure susceptibility scores
| Claim | Workshop | NeurIPS | Match? |
|-------|----------|---------|--------|
| Llama mean score | +0.422 | +0.422 | YES |
| R1-Distill mean score | -0.326 | -0.326 | YES |

---

## 2. Issues Found

### ISSUE 1 [MINOR]: SAE L0 sparsity rounding discrepancy
- **Workshop** (line 514): "mean L0 sparsity of 48 active features per input"
- **NeurIPS** appendix (line 1080): "Mean L0 (active features) 47.9 / 65,536"
- **Action needed**: Change workshop "48" to "47.9" for consistency (or add "~48"). The NeurIPS paper has the precise number; the workshop rounds it. Not a real inconsistency but should match exactly for rigor.

### ISSUE 2 [REAL INCONSISTENCY]: Heuristic family labels differ between papers
- **Workshop** introduction (line 144-146): "four heuristic families (representativeness, **loss aversion, certainty weighting, and availability**)"
- **NeurIPS** benchmark section (line 283-284): "four heuristic families: representativeness, **cognitive reflection**, **decision framing**, and **loss/availability**"
- The NeurIPS labeling is correct and matches the actual category groupings. The workshop uses different family names that don't match the NeurIPS taxonomy.
- **Action needed**: Fix workshop line 145 to match NeurIPS: "representativeness, cognitive reflection, decision framing, and loss/availability".

### ISSUE 3 [REAL INCONSISTENCY]: Workshop contributions list says "five contributions"; NeurIPS says "six contributions"
- **Workshop** (line 139): "We make five contributions"
- **NeurIPS** (line 150): "We make six contributions"
- The NeurIPS paper adds causal steering as contribution #4 (which the workshop paper discusses but does not list as a numbered contribution). The NeurIPS also has within-CoT probing as contribution #5 and the Qwen dissociation as contribution #6.
- **This is fine**: the NeurIPS paper is a strict superset. However, the workshop paper's contribution #5 (SAE features) is NOT listed as a numbered contribution in the NeurIPS intro -- it's mentioned only in passing ("Additionally, we analyze..."). This is a framing asymmetry but not a contradiction.
- **Action needed**: None strictly required, but consider whether SAE features deserve a numbered contribution in NeurIPS given they are contribution #5 in the workshop.

### ISSUE 4 [FRAMING MISMATCH]: Workshop abstract mentions natural frequency paradox; NeurIPS abstract does not
- **Workshop** abstract (lines 90-93): "natural frequency framing---which reduces base rate neglect in humans---paradoxically increases it in both models"
- **NeurIPS** abstract: No mention of natural frequency paradox.
- **NeurIPS** discussion (lines 553-556): mentions it briefly ("Llama: 84% to 100%; R1-Distill: 4% to 40%")
- The workshop abstract gives this finding prominent billing; the NeurIPS paper treats it as a discussion aside. This is fine narratively -- NeurIPS has more space for the causal steering story -- but reviewers who read both may notice the different emphasis.
- **Action needed**: None required, but be aware of the asymmetry.

---

## 3. Superset Check: Is NeurIPS a strict superset of Workshop claims?

**Yes**, with one exception noted below.

NeurIPS-only content (not in workshop):
- Causal steering results section (full dose-response analysis)
- R1-Distill steering contrast ("readable but not writable")
- Within-CoT probing as a standalone section
- Scale analysis (32B OLMo-3.1)
- Multi-seed robustness analysis (Appendix)
- Generation strategy dependence analysis (Appendix)
- Full SAE falsification protocol details (Appendix)
- Full attention entropy head-level analysis (Appendix)
- Per-layer 64-layer probe table for 32B (Appendix)

Workshop-only content NOT in NeurIPS:
- **Natural frequency paradox in the abstract** (present but not in NeurIPS abstract -- see Issue 4)
- **Immune category early-layer AUC = 1.0 observation** (workshop Section 4.2, lines 418-425): The workshop explicitly states "Probes trained on the four immune categories achieve AUC=1.0 at layers 0-1 for both models, declining to AUC ~0.95 at deeper layers" and frames this as complicating the interpretation. The NeurIPS paper does not report this immune-category early-layer AUC=1.0 finding. This is the one claim in the workshop that is not present in the NeurIPS paper.
- **Action needed**: Add the immune-category AUC=1.0 observation to the NeurIPS paper (e.g., in the probing results or appendix). A reviewer who reads both papers should not find a result in the workshop that is absent from the main paper.

---

## 4. Contradiction Check

**No contradictions found.** All shared numerical claims match exactly (with the single rounding difference on SAE L0 noted above). The narrative framing is consistent: both papers tell the same "conflict detection without resolution / blurred not sharpened / training vs inference dissociation" story.

---

## 5. Action Items (ordered by priority)

1. **Fix workshop heuristic family labels** (Issue 2) -- real inconsistency in taxonomy names
2. **Add immune-category AUC=1.0 finding to NeurIPS** -- workshop has a result NeurIPS lacks
3. **Harmonize SAE L0 sparsity** (Issue 1) -- trivial rounding, fix in workshop ("48" -> "47.9")
4. **Optional**: Consider whether natural frequency paradox deserves mention in NeurIPS abstract (Issue 4)
5. **Optional**: Consider promoting SAE features to a numbered contribution in NeurIPS (Issue 3)
