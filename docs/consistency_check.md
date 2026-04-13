# Consistency Check: Main Paper vs. Supplementary Materials

Date: 2026-04-12

## Summary

**14 inconsistencies found.** 5 are critical (wrong numbers), 4 are structural (missing supplementary content), and 5 are moderate (conflicting claims between documents).

---

## CRITICAL: Numerical Inconsistencies

### 1. Llama Peak Probe AUC and Layer (MAIN PAPER vs. SUPPLEMENTARY DATA)

**Main paper** (lines 67, 386, 434): Llama peaks at AUC = 0.974 [0.952, 0.992] at **L16**.

**Supplementary** Table C.1 (line 374): Layer 14 is **bolded as the peak** with AUC = **0.999**. L16 has AUC = 0.995.

The supplementary's own layer-by-layer data shows:
- L14: 0.999 (peak, bolded)
- L15: 0.996
- L16: 0.995
- L11: 0.979
- L12: 0.988
- L13: 0.993

**Verdict:** The main paper reports 0.974 at L16 while the supplementary data shows 0.999 at L14. These cannot both be correct. The main paper's CI [0.952, 0.992] does not even contain the supplementary's point estimate of 0.999. One of the two documents has stale numbers.

**Fix needed:** Reconcile the peak AUC value and layer. If the supplementary data is the ground truth, the main paper needs to be updated throughout: abstract (line 67), Figure 1 caption (line 377), Section 4.2 text (line 386), Table 2 (line 434), and all downstream references to "L16" and "0.974."

### 2. R1-Distill Peak Probe AUC and Layer

**Main paper** (lines 387, 435): R1-Distill peaks at AUC = 0.930 [0.894, 0.960] at **L31**.

**Supplementary** Table C.1 (line 374): L14 is bolded for R1-Distill with AUC = 0.929. But L31 shows AUC = **0.904**, not 0.930.

The highest R1-Distill AUC in the supplementary is L14 = 0.929, not 0.930 at L31 as claimed. The L31 value (0.904) is actually one of the lower values, not the peak.

**Verdict:** The main paper claims R1-Distill peaks at L31 with 0.930, but the supplementary shows L31 = 0.904 and L14 = 0.929 as the actual peak. The main paper's 0.930 appears nowhere in the supplementary layer data.

**Fix needed:** If the supplementary is the ground truth, update main paper to L14 peak with AUC = 0.929. This also changes the narrative about the "later peak layer" in the reasoning model (line 389: "The later peak layer suggests reasoning distillation shifts conflict-relevant information toward deeper layers") -- both models would peak at L14 in the supplementary data.

### 3. R1-Distill and Qwen Behavioral Lure Rates Disagree Between Tables

**Main paper** Table 1 (lines 322-326):
- R1-Distill CRT: **3%**
- R1-Distill Anchoring: **10%** (bolded)
- Qwen no-think Anchoring: **10%** (bolded)
- OLMo Instruct CRT: **3%**
- OLMo Think CRT: **3%**
- OLMo Think Syllogism: **4%**
- OLMo Instruct Base rate: **46%**
- OLMo Instruct Conjunction: **50%**

**Supplementary** Table B.1 (lines 234-256):
- R1-Distill CRT: **0%**
- R1-Distill Anchoring: **0%**
- Qwen no-think Anchoring: **0%**

**Supplementary** Table H.1 (lines 978-989):
- OLMo Instruct: **0% across ALL categories**
- OLMo Think: **0% across ALL categories**
- Overall: **0.0%** for both

**Verdict:** The supplementary shows 0% for several categories where the main paper reports non-zero values. This is the most severe inconsistency. The main paper shows OLMo overall lure rates of 14.9% and 0.9%, which the supplementary flatly contradicts with 0.0% for both. The supplementary even states "Both OLMo-3 variants show 0% S1-lure rates across all categories" (line 962-963), which is impossible to reconcile with the main paper's 14.9%/0.9%.

**Fix needed:** Determine which data is correct. If the main paper's OLMo numbers (14.9%, 0.9%) are the latest, the supplementary's Section H needs a full rewrite. If the supplementary is correct, the main paper's OLMo row and all OLMo behavioral claims need to be rewritten -- which would eliminate one of the paper's key replications.

### 4. OLMo Probe Peak AUC vs. Layer-Wise Data

**Main paper** Table 2 (line 439): OLMo Instruct peak AUC = **0.996** at L24.
**Supplementary** Table H.2 (line 1011): Peak AUC = **0.996 [0.988, 1.000]** at L24.

But the supplementary's own layer-by-layer data (lines 1066-1074) shows:
- L15: 0.997
- L16: 0.998
- L17: 0.997
- L18: 0.997
- L19: 0.998
- L20: 0.998
- L21: 0.998
- L22: 0.998
- **L24: 0.998** (not 0.996)

The peak in the layer-by-layer data is 0.998 at multiple layers (L16, L19-L24), not 0.996 at L24.

Similarly, OLMo Think:
- **Supplementary summary** (line 1012): Peak = 0.962 at L22
- **Layer-by-layer** (line 1072): L22 = 0.991, and L28 = 0.993 is actually higher

**Verdict:** The OLMo probe summary table in the supplementary is internally inconsistent with its own layer-by-layer table. The summary says 0.996 at L24 but the per-layer data shows 0.998 at L24. The Think summary says 0.962 at L22 but the layer data shows 0.991 at L22.

**Fix needed:** Reconcile the summary table with the layer-by-layer table. The bootstrap CIs in the summary may reflect fold-averaged values while the layer table shows mean AUC, but this discrepancy (0.996 vs 0.998; 0.962 vs 0.991) is too large for that explanation. If the summary CIs [0.988, 1.000] and [0.934, 0.982] are the canonical values (from bootstrap), then the layer-by-layer table needs rechecking.

### 5. Control AUC Ceiling Mismatch

**Main paper** (line 400): "Hewitt-Liang control probes ... achieve AUC <= **0.55** at all layers for both models."

**Supplementary** (lines 399-403): "aggregate control AUC <= **0.64** at all layers" and per-layer data shows control AUCs up to **0.635** (Llama L15) and **0.605** (R1-Distill L22).

**Verdict:** Main paper says <= 0.55, supplementary data shows values up to 0.635. The main paper understates the control ceiling.

**Fix needed:** Update main paper line 400 to say "<= 0.64" to match the supplementary data.

---

## CRITICAL: Selectivity Mismatch

### 6. Selectivity Claim

**Main paper** (lines 401-402): "Selectivity exceeds **40 percentage points** at peak layers for both models."

**Supplementary** (lines 402-403): "selectivity >= **29 percentage points** at every layer and >= **34 pp** at peak layers."

If peak Llama AUC is 0.999 and control is 0.612 (L14), selectivity = 38.7 pp.
If peak R1 AUC is 0.929 and control is 0.565 (L14), selectivity = 36.4 pp.

Neither exceeds 40 pp based on the supplementary's own peak-layer data. The main paper Table 2 (line 434) reports selectivity = 0.434 for Llama (43.4 pp), which uses the main paper's 0.974 and 0.54 values that already conflict with the supplementary.

**Verdict:** The "> 40pp" claim in the main paper depends on the main paper's own probe numbers (0.974 - 0.54 = 0.434). Using the supplementary's data, selectivity is 34-39 pp, not > 40.

**Fix needed:** Depends on which probe numbers are authoritative.

---

## STRUCTURAL: Missing Supplementary Sections

### 7. Ecological Rationality / Natural Frequency Analysis Missing

**Main paper** (line 542): "Full analysis is in the supplementary material" (regarding natural frequency framing increasing lure rates: Llama 84% -> 100%, R1-Distill 4% -> 40%).

**Supplementary:** No section analyzes natural frequency results. Section A mentions the 10-pair subset exists (line 100-104) but provides no results. There is no table, no analysis, and no section header for this.

**Fix needed:** Add a dedicated supplementary section (e.g., Section K) with the ecological rationality analysis, or remove the "see supplementary" claim from the main paper.

### 8. Loss Aversion / Sunk Cost / Certainty Effect / Availability Results Missing

**Main paper** Table 1 caption (line 309): "Four additional categories (loss aversion, sunk cost, certainty effect, availability) are reported in the supplementary material."

**Main paper** (lines 219-222): "Sunk cost, loss aversion, certainty effect, availability (60 pairs total): four additional categories ... Full descriptions are in the supplementary material."

**Supplementary:** No section reports results for loss aversion, certainty effect, or availability. Sunk cost appears only in Table A.3 (benchmark description, 15 pairs), Table H.1 (OLMo, all 0%), and nowhere else. The supplementary never reports Llama/R1/Qwen results on these four categories.

**Fix needed:** Add supplementary section with results for these four categories, or remove the claim from the main paper that they are reported there.

### 9. Section A.1 Category Count: "8 categories" vs. Main Paper's "7 categories"

**Supplementary** (line 67-68): "380 items (190 matched conflict/control pairs) across **8 categories** form the primary evaluation set reported in the main text."

**Main paper** (line 225-226): "All primary analyses use the original **7-category**, 165-pair subset (330 items)."

These disagree on both the number of categories (7 vs. 8) and the item count (330 vs. 380).

**Fix needed:** Align on whether the primary set is 7 or 8 categories and the corresponding pair count.

### 10. Supplementary Section A.1 "3 additional categories" vs. Main Paper's "4 additional"

**Supplementary** (lines 69-71): "The remaining 90 items (45 pairs) belong to **3 additional categories**: a natural frequency framing subset, sunk cost variants, and contamination baselines."

**Main paper** (lines 218-222): "**Sunk cost, loss aversion, certainty effect, availability** (60 pairs total): **four** additional categories."

The supplementary says 3 additional, the main paper says 4. The supplementary does not list loss aversion, certainty effect, or availability as additional categories at all.

**Fix needed:** Align the two documents on how many extended categories exist and which they are.

---

## MODERATE: Narrative Inconsistencies

### 11. Transfer Matrix Layer: L14 vs. L16

**Main paper** (line 460): Cross-prediction "At the peak layer **(L14)**" -- but earlier (line 386) says peak is at **L16**.
**Main paper** (line 468): Transfer matrix reported at "Llama, **L14**."

The main paper uses L14 for cross-prediction and transfer matrix analysis but claims the peak layer is L16 in the probing section. This is internally inconsistent within the main paper itself. The supplementary supports L14 as the peak.

**Fix needed:** Decide whether peak is L14 or L16 and use consistently throughout.

### 12. Cross-prediction Transfer AUC: "peak layer (L16)" vs. Actual Data

**Main paper** (line 460): "At the peak layer (L14), the Llama probe achieves transfer AUC = 0.378."
**Supplementary** Table D.1 (line 544): L14 transfer = 0.378. Confirmed.

But the main paper line 561 in the Discussion says "0.378 on immune categories" -- this is at L14 (confirmed by supplementary D.1), not L16. The reference to "peak layer" is correct only if the peak is L14, which contradicts the probing section's claim of L16.

### 13. Cross-prediction AUC for R1-Distill at L4

**Main paper** (line 463): "R1-Distill, results are mixed: transfer AUC = 0.878 at L4."
**Supplementary** Table D.1 (line 541): R1-Distill L4 transfer = 0.878. Confirmed -- consistent.

### 14. Attention Head Counts: Consistent

**Main paper** (line 507): R1 has 57 vs. Llama 30 S2-specialized heads (5.6% vs 2.9%).
**Supplementary** Table J.1 (line 1195): R1 = 57 (5.6%), Llama = 30 (2.9%). Confirmed -- consistent.

---

## Verified Consistent Numbers

The following numbers match between the two documents:

| Claim | Main Paper | Supplementary | Status |
|-------|-----------|---------------|--------|
| Benchmark size | 470 items, 235 pairs, 11 categories | 470 items, 11 categories | OK |
| Llama overall lure rate | 27.3% | 27.3% (Table B.1) | OK |
| R1-Distill overall lure rate | 2.4% | 2.4% (Table B.1) | OK |
| Qwen no-think overall | 21.0% | 21.0% (Table B.1) | OK |
| Qwen think overall | 7.0% | 7.0% (Table B.1) | OK |
| Qwen probe AUC both modes | 0.971 at L34 | 0.971 at L34 (Table C.3) | OK |
| Lure susceptibility Llama mean | +0.422 | +0.422 (Table G.1) | OK |
| Lure susceptibility R1 mean | -0.326 | -0.326 (Table G.1) | OK |
| SAE features: 41 significant, 0 spurious | Matches | Matches (Table I.3) | OK |
| SAE: 74% explained variance, L0=48 | 74%, 47.9 | 74.0%, 47.9 (Table I.1) | OK |
| Attention heads: 57 vs 30 | Matches | Matches (Table J.1) | OK |
| Base/conjunction transfer AUC > 0.99 | OK | 0.993/0.998 (Table E.1) | OK |
| CRT->vulnerable transfer range 0.06-0.45 | OK | 0.062-0.451 (Table E.1) | OK |
| Cross-model Llama->R1: 0.920, R1->Llama: 0.954 | Main only | No supp. section | N/A |

---

## Priority Fixes (Ordered by Severity)

1. **OLMo behavioral data** (#3): The supplementary says 0% everywhere; the main paper says 14.9%/0.9%. This is the paper's cross-architecture replication -- getting it wrong undermines the core claim. Determine which is correct and update the stale document.

2. **Llama/R1 peak probe AUC and layer** (#1, #2): 0.974@L16 vs 0.999@L14 and 0.930@L31 vs 0.929@L14. These numbers appear in the abstract, multiple figure captions, tables, and the discussion. Fix the source of truth, then propagate everywhere.

3. **OLMo probe summary vs layer-by-layer** (#4): Internal inconsistency within the supplementary itself. The summary table and layer table disagree.

4. **Missing supplementary sections** (#7, #8): Two explicit "see supplementary" claims in the main paper point to content that does not exist. Either add the content or remove the claims.

5. **Control AUC and selectivity** (#5, #6): The main paper's "<=0.55" and ">40pp" claims do not match the supplementary's data. Update the main paper.

6. **Category count** (#9, #10): Align on 7 vs 8 primary categories and 3 vs 4 additional categories.
