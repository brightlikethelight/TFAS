# Final Proofread: workshop_paper.tex

Generated: 2026-04-12
Reviewer: Claude Opus 4.6

---

## CRITICAL Issues (must fix before submission)

### C1. Table 1: R1-Distill CRT and Anchoring shown as 0% — data says otherwise
- **Line**: ~343-346 (Table 1)
- **Text**: R1-Distill row shows CRT=0 and Anchoring=0
- **Data**: `results/behavioral/r1_distill_llama_ALL.json` shows CRT=3.3% (1/30) and Anchoring=10.0% (2/20)
- **Fix**: Change R1-Distill CRT to 3 and Anchoring to 10 in Table 1. This also invalidates the claim that CRT and anchoring are "immune across all models" (line 358).
- **Severity**: CRITICAL — These are factual errors in the primary results table. Reviewers checking the data will catch this immediately.

### C2. Table 1: OLMo Think shown as 0% for Syllogism and CRT — data says otherwise
- **Line**: ~343-346 (Table 1)
- **Text**: OLMo Think row shows Syllogism=0 and CRT=0
- **Data**: `results/summary/behavioral_complete.json` shows OLMo Think Syllogism=4.0% (1/25) and CRT=3.3% (1/30)
- **Fix**: Change OLMo Think Syllogism to 4 and CRT to 3 in Table 1.
- **Severity**: CRITICAL — Same issue as C1.

### C3. Table 1: Qwen no-think Anchoring shown as 0% — data says 10%
- **Line**: ~346 (Table 1)
- **Text**: Qwen no-think Anchoring=0
- **Data**: `results/summary/behavioral_complete.json` shows Qwen no-think Anchoring=10.0% (2/20)
- **Fix**: Change Qwen no-think Anchoring to 10.
- **Severity**: CRITICAL

### C4. Transfer matrix claim "AUC >= 0.950" is false
- **Line**: ~535-536
- **Text**: "base rate neglect, conjunction, and syllogism probes transfer well to each other (AUC >= 0.950)"
- **Data**: `results/probes/transfer_matrix_l14_llama.json` shows:
  - base_rate -> syllogism: 0.594
  - conjunction -> syllogism: 0.627
  - syllogism -> syllogism: 0.936 (self, less than 0.950)
- **Fix**: The claim only holds for the base_rate<->conjunction pair (0.993 and 0.998) and syllogism->base_rate (0.950). Syllogism is the outlier — probes transfer from syllogism TO the other two, but poorly FROM the other two TO syllogism. Rewrite to: "base rate and conjunction probes transfer well to each other (AUC > 0.99) and syllogism probes transfer to base rate (0.950), but transfer to syllogism from other categories is weaker (0.594--0.627)."
- **Severity**: CRITICAL — This is a factual overclaim that a reviewer with access to the data will reject.

### C5. Cross-prediction says "peak layer (L16)" but the 0.378 value is at L14
- **Line**: ~524
- **Text**: "At the peak layer (L16), the Llama probe achieves transfer AUC = 0.378"
- **Data**: `results/probes/llama_cross_prediction.json` shows L14 transfer=0.378 but L16 transfer=0.569
- **Fix**: Either say "L14" (where the cross-prediction AUC is 0.378) or report the L16 value (0.569). Note that L16 is the bootstrap-CI peak for vulnerable categories, but L14 is the raw-AUC peak across all categories. The discrepancy exists because the paper uses two different "peak" definitions.
- **Severity**: CRITICAL — Misattributing a number to the wrong layer.

### C6. OLMo probe numbers in Table 2 and text don't match data
- **Line**: ~384, 503-504 (Table 2)
- **Text**: OLMo Instruct AUC=0.996 [0.988, 1.000] at L24; OLMo Think AUC=0.962 [0.934, 0.982] at L22
- **Data**: `results/probes/olmo3_instruct_vulnerable.json` shows peak at L21 (AUC=0.998); `results_downloaded/results/probes/olmo3_think_vulnerable.json` shows peak at L28 (AUC=0.993). L24 instruct=0.998, L22 think=0.991.
- **Fix**: The peak layers and AUC values are both wrong. At the layers the paper cites (L24 and L22), the AUCs are 0.998 and 0.991 respectively, not 0.996 and 0.962. The actual peak layers are L21 and L28. The 95% CIs reported do not appear in the probe JSON files (which only have mean and std from CV folds, not bootstrap CIs). Either re-run bootstrap CIs on the OLMo data, or report the CV mean +/- std and state it is from cross-validation rather than bootstrap.
- **Severity**: CRITICAL — Wrong numbers in a results table.

### C7. Within-model AUC values in cross-model transfer paragraph are wrong
- **Line**: ~554-556
- **Text**: "AUC=0.920 when tested on R1-Distill's activations at layer 23 (vs. within-model AUC of 0.987)" and "R1-Distill to Llama achieves AUC=0.954 at layer 15 (vs. self AUC of 0.943)"
- **Data**: At L23, Llama within-model AUC=0.963 (not 0.987). At L15, R1 within-model AUC=0.919 (not 0.943). Source: `results/probes/llama_vs_r1_layer_aucs.json`.
- **Fix**: Replace 0.987 with 0.963 and 0.943 with 0.919.
- **Severity**: CRITICAL — Incorrect comparison baselines inflate the apparent transfer success.

---

## IMPORTANT Issues (should fix)

### I1. "Immune across all models" claim is not supported
- **Line**: ~357-358
- **Text**: "vulnerability is category-specific: base rate neglect, conjunction fallacy, and syllogisms are susceptible, while CRT, arithmetic, framing, and anchoring are immune across all models"
- **Data**: R1-Distill shows CRT=3.3% and Anchoring=10%; OLMo Instruct shows CRT=3.3% and Anchoring=5%; OLMo Think shows CRT=3.3%; Qwen no-think shows Anchoring=10%.
- **Fix**: Soften to "largely immune" or "immune in the standard model" and note the low-level leakage in other models.
- **Severity**: IMPORTANT — Overclaim that contradicts the data in the same table.

### I2. CRT transfer AUC=0.062 is cherry-picked
- **Line**: ~536
- **Text**: "CRT probes do not transfer to any vulnerable category (AUC=0.062)"
- **Data**: CRT -> base_rate=0.062, CRT -> conjunction=0.223, CRT -> syllogism=0.451. The 0.451 transfer to syllogism is not trivial.
- **Fix**: Report the range: "CRT probes transfer poorly to vulnerable categories (AUC = 0.062--0.451)" or report the mean.
- **Severity**: IMPORTANT — Cherry-picking the lowest of three values to support a claim.

### I3. Llama L0 AUC stated as 0.85 but data shows 0.90
- **Line**: ~432
- **Text**: "The conflict/control signal builds rapidly from AUC=0.85 at layer 0"
- **Data**: Bootstrap CIs vulnerable L0: AUC=0.9033
- **Fix**: Change to "AUC=0.90 at layer 0"
- **Severity**: IMPORTANT — Incorrect starting value.

### I4. Missing effect sizes for confidence analysis
- **Line**: ~388-396
- **Text**: Reports first-token probability (0.751 vs 0.793) and entropy (0.948 vs 0.817) but no effect sizes or CIs.
- **Data**: `results/confidence/confidence_analysis.json` has Mann-Whitney U test results with r_rb effect sizes: first-token prob r_rb=0.135 (p=0.011), entropy r_rb=-0.121 (p=0.023).
- **Fix**: Report: "Mann-Whitney U: first-token probability significantly lower on conflict items (U=23885, p=0.011, r_rb=0.135); top-10 entropy significantly higher (U=30963, p=0.023, r_rb=0.121)." The effect sizes are small.
- **Severity**: IMPORTANT — Paper's own statistical standards require effect sizes alongside every p-value.

### I5. Qwen overall lure rates rounded inconsistently
- **Line**: ~348, and throughout
- **Text**: "21%" and "7%" for Qwen no-think/think
- **Data**: 35/165=21.2% and 12/165=7.3%
- **Fix**: Either use 21.2% and 7.3% or consistently say "~21%" and "~7%". The "3x behavioral improvement" claim (lines 455, 489) becomes 2.9x with exact numbers.
- **Severity**: IMPORTANT — Rounding makes the 3x claim slightly inflated.

### I6. "S2-specialized heads" phrasing risks anthropomorphism
- **Line**: ~505, 79
- **Text**: "S2-specialized heads" — "heads with significantly higher entropy on conflict items"
- **Fix**: Higher entropy on conflict items means the head distributes attention more broadly on conflict items. This is not necessarily "S2 specialization" — it could be confusion or difficulty rather than deliberation. The paper should note this alternative interpretation. Currently the framing implies these heads are performing deliberative reasoning, which is a strong claim.
- **Severity**: IMPORTANT — Interpretation overclaim.

### I7. N=235 matched pairs but benchmark section says different category counts
- **Line**: ~65, 198-199
- **Text**: Abstract says "235 matched pairs spanning 11 cognitive bias categories." Benchmark section lists category sizes summing to 235 pairs.
- **Issue**: The primary analyses use only 165 pairs (7 categories). The abstract should clarify this.
- **Fix**: Abstract should say "235 matched pairs (470 items) across 11 categories; primary analyses on 165 pairs across the 7 categories where both conflict and no-conflict items elicited varied responses."
- **Severity**: IMPORTANT — Reader confusion about sample size.

### I8. Table 1 mixes data from different evaluation runs
- **Line**: ~315-351
- **Issue**: The original 7 categories (base_rate through anchoring) use data from the 165-pair run (`llama31_8b_ALL.json`), while the new 4 categories (sunk_cost, availability, certainty_effect, loss_aversion) come from separate evaluation runs. The 470-item full run (`llama31_8b_470.json`) shows substantially different numbers for some categories (conjunction: 45% vs 55%; syllogism: 4% vs 52%). This suggests run-to-run variance is high, or prompt formatting differed between runs.
- **Fix**: Either use a single consistent run for all numbers, or explicitly note in the table caption that different categories were evaluated in separate runs and report the run identifiers. The discrepancy between the 470-run and 165-run for Llama syllogism (4% vs 52%) is alarming and warrants investigation.
- **Severity**: IMPORTANT — Methodological transparency issue. A reviewer noting the 470-item file would flag this.

### I9. Immune categories have AUC=1.0 at layers 0-1 claim
- **Line**: ~471
- **Text**: "Probes trained on the four immune categories achieve AUC=1.0 at layers 0-1"
- **Data**: Bootstrap CIs show immune L0 AUC=0.9994 for Llama, L1 is also <1.0. The claim of 1.0 at "layers 0-1" is slightly overstated.
- **Fix**: "AUC~1.0 at layers 0-1" or "AUC>0.999 at layers 0-1"
- **Severity**: MINOR, but reporting 1.0 for a non-exact value is imprecise.

---

## MINOR Issues (nice to fix)

### M1. Missing co-author names and advisor name
- **Line**: ~39-42
- **Text**: "[Co-authors]", "[Advisor]", "[Department]", "[advisor name]" (line 737)
- **Fix**: Fill in before submission.
- **Severity**: MINOR (obvious placeholder, but don't forget)

### M2. Incomplete figure — placeholder box
- **Line**: ~407-417
- **Text**: Figure 1 is a `\fbox` placeholder, not an actual figure
- **Fix**: Replace with `\includegraphics` of the actual probe curves PDF.
- **Severity**: MINOR (assuming you plan to generate the figure)

### M3. Table caption says "natural frequency framing" with specific numbers but no N
- **Line**: ~324
- **Text**: "Natural frequency base rate framing increases lure rates (Llama: 84% -> 100%; R1-Distill: 4% -> 40%)"
- **Fix**: Add N: "on 10 natural frequency pairs" or "(N=10 pairs)"
- **Severity**: MINOR

### M4. "System 2 module" in quotes at line 604
- **Line**: ~604
- **Text**: 'If reasoning training simply added a "System 2 module"'
- **Fix**: This is in quotes as a hypothetical, which is fine per the framing rules (it's describing a rejected hypothesis). No change needed, but flagging for awareness.
- **Severity**: MINOR — no fix needed

### M5. Table caption says "--- = not yet evaluated" but this should be clarified
- **Line**: ~325
- **Text**: "--- = not yet evaluated on the expanded set"
- **Fix**: The expanded categories (sunk_cost, certainty_effect, availability, loss_aversion) were not evaluated for Qwen and partially for OLMo. The caption is fine, but the table also uses --- for OLMo sunk_cost and certainty_effect while showing data for loss_aversion and availability. This is inconsistent — OLMo was evaluated on some expanded categories but not others. Clarify: "--- = category not evaluated for this model."
- **Severity**: MINOR

### M6. "Dashed gray line" in Figure 1 description but no actual figure
- **Line**: ~414, 427
- **Text**: References to "Gray dashed: Hewitt-Liang control task ceiling" and "Dashed gray line"
- **Fix**: Ensure the actual figure includes this when generated.
- **Severity**: MINOR

### M7. Selectivity stated as ">40 percentage points" without exact number
- **Line**: ~467
- **Text**: "Selectivity exceeds 40 percentage points at peak layers for both models"
- **Data**: Table 2 shows Llama selectivity=0.434 and R1=0.410
- **Fix**: "Selectivity is 43.4pp (Llama) and 41.0pp (R1-Distill)"
- **Severity**: MINOR

### M8. Reference to "the submission version" or supplementary not present
- **Line**: ~539 (Discussion, ecological rationality)
- **Text**: "Full analysis is in the..."
- **Issue**: This sentence appears to be cut off or references a supplement that doesn't exist yet.
- **Fix**: Check if this sentence is complete. In the text I read it appears as "This inversion warrants systematic investigation across all bias categories and models." which is fine. But the Discussion version (line ~539) has "\citep{gigerenzer1995improve}. Full analysis is in the..." — verify this is complete.
- **Severity**: MINOR

### M9. Benchmark description says certainty effect "(Kahneman & Tversky, 1979)" without a proper cite
- **Line**: ~226
- **Text**: "(Kahneman & Tversky, 1979)" in parenthetical
- **Fix**: Should be `\citep{kahneman1979prospect}` or similar bibtex key. Currently it's raw text, not a citation command.
- **Severity**: MINOR — Will show as literal text, not a formatted reference.

### M10. Line 604: "System~2 module" should be "System 2 module" (no tilde needed outside math)
- **Line**: ~604
- **Text**: Actually looking at the source: the text at this line was rewritten in the discussion. Let me flag instead: in the discussion (around line 604), the text uses "Type~2 processing" which is good.
- **Severity**: MINOR — no issue

### M11. Abstract length
- **Line**: 55-88
- **Issue**: The abstract is quite long (~300 words). Most workshop papers target 150-200 words.
- **Fix**: Consider trimming. The Qwen dissociation and natural frequency findings could be shortened.
- **Severity**: MINOR

---

## Logical Flow Issues

### L1. Contribution 5 (SAE features) gets minimal results coverage
- **Line**: ~165-169 (contribution), ~562-566 (results)
- **Issue**: The SAE analysis is listed as contribution 5 in the introduction, gets a single paragraph in results (5 sentences), and is not discussed substantively in the Discussion. It feels bolted on.
- **Fix**: Either expand the SAE section to justify its billing as a contribution, or demote it to a supporting analysis mentioned briefly.
- **Severity**: MINOR

### L2. Discussion has two separate paragraphs about the same training vs inference point
- **Line**: ~597-618 (first discussion of blurring) and ~645-663 (training vs inference paragraph)
- **Issue**: The training-changes-encoding/inference-changes-behavior point is made in detail twice: once under "conflict detection without resolution" and once under "training vs inference." The second is cleaner and more focused.
- **Fix**: Consolidate. Keep the "training vs inference" paragraph and briefly reference it from the first section.
- **Severity**: MINOR — Redundancy but each adds some unique content.

---

## Overclaiming Flags

### O1. "near-perfect conflict/control separability" for AUC=0.974
- **Line**: ~67
- **Text**: "near-perfect conflict/control separability (peak AUC=0.974)"
- **Issue**: 0.974 is high but "near-perfect" typically implies >0.99. This is a judgment call.
- **Fix**: "high conflict/control separability" would be more conservative.
- **Severity**: MINOR

### O2. Lure susceptibility sign flip framed as "flips the model's default commitment"
- **Line**: ~617-618
- **Text**: "Reasoning training doesn't just blur a boundary — it flips the model's default commitment."
- **Data**: Llama mean lure susceptibility = +0.422 (std=2.986), R1 = -0.326 (std=2.056). Cohen's d = 0.29 (small effect).
- **Issue**: The means do flip sign, but the standard deviations are huge relative to the means. The distributions heavily overlap. "Flips the model's default commitment" implies a clean reversal, but this is a small-effect-size shift in a noisy distribution.
- **Fix**: Add the effect size: "flips the mean lure susceptibility from +0.42 to -0.33 (Cohen's d=0.29, small effect), though with substantial within-model variance."
- **Severity**: IMPORTANT — Overclaiming given the small effect size.

---

## Missing Data Checks

### D1. Qwen and OLMo probe CIs not reported
- **Line**: Table 2, lines 500-504
- **Text**: Qwen rows show "---" for CI and control AUC. OLMo rows show CIs but these don't match the data files.
- **Issue**: Bootstrap CIs were computed for Llama and R1 but apparently not for Qwen or OLMo (OLMo CIs in the table don't match probe files). For a claim of "non-overlapping CIs" on OLMo (line 384), the CIs need to be real.
- **Fix**: Run bootstrap CIs for OLMo if not done. Verify the OLMo CIs.
- **Severity**: IMPORTANT

### D2. No Hewitt-Liang controls reported for Qwen or OLMo
- **Line**: Table 2
- **Issue**: Control AUC and selectivity are reported only for Llama and R1. For Qwen and OLMo, they are "---". If probing results are claimed for these models, control tasks should be run.
- **Fix**: Either run Hewitt-Liang controls on Qwen and OLMo, or note this limitation.
- **Severity**: IMPORTANT

### D3. Causal interventions mentioned in future work but not attempted
- **Line**: ~725-729
- **Issue**: The paper identifies causal interventions as "the natural next step." The infrastructure for this exists in the codebase (`src/s1s2/causal/`). If there are preliminary results, they should be mentioned.
- **Fix**: No fix needed if no results exist — the future work framing is appropriate.
- **Severity**: MINOR

---

## Summary of Changes Required

**Must fix (CRITICAL)**: 7 issues — C1-C7 involve incorrect numbers in tables or text that contradict the actual data files. These must be corrected.

**Should fix (IMPORTANT)**: 9 issues — Overclaims (I1, I2, I6, O2), missing statistical reporting (I4, D1, D2), methodological transparency (I8), and precision (I3, I5, I7).

**Nice to fix (MINOR)**: 11 issues — Placeholders (M1, M2), minor precision improvements, and structural refinements.
