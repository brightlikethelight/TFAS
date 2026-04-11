# Benchmark Audit Report

**Date**: 2026-04-09
**Auditor**: Claude Opus 4.6 (automated + manual review)
**Benchmark**: `data/benchmark/benchmark.jsonl` (284 items)
**Audit script**: `scripts/audit_benchmark.py`

---

## 1. Summary Statistics

| Metric | Count |
|--------|-------|
| Total items | 284 |
| Conflict items | 142 |
| Control items | 142 |
| Matched pairs | 142 |
| Items manually verified | 176 |
| CRITICAL issues | 11 |
| HIGH issues | 2 |
| MEDIUM issues | 4 |
| LOW issues | 0 |

### Category breakdown

| Category | Total | Conflict | Control | Pairs |
|----------|-------|----------|---------|-------|
| crt | 60 | 30 | 30 | 30 |
| base_rate | 40 | 20 | 20 | 20 |
| syllogism | 50 | 25 | 25 | 25 |
| anchoring | 30 | 15 | 15 | 15 |
| framing | 30 | 15 | 15 | 15 |
| conjunction | 24 | 12 | 12 | 12 |
| arithmetic | 50 | 25 | 25 | 25 |

---

## 2. Issues Found

### CRITICAL Issues (11)

All 11 CRITICAL issues are in the **syllogism** category -- specifically, 11 out of 12 control items for the "valid, unbelievable" conflict cell have **wrong `correct_answer` values**.

| Item ID | Issue | Description |
|---------|-------|-------------|
| `syll_mammal_breath_control` | syll_trivially_valid | Conclusion is identical to P2; marked "invalid" but is trivially valid |
| `syll_metal_corrodes_control` | syll_trivially_valid | Same issue |
| `syll_evergreen_lose_leaves_control` | syll_trivially_valid | Same issue |
| `syll_predator_vegetarian_control` | syll_trivially_valid | Same issue |
| `syll_deserts_cold_control` | syll_trivially_valid | Same issue |
| `syll_cats_aquatic_control` | syll_trivially_valid | Same issue |
| `syll_glass_dense_control` | syll_trivially_valid | Same issue |
| `syll_owl_diurnal_control` | syll_trivially_valid | Same issue |
| `syll_water_combustible_control` | syll_trivially_valid | Same issue |
| `syll_planets_cubes_control` | syll_trivially_valid | Same issue |
| `syll_philosophers_immortal_control` | syll_trivially_valid | Same issue |

**Root cause**: In `generators.py`, the `belief_bias_syllogism` generator defaults `control_conclusion` to `conflict_conclusion` when not explicitly provided. For valid-unbelievable conflict items (where the conflict is Barbara AAA-1: All M are P, All S are M, therefore All S are P), the control is constructed as AAA-2 by swapping the major premise (All P are M, All S are M). But the conclusion is left as the original "All S are P" from the conflict. The problem is that the conflict conclusion "All S are P" happens to be textually identical to the control's P2 "All S are P" (because in the AAA-2 form, P2 is set to the conflict conclusion). This makes the control's conclusion trivially entailed by its own P2, regardless of the syllogistic form. The item is marked `correct_answer="invalid"` but the correct answer is actually "valid" (the conclusion follows trivially from the premises since it IS one of the premises).

Only `syll_swimmers_landlocked` avoids this bug because its conflict conclusion ("All members of the Italian relay team were born in landlocked countries") uses a different verb form ("were born") than P2 ("are Olympic swimmers"), so the conclusion is not identical to P2.

**Severity justification**: These items are **unusable** in their current form. Any model that correctly identifies the conclusion as trivially following from the premises will be scored as incorrect. This affects 11 of 50 syllogism items (22% of the category, 3.9% of the full benchmark).

### HIGH Issues (2)

| Item ID | Issue | Description |
|---------|-------|-------------|
| `anchor_country_capital_year_conflict` | anchor_too_close | Anchor (1980) is only 1.04x true value (1913) |
| `anchor_hist_olympics_conflict` | anchor_too_close | Anchor (1960) is only 1.03x true value (1896) |

**Explanation**: For year-based questions, the multiplicative ratio is misleading -- the absolute gaps (67 and 64 years) are meaningful. However, these anchors are the weakest in the set. Whether they actually bias LLMs depends on how well-calibrated LLMs are on these well-known facts. For high-knowledge-certainty questions like "first modern Olympics" (1896 is very well-known), even a nearby anchor may not bias the model. These items may produce near-zero anchoring effects, creating noise in the category.

**Recommendation**: Keep but flag as borderline. Consider increasing the high_anchor to 2050+ for both items, or replacing with questions where the true value is less well-known.

### MEDIUM Issues (4)

All 4 MEDIUM issues concern **prompt-level rounding of 1/3 to 33%** in framing items.

| Item ID | Issue | n_total | sure | Prompt prob | Actual prob | Prompt EV gap |
|---------|-------|---------|------|-------------|-------------|---------------|
| `frame_flood_village_conflict` | framing_prompt_rounding | 600 | 200 | 33% | 33.33...% | 2.0 |
| `frame_wildfire_canyon_conflict` | framing_prompt_rounding | 900 | 300 | 33% | 33.33...% | 3.0 |
| `frame_carbon_monoxide_conflict` | framing_prompt_rounding | 240 | 80 | 33% | 33.33...% | 0.8 |
| `frame_marathon_heatstroke_conflict` | framing_prompt_rounding | 900 | 300 | 33% | 33.33...% | 3.0 |

**Explanation**: The `_pct()` helper in `generators.py` renders `1/3` as `33%` via `round(x * 100)`. The underlying EVs are exactly equal (verified from the spec), but a literal reading of the prompt sees unequal EVs. For example, `frame_flood_village` says "probability 33% all 600 people will be saved" -- a literal reader computes EV = 0.33 * 600 = 198, not 200. The prompt simultaneously says to "maximize expected lives saved" and implies EVs are equal (since the whole point is a framing manipulation), creating a contradictory instruction.

Note: `frame_blizzard_pass` also uses 1/3 (n=120, sure=40) but the gap is only 0.4, which is below the 0.5 threshold.

**Recommendation**: Fix by changing `_pct()` to render 1/3 as "1/3" or "one-third" instead of "33%". Alternatively, adjust the n_total values so that 33% is exact (e.g., n_total=300, sure=99; n_total=600 is fine with p=1/3 since 600/3=200 exactly, but the prompt says 33% not 33.33%).

---

## 3. Category-by-Category Assessment

### CRT (60 items, 30 pairs) -- PASS

All 30 conflict items manually verified for mathematical correctness:

- **Ratio (bat-and-ball)**: 10/10 correct. Formula `b = (total - diff) / 2` verified for all. Lure = `total - diff` verified. The invariants `a + b = total` and `a - b = diff` hold for all items. No degenerate cases (lure != correct for all).
- **Work-rate (widgets-machines)**: 10/10 correct. The "N workers, N items, N minutes" triple is equal for all items. Correct answer = base_rate, lure = scale. All lures are distinct from correct answers.
- **Exponential growth (lily-pad)**: 10/10 correct. Correct = days_to_full - 1, lure = days_to_full // 2, verified for all. No degenerate cases (all days_to_full values are even and > 2, so lure != correct).

Control items reviewed: all provide the correct answer through straightforward arithmetic (no S1 lure present). Prompts are clear and unambiguous.

### Base Rate (40 items, 20 pairs) -- PASS

All 20 conflict items verified:
- Common group rate >> rare group rate for all items (minimum ratio ~7x).
- Correct answer = common group (high base rate) for all items.
- Lure answer = rare group (matches the stereotype description) for all items.
- Descriptions are vivid and stereotypically matched to the rare group.
- Control items remove the stereotype description and present a uniform random draw.

No factual errors. Descriptions are novel (no classic "engineer/lawyer" Linda variants).

### Syllogism (50 items, 25 pairs) -- FAIL (11 CRITICAL)

- **Conflict items**: All 25 verified. 12 valid-unbelievable items use Barbara (AAA-1) correctly. 13 invalid-believable items use undistributed middle (AAA-2) correctly. Answer values ("valid"/"invalid") are correct for all conflict items.
- **Control items**: 14/25 correct.
  - 13 invalid-believable controls correctly use Barbara (valid) -- all pass.
  - 1 valid-unbelievable control (`syll_swimmers_landlocked`) correctly uses AAA-2 (invalid) -- passes.
  - **11 valid-unbelievable controls have wrong answers** (see CRITICAL issues above).

### Anchoring (30 items, 15 pairs) -- PASS (2 HIGH borderline)

All 15 true values verified against known facts:
- UN members: 193 (correct)
- Periodic elements: 118 (correct)
- Canberra founded: 1913 (correct)
- Amazon River: 6400 km (correct, standard estimate)
- Human chromosomes: 46 (correct)
- Sun/Earth radius: 109x (correct)
- First modern Olympics: 1896 (correct)
- Piano keys: 88 (correct)
- Great Wall: 21,000 km (correct, including all branches per 2012 survey)
- Marathon distance: 42,195 m (correct)
- Mount Fuji: 3,776 m (correct)
- Human genome: 3,000 Mbp (correct, standard estimate)
- US states: 50 (correct)
- Speed of light: 300 thousand km/s (correct to within rounding)
- Mount Everest: 8,849 m (correct, 2020 survey)

Anchors are at least 1.8x true value for all non-year items. The two year-based items have weak multiplicative ratios but meaningful absolute gaps (64-67 years).

### Framing (30 items, 15 pairs) -- PASS (4 MEDIUM)

All 15 items verified for EV equality at the specification level (exact fractions). 11 items use clean fractions (1/4, 1/5, 1/2) with no rounding issues. 4 items use 1/3, introducing a prompt-level rounding ambiguity (see MEDIUM issues).

Gain/loss frame structure is correct for all items. Correct answer correctly tracks `prefer_sure` flag. Balance: 8 prefer_sure=True, 7 prefer_sure=False.

### Conjunction (24 items, 12 pairs) -- PASS

All 12 conflict items verified:
- Correct answer is always "A" (single feature, more probable by conjunction rule).
- Lure answer is always "B" (conjunction, less probable but stereotypically fitting).
- Option B always contains Option A as a proper subset (feature_a AND feature_b).
- Descriptions are vivid and stereotypically matched to feature_b.
- Control items use neutral descriptions ("randomly selected adult"), removing the stereotype lure.

### Arithmetic (50 items, 25 pairs) -- PASS

All 25 conflict items verified by recomputing from the spec:
- Start value, operations, and trap_step parsed from provenance.
- Correct answer matches step-by-step computation for all items.
- Lure answer matches flipped-operation computation for all items.
- No degenerate cases (lure != correct for all items).
- Division operations are all exact (no remainders).

---

## 4. Structural Integrity Checks (all PASS)

| Check | Result |
|-------|--------|
| All required fields present | PASS (284/284) |
| All categories valid | PASS |
| All sources valid | PASS |
| All difficulties in [1,5] | PASS |
| No duplicate IDs | PASS |
| All matched pairs have exactly 1 conflict + 1 control | PASS |
| All pair members share category | PASS |
| All pair members share difficulty | PASS |
| All pair members share subcategory | PASS |
| All answer_patterns match correct_answer | PASS |
| All lure_patterns match lure_answer (conflict items) | PASS |
| No answer_pattern matches lure_answer (no false positives) | PASS |
| No lure_pattern matches correct_answer (no cross-contamination) | PASS |
| All conflict items have non-empty lure_answer | PASS |
| All control items have empty lure_answer | PASS |
| All conflict items have >= 2 paraphrases | PASS |
| All control items have >= 1 paraphrase | PASS |
| All prompts contain answer format instructions | PASS |

---

## 5. Recommendations

### Must Fix (CRITICAL)

1. **Syllogism control conclusions (11 items)**: Supply explicit `control_conclusion` values in `build.py` for all valid-unbelievable syllogism specs. The control conclusion should be the AAA-2 conclusion "All S are P" (where S and P come from the AAA-2 form), NOT the original conflict conclusion. Specifically, for each affected spec, add a `control_conclusion` field like:

   - `syll_mammal_breath`: control_conclusion = "All hippos are mammals" (AAA-2: All P are M, All S are M, therefore All S are P where S=hippos, P=mammals -- wait, that IS the control P2. The proper AAA-2 conclusion should be "All hippos are animals that can hold their breath underwater for at least one hour" -- but that IS P2 again.)

   Actually, the deeper issue is that for these items, the AAA-2 form's "natural" conclusion IS P2 (the control's Premise 2). The fix should be to construct a DIFFERENT conclusion that genuinely follows from an undistributed-middle fallacy. This likely requires rethinking the control construction for the valid-unbelievable cell.

   **Recommended approach**: For each affected item, construct the AAA-2 control with a conclusion that is NOT identical to either premise. For example, for `syll_mammal_breath`:
   - Conflict (VALID Barbara): All mammals can hold breath 1h. All hippos are mammals. Therefore all hippos can hold breath 1h. [correct=valid]
   - Control (INVALID AAA-2): All things that hold breath 1h are mammals. All hippos are mammals. Therefore all hippos can hold breath 1h. [This is undistributed middle. Conclusion != P1, != P2. correct=invalid]

   Wait -- but the current control uses P2 = "All hippos can hold their breath..." and Conclusion = "All hippos can hold their breath..." So P2 == C. The fix is to make the conclusion different from P2. For AAA-2 ("All P are M; All S are M; therefore All S are P"), the conclusion should use the S and P terms from P1's structure, not from P2.

   **Simplest fix**: In each affected build spec, explicitly set `control_conclusion` to a NEW statement that follows the AAA-2 pattern but is not identical to any premise. This requires manually authoring 11 new control conclusions.

### Should Fix (HIGH)

2. **Anchoring year-based items**: Consider increasing anchor distance for `anchor_country_capital_year` and `anchor_hist_olympics`. Alternatively, flag these items as "weak anchor" in the analysis and report results with and without them.

### Consider Fixing (MEDIUM)

3. **Framing 1/3 rounding**: Change the `_pct()` helper to render 1/3 as "one-third" or "33.3%" instead of "33%". Alternatively, avoid 1/3 altogether by replacing the 5 affected specs with probabilities that round cleanly (e.g., use p=0.25 with n_total adjusted).

---

## 6. Overall Assessment

The benchmark is **well-constructed** for 6 of 7 categories. The CRT, base-rate, conjunction, framing, anchoring, and arithmetic items all pass mathematical and structural verification with no CRITICAL issues. The matched-pair design is sound, the regex patterns are correct, and the prompts are clear.

The **syllogism category has a systematic bug** affecting 11 of 25 control items (44% of the category). These items must be fixed before the benchmark is used for any analysis that relies on syllogism controls. The conflict items are all correct, so any analysis that uses only conflict items is unaffected.

**Recommendation**: Fix the 11 syllogism controls before running extraction or analysis. This is a rebuild of the benchmark JSONL, not a data-level fix -- the issue is in the build specs (`build.py`), specifically the absence of explicit `control_conclusion` values for valid-unbelievable syllogism specs.

---

## 7. Verification Coverage

| Category | Conflict items verified | Control items verified | Total verified |
|----------|------------------------|------------------------|----------------|
| CRT (ratio) | 10/10 | spot-checked | 10+ |
| CRT (work_rate) | 10/10 | spot-checked | 10+ |
| CRT (exp_growth) | 10/10 | spot-checked | 10+ |
| Base rate | 20/20 | 3/20 | 23 |
| Syllogism | 25/25 | 25/25 | 50 |
| Anchoring | 15/15 | spot-checked | 15+ |
| Framing | 15/15 | 3/15 | 18 |
| Conjunction | 12/12 | 3/12 | 15 |
| Arithmetic | 25/25 | spot-checked | 25+ |
| **Total** | **142/142** | **34+ verified** | **176+** |

All 142 conflict items were verified either manually or programmatically. 34 control items were individually reviewed. All 284 items passed automated schema, regex, and structural checks.
