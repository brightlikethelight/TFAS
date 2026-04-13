# Natural Frequency "Failure Mode" Analysis: Scoring Bug, Not Model Failure

**Date**: 2026-04-12
**Analyst**: Claude (automated analysis)
**Status**: CRITICAL BUG FOUND -- previous results partially invalid

---

## Executive Summary

The reported R1-Distill natural frequency results (40% lure, 40% other, 20% correct) are **artifacts of two compounding bugs in the scoring pipeline**, not genuine model behavior. Every single verdict for the 20 natural frequency items can be predicted with 100% accuracy from the bug mechanism alone, without reference to what the model actually answered.

**The R1-Distill NF data must be re-collected before any conclusions are drawn.**

The Llama NF data (100% lure on conflict, 100% correct on control) is unaffected and valid.

---

## Bug 1: Unicode Byte-Level BPE Artifacts (Ġ characters)

### What happened

The `run_new_items.py` script decodes R1-Distill responses using:
```python
response = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
```

For R1-Distill (DeepSeek-R1-Distill-Llama-8B), the tokenizer's `decode()` output retains byte-level BPE markers:
- `Ġ` (U+0120) in place of spaces
- `Ċ` (U+010A) in place of newlines

This means `"school teachers"` in the model output becomes `"schoolĠteachers"` in the stored response.

### Why Llama is unaffected

Llama-3.1-8B-Instruct's tokenizer produces clean ASCII output. The same `decode(skip_special_tokens=True)` call works correctly for Llama but not for R1-Distill.

### Impact on scoring

The `parse_answer()` function does a case-insensitive substring search:
```python
if correct_lower in text_lower:
    return "correct"
```

Searching for `"school teacher"` (with ASCII space) inside `"schoolĠteachers"` (with Ġ) always fails. **Any multi-word answer string is guaranteed to fail matching.**

### Proof

All 50 R1-Distill responses in the new_items file contain Ġ characters. Zero of the 50 Llama responses do.

---

## Bug 2: Response Truncation to 500 Characters

### What happened

Line 108 of `run_new_items.py`:
```python
"response": response[:500],
```

R1-Distill produces chain-of-thought reasoning that typically runs 1000-4000+ tokens. The first 500 characters contain only the start of the reasoning chain, where the model **restates the problem** (including both the correct and lure occupation names from the prompt).

The model's actual answer appears at the END of its reasoning chain, which is entirely lost.

### Why this interacts with Bug 1

Even if Bug 1 were fixed, the scoring would still be wrong: `parse_answer()` would find both the correct AND lure answer in the first 500 chars (from the problem restatement), and would default to "correct" because it checks `correct_lower in text_lower` first. This would give 100% "correct" -- equally wrong, just in the other direction.

Proper scoring requires either:
1. The full response text, scored against the final answer after `</think>`
2. Or, use the regex-based `score_response()` from `src/s1s2/extract/scoring.py` which handles last-match-wins for reasoning models

### Why Llama is unaffected

Llama gives short, single-word/phrase answers (5-30 chars) with no reasoning chain. Truncation to 500 chars preserves the complete response.

---

## Complete Verdict Prediction from Bug Mechanism

The stored verdict for every NF item is **fully determined** by whether the answer strings contain spaces:

| Condition | Verdict | Mechanism |
|-----------|---------|-----------|
| `correct_answer` has no space | `correct` | Single-word match succeeds despite Ġ |
| `correct_answer` has space, `lure_answer` has no space, conflict=True | `lure` | Correct fails (Ġ), single-word lure succeeds |
| `correct_answer` has space, `lure_answer` has space or empty | `other` | Both fail (Ġ) |

### Verification: 20/20 items predicted correctly

| Item | Correct Answer | Lure Answer | Predicted | Actual |
|------|---------------|-------------|-----------|--------|
| carpenter_teacher (C) | school teacher (sp) | professional carpenter (sp) | other | other |
| carpenter_teacher (X) | school teacher (sp) | -- | other | other |
| brewer_accountant (C) | accountant | craft brewer (sp) | correct | correct |
| brewer_accountant (X) | accountant | -- | correct | correct |
| beekeeper_programmer (C) | software programmer (sp) | beekeeper | lure | lure |
| beekeeper_programmer (X) | software programmer (sp) | -- | other | other |
| tattooist_nurse (C) | nurse | tattoo artist (sp) | correct | correct |
| tattooist_nurse (X) | nurse | -- | correct | correct |
| pilot_salesperson (C) | retail salesperson (sp) | commercial pilot (sp) | other | other |
| pilot_salesperson (X) | retail salesperson (sp) | -- | other | other |
| glassblower_clerk (C) | office clerk (sp) | glassblower | lure | lure |
| glassblower_clerk (X) | office clerk (sp) | -- | other | other |
| florist_engineer (C) | mechanical engineer (sp) | professional florist (sp) | other | other |
| florist_engineer (X) | mechanical engineer (sp) | -- | other | other |
| farrier_manager (C) | office manager (sp) | farrier | lure | lure |
| farrier_manager (X) | office manager (sp) | -- | other | other |
| locksmith_analyst (C) | data analyst (sp) | locksmith | lure | lure |
| locksmith_analyst (X) | data analyst (sp) | -- | other | other |
| falconer_cashier (C) | grocery store cashier (sp) | licensed falconry practitioner (sp) | other | other |
| falconer_cashier (X) | grocery store cashier (sp) | -- | other | other |

(C) = conflict, (X) = control, (sp) = contains space

**100% prediction accuracy.** The model's actual reasoning is irrelevant to the scored verdict.

---

## What the Data Actually Shows (Bug-Free)

### Llama-3.1-8B-Instruct (unaffected by bugs)

- **Conflict NF items: 10/10 lure (100%).** The model gives the representativeness heuristic answer every time.
- **Control NF items: 10/10 correct (100%).** Without a stereotypical description, the model correctly uses base rates.
- **Standard base rate (probability format): 21/25 lure (84%).**

Llama's NF results are valid and genuinely interesting: natural frequency framing makes performance *worse* (84% -> 100% lure), the opposite of Gigerenzer's prediction in humans.

### R1-Distill-Llama-8B (INVALID -- must re-collect)

We cannot determine R1-Distill's actual NF performance from the stored data. The responses are truncated before the answer and scored with a broken matcher.

From the first 500 chars of reasoning, we can observe that R1-Distill:
1. Correctly restates the problem (base rates, description)
2. Begins reasoning about probabilities vs. description
3. Shows awareness of the base rate information

But we cannot see whether the reasoning chain ultimately produces the correct or lure answer.

### R1-Distill standard base rate (also suspect)

The `r1_distill_llama_ALL.json` shows 80% "other" on standard base rate conflict items (20/25). This file has empty response strings, suggesting it was processed by a different pipeline, but the high "other" rate is suspicious and may reflect the same or similar scoring issues.

---

## Does the Same Bug Affect Other Results?

| Dataset | Ġ Present | Truncated | Multi-word Answers | Affected |
|---------|-----------|-----------|-------------------|----------|
| `new_items_r1_distill_llama_8b.json` | Yes (all 50) | Yes (500 chars) | Yes (NF items) | **YES** |
| `new_items_llama_31_8b_instruct.json` | No | N/A (short responses) | N/A | No |
| `r1_distill_llama_ALL.json` | No | Empty responses | N/A | Different issue |
| Sunk cost items (both models) | Yes (R1) | Yes | No (Switch/Continue) | **No** -- single words work |

The sunk cost results (0% lure both models) are valid because the answers ("Switch"/"Continue") are single words unaffected by the Ġ bug.

---

## Affected Conclusions in Existing Documents

### `docs/gigerenzer_analysis.md`

- Section 1 "Result Summary": R1-Distill row (4% -> 40%) is invalid
- Section 2 point 3 "fragility of reasoning training": based on invalid data
- Section 3.3 "Fragility of reasoning training": entire section is based on invalid data
- Section 3.4 "R1-Distill's 'other' responses": the "other" responses are a Ġ artifact, not confused reasoning
- The Llama conclusions (84% -> 100%) remain valid

### `docs/preregistration.md`

- H8 "FALSIFIED (Gigerenzer prediction rejected)": The Llama half of this is correct. The R1-Distill half is invalid.
- The falsification claim should be weakened to: "Falsified for Llama; R1-Distill results pending re-collection"

### `docs/SESSION_STATE.md`

- Any references to R1-Distill NF lure rates need the caveat

---

## Required Fixes

### Immediate (before any re-analysis)

1. **Fix tokenizer decode in `run_new_items.py`**: Add explicit space normalization:
   ```python
   response = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
   response = response.replace('\u0120', ' ').replace('\u010a', '\n')
   ```
   Or better: use `tok.decode(..., clean_up_tokenization_spaces=True)` if supported.

2. **Remove response truncation**: Store the full response, or at minimum truncate to 8192 chars.

3. **Use proper scoring**: Replace the ad-hoc `parse_answer()` with the regex-based `score_response()` from `src/s1s2/extract/scoring.py`, which handles:
   - Last-match-wins for reasoning models (correct mentioned in reasoning, lure at end = lure)
   - Regex patterns instead of substring matching
   - Thinking trace separation

### Re-collection required

Re-run R1-Distill on the 20 natural frequency items with the fixed pipeline. Until then, all R1-Distill NF conclusions should be flagged as provisional.

### Audit recommendation

Check whether `r1_distill_llama_ALL.json`'s 80% "other" rate on standard base rate items reflects the same or a similar scoring bug. The empty response fields are suspicious.

---

## Answering the Original Question

> WHY does natural frequency format break R1-Distill?

**It doesn't -- or at least, we don't know yet.** The evidence that natural frequency "breaks" R1-Distill is entirely an artifact of the scoring pipeline. The 40% lure rate and 40% "other" rate are predicted with 100% accuracy by the Ġ tokenizer bug + response truncation, with no reference to actual model behavior.

The three hypothesized mechanisms (longer prompt confusing reasoning, difficulty parsing "X out of N", concrete numbers triggering different processing) cannot be evaluated from the current data.

After re-collection with a fixed pipeline, R1-Distill's actual NF performance could be anywhere from 0% to 100% lure. Given that R1-Distill achieves 4% lure on standard base rate, and given that it clearly engages in substantive reasoning about the NF problems (visible in the first 500 chars), there is no strong prior for expecting 40% lure.
