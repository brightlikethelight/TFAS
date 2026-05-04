# ICML MechInterp Workshop — Per-Venue Scorecard

## Submission Status
- **File**: `workshop_paper_icml.pdf` (9 pages)
- **Deadline**: May 8, 2026 AOE
- **Format**: NeurIPS style (allowed for initial submission; camera-ready must be ICML)
- **Non-archival**: Yes (dual submission with NeurIPS explicitly allowed)

## Key Issues (from prior audit rounds 14-22)

| Issue | Status | Round Fixed |
|---|---|---|
| Placeholder figure | FIXED | Round 16 |
| "Five contributions" (should be six) | FIXED | Round 16 |
| Unsupported 32B claim in contributions | FIXED | Round 16 |
| Stale Qwen AUC (0.971→0.954/0.970) | FIXED | Round 19 |
| Stale OLMo AUC (0.998→0.996, 0.993→0.962) | FIXED | Round 19 |
| False "non-overlapping CIs" for Llama/R1 | FIXED | Round 19 + 22 |
| "striking" AI-writing marker | FIXED | Round 15 |
| Type 3 fonts | FIXED | Round 20 |
| Table footnote "not yet evaluated" | FIXED | Round 16 |

## Remaining Concerns
1. Uses `article` document class, not `icml2026` — acceptable for submission per CFP but looks less polished
2. Missing bootstrap CIs for Qwen and OLMo in probe table (shown as "---")
3. No Conclusion section (ends with Discussion → Ack → Refs)
4. `hyperref` loaded without explicit `pdfauthor={}` (minor metadata risk)

## Venue-Specific Fit
- Directly addresses topics 1 (Understanding Model Internals) and 2 (Methods for Mechanistic Discovery)
- Activation steering is a causal mechanistic method — strong workshop fit
- SAE features and within-CoT probing are exactly what this workshop wants
- Andrew Lee (Harvard, co-organizer) is a conflict if affiliated — check

## Scorecard

```
Story & Framing             [4/5]  Strong "readable but not writable" framing; intro could be tighter
Technical Correctness       [4/5]  All numbers verified; CIs overlap issue fixed
Empirical Evidence Strength [4/5]  4 model pairs + steering + within-CoT; missing held-out template split
Novelty vs. Prior Work      [4/5]  First causal steering on cognitive biases; builds on CogBias/probe lit
Writing Clarity             [4/5]  Clean after humanizer pass; some sections dense
Figure & Table Quality      [4/5]  Real figure now; missing CIs in table
Reproducibility             [3/5]  Code not yet anonymized; benchmark JSONL not included
Venue Fit                   [5/5]  Perfect fit for mechanistic interpretability
Submission Readiness        [4/5]  Ready to submit; reciprocal reviewer needed

Overall (pre-rebuttal estimate): Weak Accept to Accept
Time-to-Submit-Ready: ~2 hours (prepare reciprocal reviewer, optional anonymous code repo)
```
