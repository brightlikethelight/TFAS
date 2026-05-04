# Consolidated Findings — P1: Readable but Not Writable

Cross-venue consolidation of all audit findings (Rounds 14-22 + current Phase 2 agents).

## Tier 1: Blockers (desk rejection risk)

| # | Issue | Venue | Effort | Status |
|---|---|---|---|---|
| 1.1 | Co-author list not finalized | NeurIPS | Human action | **OPEN** — must resolve before May 4 |
| 1.2 | OpenReview profiles for all authors | NeurIPS | Human action | **OPEN** — verify today |
| 1.3 | Type 3 fonts in figures | Both | <30min | **FIXED** Round 20 |
| 1.4 | Checklist missing | NeurIPS | <30min | **FIXED** — included since initial draft |
| 1.5 | Body >9 pages | NeurIPS | N/A | **PASS** — Discussion on p.9 |
| 1.6 | Anonymity violations | Both | <30min | **PASS** — zero identifying info |

## Tier 2: Major (materially affects acceptance)

| # | Issue | Venue | Effort | Status |
|---|---|---|---|---|
| 2.1 | False "non-overlapping CIs" claim (Llama vs R1) | Both | <30min | **FIXED** Rounds 17, 19, 22 |
| 2.2 | `≤ 0%` nonsensical expression | NeurIPS | <30min | **FIXED** Round 17 |
| 2.3 | Heuristic family names mismatch | NeurIPS | <30min | **FIXED** Round 17 |
| 2.4 | Appendix table overflow (302pt) | NeurIPS | <30min | **FIXED** Round 17 (resizebox) |
| 2.5 | Stale Qwen/OLMo numbers in workshop | Workshop | 30m-2h | **FIXED** Round 19 |
| 2.6 | Placeholder figure in workshop | Workshop | <30min | **FIXED** Round 16 |
| 2.7 | "Five contributions" (should be six) | Workshop | <30min | **FIXED** Round 16 |
| 2.8 | 32B claim in workshop not backed by table | Workshop | <30min | **FIXED** Round 16 |
| 2.9 | Missing text baselines (TF-IDF, length-only) | Both | 30m-2h | **FIXED** Round 11 |
| 2.10 | AI-writing markers (9x confirms, "dramatic", em-dashes) | NeurIPS | 30m-2h | **FIXED** Rounds 18-19 |
| 2.11 | Appendix stubs unfilled | NeurIPS | 30m-2h | **FIXED** Round 17 |
| 2.12 | bib year wrong (ma2026falsification: 2025→2026) | Both | <30min | **FIXED** Round 17 |
| 2.13 | Supplementary ZIP not prepared | NeurIPS | 2-8h | **OPEN** — needed by May 6 |
| 2.14 | Table 2 missing CIs for Qwen + OLMo | Both | 2-8h | **OPEN** — needs CPU compute or footnote |
| 2.15 | R1 L31 zero random controls | NeurIPS | 2-8h | **OPEN** — needs GPU or explicit acknowledgment |

## Tier 3: Clarity/Polish

| # | Issue | Venue | Effort | Status |
|---|---|---|---|---|
| 3.1 | Within-CoT figure not in main text | NeurIPS | <30min | **FIXED** Round 14 |
| 3.2 | Vestigial metaphor not in Discussion | Both | <30min | **FIXED** Round 10 |
| 3.3 | "Readable but not writable" not in Contribution 4 | NeurIPS | <30min | **FIXED** Round 12 |
| 3.4 | Metadata abstract stale | NeurIPS | <30min | **FIXED** Round 15 |
| 3.5 | Workshop table footnote "not yet evaluated" | Workshop | <30min | **FIXED** Round 16 |
| 3.6 | Workshop paper missing Conclusion section | Workshop | 30m-2h | **OPEN** — minor |
| 3.7 | Workshop uses `article` class not `icml2026` | Workshop | 30m-2h | **OPEN** — OK for submission per CFP |

## Tier 4: Nice-to-have (defer to v2)

| # | Issue | Venue | Effort | Status |
|---|---|---|---|---|
| 4.1 | Held-out template split experiment | Both | >1d (GPU) | DEFERRED |
| 4.2 | R1 probe score tracking during steering | NeurIPS | >1d (GPU) | DEFERRED |
| 4.3 | Forced-<think> robustness check | NeurIPS | >1d (GPU) | DEFERRED |
| 4.4 | Convert workshop to ICML template | Workshop | 2-8h | DEFERRED (camera-ready only) |
| 4.5 | Anonymous code repository | Workshop | 2-8h | OPTIONAL (helps but not required) |
| 4.6 | Pair-difference probe analysis (AUC=0.024, not understood) | Both | 2-8h | DEFERRED |

## Venue-Specific Notes

### NeurIPS-only
- Body is at exactly 9 pages — **zero room** for any text additions
- OLMo CIs are genuinely non-overlapping → strongest argument for "blurring"
- 32B scale analysis is a major strength vs typical interp papers
- Checklist is thorough (16/16 items answered with justifications)

### Workshop-only
- Non-archival, so NeurIPS dual submission is fine
- Need to register reciprocal reviewer before submission
- Workshop reviewers care more about mechanistic novelty than exhaustive scaling
- Shorter paper means some results are referenced but not shown (32B, SAE details)
