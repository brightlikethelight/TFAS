# NeurIPS 2026 Main Track — Per-Venue Scorecard

## Submission Status
- **File**: `neurips_paper_anonymous.pdf` (22 pages: 9 body + refs + appendix + checklist)
- **Deadline**: Abstract May 4 AOE, Paper May 6 AOE
- **Format**: `neurips_2026.sty` anonymous mode
- **Archival**: Yes (primary target)

## Key Issues (from prior audit rounds 14-22)

| Issue | Status | Round Fixed |
|---|---|---|
| FALSE "non-overlapping CIs" claim | FIXED | Round 17 |
| `≤ 0%` nonsensical expression | FIXED | Round 17 |
| Heuristic family names wrong in contributions | FIXED | Round 17 |
| Appendix table overflow (302pt) | FIXED | Round 17 |
| Appendix stubs (Within-CoT, Scale) | FILLED | Round 17 |
| Acknowledgments not using \begin{ack} | FIXED | Round 15 |
| Type 3 fonts in figures | FIXED | Round 20 |
| AI-writing: 9x "confirms", "dramatic", "emerge" | REDUCED | Round 18 |
| Em-dash density (45→35) | REDUCED | Round 19 |
| Metadata title/abstract stale | SYNCED | Round 15 |
| Within-CoT figure missing from main text | ADDED | Round 14 |
| Text baselines (TF-IDF, length-only) | ADDED | Round 11 |
| Vestigial metaphor | ADDED | Round 10 |
| Readable-but-not-writable in Contribution 4 | PROMOTED | Round 12 |

## Remaining Concerns (from current session audits)
1. **Co-author list pending** — blocker for May 4 abstract
2. **Supplementary ZIP not prepared** — needed by May 6
3. **Table 2**: Qwen and OLMo rows show "---" for CIs, Control AUC, Selectivity
4. **Body is at exactly 9 pages** — zero margin for any additions
5. **The central CIs (Llama vs R1) overlap by 0.008** — paper now correctly states this, but it weakens the "blurring" argument slightly (mitigated by OLMo pair which IS non-overlapping)
6. **R1 L31 steering has zero random controls** (n_directions=0 in JSON) — only L14 has random controls (5 directions). Paper uses L14 random controls for all R1 discussion, which is defensible but not ideal.

## Scorecard

```
Story & Framing             [5/5]  "Readable but not writable" is memorable and precise
Technical Correctness       [4/5]  CI overlap now disclosed; all numbers verified
Empirical Evidence Strength [4/5]  Strong (4 model pairs, steering, within-CoT, SAE, scale)
                                   Missing: held-out templates, more random directions at L31
Novelty vs. Prior Work      [4/5]  First causal steering study on cognitive biases in LLMs
                                   Concurrent: CogBias (Huang 2026), Ziabari (2025)
Writing Clarity             [4/5]  Clean after humanizer pass; some sections very dense
Figure & Table Quality      [4/5]  3 good figures; table CIs incomplete for 2 models
Reproducibility             [4/5]  HuggingFace model IDs, detailed methods, checklist complete
                                   Code not yet released
Venue Fit                   [5/5]  Interpretability + cognitive science + causal methods
Submission Readiness        [4/5]  PDF ready; co-author list and supplementary pending

Overall (pre-rebuttal estimate): Weak Accept (6/10)
Time-to-Submit-Ready: ~3 hours (co-author list, supplementary ZIP, final review)
```

## Biggest Risks for Reviewer Objections
1. **"The CIs overlap — how do you claim the reasoning model 'blurs' the direction?"** → OLMo pair IS non-overlapping; Llama/R1 point estimates differ by 0.044; the key finding is steering null, not probe AUC difference.
2. **"Only 8B models + one 32B — does this generalize?"** → OLMo cross-architecture replication helps; 32B shows same pattern; frontier models would strengthen but are out of scope for open-weight interpretability.
3. **"Linear probes find anything in high-d — why should we believe this?"** → Hewitt-Liang selectivity >34pp; random-direction controls; cross-category transfer; three text baselines all well below probe AUC.
4. **"Steering at P0 is a weak causal test — the model generates 2048 tokens after."** → Continuous 2048-token steering at all 4 R1 layers still null; Llama P0 steering produces dose-response even with short generation.
