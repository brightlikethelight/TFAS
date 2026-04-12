# ICML MechInterp Workshop Submission Checklist

**Deadline: May 8, 2026 (~24 days from Apr 14)**

---

## Pre-submission checklist

- [ ] Paper compiles without errors (pdflatex + bibtex)
- [ ] All figures generated from real data (not placeholders)
- [ ] All numbers in text match data files
- [ ] Author names and affiliations finalized
- [ ] Abstract under 250 words
- [ ] Main text under 4 pages (excluding references)
- [ ] References complete (no "???" in PDF)
- [ ] Supplementary materials compiled
- [ ] Code repository prepared (anonymized if required)
- [ ] Benchmark JSONL included in submission package

---

## Required GPU experiments (before submission)

| Priority | Experiment | Est. Time | Blocking |
|----------|-----------|-----------|----------|
| 1 | SAE Goodfire L19 | ~30 min | SAE claims in paper |
| 2 | New items behavioral (sunk_cost + natural freq) | ~1 hr | Benchmark claims |
| 3 | OLMo-3 pair | ~3 hr | Cross-architecture replication |
| 4 | Bootstrap CIs | ~30 min | Rigor claims |
| 5 | Attention entropy | ~1 hr | Stretch goal (supplement) |
| 6 | Confidence paradigm | ~1 hr | Stretch goal (supplement) |

**Critical path**: Priorities 1-4 must complete before paper can be finalized. Priorities 5-6 are supplement-only and can be cut without weakening the main paper.

**Total required GPU time**: ~5.5 hr (priorities 1-4), ~7.5 hr with stretch goals.

---

## Paper content gaps

Sections that need updates once GPU results are in:

- **Table 1 (behavioral)**: Add sunk_cost row when available from experiment #2.
- **Figure 1**: Generate from real probe data. Currently a placeholder box.
- **Supplementary**: Fill in all `[DATA]` placeholders with actual numbers.
- **Discussion**: Update cross-architecture claims if OLMo replicates (experiment #3). If it does not replicate, reframe as Llama+Qwen result with OLMo as informative negative.

---

## Submission package structure

```
submission/
├── workshop_paper.pdf
├── supplementary.pdf
├── figures/
│   ├── probe_layer_curves.pdf
│   ├── behavioral_heatmap.pdf
│   ├── cross_prediction.pdf
│   └── lure_susceptibility.pdf
├── code/  (if allowed)
│   └── s1s2/  (anonymized)
└── data/
    └── benchmark.jsonl
```

---

## Timeline (May 8 deadline)

### Week 1: Apr 14-18 — GPU experiments + figures
- Run all priority 1-4 experiments on RunPod
- Generate publication-quality figures from real data
- Replace all placeholder figures in paper
- Smoke-test stretch goals (priorities 5-6) if time permits

### Week 2: Apr 19-25 — Write final Methods + Results
- Finalize Methods section with exact hyperparameters and bootstrap CI details
- Write Results with real numbers, CIs, and effect sizes
- Fill supplementary `[DATA]` placeholders
- Complete Table 1 and all figures

### Week 3: Apr 26-May 2 — Internal review + revisions
- Full read-through for consistency (numbers in text vs. tables vs. figures)
- Get advisor/collaborator feedback
- Revise framing and discussion based on actual results
- Verify reproducibility: re-run key analyses from saved checkpoints

### Week 4: May 3-8 — Final polish + submit
- Final pdflatex + bibtex compilation check
- Verify page limits (4pp main + references + supplement)
- Assemble submission package (see structure above)
- Anonymize code repo if required
- **Submit by May 7** (one day buffer before May 8 deadline)

---

## Risk mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| SAE Goodfire fails to find interpretable features | Weakens one claim | Paper works without SAE section. Mark as future work. Core behavioral + probing results stand alone. |
| OLMo does not replicate | Weakens generality claim | Paper works with Llama + Qwen (2 model families). Report OLMo as informative negative in supplement. |
| Bootstrap CIs are wide | Weakens statistical rigor | Report honestly. Discuss statistical power limitations. Wide CIs with consistent direction still support qualitative claims. |
| RunPod GPU unavailable | Blocks all experiments | Have backup reservation. Priority 1-4 need only ~5.5 hr total; can run on any H100 instance. |
| Page limit exceeded | Rejection risk | Move detailed methods to supplement early. Keep main paper focused on key results. |
