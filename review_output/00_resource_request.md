# Resource Request

## Already Available
- [x] Compiled PDFs for all papers (zero errors)
- [x] All result JSON files (74 files, verified against paper claims)
- [x] All figures regenerated with TrueType fonts
- [x] NeurIPS style file (`neurips_2026.sty`)
- [x] Checklist completed (16/16 items)
- [x] Git repo pushed to GitHub

## Needed for Submission (HUMAN ACTION)

### By May 4 AOE (abstract deadline)
- [ ] **Co-author list finalized** — metadata says "pending advisor confirmation"
- [ ] **All co-authors have active OpenReview profiles** with institutional email (profiles without institutional email take up to 2 weeks)
- [ ] **OpenReview abstract submitted** — title, abstract, TL;DR, keywords, primary area all prepared in `docs/neurips_submission_metadata.md`

### By May 6 AOE (paper deadline)
- [ ] **Upload `submission_neurips/neurips_paper_anonymous.pdf`** to OpenReview
- [ ] **Prepare supplementary ZIP** (≤100MB): anonymized code + `data/benchmark/benchmark.jsonl` + analysis scripts
- [ ] **Contribution type selected** on OpenReview (recommend: "General")

### By May 8 AOE (workshop deadline)
- [ ] **Submit `submission/workshop_paper_anonymous.pdf`** to ICML MechInterp OpenReview
- [ ] **Register as reciprocal reviewer** via Google Form on mechinterpworkshop.com
- [ ] **Optionally**: create anonymous code repo at anonymous.4open.science (reviewers weight reproducibility)

## Needed for Proposed Experiments (if time permits)

| Experiment | GPU | Hours | Purpose | Priority |
|---|---|---|---|---|
| Held-out template split (probe on unseen templates) | 1x H100 | ~2h | Address confound concern | Medium |
| R1 probe score tracking during steering | 1x H100 | ~4h | Strongest "readable but not writable" evidence | Medium |
| Bootstrap CIs for Qwen + OLMo-32B | CPU only | ~1h | Fill Table 2 gaps | Low |
| Forced-<think> robustness check | 1x H100 | ~2h | Address empty-think bypass concern | Low |

**Total if all run**: ~9 H100-hours + 1 CPU-hour. None are blockers for submission.

## NOT Needed
- No Semantic Scholar API key (citation verification done via web search)
- No W&B API key (no new experiments planned before deadline)
- No additional API keys
- No `humanizer-academic` skill (manual pass completed in Rounds 18-19)
