# Master Action Plan — Pre-Deadline Sprint

Generated: 2026-05-03, ~20:00
Project: P1 (Readable but Not Writable)
Venues: NeurIPS 2026 Main (May 4/6), ICML MechInterp Workshop (May 8)

---

## Executive Summary

The paper is **submission-ready** for both venues after 22 rounds of fixes. All critical formatting, anonymity, citation, numerical, and AI-writing issues have been resolved. The remaining action items are primarily human logistics (co-author list, OpenReview upload, reciprocal reviewer registration) and one compute-dependent gap (Table 2 missing CIs).

**Estimated time to submit NeurIPS abstract**: 30 minutes (paste prepared text into OpenReview)
**Estimated time to submit NeurIPS paper**: 2-3 hours (prepare supplementary ZIP, final review, upload)
**Estimated time to submit workshop paper**: 1 hour (upload PDF, register reviewer)

---

## Priority Queue

### Tier 1: BLOCKERS (do before May 4 AOE)

```
[P1, neurips-main, blocker, human-action] Finalize co-author list — "pending advisor confirmation"
[P1, neurips-main, blocker, human-action] Verify all co-authors have active OpenReview profiles
[P1, neurips-main, blocker, 30min]        Submit abstract on OpenReview (text in docs/neurips_submission_metadata.md)
```

### Tier 2: MAJOR (do before May 6 AOE)

```
[P1, neurips-main, major, 2-3h]  Prepare supplementary ZIP: anonymize code, include benchmark.jsonl, write README
[P1, neurips-main, major, 30min] Upload neurips_paper_anonymous.pdf to OpenReview
[P1, neurips-main, major, 30min] Add footnote to Table 2 re: missing Qwen/OLMo CIs ("not computed; 5-fold CV AUCs reported")
```

### Tier 2b: MAJOR (do before May 8 AOE)

```
[P1, icml-ws-mech, major, 30min]  Submit workshop_paper_icml.pdf to ICML MechInterp OpenReview
[P1, icml-ws-mech, major, 15min]  Register as reciprocal reviewer (Google Form on mechinterpworkshop.com)
[P1, icml-ws-mech, major, 1-2h]   Optional: create anonymous code repo at anonymous.4open.science
```

### Tier 3: POLISH (if time permits before May 6)

```
[P1, neurips-main, minor, 1h]    Review Discussion section for any remaining hedging or overclaims
[P1, neurips-main, minor, 30min] Consider adding "Limitations" as explicit subsection heading (currently embedded in Discussion)
[P1, icml-ws-mech, minor, 30min] Add brief Conclusion paragraph to workshop paper
[P1, both, minor, 30min]         Run one final grep for any remaining AI-tell phrases (em-dashes still at 35)
```

### Tier 4: DEFERRED (post-submission / camera-ready)

```
[P1, neurips-main, polish, >1d]   Run held-out template split experiment (GPU needed)
[P1, neurips-main, polish, >1d]   Run R1 probe score tracking during steering (GPU needed)
[P1, neurips-main, polish, 2h]    Compute bootstrap CIs for Qwen + OLMo (CPU, fills Table 2 gaps)
[P1, neurips-main, polish, 4h]    Run 50 random directions at R1 L31 (GPU, currently has 0)
[P1, icml-ws-mech, polish, 2-4h]  Convert workshop to icml2026 template (camera-ready only)
[P1, both, polish, 1h]            Investigate pair-difference probe AUC=0.024 anomaly
```

---

## Resolved Issues (Rounds 14-22)

22 rounds of fixes resolved all critical and major formatting/content issues:
- False "non-overlapping CIs" claim → removed (Rounds 17, 19, 22)
- Type 3 fonts → eliminated (Round 20)
- AI-writing markers → reduced to score ~1.5/10 (Rounds 18-19)
- Stale cross-paper numbers → synced (Round 19)
- Appendix stubs → filled (Round 17)
- 5 other critical issues → fixed

Total commits this session: 9 (Rounds 14-22), all pushed to origin/main.

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Co-author list not finalized by May 4 | Medium | High (can't submit abstract) | Contact advisor NOW |
| Reviewer catches CI overlap and questions "blurring" | Medium | Medium | OLMo pair IS non-overlapping; key claim is steering null, not AUC diff |
| Reviewer wants frontier-scale models | High | Low-Medium | Acknowledge in limitations; open-weight focus is principled |
| Workshop reviewer wants circuits/SAE depth | Medium | Medium | SAE section is brief; appendix has details |
| Table 2 missing CIs flagged | Medium | Low | Add footnote explaining CV-only AUCs |
