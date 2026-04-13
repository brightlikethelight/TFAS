# Submission Timeline: ICML MechInterp Workshop

**Deadline**: May 8, 2026 AOE
**Current date**: April 12, 2026 (25 days remaining)
**Status**: Paper drafted (8pp), all experiments complete except R1-Distill multi-seed (running)

---

## Week 1: Apr 14-18 — Advisor feedback + template switch

| Day | Task | Owner | Blocker? |
|-----|------|-------|----------|
| Mon 14 | Send `advisor_summary.md` + PDF to advisor | Bright | -- |
| Mon 14 | Download R1-Distill multi-seed results from GPU | Bright | GPU job must finish |
| Tue 15 | Integrate R1 multi-seed into paper (update behavioral narrative for greedy vs. sampled) | Bright | R1 results |
| Tue 15 | Download official ICML 2026 style files; begin template conversion | Bright | -- |
| Wed 16 | Complete ICML template switch (follow `docs/icml_conversion_guide.md`) | Bright | -- |
| Wed 16 | Finalize co-author list based on advisor input | Bright | Advisor response |
| Thu 17 | Expand natural frequency to N=30 if advisor agrees it is worth the GPU time | Bright | Advisor decision |
| Fri 18 | Paper compiles cleanly in ICML 2-column format; all numbers verified | Bright | -- |

**Deliverable**: ICML-formatted draft with R1 multi-seed integrated, sent to co-authors for review.

**Risk**: If advisor feedback is delayed past Wed 16, co-author list and framing decisions slip into Week 2. Mitigate by sending the summary Mon morning.

---

## Week 2: Apr 19-25 — Co-author review + figures

| Day | Task | Owner | Blocker? |
|-----|------|-------|----------|
| Sat 19 | Address any advisor feedback on framing (title, "mechanistic" vs "representational") | Bright | Advisor feedback |
| Mon 21 | Circulate draft to all co-authors with 4-day review window | Bright | -- |
| Mon 21 | Generate all publication-quality figures from real data (probe curves, behavioral heatmap, cross-prediction, lure susceptibility) | Bright | -- |
| Tue 22 | Embed all figures into paper; remove any remaining placeholders | Bright | -- |
| Wed 23 | Run N=30 natural frequency if approved; otherwise mark as future work | Bright | GPU access |
| Thu 24 | Collect co-author feedback; triage into must-fix vs. nice-to-have | Bright | Co-author responses |
| Fri 25 | Implement must-fix feedback; update supplementary materials | Bright | -- |

**Deliverable**: Near-final draft with all figures embedded, co-author feedback addressed.

**Risk**: If co-authors are slow, compress review into Week 3 (less ideal but feasible). Natural frequency expansion is a stretch goal — cut without guilt if GPU time is scarce.

---

## Week 3: Apr 26 - May 2 — Final polish

| Day | Task | Owner | Blocker? |
|-----|------|-------|----------|
| Sat 26 | Full consistency check: numbers in text vs. tables vs. figures vs. data files | Bright | -- |
| Mon 28 | Finalize supplementary materials (all `[DATA]` placeholders filled, bootstrap CIs reported) | Bright | -- |
| Tue 29 | Trim unused bib entries (19 flagged in quality check); verify all citations resolve | Bright | -- |
| Wed 30 | Final framing pass: ensure no overclaiming per `critical_self_review.md` guidelines | Bright | -- |
| Thu 1 | Prepare anonymized code repository if required by workshop | Bright | -- |
| Fri 2 | Practice presentation if workshop requires poster/talk; prepare 5-min pitch | Bright | -- |

**Deliverable**: Submission-ready paper, supplementary, and code package.

---

## Week 4: May 3-7 — Submit

| Day | Task | Owner | Blocker? |
|-----|------|-------|----------|
| Sat 3 | Cold read-through (fresh eyes after 1 day off) | Bright | -- |
| Mon 5 | Final pdflatex + bibtex compilation; verify page limits (8pp main + refs + supplement) | Bright | -- |
| Tue 6 | Assemble submission package: PDF, supplementary PDF, code, benchmark JSONL | Bright | -- |
| Wed 7 | **SUBMIT** (1-day buffer before May 8 AOE deadline) | Bright | -- |
| Thu 8 | Deadline day — buffer for any last-minute platform issues | -- | -- |
| Fri 9 | Post Alignment Forum draft (`docs/af_post_draft.md` as starting point) | Bright | -- |

**Deliverable**: Submitted paper. AF post drafted.

---

## Decision points requiring advisor input (by Apr 16)

1. **4pp short vs. 8pp long?** Current draft is ~7.5pp. Recommend 8pp unless advisor prefers the tighter 4pp format.
2. **Title word**: "Mechanistic" or "Representational"? Critical self-review recommends the latter as more honest without causal evidence.
3. **Natural frequency expansion**: Worth GPU time for N=30, or cut to future work?
4. **Co-author order and affiliations**: Need final list before template switch.

---

## What can be cut without weakening the paper

If time runs short, drop these in order:

1. Natural frequency expansion (N=10 -> N=30) — preliminary observation either way
2. Practice presentation — can do after submission
3. Anonymized code repo — check if workshop actually requires it
4. AF post — write after submission, not before

**Do not cut**: ICML template switch, figure finalization, consistency check, co-author review.
