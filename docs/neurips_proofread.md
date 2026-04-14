# NeurIPS Paper Proofread Report

**File**: `paper/neurips_paper.tex`
**Date**: 2026-04-12

---

## CRITICAL

### 1. Wrong citation for "S2-by-default hypothesis" (line 635)

The Discussion paragraph "Readable but not writable" cites `\citep{hewitt2019control}` to support the "S2-by-default hypothesis." Hewitt & Liang (2019) is about **probe control tasks**, not the S2-by-default hypothesis. The S2-by-default hypothesis originates from De Neys --- the correct citation is `deneys2023advancing` or `deneys2017dual`.

**Fix**: Replace `\citep{hewitt2019control}` with the appropriate De Neys citation on line 635.

### 2. Heuristic family names mismatch between contributions and Section 3 (lines 150 vs 271-276)

The contributions list (line 150) names the four heuristic families as: "representativeness, **loss aversion, certainty weighting, and availability**."

Section 3 (lines 271-276) names them as: "representativeness, **cognitive reflection, decision framing, and loss/availability**."

These are completely different categorizations. Section 3's listing matches the table and the actual 11 categories. The contributions list appears to be wrong.

**Fix**: Change contributions item (1) to match Section 3: "representativeness, cognitive reflection, decision framing, and loss/availability."

### 3. "Five controls" but only four listed (line 645)

The Discussion states "We apply five controls" but then enumerates only four:
1. Hewitt-Liang selectivity >34pp
2. Cross-prediction falsification
3. Cross-model transfer (AUC > 0.92)
4. Ma et al. SAE falsification

The connective "and" before the fourth item signals the end of the list. Either a fifth control was dropped during editing, or the count should be "four."

**Fix**: Either add the missing fifth control (possibly the random-direction steering control from Section 5.3) or change "five" to "four."

### 4. Abstract vs body number mismatch: 21pp vs 21.3pp (lines 70 vs 542)

Abstract (line 70): "reduce the lure rate by **21** percentage points"
Body (line 542): "reduces lure rate by **21.3**pp"

Arithmetic confirms: 52.5% - 31.2% = 21.3pp. The abstract rounds down.

**Fix**: Change abstract to "21.3 percentage points" to match the body and arithmetic, or round both consistently to "21."

---

## IMPORTANT

### 5. CogBias method mischaracterized in Results (line 543)

Results (line 543): "comparable to CogBias's 26-32% reduction via **SAE-based** steering"
Related Work (line 223): correctly says "contrastive activation addition"

CogBias (Huang et al., 2026) used contrastive activation addition, not SAE-based steering.

**Fix**: Change "SAE-based steering" to "contrastive activation addition" on line 543.

### 6. Three stub appendices with placeholder text (lines 669-681)

Appendix A (Within-CoT): "will be reported here."
Appendix B (Scale Analysis): "appear here."
Appendix C (SAE Features): Only a one-line description, no data.

Only Appendix D (Multi-Seed Robustness) has actual content. The main text explicitly references all four appendices. Stub appendices are acceptable in a draft but would be flagged by reviewers in a submission.

**Fix**: Populate Appendices A-C with actual data/figures before submission, or remove the forward references if the data is not ready.

### 7. Direct "System 1 / System 2" usage in Related Work (line 214)

The project convention is to use "S1-like / S2-like" framing, never bare "System 1 / System 2." Four occurrences of `System~1` / `System~2` appear:

- Lines 95-96: Acceptable --- introducing the Kahneman terminology in quotes with immediate qualification that it's an oversimplification.
- **Line 214**: "System~1/System~2 distinction has a *mechanistic* counterpart" --- bare usage without S1-like/S2-like framing or quotes.
- Line 407: "a universal ``System~1''" --- in quotes, acceptable as scare-quoted.

**Fix**: Line 214: change to "the \sone{}/\stwo{} distinction" or "the fast/slow distinction."

### 8. Acknowledgments formatting (line 652)

Acknowledgments are placed as a `\paragraph{}` inside the Discussion section. NeurIPS style requires acknowledgments as a separate unnumbered section (typically `\section*{Acknowledgments}`), placed after the main body and before references. The data availability statement ("Code, benchmark, and activation data will be released upon acceptance") should also be in its own section or clearly separated.

**Fix**: Move acknowledgments to `\section*{Acknowledgments}` between Discussion and References.

### 9. Table 1 omits 4 of 11 categories (lines 382-393)

The table shows only 7 of 11 benchmark categories. The loss/availability family (sunk cost, loss aversion, certainty effect, availability -- 60 pairs) is entirely absent from the behavioral results table, despite contributing to the overall lure rate. A reviewer will notice that the per-category rows don't sum to the overall row.

**Fix**: Add the missing 4 categories to the table, even if all values are 0%, or add a note explaining their omission.

### 10. `\pending{}` macro defined but unused (line 31)

The `\pending` command for placeholder markers is defined (`\newcommand{\pending}...`) but never used. This is harmless but will render in red if accidentally invoked. It also signals "work in progress" to reviewers if they read the source.

**Fix**: Remove the `\pending` macro definition before submission to keep the preamble clean.

---

## MINOR

### 11. Four unused bib entries

The following keys are defined in `references.bib` but never cited in the paper:
- `burns2022discovering`
- `goodfire2025r1`
- `kerns2004anterior`
- `kolb2013brain`

Not a functional issue (BibTeX silently ignores them), but clutters the bib file.

### 12. Unreferenced section labels

The following `\label` definitions are never referenced with `\ref`:
`sec:benchmark`, `sec:discussion`, `sec:intro`, `sec:methods`, `sec:methods_steering`, `sec:probes`, `sec:related`, `sec:results`, `sec:results_behavioral`, `sec:results_probes`, `sec:results_sae_attention`, `sec:scale`, `sec:steering`

These are harmless (useful for future cross-references), but may generate LaTeX warnings about unused labels depending on configuration.

### 13. User's canonical numbers vs paper numbers

The user specified checking "37.5pp, 31.2%, 68.8%." The paper consistently uses **37.6pp** (not 37.5pp). This is internally consistent: 68.8% - 31.2% = 37.6pp. If 37.5pp was from an earlier draft, the paper has been updated and is now self-consistent. Verify that 37.6pp matches the actual experimental data.

### 14. Missing Conclusion section

The paper goes directly from Discussion to References with no Conclusion. While NeurIPS does not strictly require a standalone Conclusion section, the Discussion ends abruptly after the limitations paragraph. A brief concluding paragraph (or a `\paragraph{Conclusion.}` at the end of Discussion) summarizing the main takeaway would strengthen the paper.

### 15. `sanitychecks2026` bib entry has `{Anonymous}` as author (bib line 279)

This will render as "Anonymous (2026)" in the bibliography, which looks provisional. If the actual authors are known, update the entry.

### 16. NeurIPS style file not yet applied (line 21-24)

The paper uses a plain `article` class with manual geometry. The comment on lines 21-24 notes this needs to be swapped for `neurips_2026.sty`. This is a known TODO but worth tracking.

### 17. Minor grammar: "contrasting with" dangling comparison (line 508)

"...contrasting with CogBias's near-orthogonal finding on architecturally distinct models." --- The possessive on an italicized citation key is slightly awkward. Consider: "...contrasting with the near-orthogonal finding of CogBias on architecturally distinct models."

### 18. Figure paths use relative `../figures/` (lines 433, 564)

Both `\includegraphics` calls use `../figures/`. This is fine if compiling from the `paper/` directory, but will break if the build directory changes. Minor robustness concern.
