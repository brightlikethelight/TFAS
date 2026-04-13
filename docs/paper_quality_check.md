# Workshop Paper Quality Check

**File**: `paper/workshop_paper.tex`
**Date**: 2026-04-12
**Checker**: Claude Code (automated)

---

## 1. Number Consistency

### Verified CORRECT
- Llama overall lure rate: 27.3% (abstract L62, contributions L139, table L331, discussion L338, L545)
- R1-Distill overall lure rate: 2.4% (abstract L62, contributions L140, table L331, discussion L555)
- Llama base_rate: 84% (table L328)
- Llama conjunction: 55% (table L329)
- Llama syllogism: 52% (table L330)
- R1-Distill base_rate: 4% (table L328)
- OLMo Instruct overall: 14.9% (abstract L63, contributions L141, table L331, discussion L359, L366)
- OLMo Think overall: 0.9% (abstract L63, contributions L141, table L331, discussion L360)
- Probe AUC Llama: 0.974 [0.952, 0.992] at L16 (abstract L67, contributions L146, fig caption L392-393, probe results L410-411, table L476)
- Probe AUC R1-Distill: 0.930 [0.894, 0.960] at L31 (abstract L69, contributions L148, fig caption L400, probe results L418, table L477)
- OLMo Instruct probe: 0.998 at L21 (discussion L367, table L481)
- OLMo Think probe: 0.993 at L28 (discussion L367, table L482)
- Cross-prediction AUC: 0.378 (abstract L72, specificity L502, discussion L624)
- Lure susceptibility: +0.422 (L521), -0.326 (L522)
- SAE: 41 features, 0 spurious (contributions L158-162, results L532)
- Benchmark: 470 items, 235 pairs, 11 categories (abstract L65, benchmark L185-186)
- Qwen no-think: 21% lure, think: 7% lure, both AUC=0.971 at L34 (abstract L78-80, contributions L152-155, table L331, probe results L425-436)
- Pair counts: 35+20+25+30+25+20+20+15+15+15+15 = 235 (arithmetic verified)

### INCONSISTENCY FOUND
- **OLMo base rate lure**: Table (L328) says **46%**, body text (L365) says **45.7%**. The table rounds to integers (consistent style), but 45.7% rounds to 46%, so this is internally consistent. However, the user-specified reference number is "46%" -- so the text should say 46% or clarify the decimal is the exact value and the table rounds.
- **OLMo conjunction lure**: Table says **50**, text (L365) says **50.0%**. Consistent (50.0% rounds to 50%).

### POTENTIAL ISSUE
- **OLMo base rate in Table vs. text**: The text says "45.7% base rate" but the table says 46. If the true value is 46/100 = 46%, then "45.7%" in the text is wrong. If the true value is 16/35 = 45.7%, then the table's rounding to 46 is fine but should be noted. **Recommend making these consistent** -- either both 46% or clarify "45.7%" is exact.

---

## 2. Citation Check

### All cite keys resolve
Every `\citep{}` and `\citet{}` key in the .tex file has a corresponding entry in `references.bib`. No missing citations.

### Unused bib entries (19 entries)
These are defined in `references.bib` but never cited in the workshop paper:
- `benjamini1995controlling`, `cohen1988statistical`, `cover1965geometrical`
- `deneys2023dual`, `facco2017estimating`, `fang2026sae`, `fartale2025recall`
- `galichin2025reasonscore`, `gemma2`, `goodfire2025r1`, `he2024llama`
- `he2026latent`, `jian2025metacognitive`, `kornblith2019similarity`
- `lampinen2025nature`, `lieberum2024gemma`, `north2002note`
- `rao2024cobbler`, `yang2026step`

Not an error (bib files commonly contain extra entries), but worth trimming before submission to keep the .bbl file minimal.

---

## 3. Cross-Reference Check

All `\ref{}` targets have corresponding `\label{}` entries. No dangling references.

- `\ref{tab:behavioral}` -> `\label{tab:behavioral}` (L319)
- `\ref{fig:probe_curves}` -> `\label{fig:probe_curves}` (L406)
- `\ref{tab:probe_summary}` -> `\label{tab:probe_summary}` (L469)
- `\ref{sec:specificity}` -> `\label{sec:specificity}` (L488)
- `\ref{sec:probes}` -> `\label{sec:probes}` (L273)
- `\ref{sec:results_sae}` -> `\label{sec:results_sae}` (L530)

---

## 4. Placeholder Check

### PLACEHOLDERS FOUND (5 instances)

| Line | Placeholder | Context |
|------|-------------|---------|
| 39 | `[Co-authors]` | Author list: `\textbf{[Co-authors]}\textsuperscript{1}` |
| 40 | `[Advisor]` | Author list: `\textbf{[Advisor]}\textsuperscript{1,2}` |
| 42 | `[Department]` | Affiliation: `\textsuperscript{2}[Department]` |
| 682 | `[advisor name]` | Acknowledgments: "and [advisor name] for guidance" |
| 686-687 | `[URL upon acceptance]` | Code release: "released at [URL upon acceptance]" |

No `[TODO]`, `[PLACEHOLDER]`, `[DATA]`, `???`, or `XXX` found.

---

## 5. Style Check

### S1/S2 Framing
- The paper correctly uses `\sone{}` (S1-like) and `\stwo{}` (S2-like) macros throughout.
- "System 1" / "System 2" appear only in appropriate contexts:
  - L95-96: Quoted from Kahneman's framework ("System 1" / "System 2") -- correct, this is describing the original theory.
  - L352, L373: Quoted with scare quotes (`"System 1"`) when saying models don't have a universal one -- correct usage.
  - L559-560: Hypothetical framing ("If reasoning training simply added a 'System 2 module' that overrides System 1") -- correct, this is a straw-man being refuted.
- **No violations** of the "never say LLMs have System 1 and System 2" rule.

### Model Names
- Consistent throughout: "Llama-3.1-8B-Instruct", "R1-Distill-Llama-8B", "Qwen 3-8B", "OLMo-3-7B-Instruct", "OLMo-3-7B-Think".
- "R1-Distill" used as shorthand consistently.

### Orphaned Sentences / Incomplete Thoughts
- None found. All paragraphs are complete.

---

## 6. ERRORS AND ISSUES REQUIRING ACTION

### ERROR: Figure 1 X-Axis Range (L387)

The figure placeholder states:
> X-axis: layer index (0--35)

But Llama-3.1-8B-Instruct and R1-Distill-Llama-8B have **32 layers** (indices 0--31). Qwen 3-8B has 36 layers (0--35). The figure caption implies a single x-axis, but the Llama models would only span 0--31.

**Fix**: Either separate the plots or clarify "0--31 for Llama/R1-Distill, 0--35 for Qwen" in the figure description.

### ERROR: Attention Head Analysis Missing from Body

The abstract (L74-76) claims:
> "Attention analysis reveals the reasoning model has nearly twice as many S2-like-specialized heads (57 vs. 30; BH-FDR corrected), suggesting redistribution of deliberative computation across the network."

This finding appears **nowhere** in the Results or Discussion sections. It is mentioned only in the abstract. This is a serious structural problem -- the abstract claims a result that the body does not report.

**Fix**: Either (a) add an "Attention Analysis" subsection to Results with the supporting data, or (b) remove the claim from the abstract.

### WARNING: SAE "Goodfire SAE" on Llama Layer 19

The SAE results (L532) state analysis uses the "Goodfire SAE, 65,536 features" on "Llama's layer 19 residual stream." However, per the project conventions (CLAUDE.md), the Goodfire SAE (`Goodfire/DeepSeek-R1-SAE-l37`) is trained on **layer 37 of DeepSeek-R1 (671B)**, not Llama. The Llama SAE should be **Llama Scope** (`fnlp/Llama-3_1-8B-Base-LXR-32x`).

**Verify**: Which SAE was actually used? If Llama Scope, fix the name. If Goodfire, note the domain mismatch (trained on R1-671B, applied to Llama-8B) which is a significant methodological concern.

### WARNING: Future Work Section Inconsistency

The Discussion "Future work" paragraph (L665-673) says SAE analysis has *already* identified 41 features and frames causal interventions as the "natural next step." But the earlier "Future work" paragraph (L666-668) says:
> "SAE feature analysis: decompose the blurred representations into interpretable features using Llama Scope SAEs..."

This is contradictory: the paper both reports SAE results (Section 4.4) and lists SAE analysis as future work (L666-668). The older "Future work" text (L666-668) appears to be a leftover from a prior draft.

**Wait** -- re-reading more carefully, the paragraph at L665-673 is the actual Future work paragraph and it correctly frames SAE as done, with causal interventions as next. The earlier paragraph at L666-668 which mentions "Llama Scope SAEs" seems to be from the same paragraph. Let me re-check.

Actually, the Future work paragraph (L665-673) is self-consistent: it says SAE analysis is done (Section 4.4 ref) and causal interventions are next. **No issue here** -- I misread during the check. The SAE name discrepancy (Goodfire vs Llama Scope) in the results section remains the real concern.

---

## 7. Page Count Estimate

- ~697 lines of LaTeX with `article` class, 1-inch margins
- Content sections: Abstract, 4 numbered sections (Intro, Benchmark, Methods, Results), Discussion, Acknowledgments, References
- 2 tables, 1 figure placeholder
- **Estimated length: 7-8 pages** (body ~5-6 pages + references ~1.5-2 pages)
- ICML MechInterp workshop limit is 4 pages + references. The paper header says "4 pages + references" but the current content is likely **over the page limit** by 1-2 pages.

**Action needed**: Compile the paper and verify actual page count. If over 4 pages for body content, significant cuts are required.

---

## Summary

| Check | Status |
|-------|--------|
| Number consistency | PASS (one minor rounding note on OLMo base rate 46% vs 45.7%) |
| Citation keys | PASS (all cited keys exist; 19 unused bib entries) |
| Cross-references | PASS (all refs have labels) |
| Placeholders | 5 found (author names, department, advisor, URL) |
| Style/framing | PASS |
| Figure x-axis | ERROR: says 0-35 but Llama has 32 layers (0-31) |
| Attention heads | ERROR: claimed in abstract but missing from body |
| SAE identity | WARNING: "Goodfire SAE" likely wrong for Llama |
| Page count | WARNING: likely exceeds 4-page body limit |
