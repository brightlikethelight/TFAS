# ICML 2026 MechInterp Workshop -- Template Conversion Guide

Status: Pre-conversion reference. Real template switch happens closer to submission.

## Submission Requirements

- **Format**: 4pp (short) or 8pp (long), excluding references and appendix
- **Templates accepted**: ICML 2026 or NeurIPS
- **Review**: Double-blind (no author names, no self-identifying references)

## Document Class Changes

Current:
```latex
\documentclass[accepted]{article}
\usepackage[margin=1in]{geometry}
```

Target (official ICML):
```latex
\documentclass{icml2026}
\usepackage{icml2026}  % may be bundled with documentclass
```

The `[accepted]` option on `article` is a no-op (article ignores unknown options). Remove it.

### Where to Get the ICML 2026 Style File

1. **Primary**: https://icml.cc/Conferences/2026/CallForPapers -- download the LaTeX style kit
2. **Overleaf**: ICML 2026 template should appear in the Overleaf gallery
3. **Fallback**: Use NeurIPS 2026 template (also accepted by the workshop)
4. **GitHub**: Check https://github.com/ICML/ or the workshop page for a direct link

The style kit typically includes: `icml2026.sty`, `icml2026.bst`, `fancyhdr.sty` (bundled), and an example `.tex`.

## Formatting Changes

| Property | Current (`article`) | ICML 2026 |
|----------|-------------------|-----------|
| Columns | 1 | 2 |
| Font size | 10pt (default) | 10pt |
| Base font | Computer Modern | Times (via `times` or `mathptmx`) |
| Margins | 1in all sides | ~0.75in sides, ~1in top/bottom |
| Column sep | N/A | ~0.25in |
| Line spacing | single | single |
| Abstract | `\begin{abstract}` | Same, but narrower (indented) |
| References | `natbib`/`plainnat` | `natbib`/`icml2026` bst |

### Specific Changes Required

1. **Remove** `\usepackage[margin=1in]{geometry}` -- ICML class sets its own margins
2. **Remove** `\usepackage{hyperref}` -- ICML class loads it internally (double-load causes option clashes)
3. **Tables**: Change `\small` to `\footnotesize` or `\scriptsize` for 2-column fit. Table 1 (behavioral results, 6 data columns) will need `\scriptsize` or abbreviation to fit column width (~3.25in).
4. **Figures**: `\linewidth` references automatically shrink to column width. Verify placeholder boxes still look right.
5. **`\paragraph{}`**: Works fine in ICML; no change needed.
6. **Bibliography**: Switch to `\bibliographystyle{icml2026}` if provided in the kit.
7. **`enumitem`**: ICML may conflict with `enumitem`. Test; if it breaks, switch `[leftmargin=*]` to manual `\setlength`.

## Double-Blind Anonymization

### Must change:
- `\author{...}` block -> `\author{Anonymous}` or use ICML's `\icmlauthor` with anonymous option
- Remove `\thanks{...}` with email address
- Remove affiliation block (Harvard, HUSAI)
- Acknowledgments section: either remove or replace with "Acknowledgments omitted for review"

### Must check:
- No self-citations as "our prior work" or "we previously showed"
- No references to "our GitHub repo" or identifiable URLs
- "Code and Data" paragraph: replace URL with "available upon acceptance"
- Remove or anonymize any W&B project links if present

Current author block to replace:
```latex
\author{%
  \textbf{Bright Liu}\textsuperscript{1}\thanks{...} \quad
  \textbf{[Co-authors]}\textsuperscript{1} \quad
  \textbf{[Advisor]}\textsuperscript{1,2} \\[6pt]
  \textsuperscript{1}Harvard Undergraduate AI Safety (HUSAI) \quad
  \textsuperscript{2}[Department] \\
}
```

The Acknowledgments section mentions Harvard and HUSAI by name -- must be removed or anonymized for submission.

## Page Count Estimate

Current paper: **10 pages** in single-column, 10pt, 1in margins.

ICML 2-column format is roughly 1.5-1.8x denser than single-column article (same font size, but 2 columns + tighter margins = more text per page). Estimated conversion:

- **Text content**: ~10 single-col pages -> ~6-7 double-col pages
- **Tables**: Table 1 (behavioral) is wide; will need `\scriptsize` or rotation. Table 2 (probe summary) fits at `\footnotesize`. Tables add ~0.3 pages each in 2-col.
- **Figures**: Figure 1 placeholder is currently large. Real figure at column width adds ~0.4 pages.
- **References**: ~40 citations -> ~1-1.5 pages (excluded from limit)

**Estimated total: 6-8 pages** (excluding references). Should fit the 8pp long format. Tight but feasible.

## If We Exceed 8 Pages

Priority order for moving to supplementary:

1. **Natural frequency analysis** (Discussion paragraph on ecological rationality) -- interesting but tangential. Move details to appendix, keep 1-sentence mention.
2. **Immune category control analysis** (Section 4.4 details) -- move the full transfer matrix discussion to appendix, keep the conclusion.
3. **Detailed Qwen think/no-think comparison** -- compress to 1 paragraph in main text, full analysis in supplementary.
4. **SAE feature analysis** (Section 4.6) -- currently compact; could move feature-level details to appendix.
5. **Table 1 expanded categories** (sunk cost, certainty, availability) -- these are "validated but 0% lure" categories. Could be appendix-only.

Do NOT move to supplementary (core contributions):
- Behavioral validation (Table 1 core 7 categories)
- Layer-wise probe results and the Llama vs R1-Distill gap
- Cross-prediction and specificity analysis
- Training vs inference dissociation (Qwen result)

## Two-Column Approximation (Intermediate Step)

Before obtaining the official style file, we compile with:
```latex
\documentclass[twocolumn]{article}
\usepackage[margin=1in]{geometry}
\setlength{\columnsep}{0.25in}
```

This gives a conservative estimate (ICML has slightly tighter margins, so the real template will be slightly more compact). See `paper/workshop_paper_icml.tex`.

## Checklist Before Submission

- [ ] Official ICML 2026 style file obtained and compiles
- [ ] All author information removed (double-blind)
- [ ] Acknowledgments anonymized
- [ ] Self-citations checked (no "we previously...")
- [ ] Tables fit in column width
- [ ] Figures render correctly at column width
- [ ] Page count <= 8 (excluding references)
- [ ] Bibliography style switched to ICML
- [ ] No `hyperref` double-load conflicts
- [ ] Supplementary material prepared if needed
