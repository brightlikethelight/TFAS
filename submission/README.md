# Submission Package

**Paper:** "What Changes When LLMs Learn to Reason? Mechanistic Signatures of Cognitive Bias Processing"

**Venue:** ICML 2026 MechInterp Workshop (8-page long format)

**Status:** Anonymous submission (double-blind review)

## Contents

```
submission/
  workshop_paper_anonymous.tex   Main paper (anonymized)
  supplementary.tex              Supplementary materials (anonymized)
  references.bib                 Bibliography
  README.md                      This file
  figures/
    fig1_probe_auc_curves.pdf
    fig2_behavioral_heatmap.pdf
    fig3_cross_prediction.pdf
    fig4_lure_distribution.pdf
    fig5_attention_entropy.pdf
    fig5_behavioral_extended.pdf
    fig7_confidence.pdf
    fig8_cross_model_transfer.pdf
    figure1_behavioral.pdf
    figure2_probe_curves.pdf
    figure3_cross_prediction.pdf
    figure3_heatmap.pdf
    figure4_lure_distribution.pdf
```

## Compilation

### Main paper

```bash
pdflatex workshop_paper_anonymous
bibtex workshop_paper_anonymous
pdflatex workshop_paper_anonymous
pdflatex workshop_paper_anonymous
```

### Supplementary

```bash
pdflatex supplementary
bibtex supplementary
pdflatex supplementary
pdflatex supplementary
```

### Notes

- Requires a standard LaTeX distribution (TeX Live 2024+ or equivalent).
- Required packages: `amsmath`, `amssymb`, `graphicx`, `booktabs`, `enumitem`,
  `hyperref`, `natbib`, `microtype`, `longtable`, `array`.
- The main paper currently uses placeholder figure boxes (`\fbox`) rather than
  `\includegraphics`. To include actual figures, uncomment the
  `\includegraphics` lines and update paths to point to `figures/`.
- When the official ICML workshop style file is available, replace
  `\documentclass[accepted]{article}` with `\documentclass{icml2026}` and
  remove the `geometry` package.
