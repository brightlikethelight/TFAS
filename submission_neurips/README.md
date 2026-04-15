# What Changes When LLMs Learn to Reason? Mechanistic Signatures of Cognitive Bias Processing

NeurIPS 2026 Main Conference Submission (Anonymous)

## Files

- `neurips_paper_anonymous.pdf` -- the PDF to upload to OpenReview
- `checklist.tex` -- NeurIPS paper checklist (also compiled into the PDF)

## Compilation

Requires a TeX Live 2025+ installation with pdflatex and bibtex.

```bash
cd paper/
pdflatex neurips_paper_anonymous.tex
bibtex neurips_paper_anonymous
pdflatex neurips_paper_anonymous.tex
pdflatex neurips_paper_anonymous.tex
```

The paper uses `neurips_2026.sty` (anonymous mode) and references `references.bib`.
Figures are sourced from `../figures/`.

## Supplementary Materials

The appendix is included in the main PDF (pages 10--17) and does not count toward
the 9-page body limit. It contains:

- Appendix A: Within-CoT Probing Details
- Appendix B: Scale Analysis Full Results
- Appendix C: SAE Features and Attention Entropy
- Appendix D: Multi-Seed Robustness

The NeurIPS paper checklist is also compiled into the PDF.

## Code Availability

Code, benchmark, and activation data will be released upon acceptance.
The codebase includes the full analysis pipeline (probes, SAE features,
attention entropy, causal steering) and the CogBias-S1S2 benchmark builder.
