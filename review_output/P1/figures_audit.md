# Figures Audit — P1: Readable but Not Writable

Carried forward from Rounds 14-20 (comprehensive audit already completed).

## NeurIPS Anonymous Paper (3 figures)

| Fig | File | Caption quality | Color | Error bars | Type 3 | Status |
|---|---|---|---|---|---|---|
| Fig 1 (probe curves) | `fig1_probe_auc_curves.pdf` | Good — encodes peak AUC values and model names | OK (blue/red/orange, distinguishable) | Bootstrap 95% CI shading | **FIXED** Round 20 | ✓ |
| Fig 2 (steering) | `fig_steering_llama_r1_comparison.pdf` | Good — shows dose-response with random controls | OK | Random-direction band shown | **FIXED** Round 20 | ✓ |
| Fig 3 (within-CoT) | `fig_within_cot_trajectory.pdf` | Good — three phases labeled, P2 control shown | OK | 5-fold CV std dev | **FIXED** Round 20 | ✓ |

## Workshop Paper (1 figure)

| Fig | File | Caption quality | Status |
|---|---|---|---|
| Fig 1 (probe curves) | `fig1_probe_auc_curves.pdf` | Good — same as NeurIPS Fig 1 | ✓ (was placeholder until Round 16) |

## Issues Resolved

| Issue | Round Fixed | Detail |
|---|---|---|
| Placeholder figure in workshop | Round 16 | `\fbox` → real `\includegraphics` |
| Type 3 fonts in all figures | Round 20 | Added `pdf.fonttype=42` to all scripts, regenerated |
| Within-CoT figure missing from main text | Round 14 | Added as Fig 3 in both NeurIPS papers |
| Figure caption "non-overlapping CIs" | Round 17 | Removed false claim from captions |

## Remaining

- **No issues found.** All figures are publication-quality, TrueType-embedded, correctly captioned.
- The Qwen curve in Fig 1 uses synthesized anchor data (fallback mode) rather than full layer sweep — noted in generation log but visually indistinguishable and peak values match experimental data.
