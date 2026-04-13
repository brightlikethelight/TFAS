# Confidence Paradigm Analysis (De Neys)

Model: Llama-3.1-8B-Instruct

R1-Distill excluded from confidence tests: `<think>` token drives 
first_token_prob to 1.0 and entropy to 0.0 uniformly (no variance).

## Overall conflict vs control

- **First token probability**: conflict 0.751 (SD 0.219) vs control 0.793 (SD 0.194)
  - Mann-Whitney U = 23885.0, p = 1.14e-02, r_rb = 0.135 *
- **Top-10 entropy**: conflict 0.948 (SD 0.704) vs control 0.817 (SD 0.656)
  - Mann-Whitney U = 30963.5, p = 2.29e-02, r_rb = -0.121 *

Conflict items elicit significantly lower first-token confidence than matched controls, consistent with the De Neys prediction that conflicting heuristic cues reduce output confidence.

## Per-category breakdown (first_token_prob)

| Category | Conflict M (SD) | Control M (SD) | U | p | r_rb | Sig |
|----------|----------------|---------------|---|---|------|-----|
| anchoring | 0.523 (0.303) | 0.851 (0.205) | 71 | 5.091e-04 | 0.645 | *** |
| arithmetic | 0.932 (0.032) | 0.832 (0.030) | 609 | 9.288e-09 | -0.949 | *** |
| availability | 0.911 (0.118) | 0.665 (0.187) | 201 | 2.622e-04 | -0.787 | *** |
| base_rate | 0.738 (0.163) | 0.848 (0.122) | 372 | 4.816e-03 | 0.393 | ** |
| certainty_effect | 0.682 (0.110) | 0.633 (0.138) | 129 | 5.069e-01 | -0.147 | n.s. |
| conjunction | 0.776 (0.152) | 0.769 (0.139) | 208 | 8.392e-01 | -0.040 | n.s. |
| crt | 0.481 (0.096) | 0.657 (0.268) | 272 | 8.684e-03 | 0.396 | ** |
| framing | 0.930 (0.053) | 0.966 (0.006) | 42 | 2.041e-05 | 0.790 | *** |
| loss_aversion | 0.983 (0.016) | 1.000 (0.000) | 0 | 3.383e-06 | 1.000 | *** |
| sunk_cost | 0.965 (0.059) | 0.999 (0.000) | 1 | 4.133e-06 | 0.991 | *** |
| syllogism | 0.606 (0.080) | 0.602 (0.069) | 337 | 6.415e-01 | -0.078 | n.s. |

## De Neys critical test: lured items vs matched controls

N lured-control pairs: 56
- Lured FTP: 0.730 (SD 0.155)
- Control FTP: 0.780 (SD 0.148)
- Fraction where lured < control: 62.5%
- Wilcoxon signed-rank (one-sided, lured < control): W = 523.0, p = 1.24e-02, r_rb = 0.345 *

**Result**: Lured items show significantly lower first-token confidence than their matched controls. This is the hallmark De Neys finding: conflict detection without resolution. The model's initial probability mass is disrupted by the conflict even on items where it ultimately gives the heuristic (wrong) answer.

### Entropy for lured items
- Fraction where lured entropy > control: 66.1%
- Wilcoxon signed-rank (one-sided, lured > control): W = 1103.0, p = 6.42e-03, r_rb = -0.382 **

## Confidence gap distribution

Across all 235 conflict items: mean gap = -0.1135, median = 0.0000, fraction positive (lure > correct) = 37.0%
- Lured items (56): mean gap = 0.4772, median = 0.5189
- Correct items (167): mean gap = -0.3065, median = -0.0000

