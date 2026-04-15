# OLMo-32B Think Probe Results

Date: 2026-04-15 (pipeline completed 03:45 UTC)

## KEY NUMBER

**OLMo-32B Think peak probe AUC: 0.9978** (Layer 20, P0)

## Comparison: Instruct vs Think at 32B

| Metric               | Instruct          | Think             | Delta        |
|----------------------|-------------------|-------------------|--------------|
| Peak AUC (P0)        | 0.9999 (L20)      | 0.9978 (L20)      | -0.0021      |
| Mean AUC (P0, all L) | 0.9797            | 0.9736            | -0.0061      |
| Lure rate            | 19.6% (46/235)    | 0.4% (1/235)      | -19.2pp      |
| P2 (specificity)     | 0.5 (all layers)  | 0.5 (all layers)  | 0.0          |

## Is the "blurring" effect replicated at 32B?

**Barely.** The Think model's peak AUC drops by only 0.0021 relative to Instruct
(0.9978 vs 0.9999). This is a real but tiny effect -- both models have near-perfect
probe decodability of conflict/control status from pre-response activations.

For comparison:
- OLMo-7B: Instruct 0.9983 vs Think 0.9928, delta = 0.0055
- OLMo-32B: Instruct 0.9999 vs Think 0.9978, delta = 0.0021

The blurring effect is actually *smaller* at 32B than at 7B. Scaling up from 7B to
32B made the Think model's representations *more* decodable, not less. The reasoning
chain does not meaningfully obscure the vulnerability signal in the residual stream.

## Behavioral Results

### Instruct (no reasoning chain)
- Overall lure rate: 19.6%
- base_rate: 74% (26/35)
- conjunction: 50% (10/20)
- syllogism: 0% (0/25)

### Think (with reasoning chain)
- Overall lure rate: 0.4% (1/235, single sunk_cost item)
- base_rate: 0% (0/35) -- fixed by reasoning
- conjunction: 0% (0/20) -- fixed by reasoning
- syllogism: 0% (0/25) -- same as Instruct

Think dramatically reduces behavioral vulnerability (19.6% -> 0.4% lure rate) but the
probe still decodes conflict/control at 0.9978 AUC. The model "knows" it's facing a
cognitively loaded question even when it answers correctly via chain-of-thought.

## Layer-by-Layer AUC (P0, selected layers)

| Layer | Instruct | Think  | Delta   |
|-------|----------|--------|---------|
| L00   | 0.8618   | 0.8790 | +0.0172 |
| L04   | 0.9041   | 0.8737 | -0.0304 |
| L08   | 0.9187   | 0.8922 | -0.0265 |
| L12   | 0.9746   | 0.9482 | -0.0264 |
| L16   | 0.9905   | 0.9928 | +0.0023 |
| L20   | 0.9999   | 0.9978 | -0.0021 |
| L24   | 0.9973   | 0.9962 | -0.0011 |
| L28   | 0.9960   | 0.9972 | +0.0012 |
| L32   | 0.9967   | 0.9971 | +0.0004 |
| L40   | 0.9967   | 0.9966 | -0.0001 |
| L48   | 0.9970   | 0.9971 | +0.0001 |
| L56   | 0.9973   | 0.9971 | -0.0002 |
| L63   | 0.9969   | 0.9964 | -0.0005 |

The biggest Instruct-Think differences are in early/mid layers (L4-L12). By L16 both
models are at >0.99 AUC and remain there through the final layer. The Think model
catches up quickly -- the reasoning chain training doesn't fundamentally change the
representation structure in the upper layers.

## Cross-Scale Comparison (7B vs 32B)

| Model Pair      | Instruct Peak | Think Peak | Delta  |
|-----------------|---------------|------------|--------|
| OLMo-7B         | 0.9983 (L21)  | 0.9928 (L28) | 0.0055 |
| OLMo-32B        | 0.9999 (L20)  | 0.9978 (L20) | 0.0021 |

Both peak at similar relative depth (~31-33% of layers). The 32B models have even
higher absolute AUCs and smaller Instruct-Think gaps.

## Implication

The S1/S2 vulnerability signal is deeply encoded and survives reasoning-chain
training at both 7B and 32B scale. Chain-of-thought fixes behavioral outputs but
does not erase the internal representation of "this is a tricky question."
This is consistent with the monitoring-accessibility hypothesis: a probe-based
monitor can detect vulnerability even when the model answers correctly.

## Files

- Probes: `results/probes/olmo32b_think_vulnerable.json`
- Probes: `results/probes/olmo32b_instruct_vulnerable.json`
- Comparison: `results/probes/olmo32b_instruct_vs_think_comparison.json`
- Behavioral: `results/behavioral/olmo32b_think_ALL.json`
- Behavioral: `results/behavioral/olmo32b_instruct_ALL.json`
