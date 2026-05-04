# Reviewer Red Team — P1: Readable but Not Writable

## NeurIPS 2026 — Simulated Reviews

### R1 (Senior Empiricist): Score 5/10, Confidence 4/5
**Key weaknesses:**
1. R1 steering "not writable" is underpowered: 6% baseline on N=80, floor effect may swallow signal
2. Held-in steering evaluation (probe trained on same items used for steering test)
3. No multi-layer simultaneous steering for R1
4. Alpha not calibrated to activation norms across models/layers

### R2 (Theory Skeptic): Score 4/10, Confidence 4/5
**Key weaknesses:**
1. Probe detects stimulus type (conflict/no-conflict), NOT processing mode — framing is misleading
2. Cross-prediction on immune categories is tautological (items processed identically → of course no transfer)
3. Qwen cross-mode is a positional confound the paper itself acknowledges
4. "Readable but not writable" vs "probe direction misaligned with true causal direction" — not distinguished

### R3 (Domain Expert Student): Score 7/10, Confidence 3/5
**Key weaknesses:**
1. Post-hoc vulnerable/immune split is circular
2. No CoT prompting baseline (does "think step by step" in Llama reduce lures?)
3. De Neys parallel overstated (no CIs on first-token probability, no calibration analysis)
4. Conjunction 95% lure rate in Qwen needs per-item breakdown

### Workshop WR1 (Circuits): Score 5/10
- No circuit analysis; paper is representation engineering, not mechanistic interpretability
- SAE features listed but not causally tested
- "Vestigial echo" is verbal hypothesis, not mechanism

### Workshop WR2 (Safety): Score 7/10
- Good monitoring implications but undeveloped
- No adversarial robustness discussion
- Only cognitive biases tested (narrow threat model)

## Top 5 Rebuttal Priorities

| # | Weakness | Mitigation | Can fix before deadline? |
|---|---|---|---|
| 1 | R1 steering underpowered (floor effect) | Power analysis / simulation showing minimum detectable effect on N=80 with 6% baseline | CPU-only, ~1h |
| 2 | No CoT prompting baseline | Run Llama with "Let's think step by step" — report lure rates | GPU needed, ~2h |
| 3 | Probe detects stimulus, not processing mode | Train behavioral-outcome probe (correct vs lure on conflict items) | CPU if cached, ~1h |
| 4 | Alpha not calibrated to norms | Report mean L2 norms per layer per model | CPU, <30min |
| 5 | Llama/R1 probe cosine similarity | One number from existing probe weights | CPU, <10min |
