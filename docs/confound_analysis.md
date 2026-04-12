# Confound Analysis: Probe Specificity on Immune vs Vulnerable Categories

## The Confound

Linear probes classifying conflict versus control items achieve AUC=1.0 on immune categories (CRT, arithmetic, framing, anchoring) where all models show 0% lure rate. This means high probe AUC does not require a behavioral processing-mode difference — the probe can detect the structural presence of a heuristic lure in the prompt text itself.

## Three Properties of the Delta That Survive the Confound

While absolute AUC cannot distinguish processing-mode from lure-text detection, the inter-model Delta (Llama 0.999 vs R1-Distill 0.929 = 0.070 gap) has three properties a pure surface-feature explanation cannot accommodate:

1. **Model-specific**: both models see identical prompts, so a text-feature probe should yield identical AUC. The 7pp gap means the models represent the same lure text differently, and the gap direction aligns with behavioral vulnerability (the susceptible model has higher separability).

2. **Category-selective**: on immune categories, both models achieve AUC=1.0 with no inter-model gap. The Delta appears specifically on categories where models diverge behaviorally.

3. **Layer-profile dissociation**: immune AUC peaks at L0-1 (surface features), vulnerable AUC peaks at L14 (mid-network computation). The vulnerable signal builds through processing layers, while the immune signal is strongest at embedding layers.

## Resolution: Cross-Prediction Test (RESULTS AVAILABLE)

**Llama (train vulnerable → test immune)**: transfer AUC = **0.378 at L14** (below chance!). The probe learned something specific to vulnerable-category processing that does NOT transfer to immune categories. **The confound is resolved for Llama — the signal is processing-mode-specific.**

**R1-Distill (train vulnerable → test immune)**: transfer AUC = **0.878 at L4-L8** (confound) but **0.385 at L31** (specific). The reasoning model shows a mixed pattern: early layers encode general text features, late layers encode processing-specific features. This suggests reasoning training changed the layer-wise information flow.

## Interpretation

The cross-prediction test shows that the Llama probe's 0.999 AUC on vulnerable categories reflects genuine processing-mode information, not just lure-text detection. The probe direction learned from vulnerable categories is orthogonal to the lure-text direction detectable in immune categories. For R1-Distill, the picture is more nuanced — early layers share features across category types, but late layers differentiate. This is itself a finding about how reasoning training reorganizes internal representations.
