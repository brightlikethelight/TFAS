# S1/S2 Mechanistic Signatures -- Unified Report

**Generated**: {{ report.timestamp }}
**Git SHA**: `{{ report.git_sha }}`
**Models analyzed**: {{ report.models | join(', ') if report.models else 'none' }}

---

## 1. Executive Summary

{% if report.models %}
This report aggregates results from {{ report.models | length }} model(s) across the s1s2 mechanistic interpretability pipeline.
{% else %}
No models found in results. Run the pipeline first.
{% endif %}

### Hypothesis Verdicts

| Hypothesis | Verdict | Summary |
|------------|---------|---------|
{% for hid in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6'] %}
{% set h = report.hypotheses.get(hid, {}) %}
| **{{ hid }}** | **{{ h.get('verdict', 'N/A') }}** | {{ h.get('reason', 'Not evaluated') }} |
{% endfor %}

{% set verdicts = [] %}
{% for hid in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6'] %}
{% if report.hypotheses.get(hid, {}).get('verdict') == 'PASS' %}
{% if verdicts.append(hid) %}{% endif %}
{% endif %}
{% endfor %}
{% if verdicts | length == 6 %}
**Overall assessment**: Strong positive -- convergent mechanistic evidence across all workstreams.
{% elif verdicts | length >= 4 %}
**Overall assessment**: Moderate positive -- multiple lines of evidence converge.
{% elif verdicts | length >= 2 %}
**Overall assessment**: Partial positive -- some workstreams show supporting evidence.
{% elif verdicts | length >= 1 %}
**Overall assessment**: Weak positive -- limited evidence from {{ verdicts | join(', ') }}.
{% else %}
**Overall assessment**: No hypotheses pass at this time. Results may be partial or null.
{% endif %}

---

## 2. Behavioral Results

{% if report.behavioral %}
### Per-Model Accuracy

| Model | Conflict Accuracy | No-Conflict Accuracy | Lure Rate |
|-------|-------------------|----------------------|-----------|
{% for model, data in report.behavioral.items() %}
| {{ model }} | {{ "%.1f%%" | format(data.get('conflict_accuracy', 0) * 100) }} | {{ "%.1f%%" | format(data.get('no_conflict_accuracy', 0) * 100) }} | {{ "%.1f%%" | format(data.get('lure_rate', 0) * 100) }} |
{% endfor %}
{% else %}
*Behavioral results not yet available.*
{% endif %}

---

## 3. Probing Results

{% if probes_summary %}
### Peak Layer AUC per Model per Target

| Model | Target | Peak Layer | Peak AUC | Selectivity (pp) |
|-------|--------|------------|----------|-------------------|
{% for row in probes_summary %}
| {{ row.model }} | {{ row.target }} | {{ row.peak_layer }} | {{ "%.3f" | format(row.peak_auc) }} | {{ "%.1f" | format(row.selectivity_pp) }} |
{% endfor %}

{% set std_models = [] %}
{% set reas_models = [] %}
{% for row in probes_summary %}
{% if row.target == 'task_type' %}
{% set model_lower = row.model | lower %}
{% if 'r1' in model_lower %}
{% if reas_models.append(row) %}{% endif %}
{% else %}
{% if std_models.append(row) %}{% endif %}
{% endif %}
{% endif %}
{% endfor %}
{% if std_models and reas_models %}
### Standard vs Reasoning Model Comparison (task_type)

| Comparison | Standard AUC | Reasoning AUC | Delta |
|------------|--------------|---------------|-------|
{% for s in std_models %}
{% for r in reas_models %}
| {{ s.model }} vs {{ r.model }} | {{ "%.3f" | format(s.peak_auc) }} | {{ "%.3f" | format(r.peak_auc) }} | {{ "%+.3f" | format(r.peak_auc - s.peak_auc) }} |
{% endfor %}
{% endfor %}
{% endif %}
{% else %}
*Probing results not yet available.*
{% endif %}

---

## 4. SAE Results

{% if sae_summary %}
### Feature Counts

| Model | Significant (pre-falsification) | Significant (post-falsification) |
|-------|---------------------------------|----------------------------------|
{% for row in sae_summary %}
| {{ row.model }} | {{ row.n_significant_before_falsification }} | {{ row.n_significant_after_falsification }} |
{% endfor %}

### Top Features

{% for row in sae_summary %}
{% if row.top_5_features %}
**{{ row.model }}**:

| Layer | Feature ID | Log Fold Change | q-value | Effect Size | Auto-Interp |
|-------|------------|-----------------|---------|-------------|-------------|
{% for f in row.top_5_features %}
| {{ f.layer }} | {{ f.feature_id }} | {{ "%.2f" | format(f.log_fc) }} | {{ "%.1e" | format(f.q_value) }} | {{ "%.3f" | format(f.effect_size) }} | {{ f.auto_interp or '---' }} |
{% endfor %}

{% endif %}
{% endfor %}

> Volcano plots: see `figures/sae/{model}/layer_{NN}/volcano.png`
{% else %}
*SAE results not yet available.*
{% endif %}

---

## 5. Attention Results

{% if attention_summary %}
### S2-Specialized Heads

| Model | Total Heads | S2-Specialized | Proportion |
|-------|-------------|----------------|------------|
{% for row in attention_summary %}
| {{ row.model }} | {{ row.n_heads_total }} | {{ row.n_s2_specialized }} | {{ "%.1f%%" | format(row.proportion_s2 * 100) }} |
{% endfor %}

### Layer Distribution of S2-Specialized Heads

{% for row in attention_summary %}
{% if row.s2_layer_distribution %}
**{{ row.model }}**: layers {{ row.s2_layer_distribution | join(', ') }}
{% endif %}
{% endfor %}
{% else %}
*Attention results not yet available.*
{% endif %}

---

## 6. Geometry Results

{% if geometry_summary %}
### Peak Silhouette Score

| Model | Peak Layer | Silhouette | p-value | Intrinsic Dim |
|-------|------------|------------|---------|---------------|
{% for row in geometry_summary %}
| {{ row.model }} | {{ row.peak_layer }} | {{ "%.4f" | format(row.peak_silhouette) }} | {{ "%.1e" | format(row.p_value) }} | {{ "%.1f" | format(row.intrinsic_dimensionality) if row.intrinsic_dimensionality is not none else '---' }} |
{% endfor %}
{% else %}
*Geometry results not yet available.*
{% endif %}

---

## 7. Causal Results

{% if causal_summary %}
### Intervention Effects

| Model | Layer | Feature | Delta P(correct) (pp) | Random Delta (pp) | Capability OK |
|-------|-------|---------|----------------------|-------------------|---------------|
{% for row in causal_summary %}
| {{ row.model }} | {{ row.layer }} | {{ row.feature_id }} | {{ "%.1f" | format(row.best_delta_pp) if row.best_delta_pp is not none else '---' }} | {{ "%.1f" | format(row.random_delta_pp) if row.random_delta_pp is not none else '---' }} | {{ 'Yes' if row.capability_preserved else ('No' if row.capability_preserved is not none else '---') }} |
{% endfor %}
{% else %}
*Causal results not yet available.*
{% endif %}

---

## 8. Hypothesis Evaluation

{% for hid in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6'] %}
{% set h = report.hypotheses.get(hid, {}) %}
### {{ hid }}

- **Pre-registered criterion**: {{ h.get('criterion', 'N/A') }}
- **Verdict**: **{{ h.get('verdict', 'N/A') }}**
- **Explanation**: {{ h.get('reason', 'Not evaluated') }}

{% if h.get('per_model') %}
| Model | Key Metric | Value |
|-------|-----------|-------|
{% for model, metrics in h.get('per_model', {}).items() %}
{% for metric_name, metric_val in metrics.items() %}
| {{ model }} | {{ metric_name }} | {{ metric_val }} |
{% endfor %}
{% endfor %}
{% endif %}

{% endfor %}

---

*Report generated by `s1s2.report.generate_report()`.*
