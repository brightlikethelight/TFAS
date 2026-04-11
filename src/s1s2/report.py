"""Results aggregation and unified reporting for the s1s2 project.

Reads per-workstream JSON outputs from the results/ tree and produces a
unified Report object that can be serialized to JSON or rendered to
Markdown via a Jinja2 template.

Each workstream's loader tolerates missing directories gracefully — the
report is useful even when only a subset of workstreams has finished.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from beartype import beartype
from jinja2 import Environment, FileSystemLoader

from s1s2.utils.logging import get_logger

logger = get_logger("s1s2.report")

# Hypothesis thresholds from the pre-registration.
_H1_AUC_THRESHOLD = 0.6
_H1_SELECTIVITY_THRESHOLD = 5.0  # percentage points
_H1_MIN_MODELS = 2
_H2_CI_EXCLUDES_ZERO = True  # lower bound > 0
_H3_MIN_FEATURES = 5
_H3_EFFECT_SIZE_THRESHOLD = 0.3
_H4_DELTA_P_THRESHOLD = 15.0  # percentage points
_H4_RANDOM_THRESHOLD = 3.0  # percentage points
_H5_HEAD_PROPORTION_THRESHOLD = 0.05
_H6_MIN_MODELS = 2

TEMPLATE_DIR = Path(__file__).parent
TEMPLATE_NAME = "report_template.md"


# ---------------------------------------------------------------------------
# Git SHA helper
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Return the short git SHA, or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Per-workstream loaders
# ---------------------------------------------------------------------------


@beartype
def _load_behavioral(results_dir: Path) -> dict[str, Any]:
    """Load behavioral results from the HDF5-backed extraction pipeline.

    The behavioral data lives embedded in probes results (the probes CLI
    logs per-model accuracy). We also check for a standalone behavioral
    summary if one exists.
    """
    out: dict[str, Any] = {}
    behavioral_dir = results_dir / "behavioral"
    if behavioral_dir.is_dir():
        for f in sorted(behavioral_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                model = data.get("model", f.stem)
                out[model] = data
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("skipping behavioral file %s: %s", f, e)
    return out


@beartype
def _load_probes(results_dir: Path) -> dict[str, Any]:
    """Load probe results.

    Expected layout: results/probes/{model}_{target}_layer{NN}_{position}.json

    Returns a nested dict: model -> target -> list of layer results.
    """
    probes_dir = results_dir / "probes"
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if not probes_dir.is_dir():
        return out
    pattern = re.compile(
        r"^(?P<model>.+?)_(?P<target>task_type|correctness|bias_susceptible)"
        r"_layer(?P<layer>\d+)_(?P<position>\w+)\.json$"
    )
    for f in sorted(probes_dir.glob("*.json")):
        m = pattern.match(f.name)
        if m is None:
            continue
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            logger.warning("skipping malformed probe file %s", f)
            continue
        model = m.group("model")
        target = m.group("target")
        out.setdefault(model, {}).setdefault(target, []).append(data)
    return out


@beartype
def _load_sae(results_dir: Path) -> dict[str, Any]:
    """Load SAE feature_analysis.json files.

    Expected layout: results/sae/{model}/layer_{NN}/feature_analysis.json
    Returns: model -> list of per-layer dicts.
    """
    sae_dir = results_dir / "sae"
    out: dict[str, list[dict[str, Any]]] = {}
    if not sae_dir.is_dir():
        return out
    for model_dir in sorted(sae_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        layers: list[dict[str, Any]] = []
        for layer_dir in sorted(model_dir.iterdir()):
            fa = layer_dir / "feature_analysis.json"
            if fa.is_file():
                try:
                    layers.append(json.loads(fa.read_text()))
                except json.JSONDecodeError:
                    logger.warning("skipping malformed SAE file %s", fa)
        if layers:
            out[model] = layers
    return out


@beartype
def _load_attention(results_dir: Path) -> dict[str, Any]:
    """Load attention head_classifications.json files.

    Expected layout: results/attention/{model}/head_classifications.json
    """
    attn_dir = results_dir / "attention"
    out: dict[str, dict[str, Any]] = {}
    if not attn_dir.is_dir():
        return out
    for model_dir in sorted(attn_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        hc = model_dir / "head_classifications.json"
        if hc.is_file():
            try:
                out[model_dir.name] = json.loads(hc.read_text())
            except json.JSONDecodeError:
                logger.warning("skipping malformed attention file %s", hc)
    return out


@beartype
def _load_geometry(results_dir: Path) -> dict[str, Any]:
    """Load geometry results.

    Expected layout: results/geometry/{model}/layer_{NN}/geometry.json
    Returns: model -> list of per-layer dicts.
    """
    geo_dir = results_dir / "geometry"
    out: dict[str, list[dict[str, Any]]] = {}
    if not geo_dir.is_dir():
        return out
    for model_dir in sorted(geo_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        layers: list[dict[str, Any]] = []
        for layer_dir in sorted(model_dir.iterdir()):
            gf = layer_dir / "geometry.json"
            if gf.is_file():
                try:
                    layers.append(json.loads(gf.read_text()))
                except json.JSONDecodeError:
                    logger.warning("skipping malformed geometry file %s", gf)
        if layers:
            out[model] = layers
    return out


@beartype
def _load_causal(results_dir: Path) -> dict[str, Any]:
    """Load causal intervention results.

    Expected layout:
        results/causal/{model}/layer_{NN}_feature_{FFFF}/intervention_results.json
    Returns: model -> list of per-cell dicts.
    """
    causal_dir = results_dir / "causal"
    out: dict[str, list[dict[str, Any]]] = {}
    if not causal_dir.is_dir():
        return out
    for model_dir in sorted(causal_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        cells: list[dict[str, Any]] = []
        for cell_dir in sorted(model_dir.iterdir()):
            ir = cell_dir / "intervention_results.json"
            if ir.is_file():
                try:
                    cells.append(json.loads(ir.read_text()))
                except json.JSONDecodeError:
                    logger.warning("skipping malformed causal file %s", ir)
        if cells:
            out[model] = cells
    return out


# ---------------------------------------------------------------------------
# Hypothesis evaluation
# ---------------------------------------------------------------------------


@beartype
def _evaluate_h1(probes: dict[str, Any]) -> dict[str, Any]:
    """H1: Linear decodability of S1/S2 from residual stream.

    Criterion: peak-layer AUC > 0.6 AND selectivity > 5pp in >= 2 models.
    """
    if not probes:
        return {"verdict": "INCONCLUSIVE", "reason": "No probe results available."}
    passing_models: list[str] = []
    per_model: dict[str, dict[str, Any]] = {}
    for model, targets in probes.items():
        task_type_results = targets.get("task_type", [])
        if not task_type_results:
            per_model[model] = {"peak_auc": None, "selectivity": None}
            continue
        best_auc = 0.0
        best_selectivity = 0.0
        for lr in task_type_results:
            # Navigate the nested probe result structure.
            summary = _extract_probe_summary(lr)
            auc = summary.get("auc", 0.0)
            selectivity = summary.get("selectivity_pp", 0.0)
            if auc > best_auc:
                best_auc = auc
                best_selectivity = selectivity
        per_model[model] = {"peak_auc": best_auc, "selectivity": best_selectivity}
        if best_auc > _H1_AUC_THRESHOLD and best_selectivity > _H1_SELECTIVITY_THRESHOLD:
            passing_models.append(model)
    n_pass = len(passing_models)
    if n_pass >= _H1_MIN_MODELS:
        verdict = "PASS"
        reason = (
            f"{n_pass} model(s) exceed AUC>{_H1_AUC_THRESHOLD} "
            f"with selectivity>{_H1_SELECTIVITY_THRESHOLD}pp."
        )
    elif n_pass > 0:
        verdict = "INCONCLUSIVE"
        reason = (
            f"Only {n_pass} model(s) pass (need {_H1_MIN_MODELS}). "
            "Suggestive but below threshold."
        )
    else:
        verdict = "FAIL"
        reason = "No models exceed the AUC and selectivity thresholds."
    return {
        "verdict": verdict,
        "reason": reason,
        "criterion": (
            f"Peak-layer AUC > {_H1_AUC_THRESHOLD} AND selectivity > "
            f"{_H1_SELECTIVITY_THRESHOLD}pp in >= {_H1_MIN_MODELS} models"
        ),
        "per_model": per_model,
        "passing_models": passing_models,
    }


@beartype
def _evaluate_h2(probes: dict[str, Any]) -> dict[str, Any]:
    """H2: Reasoning training amplifies S1/S2 separation.

    Criterion: paired bootstrap CI for AUC(R1-Distill) - AUC(Llama) excludes zero,
    with reasoning model higher.
    """
    if not probes:
        return {"verdict": "INCONCLUSIVE", "reason": "No probe results available."}
    # Look for the architecture-matched pair.
    reasoning_key = None
    standard_key = None
    for model in probes:
        lower = model.lower()
        if "r1-distill-llama" in lower or "r1_distill_llama" in lower:
            reasoning_key = model
        elif "llama" in lower and "r1" not in lower:
            standard_key = model
    if reasoning_key is None or standard_key is None:
        return {
            "verdict": "INCONCLUSIVE",
            "reason": "Cannot find matched Llama/R1-Distill-Llama pair in probe results.",
        }
    r_auc = _peak_auc_for_model(probes, reasoning_key)
    s_auc = _peak_auc_for_model(probes, standard_key)
    if r_auc is None or s_auc is None:
        return {
            "verdict": "INCONCLUSIVE",
            "reason": "Missing task_type probes for one or both models.",
        }
    delta = r_auc - s_auc
    # Check if any result carries a bootstrap CI for the delta.
    ci_lower = _extract_delta_ci_lower(probes, reasoning_key, standard_key)
    if ci_lower is not None and ci_lower > 0:
        verdict = "PASS"
        reason = (
            f"AUC delta = {delta:.3f} (reasoning - standard), "
            f"CI lower bound = {ci_lower:.3f} > 0."
        )
    elif delta > 0:
        verdict = "INCONCLUSIVE"
        reason = (
            f"AUC delta = {delta:.3f} favoring reasoning model, "
            "but no CI data to confirm exclusion of zero."
        )
    else:
        verdict = "FAIL"
        reason = f"AUC delta = {delta:.3f}; standard model has equal or higher AUC."
    return {
        "verdict": verdict,
        "reason": reason,
        "criterion": "95% paired bootstrap CI for AUC(reasoning) - AUC(standard) excludes zero",
        "reasoning_model": reasoning_key,
        "standard_model": standard_key,
        "reasoning_peak_auc": r_auc,
        "standard_peak_auc": s_auc,
        "delta_auc": delta,
    }


@beartype
def _evaluate_h3(sae: dict[str, Any]) -> dict[str, Any]:
    """H3: SAE features differentially activate on S1 vs S2.

    Criterion: >= 5 features significant after FDR + falsification + |r_rb| > 0.3
    in at least 1 model.
    """
    if not sae:
        return {"verdict": "INCONCLUSIVE", "reason": "No SAE results available."}
    per_model: dict[str, dict[str, Any]] = {}
    any_pass = False
    for model, layers in sae.items():
        total_after_falsification = 0
        total_significant = 0
        for layer_data in layers:
            total_significant += layer_data.get("n_features_significant", 0)
            total_after_falsification += layer_data.get("n_features_after_falsification", 0)
        per_model[model] = {
            "n_significant": total_significant,
            "n_after_falsification": total_after_falsification,
        }
        if total_after_falsification >= _H3_MIN_FEATURES:
            any_pass = True
    if any_pass:
        verdict = "PASS"
        reason = (
            f"At least one model has >= {_H3_MIN_FEATURES} features "
            "surviving FDR + falsification."
        )
    else:
        verdict = "FAIL"
        reason = (
            f"No model has >= {_H3_MIN_FEATURES} features after falsification. "
            f"Counts: {per_model}"
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "criterion": (
            f">= {_H3_MIN_FEATURES} features with |r_rb| > {_H3_EFFECT_SIZE_THRESHOLD} "
            "after BH-FDR + Ma et al. falsification, in >= 1 model"
        ),
        "per_model": per_model,
    }


@beartype
def _evaluate_h4(causal: dict[str, Any]) -> dict[str, Any]:
    """H4: Causal interventions shift behavior.

    Criterion: delta P(correct) > 15pp for S2-steering,
    delta P(correct) < 3pp for random controls.
    """
    if not causal:
        return {"verdict": "INCONCLUSIVE", "reason": "No causal results available."}
    best_delta: float | None = None
    best_random_delta: float | None = None
    best_model = ""
    best_feature = ""
    for model, cells in causal.items():
        for cell in cells:
            curve = cell.get("curve", {})
            # Find best alpha's delta.
            s2_delta = curve.get("best_delta_conflict_pp", curve.get("best_delta_pp"))
            random_delta = curve.get("random_mean_delta_conflict_pp", curve.get("random_delta_pp"))
            if s2_delta is not None and (best_delta is None or s2_delta > best_delta):
                best_delta = s2_delta
                best_random_delta = random_delta
                best_model = model
                best_feature = f"L{cell.get('layer', '?')}/F{cell.get('feature_id', '?')}"
    if best_delta is None:
        return {"verdict": "INCONCLUSIVE", "reason": "Could not extract delta from causal results."}
    random_ok = best_random_delta is not None and best_random_delta < _H4_RANDOM_THRESHOLD
    if best_delta > _H4_DELTA_P_THRESHOLD and random_ok:
        verdict = "PASS"
        reason = (
            f"Best S2-steering delta = {best_delta:.1f}pp > {_H4_DELTA_P_THRESHOLD}pp, "
            f"random control delta = {best_random_delta:.1f}pp < {_H4_RANDOM_THRESHOLD}pp."
        )
    elif best_delta > _H4_DELTA_P_THRESHOLD:
        verdict = "INCONCLUSIVE"
        reason = (
            f"S2-steering delta = {best_delta:.1f}pp passes, "
            "but random control data missing or too high."
        )
    else:
        verdict = "FAIL"
        reason = f"Best S2-steering delta = {best_delta:.1f}pp <= {_H4_DELTA_P_THRESHOLD}pp."
    return {
        "verdict": verdict,
        "reason": reason,
        "criterion": (
            f"Delta P(correct) > {_H4_DELTA_P_THRESHOLD}pp (S2-steering) AND "
            f"< {_H4_RANDOM_THRESHOLD}pp (random control)"
        ),
        "best_model": best_model,
        "best_feature": best_feature,
        "best_s2_delta_pp": best_delta,
        "best_random_delta_pp": best_random_delta,
    }


@beartype
def _evaluate_h5(attention: dict[str, Any]) -> dict[str, Any]:
    """H5: Attention entropy differentiates S1/S2 in specific heads.

    Criterion: >= 5% of heads (KV-group) S2-specialized in >= 1 model.
    """
    if not attention:
        return {"verdict": "INCONCLUSIVE", "reason": "No attention results available."}
    any_pass = False
    per_model: dict[str, dict[str, Any]] = {}
    for model, data in attention.items():
        # Look at kv_group_classifications first (more conservative), fall back to head.
        classifications = data.get("kv_group_classifications") or data.get(
            "head_classifications", []
        )
        n_total = len(classifications)
        n_s2 = sum(
            1
            for c in classifications
            if c.get("classification") == "s2_specialized"
            or c.get("classification") == "S2-specialized"
        )
        proportion = n_s2 / n_total if n_total > 0 else 0.0
        per_model[model] = {
            "n_heads_total": n_total,
            "n_s2_specialized": n_s2,
            "proportion_s2": proportion,
        }
        if proportion >= _H5_HEAD_PROPORTION_THRESHOLD:
            any_pass = True
    if any_pass:
        verdict = "PASS"
        reason = f"At least one model has >= {_H5_HEAD_PROPORTION_THRESHOLD*100:.0f}% S2-specialized heads."
    else:
        verdict = "FAIL"
        reason = f"No model reaches {_H5_HEAD_PROPORTION_THRESHOLD*100:.0f}% S2-specialized heads."
    return {
        "verdict": verdict,
        "reason": reason,
        "criterion": (
            f">= {_H5_HEAD_PROPORTION_THRESHOLD*100:.0f}% of KV-group heads "
            "S2-specialized in >= 1 model"
        ),
        "per_model": per_model,
    }


@beartype
def _evaluate_h6(geometry: dict[str, Any]) -> dict[str, Any]:
    """H6: Geometric separability of S1/S2 representations.

    Criterion: silhouette > 0 with p < 0.05 (FDR-corrected) at peak layer
    in >= 2 models.
    """
    if not geometry:
        return {"verdict": "INCONCLUSIVE", "reason": "No geometry results available."}
    passing_models: list[str] = []
    per_model: dict[str, dict[str, Any]] = {}
    for model, layers in geometry.items():
        best_silhouette = -1.0
        best_p = 1.0
        for layer_data in layers:
            sil = layer_data.get("silhouette", {})
            score = sil.get("score", sil.get("silhouette_score", -1.0))
            p = sil.get("p_value", sil.get("permutation_p", 1.0))
            if score > best_silhouette:
                best_silhouette = score
                best_p = p
        per_model[model] = {"peak_silhouette": best_silhouette, "p_value": best_p}
        if best_silhouette > 0 and best_p < 0.05:
            passing_models.append(model)
    n_pass = len(passing_models)
    if n_pass >= _H6_MIN_MODELS:
        verdict = "PASS"
        reason = f"{n_pass} model(s) show significant geometric separation."
    elif n_pass > 0:
        verdict = "INCONCLUSIVE"
        reason = f"Only {n_pass} model(s) pass (need {_H6_MIN_MODELS})."
    else:
        verdict = "FAIL"
        reason = "No models show significant geometric separation."
    return {
        "verdict": verdict,
        "reason": reason,
        "criterion": (
            f"Silhouette > 0 with p < 0.05 (FDR-corrected) at peak layer "
            f"in >= {_H6_MIN_MODELS} models"
        ),
        "per_model": per_model,
        "passing_models": passing_models,
    }


# ---------------------------------------------------------------------------
# Helpers for extracting nested probe metrics
# ---------------------------------------------------------------------------


def _extract_probe_summary(layer_result: dict[str, Any]) -> dict[str, float]:
    """Extract AUC and selectivity from a probe layer result dict.

    Handles both the raw LayerResult.to_json() format and simpler test
    formats where summary is at the top level.
    """
    # Try nested path first: probes -> l2_logistic -> summary -> mean_auc
    probes = layer_result.get("probes", {})
    for probe_name in ("l2_logistic", "logistic"):
        probe = probes.get(probe_name, {})
        summary = probe.get("summary", {})
        if "mean_auc" in summary:
            auc = summary["mean_auc"]
            # Selectivity: summary may have it directly or we compute it.
            control = probe.get("control_metrics", [])
            control_auc = 0.5
            if control:
                control_auc = sum(cm.get("auc", 0.5) for cm in control) / len(control)
            selectivity = (auc - control_auc) * 100
            return {"auc": auc, "selectivity_pp": selectivity}
    # Simpler flat format used in tests.
    if "summary" in layer_result:
        s = layer_result["summary"]
        return {
            "auc": s.get("mean_auc", s.get("auc", 0.0)),
            "selectivity_pp": s.get("selectivity_pp", s.get("selectivity", 0.0)),
        }
    # Absolute fallback.
    return {
        "auc": layer_result.get("auc", layer_result.get("mean_auc", 0.0)),
        "selectivity_pp": layer_result.get("selectivity_pp", 0.0),
    }


def _peak_auc_for_model(probes: dict[str, Any], model: str) -> float | None:
    """Get the peak task_type AUC across layers for a model."""
    targets = probes.get(model, {})
    task_type = targets.get("task_type", [])
    if not task_type:
        return None
    best = 0.0
    for lr in task_type:
        s = _extract_probe_summary(lr)
        best = max(best, s.get("auc", 0.0))
    return best


def _extract_delta_ci_lower(probes: dict[str, Any], reasoning: str, standard: str) -> float | None:
    """Try to find a pre-computed bootstrap CI lower bound for the delta AUC.

    This is stored in some probe result formats but not all.
    """
    # Walk reasoning model results looking for a comparison field.
    targets = probes.get(reasoning, {})
    for _target, layers in targets.items():
        for lr in layers:
            comp = lr.get("comparison", lr.get("delta_ci", {}))
            if isinstance(comp, dict) and "ci_lower" in comp:
                return float(comp["ci_lower"])
    return None


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class Report:
    """Unified report aggregating all workstream results."""

    timestamp: str
    git_sha: str
    models: list[str]
    behavioral: dict[str, Any] = field(default_factory=dict)
    probes: dict[str, Any] = field(default_factory=dict)
    sae: dict[str, Any] = field(default_factory=dict)
    attention: dict[str, Any] = field(default_factory=dict)
    geometry: dict[str, Any] = field(default_factory=dict)
    causal: dict[str, Any] = field(default_factory=dict)
    hypotheses: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize the full report to a JSON string."""
        return json.dumps(self._as_dict(), indent=2, default=str)

    def to_markdown(self) -> str:
        """Render the report to Markdown via the Jinja2 template."""
        template_dir = TEMPLATE_DIR
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        tmpl = env.get_template(TEMPLATE_NAME)
        return tmpl.render(report=self, **self._template_helpers())

    def save(self, output_dir: Path) -> None:
        """Write both JSON and Markdown reports to output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "report.json"
        json_path.write_text(self.to_json())
        md_path = output_dir / "report.md"
        md_path.write_text(self.to_markdown())
        logger.info("Report saved to %s and %s", json_path, md_path)

    def _as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "git_sha": self.git_sha,
            "models": self.models,
            "behavioral": self.behavioral,
            "probes": self.probes,
            "sae": self.sae,
            "attention": self.attention,
            "geometry": self.geometry,
            "causal": self.causal,
            "hypotheses": self.hypotheses,
        }

    def _template_helpers(self) -> dict[str, Any]:
        """Extra values/functions for the template."""
        return {
            "probes_summary": self._probes_summary(),
            "sae_summary": self._sae_summary(),
            "attention_summary": self._attention_summary(),
            "geometry_summary": self._geometry_summary(),
            "causal_summary": self._causal_summary(),
        }

    def _probes_summary(self) -> list[dict[str, Any]]:
        """Flatten probes into a list of per-model-per-target peak results."""
        rows: list[dict[str, Any]] = []
        for model, targets in self.probes.items():
            for target, layers in targets.items():
                best_auc = 0.0
                best_layer = -1
                best_selectivity = 0.0
                for lr in layers:
                    s = _extract_probe_summary(lr)
                    auc = s.get("auc", 0.0)
                    if auc > best_auc:
                        best_auc = auc
                        best_layer = lr.get("layer", -1)
                        best_selectivity = s.get("selectivity_pp", 0.0)
                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "peak_layer": best_layer,
                        "peak_auc": best_auc,
                        "selectivity_pp": best_selectivity,
                    }
                )
        return rows

    def _sae_summary(self) -> list[dict[str, Any]]:
        """Per-model SAE summary (aggregated across layers)."""
        rows: list[dict[str, Any]] = []
        for model, layers in self.sae.items():
            n_sig = sum(ld.get("n_features_significant", 0) for ld in layers)
            n_after = sum(ld.get("n_features_after_falsification", 0) for ld in layers)
            # Collect top features across layers.
            top_feats: list[dict[str, Any]] = []
            for ld in layers:
                for feat in ld.get("top_features", []):
                    top_feats.append(
                        {
                            "layer": ld.get("layer", -1),
                            "feature_id": feat.get("feature_id", -1),
                            "log_fc": feat.get("log_fc", 0.0),
                            "q_value": feat.get("q_value", 1.0),
                            "effect_size": feat.get("effect_size", 0.0),
                            "auto_interp": feat.get(
                                "auto_interp_label", feat.get("auto_interp", "")
                            ),
                        }
                    )
            # Sort by absolute effect size descending, take top 5.
            top_feats.sort(key=lambda x: abs(x.get("effect_size", 0.0)), reverse=True)
            rows.append(
                {
                    "model": model,
                    "n_significant_before_falsification": n_sig,
                    "n_significant_after_falsification": n_after,
                    "top_5_features": top_feats[:5],
                }
            )
        return rows

    def _attention_summary(self) -> list[dict[str, Any]]:
        """Per-model attention summary."""
        rows: list[dict[str, Any]] = []
        for model, data in self.attention.items():
            classifications = data.get("kv_group_classifications") or data.get(
                "head_classifications", []
            )
            n_total = len(classifications)
            n_s2 = sum(
                1
                for c in classifications
                if c.get("classification") in ("s2_specialized", "S2-specialized")
            )
            proportion = n_s2 / n_total if n_total > 0 else 0.0
            # Layer distribution of S2 heads.
            s2_layers = [
                c.get("layer", -1)
                for c in classifications
                if c.get("classification") in ("s2_specialized", "S2-specialized")
            ]
            rows.append(
                {
                    "model": model,
                    "n_heads_total": n_total,
                    "n_s2_specialized": n_s2,
                    "proportion_s2": proportion,
                    "s2_layer_distribution": s2_layers,
                }
            )
        return rows

    def _geometry_summary(self) -> list[dict[str, Any]]:
        """Per-model geometry summary (peak silhouette, CKA, intrinsic dim)."""
        rows: list[dict[str, Any]] = []
        for model, layers in self.geometry.items():
            best_sil = -1.0
            best_p = 1.0
            best_layer = -1
            best_id = None
            for ld in layers:
                sil = ld.get("silhouette", {})
                score = sil.get("score", sil.get("silhouette_score", -1.0))
                p = sil.get("p_value", sil.get("permutation_p", 1.0))
                if score > best_sil:
                    best_sil = score
                    best_p = p
                    best_layer = ld.get("layer", -1)
                    best_id = ld.get("intrinsic_dim_two_nn")
            rows.append(
                {
                    "model": model,
                    "peak_layer": best_layer,
                    "peak_silhouette": best_sil,
                    "p_value": best_p,
                    "intrinsic_dimensionality": best_id,
                }
            )
        return rows

    def _causal_summary(self) -> list[dict[str, Any]]:
        """Per-model causal summary."""
        rows: list[dict[str, Any]] = []
        for model, cells in self.causal.items():
            for cell in cells:
                curve = cell.get("curve", {})
                rows.append(
                    {
                        "model": model,
                        "layer": cell.get("layer", -1),
                        "feature_id": cell.get("feature_id", -1),
                        "best_delta_pp": curve.get(
                            "best_delta_conflict_pp", curve.get("best_delta_pp")
                        ),
                        "random_delta_pp": curve.get(
                            "random_mean_delta_conflict_pp", curve.get("random_delta_pp")
                        ),
                        "capability_preserved": (
                            all(
                                not c.get("exceeded_max_drop", True)
                                for c in cell.get("capability", [])
                            )
                            if cell.get("capability")
                            else None
                        ),
                    }
                )
        return rows


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class ResultsAggregator:
    """Reads per-workstream JSON results and builds a unified Report."""

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)

    @classmethod
    def from_directory(cls, results_dir: str | Path) -> "ResultsAggregator":
        """Factory constructor from a path."""
        return cls(Path(results_dir))

    def aggregate(self) -> Report:
        """Load all available results and produce a Report."""
        behavioral = _load_behavioral(self.results_dir)
        probes = _load_probes(self.results_dir)
        sae = _load_sae(self.results_dir)
        attention = _load_attention(self.results_dir)
        geometry = _load_geometry(self.results_dir)
        causal = _load_causal(self.results_dir)

        # Collect all model names seen across workstreams.
        all_models: set[str] = set()
        all_models.update(behavioral.keys())
        all_models.update(probes.keys())
        all_models.update(sae.keys())
        all_models.update(attention.keys())
        all_models.update(geometry.keys())
        all_models.update(causal.keys())
        models = sorted(all_models)

        # Evaluate hypotheses.
        hypotheses = {
            "H1": _evaluate_h1(probes),
            "H2": _evaluate_h2(probes),
            "H3": _evaluate_h3(sae),
            "H4": _evaluate_h4(causal),
            "H5": _evaluate_h5(attention),
            "H6": _evaluate_h6(geometry),
        }

        return Report(
            timestamp=datetime.now(UTC).isoformat(),
            git_sha=_git_sha(),
            models=models,
            behavioral=behavioral,
            probes=probes,
            sae=sae,
            attention=attention,
            geometry=geometry,
            causal=causal,
            hypotheses=hypotheses,
        )


@beartype
def generate_report(results_dir: str | Path) -> Report:
    """Top-level convenience function: load results and return a Report."""
    agg = ResultsAggregator.from_directory(results_dir)
    return agg.aggregate()
