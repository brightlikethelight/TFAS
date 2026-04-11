#!/usr/bin/env python3
"""Dry-run the full s1s2 pipeline with a tiny model on CPU.

Loads N benchmark items, runs extraction with sshleifer/tiny-gpt2,
then exercises all CPU-feasible analysis workstreams (probes, SAE
with MockSAE, attention entropy, geometry). Prints a rich summary
and saves all artifacts to --output-dir for manual inspection.

This is useful for:
- Verifying the pipeline works before getting GPU access
- CI validation of the full chain (with --n-items 2 for speed)
- Debugging pipeline issues

Usage:
    python scripts/dry_run.py                         # 5 items, data/dry_run/
    python scripts/dry_run.py --n-items 2 --output-dir /tmp/dr
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# macOS OpenMP guard
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Allow import without pip install -e .
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402 — sys.path must be patched first

BENCHMARK_PATH = _REPO / "data" / "benchmark" / "benchmark.jsonl"
TINY_MODEL_ID = "sshleifer/tiny-gpt2"
TINY_HDF5_KEY = "sshleifer_tiny-gpt2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_items(n_items: int) -> list[SimpleNamespace]:
    """Load a balanced subset of benchmark items."""
    all_items: list[dict] = []
    with BENCHMARK_PATH.open() as f:
        for line in f:
            all_items.append(json.loads(line))

    # Pick roughly half conflict, half control, from different categories
    n_conflict = max(1, n_items // 2)
    n_control = n_items - n_conflict

    conflict_items: list[dict] = []
    control_items: list[dict] = []
    seen_cats_c: set[str] = set()
    seen_cats_nc: set[str] = set()

    for item in all_items:
        cat = item["category"]
        if item["conflict"] and len(conflict_items) < n_conflict and (cat not in seen_cats_c or len(conflict_items) < n_conflict):
            conflict_items.append(item)
            seen_cats_c.add(cat)
        elif not item["conflict"] and len(control_items) < n_control and (cat not in seen_cats_nc or len(control_items) < n_control):
            control_items.append(item)
            seen_cats_nc.add(cat)
        if len(conflict_items) >= n_conflict and len(control_items) >= n_control:
            break

    items = conflict_items + control_items
    if len(items) < n_items:
        print(f"[warn] only found {len(items)} items (requested {n_items})")
    return [SimpleNamespace(**d) for d in items]


def fmt_time(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def run_extraction(items: list[SimpleNamespace], output_dir: Path) -> Path:
    """Run the extraction pipeline and return the HDF5 path."""
    import torch

    from s1s2.extract import (
        ActivationWriter,
        ExtractionConfig,
        GenerationConfig,
        ModelSpec,
        RunMetadata,
        SingleModelExtractor,
        build_problem_metadata_from_items,
    )

    spec = ModelSpec(
        key="tiny-gpt2",
        hdf5_key=TINY_HDF5_KEY,
        hf_id=TINY_MODEL_ID,
        family="gpt2",
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        hidden_dim=2,
        head_dim=1,
        is_reasoning=False,
    )
    gen_cfg = GenerationConfig(
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
        seed=42,
    )
    extr_cfg = ExtractionConfig(
        dtype="float16",
        attn_implementation="eager",
        log_every=1,
    )

    extractor = SingleModelExtractor(
        spec=spec,
        generation_cfg=gen_cfg,
        extraction_cfg=extr_cfg,
        device="cpu",
        torch_dtype=torch.float32,
    )
    extractor.load()

    hidden_dim = extractor.model.config.hidden_size
    n_layers = spec.n_layers
    n_heads = spec.n_heads

    out_path = output_dir / "activations.h5"
    with ActivationWriter(out_path) as writer:
        writer.write_run_metadata(
            RunMetadata.build(
                benchmark_path=str(BENCHMARK_PATH),
                seed=42,
                config_json=json.dumps({"kind": "dry_run", "model": TINY_MODEL_ID}),
            )
        )

        prompt_counts = []
        for item in items:
            ids = extractor.tokenizer(item.prompt, return_tensors="pt").input_ids
            prompt_counts.append(int(ids.shape[1]))

        writer.write_problems(build_problem_metadata_from_items(items, prompt_counts))

        writer.create_model_group(
            model_key=spec.hdf5_key,
            hf_model_id=spec.hf_id,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=spec.n_kv_heads,
            hidden_dim=hidden_dim,
            head_dim=spec.head_dim,
            dtype="float16",
            is_reasoning_model=False,
        )

        extractor.run(
            items=items,
            writer=writer,
            effective_n_layers=n_layers,
            effective_n_heads=n_heads,
            effective_hidden_dim=hidden_dim,
        )

    extractor.unload()
    return out_path


def run_probes(hdf5_path: Path, output_dir: Path) -> dict:
    """Run probing analysis. Returns summary dict.

    Probes need at least 4 items (2 per class minimum for 2-fold CV).
    Skips gracefully if there aren't enough items.
    """
    from s1s2.probes.core import ProbeRunner, RunnerConfig, save_layer_result
    from s1s2.probes.targets import build_target
    from s1s2.utils.io import get_residual, open_activations

    probe_dir = output_dir / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)

    with open_activations(hdf5_path) as f:
        td = build_target("task_type", f, TINY_HDF5_KEY)
        n_layers = int(f[f"/models/{TINY_HDF5_KEY}/metadata"].attrs["n_layers"])

        # Check minimum class counts -- need >= 2 per class for 2-fold CV
        n_pos = int(td.y.sum())
        n_neg = int(len(td.y) - n_pos)
        if n_pos < 2 or n_neg < 2:
            print(f"  [skip] probes need >= 2 items per class, got {n_pos} conflict + {n_neg} control")
            return {"n_layers": n_layers, "layer_aucs": {}, "skipped": True, "output_dir": str(probe_dir)}

        config = RunnerConfig(
            probes=("logistic",),
            n_folds=2,
            n_seeds=1,
            control_enabled=False,
            n_permutations=10,
            n_bootstrap=10,
            run_loco=False,
            seed=0,
            probe_kwargs={"logistic": {"cv": 2}},
        )
        runner = ProbeRunner(config)

        layer_aucs: dict[int, dict[str, float]] = {}
        for layer in range(n_layers):
            X = get_residual(f, TINY_HDF5_KEY, layer=layer, position="P0")
            X = X.astype(np.float32, copy=False)
            try:
                result = runner.run(
                    X=X, target_data=td, model=TINY_HDF5_KEY, layer=layer, position="P0"
                )
                save_layer_result(result, probe_dir)
                layer_aucs[layer] = {
                    name: pr.summary.get("roc_auc", float("nan"))
                    for name, pr in result.probes.items()
                }
            except ValueError as e:
                # Degenerate fold: too few items per class for the inner CV.
                # Expected with tiny item counts -- log and continue.
                print(f"  [warn] layer {layer} skipped: {e}")
                layer_aucs[layer] = {"logistic": float("nan")}

    return {"n_layers": n_layers, "layer_aucs": layer_aucs, "output_dir": str(probe_dir)}


def run_sae(hdf5_path: Path, output_dir: Path) -> dict:
    """Run SAE differential analysis with MockSAE. Returns summary dict."""
    from s1s2.sae.differential import differential_activation, encode_batched
    from s1s2.sae.loaders import MockSAE, reconstruction_report
    from s1s2.utils.io import get_residual, open_activations

    sae_dir = output_dir / "sae"
    sae_dir.mkdir(parents=True, exist_ok=True)

    with open_activations(hdf5_path) as f:
        hidden_dim = int(f[f"/models/{TINY_HDF5_KEY}/metadata"].attrs["hidden_dim"])
        X = get_residual(f, TINY_HDF5_KEY, layer=0, position="P0")
        X = X.astype(np.float32, copy=False)
        conflict = f["/problems/conflict"][:].astype(bool)

    n_features = max(64, hidden_dim * 2)
    sae = MockSAE(hidden_dim=hidden_dim, n_features=n_features, layer=0, seed=0)

    report = reconstruction_report(sae, X, min_explained_variance=0.01, n_samples=min(5, X.shape[0]))
    feature_acts = encode_batched(sae, X)
    result = differential_activation(feature_acts, conflict, fdr_q=0.05, subset_label="all")

    # Save the differential results
    result.df.to_csv(sae_dir / "differential.csv", index=False)

    return {
        "n_features": sae.n_features,
        "n_significant": int(result.df["significant"].sum()),
        "explained_variance": report.explained_variance,
        "is_poor_fit": report.is_poor_fit,
        "output_dir": str(sae_dir),
    }


def run_attention(hdf5_path: Path, output_dir: Path) -> dict:
    """Run attention differential analysis. Returns summary dict."""
    from s1s2.attention.core import ModelAttentionData
    from s1s2.attention.heads import run_all_head_differential_tests
    from s1s2.utils.io import (
        get_attention_metric,
        open_activations,
        position_labels,
        position_valid,
    )

    attn_dir = output_dir / "attention"
    attn_dir.mkdir(parents=True, exist_ok=True)

    with open_activations(hdf5_path) as f:
        labels = position_labels(f, TINY_HDF5_KEY)
        valid = position_valid(f, TINY_HDF5_KEY)
        n_layers = int(f[f"/models/{TINY_HDF5_KEY}/metadata"].attrs["n_layers"])
        n_heads = int(f[f"/models/{TINY_HDF5_KEY}/metadata"].attrs["n_heads"])
        n_kv_heads = int(f[f"/models/{TINY_HDF5_KEY}/metadata"].attrs["n_kv_heads"])
        conflict = f["/problems/conflict"][:].astype(bool)

        selected = []
        selected_idx = []
        for pos in ("P0", "P2"):
            if pos in labels:
                idx = labels.index(pos)
                if valid[:, idx].any():
                    selected.append(pos)
                    selected_idx.append(idx)

        metrics: dict[str, np.ndarray] = {}
        for name in ("entropy", "entropy_normalized", "gini", "max_attn", "focus_5"):
            arr = get_attention_metric(f, TINY_HDF5_KEY, name)
            metrics[name] = arr[..., selected_idx].astype(np.float32, copy=False)

    data = ModelAttentionData(
        model_key=TINY_HDF5_KEY,
        model_config_key="tiny-gpt2-dryrun",
        family="gpt2",
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        is_reasoning_model=False,
        position_labels=labels,
        selected_positions=selected,
        metrics=metrics,
        conflict=conflict,
    )

    df = run_all_head_differential_tests(data, metrics=("entropy", "gini"))

    # Save
    try:
        df.to_parquet(attn_dir / "head_tests.parquet", index=False)
    except (ImportError, Exception):
        df.to_csv(attn_dir / "head_tests.csv", index=False)

    n_sig = int(df["significant"].sum()) if "significant" in df.columns else 0
    return {
        "n_tests": len(df),
        "n_significant_raw": n_sig,
        "positions_analyzed": selected,
        "output_dir": str(attn_dir),
    }


def run_geometry(hdf5_path: Path, output_dir: Path) -> dict:
    """Run geometry analysis. Returns summary dict."""
    from s1s2.geometry.clusters import (
        calinski_harabasz,
        cosine_silhouette,
        davies_bouldin,
    )
    from s1s2.geometry.separability import linear_separability_with_d_gg_n_fix
    from s1s2.utils.io import get_residual, open_activations

    geo_dir = output_dir / "geometry"
    geo_dir.mkdir(parents=True, exist_ok=True)

    with open_activations(hdf5_path) as f:
        n_layers = int(f[f"/models/{TINY_HDF5_KEY}/metadata"].attrs["n_layers"])
        conflict = f["/problems/conflict"][:].astype(bool)

        per_layer: list[dict] = []
        for layer in range(n_layers):
            X = get_residual(f, TINY_HDF5_KEY, layer=layer, position="P0")
            X = X.astype(np.float32, copy=False)
            labels = conflict.astype(np.int64)

            sil = cosine_silhouette(X, labels)
            ch = calinski_harabasz(X, labels)
            db = davies_bouldin(X, labels)

            # Separability: need >= 4 samples
            sep_result = None
            if X.shape[0] >= 4:
                try:
                    sep_result = linear_separability_with_d_gg_n_fix(
                        X, labels,
                        pca_dim=min(X.shape[1] - 1, 50) if X.shape[1] > 1 else 1,
                        n_shuffles=5,
                        n_folds=2,
                        seed=0,
                    )
                except Exception as e:
                    sep_result = None
                    print(f"  [warn] separability failed on layer {layer}: {e}")

            entry = {
                "layer": layer,
                "silhouette": sil,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
            }
            if sep_result is not None:
                entry["pca_cv_accuracy"] = sep_result.pca_cv_accuracy
                entry["margin_real"] = sep_result.margin_real
            per_layer.append(entry)

    # Save
    with (geo_dir / "geometry_results.json").open("w") as fh:
        json.dump(per_layer, fh, indent=2, default=float)

    return {
        "n_layers": n_layers,
        "per_layer": per_layer,
        "output_dir": str(geo_dir),
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(
    n_items: int,
    extraction_time: float,
    hdf5_path: Path,
    probe_result: dict | None,
    sae_result: dict | None,
    attn_result: dict | None,
    geo_result: dict | None,
    stage_times: dict[str, float],
) -> None:
    """Print a summary table of all pipeline stages."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table(title="s1s2 Dry Run Summary", show_lines=True)
        table.add_column("Stage", style="bold")
        table.add_column("Status")
        table.add_column("Time")
        table.add_column("Details")

        # Extraction
        table.add_row(
            "Extraction",
            "[green]OK[/green]",
            fmt_time(extraction_time),
            f"{n_items} items -> {hdf5_path.name} ({hdf5_path.stat().st_size / 1024:.1f} KB)",
        )

        # Probes
        if probe_result:
            detail = ", ".join(
                f"L{layer}: {'/'.join(f'{k}={v:.2f}' for k, v in aucs.items())}"
                for layer, aucs in probe_result["layer_aucs"].items()
            )
            table.add_row(
                "Probes",
                "[green]OK[/green]",
                fmt_time(stage_times.get("probes", 0)),
                detail,
            )
        else:
            table.add_row("Probes", "[red]FAIL[/red]", "-", "-")

        # SAE
        if sae_result:
            table.add_row(
                "SAE (MockSAE)",
                "[green]OK[/green]",
                fmt_time(stage_times.get("sae", 0)),
                f"{sae_result['n_features']} features, "
                f"{sae_result['n_significant']} significant, "
                f"EV={sae_result['explained_variance']:.3f}",
            )
        else:
            table.add_row("SAE", "[red]FAIL[/red]", "-", "-")

        # Attention
        if attn_result:
            table.add_row(
                "Attention",
                "[green]OK[/green]",
                fmt_time(stage_times.get("attention", 0)),
                f"{attn_result['n_tests']} tests, "
                f"positions={attn_result['positions_analyzed']}",
            )
        else:
            table.add_row("Attention", "[red]FAIL[/red]", "-", "-")

        # Geometry
        if geo_result:
            sils = [f"L{e['layer']}={e['silhouette']:.3f}" for e in geo_result["per_layer"]]
            table.add_row(
                "Geometry",
                "[green]OK[/green]",
                fmt_time(stage_times.get("geometry", 0)),
                f"silhouette: {', '.join(sils)}",
            )
        else:
            table.add_row("Geometry", "[red]FAIL[/red]", "-", "-")

        total = sum(stage_times.values()) + extraction_time
        table.add_row(
            "[bold]Total[/bold]",
            "",
            f"[bold]{fmt_time(total)}[/bold]",
            "",
        )

        console.print(table)

    except ImportError:
        # Fallback: plain text
        print("\n" + "=" * 60)
        print("s1s2 Dry Run Summary")
        print("=" * 60)
        print(f"Extraction: OK ({fmt_time(extraction_time)}) - {n_items} items")
        if probe_result:
            print(f"Probes:     OK ({fmt_time(stage_times.get('probes', 0))})")
        else:
            print("Probes:     FAIL")
        if sae_result:
            print(f"SAE:        OK ({fmt_time(stage_times.get('sae', 0))}) - {sae_result['n_significant']} sig features")
        else:
            print("SAE:        FAIL")
        if attn_result:
            print(f"Attention:  OK ({fmt_time(stage_times.get('attention', 0))}) - {attn_result['n_tests']} tests")
        else:
            print("Attention:  FAIL")
        if geo_result:
            print(f"Geometry:   OK ({fmt_time(stage_times.get('geometry', 0))})")
        else:
            print("Geometry:   FAIL")
        total = sum(stage_times.values()) + extraction_time
        print(f"Total:      {fmt_time(total)}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run the full s1s2 pipeline with a tiny model on CPU."
    )
    parser.add_argument(
        "--n-items", type=int, default=5,
        help="Number of benchmark items to process (default: 5).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(_REPO / "data" / "dry_run"),
        help="Directory to save outputs (default: data/dry_run/).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.n_items} benchmark items from {BENCHMARK_PATH}")
    items = load_items(args.n_items)
    n_conflict = sum(1 for it in items if it.conflict)
    n_control = len(items) - n_conflict
    print(f"  {n_conflict} conflict + {n_control} no-conflict from "
          f"{len({it.category for it in items})} categories")

    # --- Extraction ---
    print("\n--- Extraction ---")
    t0 = time.time()
    hdf5_path = run_extraction(items, output_dir)
    extraction_time = time.time() - t0
    print(f"  HDF5 written to {hdf5_path} ({hdf5_path.stat().st_size / 1024:.1f} KB)")

    # Validate schema
    from s1s2.extract import validate_file
    errors = validate_file(hdf5_path)
    if errors:
        print(f"  [ERROR] Schema validation failed: {errors}")
    else:
        print("  Schema validation: PASS")

    stage_times: dict[str, float] = {}

    # --- Probes ---
    print("\n--- Probes ---")
    probe_result = None
    try:
        t0 = time.time()
        probe_result = run_probes(hdf5_path, output_dir)
        stage_times["probes"] = time.time() - t0
        print(f"  Completed in {fmt_time(stage_times['probes'])}")
    except Exception as e:
        stage_times["probes"] = 0
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

    # --- SAE ---
    print("\n--- SAE (MockSAE) ---")
    sae_result = None
    try:
        t0 = time.time()
        sae_result = run_sae(hdf5_path, output_dir)
        stage_times["sae"] = time.time() - t0
        print(f"  Completed in {fmt_time(stage_times['sae'])}")
    except Exception as e:
        stage_times["sae"] = 0
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

    # --- Attention ---
    print("\n--- Attention ---")
    attn_result = None
    try:
        t0 = time.time()
        attn_result = run_attention(hdf5_path, output_dir)
        stage_times["attention"] = time.time() - t0
        print(f"  Completed in {fmt_time(stage_times['attention'])}")
    except Exception as e:
        stage_times["attention"] = 0
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

    # --- Geometry ---
    print("\n--- Geometry ---")
    geo_result = None
    try:
        t0 = time.time()
        geo_result = run_geometry(hdf5_path, output_dir)
        stage_times["geometry"] = time.time() - t0
        print(f"  Completed in {fmt_time(stage_times['geometry'])}")
    except Exception as e:
        stage_times["geometry"] = 0
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

    # --- Summary ---
    print()
    print_summary(
        n_items=len(items),
        extraction_time=extraction_time,
        hdf5_path=hdf5_path,
        probe_result=probe_result,
        sae_result=sae_result,
        attn_result=attn_result,
        geo_result=geo_result,
        stage_times=stage_times,
    )

    # Save a machine-readable summary
    summary = {
        "n_items": len(items),
        "extraction_time_s": extraction_time,
        "hdf5_path": str(hdf5_path),
        "stages": {
            "probes": probe_result,
            "sae": sae_result,
            "attention": attn_result,
            "geometry": geo_result,
        },
        "stage_times": stage_times,
        "all_passed": all(
            r is not None for r in [probe_result, sae_result, attn_result, geo_result]
        ),
    }
    summary_path = output_dir / "dry_run_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
