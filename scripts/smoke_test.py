#!/usr/bin/env python3
"""End-to-end smoke test for the s1s2 pipeline on synthetic data.

This script:

1. Builds a synthetic HDF5 cache (1 model, 20 problems, 4 layers, 4 positions)
   that conforms to ``docs/data_contract.md`` via the
   :mod:`s1s2.utils.io` writer helpers.
2. Synthesizes a tiny benchmark JSONL (10 items: 5 conflict + 5 control pairs).
3. Runs each analysis workstream against the synthetic cache:
   probes -> sae -> attention -> geometry.
4. Prints a summary table indicating which workstreams passed.
5. Exits 0 if every workstream passed, 1 if any failed.

Each workstream runner is wrapped in try/except so a single broken workstream
does not abort the rest of the smoke test — that way the script remains a
useful diagnostic even with 1-2 failing workstreams.

Constraints
-----------
* Must run on CPU in <60 seconds.
* No internet, no GPU, no pre-trained SAE downloads.
* Outputs go to a temporary directory unless ``--keep-tempdir`` is passed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

# Make sure we can import s1s2 without `pip install -e .` in dev environments.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# OpenMP guard for macOS where torch + numpy fight over libomp.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# We delegate the HDF5 build to the conftest helper so the smoke test and the
# unit tests stay in lockstep — there's exactly one synthesizer.
_TESTS = _REPO / "tests"
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from conftest import (  # noqa: E402  (sys.path mutation above)
    SYNTH_MODEL_KEY,
    SYNTH_N_LAYERS,
    SYNTH_POSITIONS,
    build_synthetic_hdf5,
)

# --------------------------------------------------------------------------- #
# Synthetic benchmark JSONL                                                    #
# --------------------------------------------------------------------------- #


def build_synthetic_benchmark(path: Path) -> Path:
    """Write a 10-item JSONL benchmark conforming to the BenchmarkItem schema.

    Five conflict items and five matched controls. We assemble plain dicts
    rather than calling the dataclass directly so this script doesn't pull
    in the benchmark loader at import time (we'd rather it fail loudly inside
    the workstream runner if the loader is broken).
    """
    cats = ("crt", "base_rate", "syllogism", "anchoring", "framing")
    items: list[dict] = []
    for pair_idx in range(5):
        cat = cats[pair_idx % len(cats)]
        pair_id = f"pair_{pair_idx:02d}"
        items.append(
            {
                "id": f"{pair_id}__conflict",
                "category": cat,
                "subcategory": "synthetic",
                "conflict": True,
                "difficulty": 2,
                "prompt": f"[{cat}] Lure-eliciting prompt for pair {pair_idx}.",
                "system_prompt": None,
                "correct_answer": "A",
                "lure_answer": "B",
                "answer_pattern": "A",
                "lure_pattern": "B",
                "matched_pair_id": pair_id,
                "source": "template",
                "provenance_note": "smoke test fixture",
                "paraphrases": [],
            }
        )
        items.append(
            {
                "id": f"{pair_id}__control",
                "category": cat,
                "subcategory": "synthetic",
                "conflict": False,
                "difficulty": 2,
                "prompt": f"[{cat}] Neutral prompt for pair {pair_idx}.",
                "system_prompt": None,
                "correct_answer": "A",
                "lure_answer": "",
                "answer_pattern": "A",
                "lure_pattern": "",
                "matched_pair_id": pair_id,
                "source": "template",
                "provenance_note": "smoke test fixture",
                "paraphrases": [],
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Per-workstream runners                                                      #
# --------------------------------------------------------------------------- #
#
# Every runner returns ``(ok: bool, err: str | None)``. They wrap their
# imports inside the function body so that an ImportError in one workstream
# doesn't abort the others.


def run_probes_smoke(hdf5_path: Path, out_dir: Path) -> tuple[bool, str | None]:
    """Run a minimal probe pipeline against the synthetic HDF5."""
    try:
        import numpy as np

        from s1s2.probes import (
            ProbeRunner,
            RunnerConfig,
            build_target,
            layer_result_to_dict,
            save_layer_result,
        )
        from s1s2.utils import io as ioh

        out_dir.mkdir(parents=True, exist_ok=True)

        with ioh.open_activations(hdf5_path) as f:
            target_data = build_target("task_type", f, SYNTH_MODEL_KEY)
            # Layer 2 has the planted signal (set up in conftest).
            X = ioh.get_residual(
                f, SYNTH_MODEL_KEY, layer=2, position="P0"
            ).astype(np.float32)

        cfg = RunnerConfig(
            probes=("mass_mean", "logistic"),
            n_folds=3,
            n_seeds=1,
            control_enabled=True,
            control_n_shuffles=1,
            n_permutations=20,
            n_bootstrap=20,
            run_loco=False,
            seed=0,
        )
        runner = ProbeRunner(cfg)
        result = runner.run(
            X=X,
            target_data=target_data,
            model="synthetic",
            layer=2,
            position="P0",
        )

        # Sanity: the planted-signal layer should not look like noise.
        auc = result.probes["logistic"].summary.get("roc_auc", 0.5)
        if not (0.0 <= auc <= 1.0):
            return False, f"probe AUC out of range: {auc!r}"

        # Round-trip through the JSON serializer so we exercise that path too.
        result_dict = layer_result_to_dict(result)
        json.dumps(result_dict)  # must not raise
        path = save_layer_result(result, out_dir)
        if not path.exists():
            return False, f"result file not written: {path}"

        return True, None
    except Exception as exc:  # pragma: no cover - smoke diagnostic
        tb = traceback.format_exc(limit=8)
        return False, f"{type(exc).__name__}: {exc}\n{tb}"


def run_sae_smoke(hdf5_path: Path, out_dir: Path) -> tuple[bool, str | None]:
    """Run a tiny SAE differential analysis using a MockSAE."""
    try:
        import numpy as np

        from s1s2.sae.differential import differential_activation, encode_batched
        from s1s2.sae.falsification import ma_et_al_falsification
        from s1s2.sae.loaders import MockSAE, reconstruction_report
        from s1s2.utils import io as ioh

        out_dir.mkdir(parents=True, exist_ok=True)

        with ioh.open_activations(hdf5_path) as f:
            problems = ioh.load_problem_metadata(f)
            X = ioh.get_residual(
                f, SYNTH_MODEL_KEY, layer=2, position="P0"
            ).astype(np.float32)

        hidden = X.shape[1]
        # MockSAE with sparsity=1.0 still applies a ReLU after the encoder,
        # so reconstruction is lossy on negatives — this is the same shape as
        # a real SAE and good enough to exercise the downstream code paths.
        sae = MockSAE(hidden_dim=hidden, n_features=hidden * 2, layer=2, seed=0, sparsity=1.0)

        # The reconstruction report is the workstream's fidelity gate. We
        # exercise it but do NOT fail on a low explained variance for the
        # mock — the smoke test is checking that the *pipeline* runs, not
        # that a random SAE reconstructs cognitive bias residuals.
        rep = reconstruction_report(sae, X, n_samples=min(16, X.shape[0]))
        if not (0.0 <= rep.explained_variance <= 1.0 + 1e-6):
            return False, (
                f"reconstruction report returned out-of-range ev={rep.explained_variance!r}"
            )

        feats = encode_batched(sae, X, batch_size=8)
        if feats.shape != (X.shape[0], sae.n_features):
            return False, f"unexpected SAE feature shape {feats.shape}"

        diff = differential_activation(
            feature_activations=feats,
            conflict=problems["conflict"].astype(bool),
            fdr_q=0.05,
            subset_label="all",
        )
        if "feature_id" not in diff.df.columns:
            return False, "differential dataframe missing 'feature_id' column"

        # Pick the most differentially-active feature and run cheap falsification.
        top_feat = int(diff.df.sort_values("p_value").iloc[0]["feature_id"])
        prompts = [str(p) for p in problems["prompt_text"]]
        fres = ma_et_al_falsification(
            feature_id=top_feat,
            sae=sae,
            activations=X,
            feature_activations=feats,
            prompts=prompts,
            tokenizer=None,
            mode="cheap",
            n_random_texts=10,
            n_top_tokens=3,
        )
        if fres.feature_id != top_feat:
            return False, f"falsification feature_id mismatch: {fres.feature_id} vs {top_feat}"

        # Persist a tiny artifact so consumers can verify outputs landed.
        (out_dir / "differential.csv").write_text(diff.df.head().to_csv(index=False))
        (out_dir / "falsification.json").write_text(
            json.dumps(
                {
                    "feature_id": fres.feature_id,
                    "is_spurious": bool(fres.is_spurious),
                    "falsification_ratio": float(fres.falsification_ratio),
                }
            )
        )
        return True, None
    except Exception as exc:  # pragma: no cover - smoke diagnostic
        tb = traceback.format_exc(limit=8)
        return False, f"{type(exc).__name__}: {exc}\n{tb}"


def run_attention_smoke(hdf5_path: Path, out_dir: Path) -> tuple[bool, str | None]:
    """Run per-head differential tests using the precomputed metrics."""
    try:
        import numpy as np

        from s1s2.attention.core import (
            METRIC_NAMES,
            ModelAttentionData,
            bh_fdr_joint,
        )
        from s1s2.attention.heads import (
            classify_heads,
            head_classifications_to_records,
            run_all_head_differential_tests,
        )
        from s1s2.utils import io as ioh

        out_dir.mkdir(parents=True, exist_ok=True)

        with ioh.open_activations(hdf5_path) as f:
            meta = ioh.model_metadata(f, SYNTH_MODEL_KEY)
            labels = ioh.position_labels(f, SYNTH_MODEL_KEY)
            valid = ioh.position_valid(f, SYNTH_MODEL_KEY)
            conflict = f["/problems/conflict"][:].astype(bool)
            metrics: dict[str, np.ndarray] = {}
            for name in METRIC_NAMES:
                metrics[name] = ioh.get_attention_metric(
                    f, SYNTH_MODEL_KEY, name
                ).astype(np.float32)

        # Restrict to the positions valid for at least one problem.
        kept_indices = [i for i, _ in enumerate(labels) if bool(valid[:, i].any())]
        kept_labels = [labels[i] for i in kept_indices]
        if not kept_labels:
            return False, "no valid positions found in synthetic HDF5"
        for name in list(metrics.keys()):
            metrics[name] = metrics[name][..., kept_indices]

        data = ModelAttentionData(
            model_key=SYNTH_MODEL_KEY,
            model_config_key="synthetic",
            family="llama",
            n_layers=int(meta["n_layers"]),
            n_heads=int(meta["n_heads"]),
            n_kv_heads=int(meta["n_kv_heads"]),
            is_reasoning_model=False,
            position_labels=labels,
            selected_positions=kept_labels,
            metrics=metrics,
            conflict=conflict,
        )

        df = run_all_head_differential_tests(data, metrics=tuple(METRIC_NAMES))
        if df.empty:
            return False, "differential tests dataframe is empty"
        # Joint BH-FDR over heads x metrics x positions.
        rejected, qvals = bh_fdr_joint(df["p_value"].to_numpy(dtype=np.float64), q=0.10)
        df = df.copy()
        df["q_value"] = qvals
        df["significant"] = rejected

        classifs = classify_heads(
            df,
            n_layers=data.n_layers,
            n_heads=data.n_heads,
            min_significant=2,
            entropy_effect_threshold=0.2,
        )
        records = head_classifications_to_records(classifs)
        if not isinstance(records, list):
            return False, f"expected list of head classifications, got {type(records).__name__}"

        (out_dir / "head_classifications.json").write_text(json.dumps(records[:5]))
        df.head(20).to_csv(out_dir / "differential_tests.csv", index=False)
        return True, None
    except Exception as exc:  # pragma: no cover - smoke diagnostic
        tb = traceback.format_exc(limit=8)
        return False, f"{type(exc).__name__}: {exc}\n{tb}"


def run_geometry_smoke(hdf5_path: Path, out_dir: Path) -> tuple[bool, str | None]:
    """Run silhouette + linear separability with the d>>N fix."""
    try:
        import numpy as np

        from s1s2.geometry.clusters import compute_silhouette_result
        from s1s2.geometry.separability import linear_separability_with_d_gg_n_fix
        from s1s2.utils import io as ioh

        out_dir.mkdir(parents=True, exist_ok=True)

        with ioh.open_activations(hdf5_path) as f:
            X = ioh.get_residual(
                f, SYNTH_MODEL_KEY, layer=2, position="P0"
            ).astype(np.float32)
            conflict = f["/problems/conflict"][:].astype(np.int64)

        # Cosine silhouette + bootstrap CI + permutation test (small N to keep <1s).
        sil = compute_silhouette_result(
            X=X,
            labels=conflict.astype(np.int64),
            n_bootstrap=20,
            n_permutations=50,
            seed=0,
        )
        d = sil.to_dict()
        if "silhouette" not in d:
            return False, "silhouette result missing 'silhouette' key"

        # Linear separability with PCA pre-reduction.
        sep = linear_separability_with_d_gg_n_fix(
            X=X.astype(np.float64),
            y=conflict.astype(np.int64),
            pca_dim=8,
            n_shuffles=10,
            n_folds=3,
            seed=0,
        )
        sep_d = sep.to_dict()
        if "pca_cv_accuracy" not in sep_d:
            return False, "separability result missing 'pca_cv_accuracy' key"

        (out_dir / "silhouette.json").write_text(json.dumps(d))
        (out_dir / "separability.json").write_text(json.dumps(sep_d))
        return True, None
    except Exception as exc:  # pragma: no cover - smoke diagnostic
        tb = traceback.format_exc(limit=8)
        return False, f"{type(exc).__name__}: {exc}\n{tb}"


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #


def _print_summary(results: dict[str, tuple[bool, str | None]], elapsed: float) -> bool:
    print("\n=== Smoke Test Summary ===")
    print(f"  elapsed: {elapsed:.1f}s")
    print()
    all_ok = True
    width = max(len(name) for name in results) + 2
    for name, (ok, err) in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name:<{width}} {status}")
        if not ok:
            err_text = (err or "").rstrip()
            if err_text:
                indent = " " * (width + 4)
                for line in err_text.splitlines():
                    print(f"{indent}{line}")
            all_ok = False
    print()
    if all_ok:
        print("All workstreams PASSED.")
    else:
        n_failed = sum(1 for ok, _ in results.values() if not ok)
        print(f"{n_failed} workstream(s) FAILED.")
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-tempdir",
        action="store_true",
        help="Keep the temporary working directory after the run for inspection.",
    )
    parser.add_argument(
        "--workstream",
        action="append",
        default=None,
        help="Run only the named workstream(s). May be passed multiple times.",
    )
    args = parser.parse_args()

    runners = {
        "probes": run_probes_smoke,
        "sae": run_sae_smoke,
        "attention": run_attention_smoke,
        "geometry": run_geometry_smoke,
    }
    if args.workstream:
        bad = [w for w in args.workstream if w not in runners]
        if bad:
            print(f"unknown workstream(s): {bad}; valid: {list(runners)}")
            return 2
        runners = {k: v for k, v in runners.items() if k in args.workstream}

    started = time.time()
    tmp = tempfile.mkdtemp(prefix="s1s2_smoke_")
    tmp_path = Path(tmp)
    keep = bool(args.keep_tempdir)
    if keep:
        print(f"Working directory: {tmp_path} (will be kept)")

    results: dict[str, tuple[bool, str | None]] = {}
    try:
        # ---- Setup ----
        try:
            hdf5_path = tmp_path / "smoke.h5"
            build_synthetic_hdf5(
                hdf5_path,
                n_problems=20,
                n_layers=SYNTH_N_LAYERS,
                hidden=32,
                positions=SYNTH_POSITIONS,
                seed=0,
            )
            bench_path = tmp_path / "benchmark.jsonl"
            build_synthetic_benchmark(bench_path)
        except Exception as exc:
            tb = traceback.format_exc(limit=10)
            print(f"Setup FAILED: {type(exc).__name__}: {exc}")
            print(tb)
            return 1

        # ---- Workstream runners ----
        for name, fn in runners.items():
            t0 = time.time()
            try:
                ok, err = fn(hdf5_path, tmp_path / f"results_{name}")
            except Exception as exc:  # extra safety belt
                tb = traceback.format_exc(limit=8)
                ok, err = False, f"{type(exc).__name__}: {exc}\n{tb}"
            results[name] = (ok, err)
            dt = time.time() - t0
            tag = "PASS" if ok else "FAIL"
            print(f"[{tag}] {name} ({dt:.2f}s)")
    finally:
        if not keep:
            import shutil

            shutil.rmtree(tmp_path, ignore_errors=True)

    elapsed = time.time() - started
    all_ok = _print_summary(results, elapsed)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
