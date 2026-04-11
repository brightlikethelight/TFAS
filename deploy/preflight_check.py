#!/usr/bin/env python3
"""Automated pre-flight verification for the s1s2 pipeline.

Checks everything needed before starting GPU extraction: Python version,
CUDA availability, VRAM, package imports, HuggingFace token, disk space,
tokenizer loading, benchmark validation, and a smoke test.

Prints a rich table of pass/fail/warn for each check. Exits 0 if all critical
checks pass, 1 if any critical check fails, 2 on hard error.

Usage:
    python deploy/preflight_check.py
    python deploy/preflight_check.py --cache-dir /workspace/hf_cache
    python deploy/preflight_check.py --skip-tokenizer   # offline cluster
    python deploy/preflight_check.py --skip-smoke        # fast check only
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

# OpenMP guard for macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_REPO = Path(__file__).resolve().parent.parent

# ---- Types -------------------------------------------------------------------

Status = Literal["PASS", "FAIL", "WARN", "SKIP"]


class CheckResult:
    """Result of a single pre-flight check."""

    def __init__(
        self,
        name: str,
        status: Status,
        detail: str,
        critical: bool = True,
    ) -> None:
        self.name = name
        self.status = status
        self.detail = detail
        self.critical = critical


# ---- Individual checks -------------------------------------------------------


def check_python_version() -> CheckResult:
    """Python >= 3.11 required."""
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 11):
        return CheckResult("Python >= 3.11", "PASS", version_str)
    return CheckResult("Python >= 3.11", "FAIL", f"{version_str} (need 3.11+)")


def check_cuda() -> CheckResult:
    """CUDA available via PyTorch."""
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            return CheckResult("CUDA available", "PASS", device)
        return CheckResult("CUDA available", "WARN", "no GPU detected (CPU-only mode)")
    except ImportError:
        return CheckResult("CUDA available", "FAIL", "torch not installed")


def check_vram() -> CheckResult:
    """VRAM >= 40 GB (needed for 8B models in bf16 + extraction overhead)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return CheckResult("VRAM >= 40 GB", "SKIP", "no GPU")
        total = torch.cuda.get_device_properties(0).total_mem
        total_gb = total / (1024**3)
        if total_gb >= 40.0:
            return CheckResult("VRAM >= 40 GB", "PASS", f"{total_gb:.1f} GB")
        if total_gb >= 24.0:
            return CheckResult(
                "VRAM >= 40 GB",
                "WARN",
                f"{total_gb:.1f} GB (may work with careful memory management)",
            )
        return CheckResult(
            "VRAM >= 40 GB", "FAIL", f"{total_gb:.1f} GB (insufficient for 8B models)"
        )
    except ImportError:
        return CheckResult("VRAM >= 40 GB", "FAIL", "torch not installed")


def check_s1s2_import() -> CheckResult:
    """s1s2 package is importable."""
    # Make sure src/ is on path for editable installs.
    src = str(_REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        import s1s2  # noqa: F401

        return CheckResult("s1s2 installed", "PASS", "OK")
    except ImportError as e:
        return CheckResult("s1s2 installed", "FAIL", f"ImportError: {e}")


def check_core_packages() -> CheckResult:
    """Critical packages: torch, transformers, h5py, hydra, rich."""
    missing: list[str] = []
    versions: list[str] = []
    for pkg in ("torch", "transformers", "h5py", "hydra", "rich"):
        try:
            mod = __import__(pkg if pkg != "hydra" else "hydra")
            v = getattr(mod, "__version__", "?")
            versions.append(f"{pkg}={v}")
        except ImportError:
            missing.append(pkg)
    if missing:
        return CheckResult(
            "Core packages",
            "FAIL",
            f"missing: {', '.join(missing)}",
        )
    return CheckResult("Core packages", "PASS", "; ".join(versions))


def check_hf_token() -> CheckResult:
    """HuggingFace token set (needed for gated models like Llama)."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        masked = token[:6] + "..." + token[-4:]
        return CheckResult("HF token set", "PASS", masked, critical=False)
    # Also check the HF CLI cache.
    try:
        from huggingface_hub import HfFolder

        cached = HfFolder.get_token()
        if cached:
            return CheckResult("HF token set", "PASS", "found in HF cache", critical=False)
    except Exception:
        pass
    return CheckResult(
        "HF token set",
        "WARN",
        "not set (gated models like Llama will fail to download). "
        "Set HF_TOKEN or run `huggingface-cli login`.",
        critical=False,
    )


def check_disk_space(scratch_dir: str | None = None) -> CheckResult:
    """Disk space >= 100 GB on the target storage."""
    target = scratch_dir or str(_REPO)
    try:
        usage = shutil.disk_usage(target)
        free_gb = usage.free / (1024**3)
        if free_gb >= 100.0:
            return CheckResult("Disk space >= 100 GB", "PASS", f"{free_gb:.0f} GB free at {target}")
        if free_gb >= 50.0:
            return CheckResult(
                "Disk space >= 100 GB",
                "WARN",
                f"{free_gb:.0f} GB free (tight; models alone are ~60 GB)",
            )
        return CheckResult(
            "Disk space >= 100 GB",
            "FAIL",
            f"{free_gb:.0f} GB free at {target} (need ~100 GB for models + activations)",
        )
    except Exception as e:
        return CheckResult("Disk space >= 100 GB", "WARN", f"could not check: {e}", critical=False)


def check_tokenizer(cache_dir: str | None = None) -> CheckResult:
    """Can load tokenizers for all 4 models (tests HF access + cache)."""
    import yaml

    cfg_path = _REPO / "configs" / "models.yaml"
    if not cfg_path.exists():
        return CheckResult("Load tokenizers", "FAIL", f"models.yaml not found at {cfg_path}")

    with open(cfg_path) as f:
        models_cfg = yaml.safe_load(f)["models"]

    loaded: list[str] = []
    failed: list[str] = []
    kwargs: dict = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    try:
        from transformers import AutoTokenizer
    except ImportError:
        return CheckResult("Load tokenizers", "FAIL", "transformers not installed")

    for key, cfg in models_cfg.items():
        hf_id = cfg["hf_id"]
        try:
            AutoTokenizer.from_pretrained(hf_id, **kwargs)
            loaded.append(key)
        except Exception as e:
            err_str = str(e)
            # Distinguish between "not downloaded yet" and "access denied"
            if "gated" in err_str.lower() or "access" in err_str.lower():
                failed.append(f"{key} (gated: need HF token + access request)")
            elif "does not appear to have" in err_str.lower() or "not found" in err_str.lower():
                failed.append(f"{key} (not cached: run deploy/download_models.py first)")
            else:
                failed.append(f"{key} ({e})")

    if not failed:
        return CheckResult("Load tokenizers", "PASS", f"all {len(loaded)} models OK")
    if loaded:
        return CheckResult(
            "Load tokenizers",
            "WARN",
            f"{len(loaded)} OK, {len(failed)} failed: {'; '.join(failed)}",
            critical=False,
        )
    return CheckResult(
        "Load tokenizers",
        "WARN",
        f"all failed: {'; '.join(failed)} (run deploy/download_models.py)",
        critical=False,
    )


def check_benchmark() -> CheckResult:
    """Benchmark file exists and validates."""
    bench_path = _REPO / "data" / "benchmark" / "benchmark.jsonl"
    if not bench_path.exists():
        return CheckResult("Benchmark validates", "FAIL", f"not found at {bench_path}")

    try:
        import json

        items = []
        with open(bench_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))

        n_total = len(items)
        n_conflict = sum(1 for it in items if it.get("conflict"))
        n_control = sum(1 for it in items if not it.get("conflict"))

        if n_total != 284:
            return CheckResult(
                "Benchmark validates",
                "WARN",
                f"{n_total} items (expected 284)",
                critical=False,
            )
        if n_conflict != 142 or n_control != 142:
            return CheckResult(
                "Benchmark validates",
                "WARN",
                f"{n_conflict} conflict + {n_control} control (expected 142 + 142)",
                critical=False,
            )
        return CheckResult(
            "Benchmark validates",
            "PASS",
            f"{n_total} items ({n_conflict} conflict, {n_control} control)",
        )
    except Exception as e:
        return CheckResult("Benchmark validates", "FAIL", f"parse error: {e}")


def check_smoke() -> CheckResult:
    """Run the CPU smoke test (scripts/smoke_test.py) with a timeout."""
    smoke_script = _REPO / "scripts" / "smoke_test.py"
    if not smoke_script.exists():
        return CheckResult("Smoke test", "WARN", "smoke_test.py not found", critical=False)
    try:
        result = subprocess.run(
            [sys.executable, str(smoke_script)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(_REPO),
            env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
        )
        if result.returncode == 0:
            return CheckResult("Smoke test", "PASS", "all workstreams passed")
        # Extract the summary line from output.
        lines = (result.stdout + result.stderr).strip().splitlines()
        summary = lines[-1] if lines else "unknown failure"
        return CheckResult("Smoke test", "WARN", f"exit {result.returncode}: {summary}")
    except subprocess.TimeoutExpired:
        return CheckResult("Smoke test", "WARN", "timed out after 120s", critical=False)
    except Exception as e:
        return CheckResult("Smoke test", "WARN", f"could not run: {e}", critical=False)


# ---- Rich table printer ------------------------------------------------------


def _print_results(results: list[CheckResult]) -> None:
    """Print results as a formatted table. Uses rich if available, falls back to plain."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="s1s2 Pre-Flight Check", show_lines=True)
        table.add_column("Check", style="bold", width=24)
        table.add_column("Status", width=6, justify="center")
        table.add_column("Detail")

        status_styles = {
            "PASS": "green",
            "FAIL": "bold red",
            "WARN": "yellow",
            "SKIP": "dim",
        }
        for r in results:
            style = status_styles.get(r.status, "")
            table.add_row(r.name, f"[{style}]{r.status}[/{style}]", r.detail)

        console.print(table)
    except ImportError:
        # Fallback: plain text table.
        print("\n" + "=" * 78)
        print("  s1s2 Pre-Flight Check")
        print("=" * 78)
        width = max(len(r.name) for r in results) + 2
        for r in results:
            print(f"  {r.name:<{width}} [{r.status}]  {r.detail}")
        print("=" * 78)


def _print_verdict(results: list[CheckResult]) -> None:
    critical_failures = [r for r in results if r.status == "FAIL" and r.critical]
    warnings = [r for r in results if r.status == "WARN"]

    print()
    if critical_failures:
        print("VERDICT: FAIL -- critical checks did not pass.")
        print("  Fix these before proceeding:")
        for r in critical_failures:
            print(f"    - {r.name}: {r.detail}")
    elif warnings:
        print("VERDICT: PASS with warnings.")
        print("  Non-critical issues:")
        for r in warnings:
            print(f"    - {r.name}: {r.detail}")
    else:
        print("VERDICT: ALL CLEAR -- ready for extraction.")
    print()


# ---- Main --------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-flight check for the s1s2 GPU pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME"),
        help="HuggingFace cache directory (for tokenizer check). Default: $HF_HOME.",
    )
    parser.add_argument(
        "--scratch-dir",
        default=os.environ.get("S1S2_SCRATCH"),
        help="Scratch storage path (for disk space check). Default: $S1S2_SCRATCH.",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip tokenizer loading check (for offline clusters).",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip CPU smoke test (saves ~60s).",
    )
    args = parser.parse_args()

    print("s1s2 Pre-Flight Check")
    print(f"  Repo: {_REPO}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if args.cache_dir:
        print(f"  HF cache: {args.cache_dir}")
    if args.scratch_dir:
        print(f"  Scratch: {args.scratch_dir}")
    print()

    results: list[CheckResult] = []

    # --- Critical checks ---
    results.append(check_python_version())
    results.append(check_cuda())
    results.append(check_vram())
    results.append(check_s1s2_import())
    results.append(check_core_packages())

    # --- Environment checks ---
    results.append(check_hf_token())
    results.append(check_disk_space(args.scratch_dir))

    # --- Data checks ---
    if not args.skip_tokenizer:
        results.append(check_tokenizer(args.cache_dir))
    else:
        results.append(CheckResult("Load tokenizers", "SKIP", "skipped (--skip-tokenizer)"))

    results.append(check_benchmark())

    # --- Integration check ---
    if not args.skip_smoke:
        results.append(check_smoke())
    else:
        results.append(CheckResult("Smoke test", "SKIP", "skipped (--skip-smoke)"))

    _print_results(results)
    _print_verdict(results)

    critical_failures = [r for r in results if r.status == "FAIL" and r.critical]
    return 1 if critical_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
