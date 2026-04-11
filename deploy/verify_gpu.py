#!/usr/bin/env python3
"""Quick GPU verification for deployment environments.

Checks: CUDA availability, VRAM capacity, tensor allocation, and
transformers import. Prints a summary table and exits 0 on success,
1 if GPU is missing (non-fatal — just a warning), 2 on hard failure.
"""
from __future__ import annotations

import os
import sys

# Prevent OpenMP double-init crash on macOS (torch + numpy fight over libomp).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _bytes_to_gib(b: int | float) -> float:
    return b / (1024**3)


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    # --- Check 1: torch import ---
    try:
        import torch

        checks.append(("torch import", True, f"v{torch.__version__}"))
    except ImportError as e:
        checks.append(("torch import", False, str(e)))
        _print_summary(checks)
        return 2

    # --- Check 2: CUDA available ---
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        checks.append(("CUDA available", True, device_name))
    else:
        checks.append(("CUDA available", False, "no GPU detected"))
        # Still run remaining checks that don't need GPU

    # --- Check 3: VRAM capacity ---
    if cuda_ok:
        total = torch.cuda.get_device_properties(0).total_mem
        total_gib = _bytes_to_gib(total)
        vram_ok = total_gib >= 40.0
        checks.append((
            "VRAM >= 40 GiB",
            vram_ok,
            f"{total_gib:.1f} GiB" + ("" if vram_ok else " (INSUFFICIENT for 8B models)"),
        ))
    else:
        checks.append(("VRAM >= 40 GiB", False, "skipped — no GPU"))

    # --- Check 4: tensor allocation ---
    if cuda_ok:
        try:
            # Allocate 1 GiB tensor on GPU, verify it works
            t = torch.zeros(256, 1024, 1024, dtype=torch.float32, device="cuda")  # 1 GiB
            assert t.sum().item() == 0.0
            del t
            torch.cuda.empty_cache()
            checks.append(("1 GiB tensor alloc", True, "OK"))
        except Exception as e:
            checks.append(("1 GiB tensor alloc", False, str(e)))
    else:
        checks.append(("1 GiB tensor alloc", False, "skipped — no GPU"))

    # --- Check 5: BF16 support ---
    if cuda_ok:
        bf16_ok = torch.cuda.is_bf16_supported()
        checks.append(("BF16 support", bf16_ok, "OK" if bf16_ok else "not supported"))
    else:
        checks.append(("BF16 support", False, "skipped — no GPU"))

    # --- Check 6: transformers import ---
    try:
        import transformers

        checks.append(("transformers import", True, f"v{transformers.__version__}"))
    except ImportError as e:
        checks.append(("transformers import", False, str(e)))

    # --- Check 7: s1s2 import ---
    try:
        import s1s2  # noqa: F401

        checks.append(("s1s2 import", True, "OK"))
    except ImportError as e:
        checks.append(("s1s2 import", False, str(e)))

    # --- Check 8: multi-GPU info ---
    if cuda_ok:
        n_gpus = torch.cuda.device_count()
        gpu_info = ", ".join(
            f"GPU{i}: {torch.cuda.get_device_name(i)}" for i in range(n_gpus)
        )
        checks.append((f"GPU count ({n_gpus})", True, gpu_info))

    _print_summary(checks)

    # Exit 0 if at least CUDA works, 1 if no GPU (still usable for CPU work)
    if not cuda_ok:
        print("\nNo GPU found. CPU-only mode is available for probes/geometry.")
        return 1
    return 0


def _print_summary(checks: list[tuple[str, bool, str]]) -> None:
    print("\n" + "=" * 70)
    print("  GPU Verification Summary")
    print("=" * 70)
    width = max(len(name) for name, _, _ in checks) + 2
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  {name:<{width}} [{status}]  {detail}")
    print("=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())
