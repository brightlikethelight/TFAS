#!/usr/bin/env bash
# Setup script for a fresh RunPod A100/H100 pod.
#
# Prerequisites:
#   - Pod has persistent storage mounted at /workspace
#   - Repo has been SCP'd to /workspace/s1s2 (Bright's preferred method)
#   - HF_TOKEN is set in pod environment (for gated models like Llama)
#
# Usage:
#   scp -r ./s1s2 root@<pod-ip>:/workspace/
#   ssh root@<pod-ip> "bash /workspace/s1s2/deploy/setup_runpod.sh"
set -euo pipefail

REPO_DIR="${S1S2_REPO:-/workspace/s1s2}"
HF_CACHE="${HF_CACHE_DIR:-/workspace/hf_cache}"
CONDA_ENV="s1s2"

echo "============================================"
echo "  s1s2 RunPod Setup"
echo "  Repo: ${REPO_DIR}"
echo "  HF cache: ${HF_CACHE}"
echo "============================================"

# ---- 1. System dependencies ------------------------------------------------
echo "[1/6] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git htop tmux nvtop 2>/dev/null || true

# ---- 2. Verify repo exists -------------------------------------------------
if [ ! -d "${REPO_DIR}" ]; then
    echo "ERROR: Repo not found at ${REPO_DIR}"
    echo "SCP the repo first: scp -r ./s1s2 root@<pod-ip>:/workspace/"
    exit 1
fi
cd "${REPO_DIR}"
echo "[2/6] Repo found at ${REPO_DIR}"

# ---- 3. Conda environment --------------------------------------------------
echo "[3/6] Setting up conda environment..."

# RunPod images typically have conda pre-installed
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install miniconda first."
    exit 1
fi

# Create or update the conda env
if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "  Updating existing ${CONDA_ENV} environment..."
    conda env update -n "${CONDA_ENV}" -f environment.yml --prune
else
    echo "  Creating ${CONDA_ENV} environment..."
    conda env create -f environment.yml
fi

# Activate — need to source conda's shell functions
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# Install the package in editable mode with optional deps
pip install -e ".[sae,dev]" --quiet

echo "  Python: $(python --version)"
echo "  Torch: $(python -c 'import torch; print(torch.__version__)')"

# ---- 4. Verify GPU ---------------------------------------------------------
echo "[4/6] Verifying GPU..."
python deploy/verify_gpu.py

# ---- 5. Download models ----------------------------------------------------
echo "[5/6] Downloading models to ${HF_CACHE}..."
mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"

# Check for HF token (needed for gated models like Llama)
if [ -z "${HF_TOKEN:-}" ]; then
    echo "  WARNING: HF_TOKEN not set. Gated models (Llama) will fail to download."
    echo "  Set it: export HF_TOKEN=hf_..."
fi

python deploy/download_models.py --cache-dir "${HF_CACHE}"

# ---- 6. Smoke test ---------------------------------------------------------
echo "[6/6] Running smoke test..."
python scripts/smoke_test.py

echo ""
echo "============================================"
echo "  Pod is ready."
echo ""
echo "  Activate env:  conda activate ${CONDA_ENV}"
echo "  HF cache:      export HF_HOME=${HF_CACHE}"
echo ""
echo "  Run extraction:"
echo "    python scripts/extract_all.py run_name=main"
echo ""
echo "  Or use tmux for long runs:"
echo "    tmux new -s extract"
echo "    python scripts/extract_all.py run_name=main"
echo "============================================"
