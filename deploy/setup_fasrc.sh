#!/usr/bin/env bash
# Setup script for Harvard FASRC Cannon / Kempner cluster.
#
# Prerequisites:
#   - FASRC account with Kempner PI sponsorship (for gpu_requeue)
#   - Repo cloned to home or lab storage
#   - Scratch space at /n/holyscratch01/$USER/
#
# Usage:
#   # On a login node (NOT a compute node):
#   bash deploy/setup_fasrc.sh
set -euo pipefail

REPO_DIR="${S1S2_REPO:-$(pwd)}"
SCRATCH="${S1S2_SCRATCH:-/n/holyscratch01/${USER}/s1s2}"
CONDA_ENV="s1s2"

echo "============================================"
echo "  s1s2 FASRC Setup"
echo "  Repo: ${REPO_DIR}"
echo "  Scratch: ${SCRATCH}"
echo "============================================"

# ---- 1. Load modules -------------------------------------------------------
echo "[1/5] Loading modules..."
module load python/3.11.0-fasrc01
module load cuda/12.1.0-fasrc01
echo "  Python: $(python3 --version)"
echo "  CUDA: $(nvcc --version 2>/dev/null | tail -1 || echo 'nvcc not in PATH (normal on login node)')"

# ---- 2. Create conda environment -------------------------------------------
echo "[2/5] Setting up conda environment..."

if conda env list 2>/dev/null | grep -q "^${CONDA_ENV} "; then
    echo "  Updating existing ${CONDA_ENV} environment..."
    conda env update -n "${CONDA_ENV}" -f environment.yml --prune
else
    echo "  Creating ${CONDA_ENV} environment..."
    conda create -n "${CONDA_ENV}" python=3.11 -y
fi

# Activate
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# PyTorch with CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet

# Project dependencies
pip install -e ".[sae,dev]" --quiet

echo "  Python: $(python --version)"
echo "  Torch: $(python -c 'import torch; print(torch.__version__)')"

# ---- 3. Create scratch directories -----------------------------------------
echo "[3/5] Creating scratch directories..."
mkdir -p "${SCRATCH}/activations"
mkdir -p "${SCRATCH}/checkpoints"
mkdir -p "${SCRATCH}/hf_cache"
mkdir -p "${SCRATCH}/logs"
echo "  Activations: ${SCRATCH}/activations"
echo "  Checkpoints: ${SCRATCH}/checkpoints"
echo "  HF cache:    ${SCRATCH}/hf_cache"
echo "  Logs:        ${SCRATCH}/logs"

# ---- 4. Create logs directory in repo (for SLURM output) -------------------
echo "[4/5] Creating local log directory..."
mkdir -p "${REPO_DIR}/logs"

# ---- 5. Print next steps ---------------------------------------------------
echo "[5/5] Setup complete."
echo ""
echo "============================================"
echo "  Next steps:"
echo ""
echo "  1. Download models (run on a compute node with internet):"
echo "     srun --partition=test --time=01:00:00 --mem=32G \\"
echo "       python deploy/download_models.py --cache-dir ${SCRATCH}/hf_cache"
echo ""
echo "  2. Submit extraction job:"
echo "     sbatch deploy/slurm_extract.sh"
echo ""
echo "  3. Submit full pipeline:"
echo "     sbatch deploy/slurm_full_pipeline.sh"
echo ""
echo "  Environment variables to set in your ~/.bashrc:"
echo "    export S1S2_REPO=${REPO_DIR}"
echo "    export S1S2_SCRATCH=${SCRATCH}"
echo "    export HF_HOME=${SCRATCH}/hf_cache"
echo "============================================"
