#!/bin/bash
#SBATCH --job-name=s1s2_probes
#SBATCH --partition=kempner_requeue
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/probes_%j.out
#SBATCH --error=logs/probes_%j.err
#
# Probe analysis — CPU-only job. Reads cached activations from HDF5 and
# trains linear probes (logistic regression, mass-mean difference).
# No GPU needed; the bottleneck is sklearn cross-validation.
#
# Usage:
#   sbatch deploy/slurm_probes.sh
set -euo pipefail

# ---- Environment ------------------------------------------------------------
REPO_DIR="${S1S2_REPO:-$(pwd)}"
SCRATCH="${S1S2_SCRATCH:-/n/holyscratch01/${USER}/s1s2}"

module load python/3.11.0-fasrc01

eval "$(conda shell.bash hook)"
conda activate s1s2

# ---- Job info ---------------------------------------------------------------
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Partition:    ${SLURM_JOB_PARTITION}"
echo "Start time:   $(date)"

cd "${REPO_DIR}"

# ---- Run probes for all models ----------------------------------------------
# The probe CLI iterates over models and layers internally.
for MODEL in llama-3.1-8b-instruct gemma-2-9b-it r1-distill-llama-8b r1-distill-qwen-7b; do
    echo ""
    echo "=== Probes: ${MODEL} ==="
    python scripts/run_probes.py \
        model="${MODEL}" \
        output_dir="${SCRATCH}/activations" \
        hydra.run.dir="${SCRATCH}/hydra/probes/${MODEL}" \
        || echo "WARNING: probes failed for ${MODEL}"
done

echo ""
echo "End time: $(date)"
echo "Job ${SLURM_JOB_ID} finished."
