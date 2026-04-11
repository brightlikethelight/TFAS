#!/bin/bash
#SBATCH --job-name=s1s2_extract
#SBATCH --partition=kempner_requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#
# Checkpoint-aware extraction for preemptible Kempner gpu_requeue.
# SIGUSR1 is sent 120s before wall time — the checkpoint wrapper
# finishes the current model and exits cleanly. Resubmit to continue.
#
# Usage:
#   sbatch deploy/slurm_extract.sh
#   sbatch --partition=kempner deploy/slurm_extract.sh  # non-preemptible
#
# To extract specific models:
#   sbatch deploy/slurm_extract.sh --models llama-3.1-8b-instruct r1-distill-llama-8b
set -euo pipefail

# ---- Environment ------------------------------------------------------------
REPO_DIR="${S1S2_REPO:-$(pwd)}"
SCRATCH="${S1S2_SCRATCH:-/n/holyscratch01/${USER}/s1s2}"

module load python/3.11.0-fasrc01
module load cuda/12.1.0-fasrc01

eval "$(conda shell.bash hook)"
conda activate s1s2

export HF_HOME="${SCRATCH}/hf_cache"

# ---- Job info ---------------------------------------------------------------
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Partition:    ${SLURM_JOB_PARTITION}"
echo "Repo:         ${REPO_DIR}"
echo "Scratch:      ${SCRATCH}"
echo "HF cache:     ${HF_HOME}"
echo "Start time:   $(date)"

cd "${REPO_DIR}"

# ---- Run extraction with checkpoint support ---------------------------------
# All 4 models by default. Override with --models flag.
python deploy/checkpoint_extract.py \
    --config configs/extract.yaml \
    --checkpoint-dir "${SCRATCH}/checkpoints" \
    --output-dir "${SCRATCH}/activations" \
    --models llama-3.1-8b-instruct gemma-2-9b-it r1-distill-llama-8b r1-distill-qwen-7b \
    "$@"

echo "End time: $(date)"
echo "Job ${SLURM_JOB_ID} finished."
