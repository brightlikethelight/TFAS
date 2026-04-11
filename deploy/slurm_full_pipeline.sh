#!/bin/bash
#SBATCH --job-name=s1s2_full
#SBATCH --partition=kempner_requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#
# Full s1s2 pipeline: extract -> probes -> sae -> attention -> geometry
#
# This script runs stages sequentially. Each stage writes outputs to scratch.
# If preempted, re-submit — extraction resumes from checkpoints, and
# downstream stages are skipped if their outputs already exist.
#
# For the full pipeline across all models, expect ~7 hours on A100.
# On Kempner gpu_requeue (7hr max), the pipeline may need 2 submissions
# if attention re-extraction is included.
#
# Usage:
#   sbatch deploy/slurm_full_pipeline.sh
#   sbatch --partition=kempner deploy/slurm_full_pipeline.sh  # non-preemptible, no 7hr limit
set -euo pipefail

# ---- Preemption handler (propagates SIGUSR1 to child processes) -------------
_CHILD_PID=""
_handle_preempt() {
    echo "[PIPELINE] Preemption signal received at $(date)"
    if [ -n "${_CHILD_PID}" ]; then
        echo "[PIPELINE] Forwarding SIGUSR1 to child PID ${_CHILD_PID}"
        kill -USR1 "${_CHILD_PID}" 2>/dev/null || true
    fi
}
trap _handle_preempt USR1

# ---- Environment ------------------------------------------------------------
REPO_DIR="${S1S2_REPO:-$(pwd)}"
SCRATCH="${S1S2_SCRATCH:-/n/holyscratch01/${USER}/s1s2}"
MODELS="llama-3.1-8b-instruct gemma-2-9b-it r1-distill-llama-8b r1-distill-qwen-7b"

module load python/3.11.0-fasrc01
module load cuda/12.1.0-fasrc01

eval "$(conda shell.bash hook)"
conda activate s1s2

export HF_HOME="${SCRATCH}/hf_cache"

# ---- Job info ---------------------------------------------------------------
echo "============================================"
echo "  s1s2 Full Pipeline"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Node:       $(hostname)"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Partition:  ${SLURM_JOB_PARTITION}"
echo "  Start time: $(date)"
echo "============================================"

cd "${REPO_DIR}"

# ---- Stage marker helper ----------------------------------------------------
# Writes a marker file when a stage completes so re-submissions skip it.
stage_done() {
    local stage="$1"
    [ -f "${SCRATCH}/checkpoints/stage_${stage}.done" ]
}

mark_stage_done() {
    local stage="$1"
    mkdir -p "${SCRATCH}/checkpoints"
    date > "${SCRATCH}/checkpoints/stage_${stage}.done"
    echo "[PIPELINE] Stage '${stage}' marked complete."
}

# ---- Stage 1: Extraction ----------------------------------------------------
echo ""
echo "=== Stage 1/5: Extraction ==="
if stage_done "extract"; then
    echo "[SKIP] Extraction already complete."
else
    python deploy/checkpoint_extract.py \
        --config configs/extract.yaml \
        --checkpoint-dir "${SCRATCH}/checkpoints" \
        --output-dir "${SCRATCH}/activations" \
        --models ${MODELS} &
    _CHILD_PID=$!
    wait ${_CHILD_PID}
    _CHILD_PID=""

    # Check if all models completed (checkpoint_extract exits 0 only if all done)
    ALL_DONE=true
    for MODEL in ${MODELS}; do
        if [ ! -f "${SCRATCH}/checkpoints/${MODEL}.done" ]; then
            ALL_DONE=false
            break
        fi
    done
    if ${ALL_DONE}; then
        mark_stage_done "extract"
    else
        echo "[PIPELINE] Extraction incomplete (likely preempted). Resubmit to continue."
        exit 0
    fi
fi

# ---- Stage 2: Probes -------------------------------------------------------
echo ""
echo "=== Stage 2/5: Probes ==="
if stage_done "probes"; then
    echo "[SKIP] Probes already complete."
else
    for MODEL in ${MODELS}; do
        echo "--- Probes: ${MODEL} ---"
        python scripts/run_probes.py \
            model="${MODEL}" \
            output_dir="${SCRATCH}/activations" \
            hydra.run.dir="${SCRATCH}/hydra/probes/${MODEL}" \
            || echo "WARNING: probes failed for ${MODEL}"
    done
    mark_stage_done "probes"
fi

# ---- Stage 3: SAE -----------------------------------------------------------
echo ""
echo "=== Stage 3/5: SAE Analysis ==="
if stage_done "sae"; then
    echo "[SKIP] SAE already complete."
else
    for MODEL in ${MODELS}; do
        echo "--- SAE: ${MODEL} ---"
        python scripts/run_sae.py \
            model="${MODEL}" \
            output_dir="${SCRATCH}/activations" \
            hydra.run.dir="${SCRATCH}/hydra/sae/${MODEL}" \
            || echo "WARNING: SAE failed for ${MODEL}"
    done
    mark_stage_done "sae"
fi

# ---- Stage 4: Attention -----------------------------------------------------
echo ""
echo "=== Stage 4/5: Attention ==="
if stage_done "attention"; then
    echo "[SKIP] Attention already complete."
else
    for MODEL in ${MODELS}; do
        echo "--- Attention: ${MODEL} ---"
        python scripts/run_attention.py \
            model="${MODEL}" \
            output_dir="${SCRATCH}/activations" \
            hydra.run.dir="${SCRATCH}/hydra/attention/${MODEL}" \
            || echo "WARNING: Attention failed for ${MODEL}"
    done
    mark_stage_done "attention"
fi

# ---- Stage 5: Geometry ------------------------------------------------------
echo ""
echo "=== Stage 5/5: Geometry ==="
if stage_done "geometry"; then
    echo "[SKIP] Geometry already complete."
else
    for MODEL in ${MODELS}; do
        echo "--- Geometry: ${MODEL} ---"
        python scripts/run_geometry.py \
            model="${MODEL}" \
            output_dir="${SCRATCH}/activations" \
            hydra.run.dir="${SCRATCH}/hydra/geometry/${MODEL}" \
            || echo "WARNING: Geometry failed for ${MODEL}"
    done
    mark_stage_done "geometry"
fi

# ---- Done -------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Pipeline complete."
echo "  End time: $(date)"
echo "  Results:  ${SCRATCH}/activations/"
echo "============================================"
