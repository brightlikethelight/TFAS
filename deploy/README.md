# Deployment Guide

GPU deployment infrastructure for the s1s2 activation extraction and analysis pipeline.

## Platform Comparison

| | RunPod | FASRC / Kempner |
|---|---|---|
| **Best for** | Quick iteration, full control | Long runs, free compute |
| **GPUs** | A100-80GB, H100-80GB | A100-80GB, H100, H200 |
| **Cost** | $0.82--2.69/hr | Free (PI allocation SUs) |
| **Max wall time** | Unlimited | 7hr (gpu_requeue) |
| **Setup time** | ~30 min | ~15 min + queue wait |
| **Preemption** | No | Yes (gpu_requeue) |
| **Storage** | /workspace (persistent) | /n/holyscratch01/ (90-day) |

**Recommendation:** Use RunPod for development and debugging. Use Kempner for
production runs (free compute, but must handle preemption).

## Quick Start: RunPod

```bash
# 1. Create an A100-80GB pod on RunPod (Community Cloud for cheaper rates)
#    Select a PyTorch template with CUDA 12.1+

# 2. SCP the repo to the pod
scp -r ./s1s2 root@<pod-ip>:/workspace/

# 3. SSH in and run setup
ssh root@<pod-ip>
export HF_TOKEN=hf_...  # needed for gated models (Llama)
bash /workspace/s1s2/deploy/setup_runpod.sh

# 4. Run extraction (in tmux for safety)
tmux new -s extract
conda activate s1s2
export HF_HOME=/workspace/hf_cache
python scripts/extract_all.py run_name=main

# 5. Retrieve results
# From local machine:
scp root@<pod-ip>:/workspace/s1s2/data/activations/main.h5 ./data/activations/
```

## Quick Start: FASRC / Kempner

```bash
# 1. SSH to FASRC login node
ssh <username>@login.rc.fas.harvard.edu

# 2. Clone repo and run setup
cd ~/projects
git clone <repo-url> s1s2
cd s1s2
bash deploy/setup_fasrc.sh

# 3. Download models (on a compute node with internet)
srun --partition=test --time=01:00:00 --mem=32G \
    python deploy/download_models.py --cache-dir /n/holyscratch01/$USER/s1s2/hf_cache

# 4. Submit extraction job
sbatch deploy/slurm_extract.sh

# 5. Check job status
squeue -u $USER
sacct -j <job-id> --format=JobID,State,Elapsed,MaxRSS

# 6. If preempted, just resubmit — checkpointing handles resume
sbatch deploy/slurm_extract.sh

# 7. Run full pipeline (after extraction completes)
sbatch deploy/slurm_full_pipeline.sh
```

## File Descriptions

| File | Purpose |
|---|---|
| `setup_runpod.sh` | One-shot setup for a fresh RunPod pod |
| `setup_fasrc.sh` | Environment setup on FASRC Cannon cluster |
| `slurm_extract.sh` | SLURM job for activation extraction (preemptible) |
| `slurm_probes.sh` | SLURM job for linear probing (CPU-only) |
| `slurm_full_pipeline.sh` | SLURM job for extract + probes + sae + attention + geometry |
| `verify_gpu.py` | Quick GPU sanity check (CUDA, VRAM, tensor alloc) |
| `download_models.py` | Pre-download all HF models and SAEs |
| `checkpoint_extract.py` | Extraction wrapper with per-model checkpointing |
| `cost_estimate.py` | Compute cost estimator for the full pipeline |

## Checkpointing Strategy

The Kempner `gpu_requeue` partition has a 7-hour wall time limit and can
preempt jobs at any time. The extraction checkpoint system handles this:

1. **Per-model granularity:** extraction processes one model at a time.
   After each model completes, a `.done` marker is written to the checkpoint
   directory.

2. **SIGUSR1 handling:** SLURM sends SIGUSR1 120 seconds before wall time
   (`--signal=B:USR1@120`). The checkpoint wrapper catches this, finishes
   the current model, saves, and exits cleanly.

3. **Automatic resume:** when the job is resubmitted, `checkpoint_extract.py`
   skips models that already have `.done` markers.

4. **Pipeline stages:** `slurm_full_pipeline.sh` uses stage markers
   (`stage_extract.done`, etc.) so re-submissions skip completed stages.

Worst case: a single model extraction is interrupted mid-way and must restart
from scratch. At ~10 minutes per model, this wastes at most ~10 minutes.

## Cost Estimates

Run `python deploy/cost_estimate.py` for a detailed breakdown. Summary:

| Stage | GPU hours | Wall time |
|---|---|---|
| Model download | 0 | ~30 min |
| Extraction (4 models, 284 problems) | ~0.7 hr | ~40 min |
| Probes | 0 | ~5 min |
| SAE analysis | ~2 hr | ~2 hr |
| Attention re-extraction | ~4 hr | ~4 hr |
| Geometry | 0 | ~10 min |
| **Total** | **~6.7 hr** | **~7.3 hr** |

**RunPod cost:** $6--$10 on A100-80GB (Community Cloud).
**FASRC cost:** ~67 SUs (covered by PI allocation).

## Environment Variables

All scripts respect these environment variables for path configuration:

| Variable | Default | Description |
|---|---|---|
| `S1S2_REPO` | `$(pwd)` / `/workspace/s1s2` | Path to repo root |
| `S1S2_SCRATCH` | `/n/holyscratch01/$USER/s1s2` | Scratch storage (FASRC) |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache directory |
| `HF_CACHE_DIR` | `/workspace/hf_cache` | Alias for HF cache (RunPod) |
| `HF_TOKEN` | (unset) | HuggingFace token for gated models |

## Retrieving Results

Results are written to:
- **Activations:** `$S1S2_SCRATCH/activations/main.h5` (FASRC) or `/workspace/s1s2/data/activations/main.h5` (RunPod)
- **Per-workstream:** `results/{workstream}/` in the repo directory
- **Figures:** `figures/` in the repo directory

To copy results off a RunPod pod:
```bash
scp root@<pod-ip>:/workspace/s1s2/data/activations/main.h5 ./data/activations/
scp -r root@<pod-ip>:/workspace/s1s2/results/ ./results/
```

To copy results from FASRC:
```bash
scp <user>@login.rc.fas.harvard.edu:/n/holyscratch01/<user>/s1s2/activations/main.h5 ./data/activations/
```

## Troubleshooting

**"CUDA out of memory"**: unlikely with 8B models on A100-80GB (~16GB per model).
If it happens, check that only one model is loaded at a time. The extraction
pipeline loads, extracts, and unloads models sequentially.

**"Gated model" download error**: set `HF_TOKEN` to a token with access to
`meta-llama/Llama-3.1-8B-Instruct`. Request access at https://huggingface.co/meta-llama.

**SAE download fails**: SAE repos are optional for initial extraction. The
`download_models.py` script treats SAE failures as warnings, not errors.

**Preempted mid-model**: the current model's extraction is lost, but all
previously completed models are checkpointed. Resubmit the job.

**SLURM job pending**: Kempner `gpu_requeue` is backfill. Wait time depends on
cluster load. For faster scheduling, use `--partition=kempner` (non-preemptible)
if your allocation allows.
