# Runbook: First GPU Day (RunPod)

RunPod variant of the first-GPU-day runbook. Use this when FASRC is unavailable or when you need faster iteration without SLURM queues.

**Key differences from FASRC**:
- No SLURM -- run scripts directly (in tmux for safety).
- Persistent storage at `/workspace/` -- survives pod restarts.
- You pay per hour -- track GPU time and terminate pods when idle.
- No preemption -- jobs run to completion.
- No `module load` -- pods come with CUDA pre-installed.

**Cost estimate**: $6--$10 on A100-80GB Community Cloud for the full pipeline (~7 GPU-hours).

---

## Phase 0: Pre-Flight (30 min)

### 0.1 Create a RunPod pod

1. Go to https://www.runpod.io/console/pods and click "Deploy."
2. Select **A100-80GB** (Community Cloud for cheaper rates, ~$0.82/hr).
3. Template: **RunPod PyTorch 2.x** (any CUDA 12.1+ template works).
4. Disk: at least **200 GB** persistent storage at `/workspace/`.
5. Deploy.

### 0.2 Transfer the repo

From your local machine:

```bash
# SCP is Bright's preferred method (no quoting issues):
scp -r ./s1s2 root@<pod-ip>:/workspace/
```

Or if the repo is on GitHub:

```bash
ssh root@<pod-ip>
cd /workspace
git clone <repo-url> s1s2
```

### 0.3 Set the HF token

```bash
ssh root@<pod-ip>
export HF_TOKEN=hf_...
```

### 0.4 Run the setup script

```bash
bash /workspace/s1s2/deploy/setup_runpod.sh
```

This installs system dependencies, creates the conda environment, verifies the GPU, downloads all models, and runs a smoke test. It takes ~30 min (mostly model downloads).

If you want to do it step-by-step instead:

```bash
cd /workspace/s1s2

# Install deps
apt-get update -qq && apt-get install -y -qq tmux nvtop htop

# Conda env
conda env create -f environment.yml
conda activate s1s2
pip install -e ".[sae,dev]"

# Verify GPU
python deploy/verify_gpu.py

# Download models
export HF_HOME=/workspace/hf_cache
python deploy/download_models.py --cache-dir /workspace/hf_cache

# Preflight
python deploy/preflight_check.py --cache-dir /workspace/hf_cache --scratch-dir /workspace
```

### 0.5 Start a tmux session

Always use tmux for GPU work. If your SSH connection drops, the job continues.

```bash
tmux new -s s1s2
conda activate s1s2
export HF_HOME=/workspace/hf_cache
export HF_TOKEN=hf_...
cd /workspace/s1s2
```

### 0.6 Start the cost clock

Track GPU-hours to avoid bill shock:

```bash
echo "Pod started: $(date)" >> /workspace/cost_log.txt
```

At the end of each session:

```bash
echo "Pod stopped: $(date)" >> /workspace/cost_log.txt
```

**Cost tracking rule**: terminate the pod (or at least stop it) whenever you are not actively using it. A100-80GB at $0.82/hr = $19.68/day if left running.

---

## Phase 1: Behavioral Validation -- The Week 2 Gate (1 hr)

### 1.1 Run behavioral extraction

No SLURM -- run directly:

```bash
python scripts/extract_all.py \
    run_name=behavioral \
    models_to_extract='[llama-3.1-8b-instruct,gemma-2-9b-it,r1-distill-llama-8b,r1-distill-qwen-7b]' \
    generation.max_new_tokens_standard=128 \
    generation.max_new_tokens_reasoning=512 \
    output_dir=/workspace/s1s2/data/activations \
    extraction.layers=none \
    device=cuda
```

**Estimated time**: ~30 min for all 4 models.

### 1.2 Check the go/no-go gate

```bash
python scripts/audit_benchmark.py
```

**GO criterion** (from pre-registration):
- At least 1 standard model shows >30% lure responses on conflict items.
- If models are >90% correct everywhere, the benchmark is too easy -- stop and adjust.

See the FASRC runbook Phase 1.3 for the full decision matrix.

---

## Phase 2: Full Extraction -- Llama-3.1-8B-Instruct First (1 hr)

### 2.1 Extract

```bash
python scripts/extract_all.py \
    run_name=main \
    models_to_extract='[llama-3.1-8b-instruct]' \
    output_dir=/workspace/s1s2/data/activations \
    device=cuda
```

### 2.2 Validate

```bash
python -c "
from s1s2.extract import validate_file
errors = validate_file('/workspace/s1s2/data/activations/main.h5')
print(f'{len(errors)} errors' if errors else 'HDF5 validates OK')
"
```

### 2.3 Quick probe sanity check

```bash
python scripts/run_probes.py \
    model=llama-3.1-8b-instruct \
    activations_path=/workspace/s1s2/data/activations/main.h5 \
    layers=[16] \
    targets=[task_type]
```

**Check**: AUC > 0.55 at layer 16 = promising. AUC near 0.50 = null result (valid but double-check extraction).

---

## Phase 3: R1-Distill-Llama-8B Extraction (2--4 hrs)

```bash
python scripts/extract_all.py \
    run_name=main \
    models_to_extract='[r1-distill-llama-8b]' \
    output_dir=/workspace/s1s2/data/activations \
    device=cuda
```

This takes longer than Llama because of `<think>` trace generation with `max_new_tokens_reasoning=4096` and `temperature=0.6`.

Validate and probe-check as in Phase 2.

---

## Phase 4: First Analysis Pass (1 hr)

### 4.1 Run probes on both models

```bash
python scripts/run_probes.py \
    model=llama-3.1-8b-instruct \
    activations_path=/workspace/s1s2/data/activations/main.h5 \
    layers=all \
    targets=[task_type]

python scripts/run_probes.py \
    model=r1-distill-llama-8b \
    activations_path=/workspace/s1s2/data/activations/main.h5 \
    layers=all \
    targets=[task_type]
```

### 4.2 Generate comparison figure

```bash
python scripts/generate_figures.py \
    --workstream probes \
    --models llama-3.1-8b-instruct,r1-distill-llama-8b \
    --results-dir results/probes \
    --figures-dir figures
```

### 4.3 Retrieve figures to local machine

From your local machine:

```bash
scp root@<pod-ip>:/workspace/s1s2/figures/*.pdf ./figures/
scp root@<pod-ip>:/workspace/s1s2/results/probes/*.json ./results/probes/
```

---

## Phase 5: Remaining Models + Full Analysis (4 hrs)

### 5.1 Extract remaining models

```bash
python scripts/extract_all.py \
    run_name=main \
    models_to_extract='[gemma-2-9b-it,r1-distill-qwen-7b]' \
    output_dir=/workspace/s1s2/data/activations \
    device=cuda
```

### 5.2 Run full pipeline

```bash
python scripts/run_pipeline.py \
    --stages probes,sae,attention,geometry \
    --activations /workspace/s1s2/data/activations/main.h5 \
    --results-dir results \
    --figures-dir figures
```

### 5.3 Generate all figures

```bash
python scripts/generate_figures.py \
    --workstream all \
    --results-dir results \
    --figures-dir figures
```

---

## Phase 6: Results Check + Cleanup

### 6.1 Audit and compare with pre-registration

```bash
python scripts/audit_benchmark.py
```

See the FASRC runbook Phase 6.2 for the hypothesis checklist.

### 6.2 Retrieve all results

From your local machine:

```bash
# Activations (large -- only if you need them locally):
scp root@<pod-ip>:/workspace/s1s2/data/activations/main.h5 ./data/activations/

# Results and figures (small):
scp -r root@<pod-ip>:/workspace/s1s2/results/ ./results/
scp -r root@<pod-ip>:/workspace/s1s2/figures/ ./figures/
```

### 6.3 Log GPU-hours and cost

```bash
echo "Session end: $(date)" >> /workspace/cost_log.txt
cat /workspace/cost_log.txt
```

Compute cost manually: hours x $0.82 (A100 Community Cloud) or hours x $1.25 (H100).

### 6.4 Stop or terminate the pod

**Stop** (preserves disk, stops billing GPU-hours, still billed for disk):
```bash
# Use the RunPod console or API
```

**Terminate** (deletes everything -- make sure you retrieved results first):
```bash
# Use the RunPod console
```

### 6.5 Update session state

```bash
# On your local machine after retrieving results:
$EDITOR docs/SESSION_STATE.md
```

---

## Troubleshooting

### SSH connection drops
That is why you use tmux. Reconnect and reattach:
```bash
ssh root@<pod-ip>
tmux attach -t s1s2
```

### Pod runs out of disk
Check usage:
```bash
df -h /workspace
du -sh /workspace/*
```
The HF cache is the biggest consumer. If you are tight on space:
```bash
# Delete model shards you no longer need:
du -sh /workspace/hf_cache/models--*
```

### CUDA out of memory
Unlikely on A100-80GB. Same guidance as FASRC runbook. Additionally:
```bash
# Check if another process is using the GPU:
nvidia-smi
# Kill any stale Python processes:
pkill -f "python.*extract"
```

### Pod was terminated unexpectedly
If you used persistent storage at `/workspace/`, your data survives. Create a new pod with the same persistent volume. If not, data is lost -- always retrieve results before terminating.

### Cost is higher than expected
Check your RunPod billing dashboard. Common causes:
- Pod left running overnight.
- Selected On-Demand instead of Community Cloud.
- Selected H100 ($2.69/hr) instead of A100 ($0.82/hr).

---

## Quick Reference: Key Paths (RunPod)

| What | Path |
|---|---|
| Repo | `/workspace/s1s2` |
| HF cache | `/workspace/hf_cache` |
| Activations | `/workspace/s1s2/data/activations/main.h5` |
| Probe results | `/workspace/s1s2/results/probes/` |
| Figures | `/workspace/s1s2/figures/` |
| Benchmark | `/workspace/s1s2/data/benchmark/benchmark.jsonl` |
| Cost log | `/workspace/cost_log.txt` |
| Models config | `/workspace/s1s2/configs/models.yaml` |
| Extraction config | `/workspace/s1s2/configs/extract.yaml` |

## Quick Reference: Environment Setup

```bash
# Every new shell / tmux pane:
conda activate s1s2
export HF_HOME=/workspace/hf_cache
export HF_TOKEN=hf_...
cd /workspace/s1s2
```
