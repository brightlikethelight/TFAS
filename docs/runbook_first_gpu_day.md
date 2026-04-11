# Runbook: First GPU Day (FASRC / Kempner)

Step-by-step guide to go from "I have a GPU allocation" to "first real results."

**Total estimated time**: 18--20 hours of wall clock (much of it is unattended extraction).
You can get through Phase 0--1 in a single afternoon. Phases 2--5 can run overnight.

**Pre-requisite**: FASRC account with Kempner PI sponsorship and access to `kempner_requeue` partition.

---

## Phase 0: Pre-Flight (30 min)

### 0.1 Clone the repo

```bash
ssh <username>@login.rc.fas.harvard.edu

cd ~/projects   # or wherever you keep repos
git clone <repo-url> s1s2
cd s1s2
```

### 0.2 Run the setup script

This creates the conda environment, installs PyTorch with CUDA 12.1, and sets up scratch directories.

```bash
bash deploy/setup_fasrc.sh
```

After it finishes, add the environment variables to your shell profile:

```bash
# Add to ~/.bashrc (the setup script prints these):
export S1S2_REPO=~/projects/s1s2
export S1S2_SCRATCH=/n/holyscratch01/${USER}/s1s2
export HF_HOME=/n/holyscratch01/${USER}/s1s2/hf_cache
```

Reload:

```bash
source ~/.bashrc
```

### 0.3 Set HuggingFace token

Llama-3.1-8B-Instruct is a gated model. You need a HuggingFace token with access granted.

1. Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and request access (usually instant).
2. Create an access token at https://huggingface.co/settings/tokens.
3. Set it:

```bash
export HF_TOKEN=hf_...
# Also add to ~/.bashrc for persistence across jobs.
```

### 0.4 Run pre-flight check

```bash
conda activate s1s2
python deploy/preflight_check.py --scratch-dir ${S1S2_SCRATCH} --skip-tokenizer
```

This checks Python version, CUDA (will show "no GPU" on a login node -- that is fine), s1s2 import, packages, disk space, and benchmark validation. All critical checks except CUDA/VRAM should pass on the login node.

### 0.5 Download all models + SAEs

Models are ~16 GB each. Download on a compute node (login nodes have restricted bandwidth):

```bash
srun --partition=test --time=01:00:00 --mem=32G \
    python deploy/download_models.py --cache-dir ${S1S2_SCRATCH}/hf_cache
```

This downloads all 4 models and 2 SAE repos:
- `meta-llama/Llama-3.1-8B-Instruct` (~16 GB)
- `google/gemma-2-9b-it` (~18 GB)
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (~16 GB)
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (~14 GB)
- `fnlp/Llama-3_1-8B-Base-LXR-32x` (Llama Scope SAE)
- `google/gemma-scope-9b-it-res` (Gemma Scope SAE)

**Estimated time**: 20--30 min on a fast connection.

### 0.6 Verify GPU access

Grab a quick interactive GPU session and run the full preflight:

```bash
srun --partition=kempner_requeue --gres=gpu:1 --time=00:15:00 --mem=32G --pty bash

conda activate s1s2
export HF_HOME=${S1S2_SCRATCH}/hf_cache

python deploy/preflight_check.py \
    --cache-dir ${S1S2_SCRATCH}/hf_cache \
    --scratch-dir ${S1S2_SCRATCH}
```

All checks should now pass (CUDA available, VRAM >= 40 GB, tokenizers load, smoke test passes).

**If the smoke test fails**: read the error output. The most common issues are:
- Missing package: `pip install -e ".[sae,dev]"` in the conda env.
- h5py version mismatch: `pip install --upgrade h5py`.
- Torch version without CUDA: reinstall with `pip install torch --index-url https://download.pytorch.org/whl/cu121`.

Exit the interactive session when done: `exit`.

---

## Phase 1: Behavioral Validation -- The Week 2 Gate (2 hrs)

This is the first **go/no-go gate** from the pre-registration (Section 5, "Pre-requisite: Behavioral Validation"). Do NOT proceed to full extraction if this gate fails.

### 1.1 Run behavioral extraction on all 4 models

We use short generations (`max_new_tokens=128`) to get behavioral responses without burning hours on full reasoning traces. This is enough to score correct/lure/other for each item.

```bash
# Submit as a SLURM job (more reliable than interactive for 4-model run):
sbatch <<'SLURM'
#!/bin/bash
#SBATCH --job-name=s1s2_behavioral
#SBATCH --partition=kempner_requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/behavioral_%j.out
#SBATCH --error=logs/behavioral_%j.err

module load python/3.11.0-fasrc01
module load cuda/12.1.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate s1s2
export HF_HOME=${S1S2_SCRATCH}/hf_cache
cd ${S1S2_REPO}

# Short generations for behavioral scoring only.
# Override max_new_tokens to 128 for standard models, 512 for reasoning
# (reasoning models need some room for <think> blocks).
python scripts/extract_all.py \
    run_name=behavioral \
    models_to_extract='[llama-3.1-8b-instruct,gemma-2-9b-it,r1-distill-llama-8b,r1-distill-qwen-7b]' \
    generation.max_new_tokens_standard=128 \
    generation.max_new_tokens_reasoning=512 \
    output_dir=${S1S2_SCRATCH}/activations \
    extraction.layers=none \
    device=cuda
SLURM
```

**Note**: `extraction.layers=none` skips activation extraction entirely -- we only need the behavioral responses (stored as text in the HDF5) for this gate.

**Estimated time**: ~30 min for all 4 models (short generations, no activation hooks).

### 1.2 Score behavioral responses

After the job completes, run the benchmark audit script to compute behavioral metrics:

```bash
python scripts/audit_benchmark.py
```

Then compute the per-model behavioral summary from the extraction outputs:

```bash
python -c "
from s1s2.extract import validate_file
from s1s2.utils.io import open_activations
import json, os

h5_path = os.path.join(os.environ['S1S2_SCRATCH'], 'activations', 'behavioral.h5')
errors = validate_file(h5_path)
if errors:
    print(f'Validation errors: {errors}')
else:
    print('HDF5 validates OK')
"
```

### 1.3 Evaluate the go/no-go criterion

**GO criterion** (from pre-registration):
- At least 1 standard model shows >30% lure responses on conflict items.
- Models are not >90% correct on conflict items (benchmark would be too easy).

**Decision matrix**:

| Lure rate on conflict items | Accuracy on no-conflict | Action |
|---|---|---|
| > 30% (at least 1 standard model) | > 70% | **GO** -- proceed to Phase 2 |
| 10--30% | > 70% | **CAUTION** -- may need to descope to categories with highest lure rates |
| < 10% | > 90% | **NO-GO** -- benchmark is too easy, need harder items |
| < 10% | < 70% | **NO-GO** -- models are confused, benchmark needs debugging |

If the gate passes, proceed. If it fails, stop and reassess the benchmark design before burning GPU hours on full extraction.

---

## Phase 2: Full Extraction -- Llama-3.1-8B-Instruct First (4 hrs)

We start with Llama because it is the simpler model in the headline comparison (Llama vs R1-Distill-Llama). Getting this right first validates the entire extraction pipeline.

### 2.1 Run full extraction

```bash
sbatch <<'SLURM'
#!/bin/bash
#SBATCH --job-name=s1s2_llama
#SBATCH --partition=kempner_requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/extract_llama_%j.out
#SBATCH --error=logs/extract_llama_%j.err
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

module load python/3.11.0-fasrc01
module load cuda/12.1.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate s1s2
export HF_HOME=${S1S2_SCRATCH}/hf_cache
cd ${S1S2_REPO}

python deploy/checkpoint_extract.py \
    --config configs/extract.yaml \
    --checkpoint-dir ${S1S2_SCRATCH}/checkpoints \
    --output-dir ${S1S2_SCRATCH}/activations \
    --models llama-3.1-8b-instruct
SLURM
```

**Estimated time**: ~40 min for 284 problems (batch_size=1, greedy decoding, all layers extracted).

### 2.2 Validate the HDF5 output

```bash
python -c "
from s1s2.extract import validate_file
errors = validate_file('${S1S2_SCRATCH}/activations/main.h5')
if errors:
    for e in errors:
        print(f'  ERROR: {e}')
else:
    print('HDF5 validates OK')
"
```

The validator checks: schema version, model groups, activation shapes, position labels, problem metadata, and text field completeness.

### 2.3 Quick probe sanity check

Run a single-layer probe to verify the extracted activations contain usable signal:

```bash
python scripts/run_probes.py \
    model=llama-3.1-8b-instruct \
    activations_path=${S1S2_SCRATCH}/activations/main.h5 \
    layers=[16] \
    targets=[task_type] \
    output_dir=results/probes
```

**What to look for**:
- **AUC > 0.55 at any layer** -> promising, the residual stream encodes something about conflict vs. no-conflict. Continue.
- **AUC near 0.50 at all layers** -> the probe finds nothing. This is a valid null result for H1 but worth double-checking the extraction. Re-run with `layers=[0,8,16,24,31]` to check the full depth.
- **AUC > 0.80 at layer 16** -> suspiciously high for a first pass. Verify the probe is not fitting on surface features: check Hewitt-Liang selectivity in the output JSON. Selectivity < 5pp means the signal is likely probe expressiveness, not representation.

Also run the positive control (7-way task category classification -- this should get high AUC trivially because surface features differ across categories):

```bash
python scripts/run_probes.py \
    model=llama-3.1-8b-instruct \
    activations_path=${S1S2_SCRATCH}/activations/main.h5 \
    layers=[16] \
    targets=[category]
```

If category probe AUC is low (<0.7), the extraction pipeline has a bug.

---

## Phase 3: R1-Distill-Llama-8B Extraction (4 hrs)

This is the headline comparison model. Same Llama architecture, but distilled from DeepSeek-R1 reasoning training. Differences between this and Phase 2 are attributable to reasoning distillation, not architecture.

### 3.1 Run extraction

```bash
sbatch <<'SLURM'
#!/bin/bash
#SBATCH --job-name=s1s2_r1llama
#SBATCH --partition=kempner_requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH --output=logs/extract_r1llama_%j.out
#SBATCH --error=logs/extract_r1llama_%j.err
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

module load python/3.11.0-fasrc01
module load cuda/12.1.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate s1s2
export HF_HOME=${S1S2_SCRATCH}/hf_cache
cd ${S1S2_REPO}

python deploy/checkpoint_extract.py \
    --config configs/extract.yaml \
    --checkpoint-dir ${S1S2_SCRATCH}/checkpoints \
    --output-dir ${S1S2_SCRATCH}/activations \
    --models r1-distill-llama-8b
SLURM
```

**Longer wall time**: R1-Distill generates `<think>...</think>` traces before answering. With `max_new_tokens_reasoning=4096` and `temperature=0.6` (sampling, not greedy), each problem takes longer. Budget 2--4 hours.

**If preempted**: the checkpoint system handles this. Just resubmit the same command. It skips the completed model if the `.done` marker exists.

### 3.2 Validate and quick-check

```bash
python -c "
from s1s2.extract import validate_file
errors = validate_file('${S1S2_SCRATCH}/activations/main.h5')
print(f'{len(errors)} errors' if errors else 'HDF5 validates OK')
"

python scripts/run_probes.py \
    model=r1-distill-llama-8b \
    activations_path=${S1S2_SCRATCH}/activations/main.h5 \
    layers=[16] \
    targets=[task_type]
```

---

## Phase 4: First Analysis Pass (2 hrs)

With both Llama models extracted, we can now run the core comparison that is the paper's central finding.

### 4.1 Run probes on both models (all layers)

```bash
# Llama -- all layers
python scripts/run_probes.py \
    model=llama-3.1-8b-instruct \
    activations_path=${S1S2_SCRATCH}/activations/main.h5 \
    layers=all \
    targets=[task_type] \
    output_dir=results/probes

# R1-Distill-Llama -- all layers
python scripts/run_probes.py \
    model=r1-distill-llama-8b \
    activations_path=${S1S2_SCRATCH}/activations/main.h5 \
    layers=all \
    targets=[task_type] \
    output_dir=results/probes
```

### 4.2 Compare layer-wise accuracy curves (the paper's Figure 2)

This is the money comparison. Plot AUC vs. layer for both models on the same axes:

```bash
python scripts/generate_figures.py \
    --workstream probes \
    --models llama-3.1-8b-instruct,r1-distill-llama-8b \
    --results-dir results/probes \
    --figures-dir figures
```

**What to look for**:

- **Different peak layers or AUC profiles** -> This IS the finding for H2. Reasoning distillation changed the internal S1/S2 separation.
- **R1-Distill has higher peak AUC** -> Consistent with H2 (reasoning training amplifies S1/S2 distinction).
- **Same profiles** -> H2 fails. Still publishable: reasoning distillation did not change internal representations despite changing external behavior.
- **Both flat at chance** -> H1 fails. The dual-process distinction is not linearly decodable.

### 4.3 Quick attention entropy check

If probes are promising, run attention on both models to start building the convergent evidence story:

```bash
python scripts/run_attention.py \
    model=llama-3.1-8b-instruct \
    activations_path=${S1S2_SCRATCH}/activations/main.h5

python scripts/run_attention.py \
    model=r1-distill-llama-8b \
    activations_path=${S1S2_SCRATCH}/activations/main.h5
```

---

## Phase 5: Remaining Models + Full Analysis (6 hrs)

### 5.1 Extract Gemma-2-9B-IT and R1-Distill-Qwen-7B

Use the SLURM batch script with all remaining models:

```bash
sbatch deploy/slurm_extract.sh --models gemma-2-9b-it r1-distill-qwen-7b
```

Or submit them as separate jobs to run in parallel (if you have the allocation):

```bash
sbatch <<'SLURM'
#!/bin/bash
#SBATCH --job-name=s1s2_gemma
#SBATCH --partition=kempner_requeue
#SBATCH --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=04:00:00
#SBATCH --output=logs/extract_gemma_%j.out
#SBATCH --signal=B:USR1@120 --requeue
module load python/3.11.0-fasrc01 cuda/12.1.0-fasrc01
eval "$(conda shell.bash hook)" && conda activate s1s2
export HF_HOME=${S1S2_SCRATCH}/hf_cache
cd ${S1S2_REPO}
python deploy/checkpoint_extract.py \
    --config configs/extract.yaml \
    --checkpoint-dir ${S1S2_SCRATCH}/checkpoints \
    --output-dir ${S1S2_SCRATCH}/activations \
    --models gemma-2-9b-it
SLURM

sbatch <<'SLURM'
#!/bin/bash
#SBATCH --job-name=s1s2_r1qwen
#SBATCH --partition=kempner_requeue
#SBATCH --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=07:00:00
#SBATCH --output=logs/extract_r1qwen_%j.out
#SBATCH --signal=B:USR1@120 --requeue
module load python/3.11.0-fasrc01 cuda/12.1.0-fasrc01
eval "$(conda shell.bash hook)" && conda activate s1s2
export HF_HOME=${S1S2_SCRATCH}/hf_cache
cd ${S1S2_REPO}
python deploy/checkpoint_extract.py \
    --config configs/extract.yaml \
    --checkpoint-dir ${S1S2_SCRATCH}/checkpoints \
    --output-dir ${S1S2_SCRATCH}/activations \
    --models r1-distill-qwen-7b
SLURM
```

### 5.2 Run all workstreams

Once all 4 models are extracted, run the full pipeline:

```bash
python scripts/run_pipeline.py \
    --stages probes,sae,attention,geometry \
    --activations ${S1S2_SCRATCH}/activations/main.h5
```

Or submit via SLURM for longer analysis stages:

```bash
sbatch deploy/slurm_full_pipeline.sh
```

### 5.3 Generate figures

```bash
python scripts/generate_figures.py \
    --workstream all \
    --results-dir results \
    --figures-dir figures
```

---

## Phase 6: Results Check

### 6.1 Audit behavioral results

```bash
python scripts/audit_benchmark.py
```

### 6.2 Compare with pre-registration predictions

Open `docs/preregistration.md` and check each hypothesis:

| Hypothesis | What to check | Where |
|---|---|---|
| H1 (linear decodability) | Peak-layer AUC > 0.6 in >= 2 models | `results/probes/` |
| H2 (reasoning amplification) | R1-Distill-Llama peak AUC > Llama peak AUC | `results/probes/` |
| H3 (SAE features) | >= 5 non-falsified differential features | `results/sae/` |
| H4 (causal) | Steering Delta P(correct) > 15pp | `results/causal/` (Phase 5+) |
| H5 (attention entropy) | >= 5% S2-specialized heads | `results/attention/` |
| H6 (geometric separability) | Silhouette > 0, p < 0.05 in >= 2 models | `results/geometry/` |

### 6.3 Update session state

```bash
# Update the session state with your findings:
$EDITOR docs/SESSION_STATE.md
```

Record:
- Which models were extracted successfully.
- Behavioral gate result (GO / NO-GO).
- Preliminary probe AUCs (any promising layers?).
- Any issues encountered.
- Modified files.

---

## Troubleshooting

### Job stays in PENDING state
Kempner `kempner_requeue` is backfill-scheduled. Check queue:
```bash
squeue -u $USER
squeue -p kempner_requeue --format="%.18i %.30j %.8u %.8T %.10M %.6D %R" | head -20
```
For faster scheduling, use `--partition=kempner` (non-preemptible, if your allocation allows).

### Job preempted mid-extraction
This is expected on `kempner_requeue`. The checkpoint system handles it:
```bash
# Check which models completed:
ls ${S1S2_SCRATCH}/checkpoints/*.done

# Resubmit -- it resumes from the last checkpoint:
sbatch deploy/slurm_extract.sh
```

### CUDA out of memory
Unlikely with 8B models on A100-80GB (~16 GB per model in bf16). If it happens:
1. Check that only one model is loaded: `nvidia-smi` while the job is running.
2. Verify `batch_size: 1` in `configs/extract.yaml`.
3. Try `torch_dtype: float16` instead of `bfloat16`.

### Gated model download error
```bash
# Check token is set:
echo $HF_TOKEN

# Request access (usually instant):
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

# Re-download:
python deploy/download_models.py --cache-dir ${S1S2_SCRATCH}/hf_cache --models llama-3.1-8b-instruct
```

### HDF5 validation fails
```bash
python -c "
from s1s2.extract import validate_file
errors = validate_file('${S1S2_SCRATCH}/activations/main.h5')
for e in errors:
    print(e)
"
```
Common causes: extraction was interrupted before writing problem metadata, or a model group is incomplete.

### Scratch storage running low
Holyscratch has a 90-day retention policy. Check usage:
```bash
du -sh ${S1S2_SCRATCH}/*
```
The full pipeline (4 models, all activations) uses roughly:
- Models cache: ~60 GB
- Activations HDF5: ~20 GB
- SAE cache: ~30 GB
- Total: ~110 GB

---

## Quick Reference: Key Paths

| What | Path |
|---|---|
| Repo | `${S1S2_REPO}` (default: `~/projects/s1s2`) |
| Scratch | `${S1S2_SCRATCH}` (default: `/n/holyscratch01/${USER}/s1s2`) |
| HF cache | `${S1S2_SCRATCH}/hf_cache` |
| Activations | `${S1S2_SCRATCH}/activations/main.h5` |
| Checkpoints | `${S1S2_SCRATCH}/checkpoints/` |
| SLURM logs | `${S1S2_REPO}/logs/` |
| Probe results | `${S1S2_REPO}/results/probes/` |
| Figures | `${S1S2_REPO}/figures/` |
| Benchmark | `${S1S2_REPO}/data/benchmark/benchmark.jsonl` |
| Models config | `${S1S2_REPO}/configs/models.yaml` |
| Extraction config | `${S1S2_REPO}/configs/extract.yaml` |

## Quick Reference: Key Commands

```bash
# Pre-flight
python deploy/preflight_check.py --scratch-dir ${S1S2_SCRATCH} --cache-dir ${S1S2_SCRATCH}/hf_cache

# Download models
srun --partition=test --time=01:00:00 --mem=32G python deploy/download_models.py --cache-dir ${S1S2_SCRATCH}/hf_cache

# Extraction (single model)
python deploy/checkpoint_extract.py --config configs/extract.yaml --checkpoint-dir ${S1S2_SCRATCH}/checkpoints --output-dir ${S1S2_SCRATCH}/activations --models llama-3.1-8b-instruct

# Extraction (all models via SLURM)
sbatch deploy/slurm_extract.sh

# Validate HDF5
python -c "from s1s2.extract import validate_file; print(validate_file('${S1S2_SCRATCH}/activations/main.h5'))"

# Probes (single layer)
python scripts/run_probes.py model=llama-3.1-8b-instruct activations_path=${S1S2_SCRATCH}/activations/main.h5 layers=[16] targets=[task_type]

# Full pipeline
python scripts/run_pipeline.py --activations ${S1S2_SCRATCH}/activations/main.h5

# Monitor jobs
squeue -u $USER
sacct -j <job-id> --format=JobID,State,Elapsed,MaxRSS
```
