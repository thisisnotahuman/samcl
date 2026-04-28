#!/bin/bash
#SBATCH --job-name=hf_cc3m_pull
#SBATCH --output=/scratch/xj2366/samcl/logs/%x_%j.out
#SBATCH --error=/scratch/xj2366/samcl/logs/%x_%j.err
#SBATCH --account=torch_pr_36_mren
#SBATCH --partition=l40s_mren
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xj2366@nyu.edu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#
# Download CC3M WebDataset from Hugging Face inside Apptainer + conda (same pattern as your mcl jobs).
# No GPU: do NOT use --gres=gpu; apptainer exec without --nv (CPU node friendly).
# Re-submit to resume if walltime runs out.
#
# Dataset: https://huggingface.co/datasets/pixparse/cc3m-wds

set -euo pipefail

module purge

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Same overlay + SIF as your torch jobs (conda env has pip)
OVERLAY_ENV=/scratch/xj2366/mcl/mcl-env/mcl/mcl.ext3
SIF=/scratch/xj2366/containers/cuda12.1.1-cudnn8-devel-ubuntu22.04.sif

TMP_BASE="/tmp/$USER/apptainer"
mkdir -p "$TMP_BASE/tmp" "$TMP_BASE/cache"
export APPTAINER_TMPDIR="$TMP_BASE/tmp"
export APPTAINER_CACHEDIR="$TMP_BASE/cache"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"

export HF_CC3M_REPO="${HF_CC3M_REPO:-pixparse/cc3m-wds}"
export HF_CC3M_DIR="${HF_CC3M_DIR:-/scratch/xj2366/samcl/data/cc3m_wds}"
export PYTHONUNBUFFERED=1

mkdir -p /scratch/xj2366/samcl/logs
mkdir -p "$HF_CC3M_DIR"

echo "SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname)  DATE=$(date -Is)"
echo "HF_CC3M_REPO=$HF_CC3M_REPO"
echo "HF_CC3M_DIR=$HF_CC3M_DIR"

echo "---- connectivity (host) ----"
if ! curl -sS --max-time 30 -o /dev/null -w "huggingface.co HTTP %{http_code}\n" https://huggingface.co; then
  echo "WARNING: curl to huggingface.co failed; download may still work."
fi

apptainer exec \
  --env HF_CC3M_REPO="$HF_CC3M_REPO" \
  --env HF_CC3M_DIR="$HF_CC3M_DIR" \
  --env PYTHONUNBUFFERED=1 \
  --overlay "${OVERLAY_ENV}:ro" \
  "${SIF}" \
  /bin/bash -lc '
    set -euo pipefail
    echo "[container] $(hostname) python: $(which python)"
    source /ext3/env.sh
    conda activate mcl
    python -m pip install -U pip
    python -m pip install -U "huggingface_hub[cli]"
    huggingface-cli download "${HF_CC3M_REPO}" \
      --repo-type dataset \
      --local-dir "${HF_CC3M_DIR}"
  '

echo "---- done (host) ----"
echo "Sample files under ${HF_CC3M_DIR}:"
find "${HF_CC3M_DIR}" -maxdepth 2 -type f \( -name "*.tar" -o -name "*.json" \) 2>/dev/null | head -n 20
echo "Next: OpenCLIP --dataset-type webdataset --train-data '<shard brace pattern under ${HF_CC3M_DIR}>'"
