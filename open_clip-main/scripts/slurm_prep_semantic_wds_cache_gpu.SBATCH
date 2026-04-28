#!/bin/bash
# GPU-accelerated prep: same outputs as slurm_prep_semantic_wds_cache.SBATCH but runs text+image
# teachers on CUDA (ResNet image encoding is usually ~5–20x faster than CPU for full CC3M).
#
# WDS index still lives under CC3M_WDS/.openclip_semantic_index/
# Teacher caches still under TEACHER_CACHE_SHARED/<hash>/ — use the same paths as CPU prep / training.
#
# Raise --teacher-image-batch-size if you have VRAM headroom; --prep-wds-decode-workers uses extra CPUs for JPEG decode.
# Optional: --prep-gpu-log-util-interval 120 --prep-gpu-log-util-subsample-sec 5 (mean SM%%; lower subsample = more fork noise).
# Optional: --prep-debug-timings (INFO logs: where time goes, WDS decode vs GPU encode split).
# If the scheduler still requires artificial load, you can add --prep-gpu-util-boost (slows prep).
#
#SBATCH --job-name=prep_samcl_wds_gpu
#SBATCH --output=/scratch/xj2366/samcl/logs/%x_%j.out
#SBATCH --error=/scratch/xj2366/samcl/logs/%x_%j.err
#SBATCH --account=torch_pr_36_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xj2366@nyu.edu

set -euo pipefail

module purge

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

OVERLAY_ENV=/scratch/xj2366/mcl/mcl-env/mcl/mcl_.ext3
SIF=/scratch/xj2366/containers/cuda12.1.1-cudnn8-devel-ubuntu22.04.sif

REPO_DIR="${REPO_DIR:-/scratch/xj2366/samcl/SAMCL/open_clip-main}"
CC3M_WDS="${CC3M_WDS:-/scratch/xj2366/samcl/data/cc3m_wds}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/xj2366/samcl/cache/samcl_teacher_wds}"
TEACHER_CACHE_SHARED="${TEACHER_CACHE_SHARED:-$CACHE_ROOT/cc3m_semantic_shared_teacher}"

mkdir -p /scratch/xj2366/samcl/logs "$CACHE_ROOT" "$TEACHER_CACHE_SHARED"

TMP_BASE="/tmp/$USER/apptainer"
mkdir -p "$TMP_BASE/tmp" "$TMP_BASE/cache"
export APPTAINER_TMPDIR="$TMP_BASE/tmp"
export APPTAINER_CACHEDIR="$TMP_BASE/cache"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"

export PYTHONUNBUFFERED=1
# Quiet HuggingFace tokenizers when nvidia-smi (subprocess/fork) runs after text teacher init.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "JOBID=$SLURM_JOB_ID NODE=$(hostname) DATE=$(date -Is)"
echo "REPO_DIR=$REPO_DIR CC3M_WDS=$CC3M_WDS TEACHER_CACHE_SHARED=$TEACHER_CACHE_SHARED (GPU prep)"

apptainer exec --nv \
  --env REPO_DIR="$REPO_DIR" \
  --env CC3M_WDS="$CC3M_WDS" \
  --env CACHE_ROOT="$CACHE_ROOT" \
  --env TEACHER_CACHE_SHARED="$TEACHER_CACHE_SHARED" \
  --env PYTHONUNBUFFERED=1 \
  --env TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}" \
  --overlay "${OVERLAY_ENV}:ro" \
  "${SIF}" \
  /bin/bash -lc '
    set -euo pipefail
    source /ext3/env.sh
    conda activate mcl
    export PYTHONPATH=/scratch/xj2366/python_userbase:${PYTHONPATH:-}
    cd "$REPO_DIR"
    export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

    python -u -m open_clip_train.prep_semantic_cache \
      --semantic-sampling \
      --train-data "$CC3M_WDS" \
      --semantic-wds-shard-glob "cc3m-train-*.tar" \
      --device cuda \
      --semantic-sampler-compute-device cuda \
      --cache-dir "$CACHE_ROOT" \
      --teacher-cache-dir "$TEACHER_CACHE_SHARED" \
      --teacher-text-model sentence-transformers/all-MiniLM-L6-v2 \
      --teacher-image-arch resnet50 \
      --teacher-text-batch-size 128 \
      --teacher-image-batch-size 512 \
      --prep-wds-decode-workers 8
  '

echo "---- prep (GPU) done ----"
echo "Next: slurm_cc3m_semantic.SBATCH with same TEACHER_CACHE_SHARED."
