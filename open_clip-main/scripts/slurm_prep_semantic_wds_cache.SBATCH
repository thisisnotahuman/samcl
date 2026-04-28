#!/bin/bash
# CPU-only: build CC3M WDS pair index under CC3M_WDS/.openclip_semantic_index/
# and teacher embedding caches under TEACHER_CACHE_SHARED/<hash>/.
#
# Run this BEFORE slurm_cc3m_semantic.SBATCH (GPU). Full CC3M on CPU is slow (~24h+ image ResNet);
# same cache paths as the GPU prep script — pick CPU if you prefer not to hold a GPU or I/O-bound prep.
# Use --semantic-max-pairs for a smoke test inside the python line below.
#
# Sharing across array tasks 0–3:
#   - WDS index: same for all jobs if same CC3M_WDS, shard glob, and semantic-max-pairs.
#   - Teacher .pt files: same if same --train-data / glob / max_pairs AND same
#     --teacher-cache-dir (use TEACHER_CACHE_SHARED, not per-RUN_NAME folders).
#
#SBATCH --job-name=prep_samcl_wds
#SBATCH --output=/scratch/xj2366/samcl/logs/%x_%j.out
#SBATCH --error=/scratch/xj2366/samcl/logs/%x_%j.err
#SBATCH --account=torch_pr_36_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xj2366@nyu.edu
#
# No GPU (--gres unset). apptainer exec without --nv.

set -euo pipefail

module purge

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

OVERLAY_ENV=/scratch/xj2366/mcl/mcl-env/mcl/mcl_.ext3
SIF=/scratch/xj2366/containers/cuda12.1.1-cudnn8-devel-ubuntu22.04.sif

REPO_DIR="${REPO_DIR:-/scratch/xj2366/samcl/SAMCL/open_clip-main}"
CC3M_WDS="${CC3M_WDS:-/scratch/xj2366/samcl/data/cc3m_wds}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/xj2366/samcl/cache/samcl_teacher_wds}"
# Must match training script --teacher-cache-dir parent for shared teacher weights.
TEACHER_CACHE_SHARED="${TEACHER_CACHE_SHARED:-$CACHE_ROOT/cc3m_semantic_shared_teacher}"

mkdir -p /scratch/xj2366/samcl/logs "$CACHE_ROOT" "$TEACHER_CACHE_SHARED"

TMP_BASE="/tmp/$USER/apptainer"
mkdir -p "$TMP_BASE/tmp" "$TMP_BASE/cache"
export APPTAINER_TMPDIR="$TMP_BASE/tmp"
export APPTAINER_CACHEDIR="$TMP_BASE/cache"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "JOBID=$SLURM_JOB_ID NODE=$(hostname) DATE=$(date -Is)"
echo "REPO_DIR=$REPO_DIR CC3M_WDS=$CC3M_WDS TEACHER_CACHE_SHARED=$TEACHER_CACHE_SHARED"

apptainer exec \
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
      --device cpu \
      --semantic-sampler-compute-device cpu \
      --cache-dir "$CACHE_ROOT" \
      --teacher-cache-dir "$TEACHER_CACHE_SHARED" \
      --teacher-text-model sentence-transformers/all-MiniLM-L6-v2 \
      --teacher-image-arch resnet50 \
      --teacher-text-batch-size 64 \
      --teacher-image-batch-size 256 \
      --prep-wds-decode-workers 8
  '

echo "---- prep done (host) ----"
echo "Next: submit GPU job slurm_cc3m_semantic.SBATCH with the same TEACHER_CACHE_SHARED path."
