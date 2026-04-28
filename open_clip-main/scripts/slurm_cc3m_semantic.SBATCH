#!/bin/bash
# GPU training only. Run scripts/slurm_prep_semantic_wds_cache.SBATCH first (CPU) to build
# WDS index + teacher caches; set TEACHER_CACHE_SHARED to the same path in both scripts.
#
#SBATCH --job-name=oc_samcl_cc3m
#SBATCH --output=/scratch/xj2366/samcl/SAMCL/open_clip-main/logs_new/cc3m/%x_%A_%a.out
#SBATCH --error=/scratch/xj2366/samcl/SAMCL/open_clip-main/logs_new/cc3m/%x_%A_%a.err
#SBATCH --account=torch_pr_36_mren
#SBATCH --partition=l40s_mren
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xj2366@nyu.edu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --array=0-3

set -euo pipefail

module purge

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# ---------- Paths ----------
OVERLAY_ENV=/scratch/xj2366/mcl/mcl-env/mcl/mcl_.ext3
SIF=/scratch/xj2366/containers/cuda12.1.1-cudnn8-devel-ubuntu22.04.sif

REPO_DIR=/scratch/xj2366/samcl/SAMCL/open_clip-main
CC3M_WDS=/scratch/xj2366/samcl/data/cc3m_wds

LOG_ROOT=/scratch/xj2366/samcl/SAMCL/open_clip-main/logs/cc3m
TEACHER_CACHE_ROOT=/scratch/xj2366/samcl/cache/samcl_teacher_wds
# Shared across all SLURM array tasks (relation mode does not change teacher embeddings).
TEACHER_CACHE_SHARED="${TEACHER_CACHE_SHARED:-$TEACHER_CACHE_ROOT/cc3m_semantic_shared_teacher}"

mkdir -p "$LOG_ROOT" "$TEACHER_CACHE_ROOT" "$TEACHER_CACHE_SHARED"

export PYTHONUNBUFFERED=1

# ---------- Apptainer tmp ----------
TMP_BASE="/tmp/$USER/apptainer"
mkdir -p "$TMP_BASE/tmp" "$TMP_BASE/cache"
export APPTAINER_TMPDIR="$TMP_BASE/tmp"
export APPTAINER_CACHEDIR="$TMP_BASE/cache"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"

# ---------- Experiment config ----------
SEED=1337

case $SLURM_ARRAY_TASK_ID in
  0)
    REL_MODE="text_only"
    MIX_SIMILAR=1.0; MIX_DIFFERENT=0.0
    EXP_NAME="text_only_similar"
    ;;
  1)
    REL_MODE="text_only"
    MIX_SIMILAR=0.0; MIX_DIFFERENT=1.0
    EXP_NAME="text_only_different"
    ;;
  2)
    REL_MODE="image_only"
    MIX_SIMILAR=1.0; MIX_DIFFERENT=0.0
    EXP_NAME="image_only_similar"
    ;;
  3)
    REL_MODE="image_only"
    MIX_SIMILAR=0.0; MIX_DIFFERENT=1.0
    EXP_NAME="image_only_different"
    ;;
esac

# Stable experiment name per array task (EXP_NAME already differs by SLURM_ARRAY_TASK_ID).
# Checkpoints: LOG_ROOT/RUN_NAME/checkpoints/epoch_latest.pt (--save-most-recent + --resume latest).
# Re-submit the same array task to resume optimizer+weights. New experiment: override RUN_NAME.
# Note: open_clip saves epoch_latest at end of each epoch; with --epochs 1, a mid-epoch kill leaves
# no completed checkpoint unless you add intra-epoch saving elsewhere.
RUN_NAME="${RUN_NAME:-${EXP_NAME}_seed${SEED}_v1}"

# ---------- Run ----------
apptainer exec --nv \
  --env REPO_DIR="$REPO_DIR" \
  --env CC3M_WDS="$CC3M_WDS" \
  --env LOG_ROOT="$LOG_ROOT" \
  --env TEACHER_CACHE_ROOT="$TEACHER_CACHE_ROOT" \
  --env TEACHER_CACHE_SHARED="$TEACHER_CACHE_SHARED" \
  --env RUN_NAME="$RUN_NAME" \
  --env REL_MODE="$REL_MODE" \
  --env MIX_SIMILAR="$MIX_SIMILAR" \
  --env MIX_DIFFERENT="$MIX_DIFFERENT" \
  --env SEED="$SEED" \
  --env PYTHONUNBUFFERED=1 \
  --overlay "${OVERLAY_ENV}:ro" \
  "${SIF}" \
  /bin/bash -lc '
    set -euo pipefail

    source /ext3/env.sh
    conda activate mcl

    # ---- PYTHONPATH (critical) ----
    export PYTHONPATH=/scratch/xj2366/python_userbase:$PYTHONPATH
    cd "$REPO_DIR"
    export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

    # ---- sanity check ----
    python -c "import sentence_transformers; print(\"sentence-transformers OK\")"

    # ---- training ----
    python -u -m open_clip_train.main \
      --semantic-sampling \
      --train-data "$CC3M_WDS" \
      --semantic-wds-shard-glob "cc3m-train-*.tar" \
      --semantic-relation-mode "$REL_MODE" \
      --mix-similar "$MIX_SIMILAR" \
      --mix-different "$MIX_DIFFERENT" \
      --semantic-sampler-mode global \
      --semantic-sampler-compute-device cuda \
      --semantic-sampler-cache-teacher-on-device \
      --semantic-sampler-num-anchors 3 \
      --semantic-sampler-global-num-candidates 64 \
      --sampler-max-tries 3000 \
      --train-eval-type four_afc \
      --seed "$SEED" \
      --batch-size 128 \
      --train-num-samples 2560000 \
      --epochs 1 \
      --workers 8 \
      --prefetch-factor 6 \
      --dataloader-worker-torch-threads 1 \
      --model ViT-B-32 \
      --pretrained "" \
      --lr 1e-3 \
      --wd 0.1 \
      --warmup 1000 \
      --logs "$LOG_ROOT" \
      --name "$RUN_NAME" \
      --save-most-recent \
      --save-most-recent-every-n-steps 500 \
      --resume latest \
      --cache-dir "$TEACHER_CACHE_ROOT" \
      --teacher-cache-dir "$TEACHER_CACHE_SHARED" \
      --teacher-text-model sentence-transformers/all-MiniLM-L6-v2 \
      --teacher-image-arch resnet50 \
      --extra-gpu-reserve-mib 9216 \
      --extra-gpu-util-boost \
      --extra-gpu-boost-idle-only \
      --extra-gpu-boost-matmul-iters 12 \
      --extra-gpu-boost-duty 0.52 \
      --text-sim-threshold 0.30 \
      --image-sim-threshold 0.65 \
      --four-afc-eval-json "$REPO_DIR/data/object_categories_eval/eval_object_categories_4150.json" \
      --four-afc-eval-every-n-steps 50 \
      --four-afc-eval-type text \
      --four-afc-eval-max-trials 512 \
      --report-to none \
      --log-every-n-steps 20
  '
# 4AFC: JSON lives under REPO_DIR; trial image paths inside JSON must exist on the node (e.g.
# /scratch/xj2366/mcl/saycam_labeled/...). If reads fail inside Apptainer, add e.g.
#   --bind /scratch/xj2366/mcl/saycam_labeled:/scratch/xj2366/mcl/saycam_labeled
# to the apptainer exec line above.

echo "---- DONE: TASK $SLURM_ARRAY_TASK_ID ----"