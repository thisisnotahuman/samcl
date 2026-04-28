#!/bin/bash
# GPU training only. Prep teacher caches first (CPU/GPU prep job); set TEACHER_CACHE_SHARED (or this script's
# default under CACHE_DIR/openclip_coco_semantic_shared_teacher) to the same path as prep.
#
# OpenCLIP trains the full CLIP stack (not the legacy HF dual-tower recipe). Do not use --samcl-train-align here:
# that path forces const schedule + lr=1e-5 and is a poor match for ViT-B-32 from scratch. This script uses the
# OpenCLIP ViT defaults from the CLIP paper (lr=5e-4, cosine, wd=0.2). Override: export LR=... .
# OpenCLIP's global default warmup is 10000 steps (meant for long LAION-style runs). For short jobs (e.g.
# MAX_STEPS=20000), this script sets a smaller default WARMUP; override: export WARMUP=... .
# Before sbatch: export MAX_STEPS=... / EPOCHS=... to change length.
#
#SBATCH --job-name=oc_samcl_coco
#SBATCH --output=/scratch/xj2366/samcl/SAMCL/open_clip-main/logs_new/coco/%x_%A_%a.out
#SBATCH --error=/scratch/xj2366/samcl/SAMCL/open_clip-main/logs_new/coco/%x_%A_%a.err
#SBATCH --account=torch_pr_36_general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xj2366@nyu.edu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --array=0-3

set -euo pipefail

module purge

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# ---------- Paths (align with scripts/slurm_text_image_only.SBATCH: COCO + cache_dir) ----------
OVERLAY_ENV=/scratch/xj2366/mcl/mcl-env/mcl/mcl.ext3
SIF=/scratch/xj2366/containers/cuda12.1.1-cudnn8-devel-ubuntu22.04.sif

REPO_DIR=/scratch/xj2366/samcl/SAMCL/open_clip-main
# Same COCO layout as samcl.train --coco_images_dir / --coco_captions_json in slurm_text_image_only.SBATCH
COCO_IMAGES=/scratch/xj2366/samcl/data/coco2017/images/train2017
COCO_CAPTIONS_JSON=/scratch/xj2366/samcl/data/coco2017/annotations/captions_train2017.json

LOG_ROOT=/scratch/xj2366/samcl/SAMCL/open_clip-main/logs
# OpenCLIP: HF/tokenizer cache + teacher-cache parent; same root as SAMCL --cache_dir
CACHE_DIR=/scratch/xj2366/samcl/cache/default
# Shared teacher embedding dir for all array tasks (relation mode does not change teacher embeddings).
TEACHER_CACHE_SHARED="${TEACHER_CACHE_SHARED:-$CACHE_DIR/openclip_coco_semantic_shared_teacher}"

mkdir -p "$LOG_ROOT" "$CACHE_DIR" "$TEACHER_CACHE_SHARED"

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

RUN_NAME="${RUN_NAME:-${EXP_NAME}_seed${SEED}_v1}"
# OpenCLIP ViT-B-32 default LR (params.get_default_params); override before sbatch: export LR=3e-4
LR="${LR:-5e-4}"
# Cosine linear-warmup steps (optimizer steps). Code default 10000 is too long when MAX_STEPS~20k.
WARMUP="${WARMUP:-2000}"
# Legacy samcl.train default: 2000 optimizer steps. Override: export MAX_STEPS=20000 before sbatch.
MAX_STEPS="${MAX_STEPS:-20000}"
# One epoch ~= len(COCO_pairs)//batch_size batches; default max_steps=2000 stops inside epoch 0.
# Only raise EPOCHS when max_steps is larger than (batches-per-epoch * epochs) without early stop.
EPOCHS="${EPOCHS:-8}"

# ---------- Run ----------
apptainer exec --nv \
  --env REPO_DIR="$REPO_DIR" \
  --env COCO_IMAGES="$COCO_IMAGES" \
  --env COCO_CAPTIONS_JSON="$COCO_CAPTIONS_JSON" \
  --env LOG_ROOT="$LOG_ROOT" \
  --env CACHE_DIR="$CACHE_DIR" \
  --env TEACHER_CACHE_SHARED="$TEACHER_CACHE_SHARED" \
  --env RUN_NAME="$RUN_NAME" \
  --env REL_MODE="$REL_MODE" \
  --env MIX_SIMILAR="$MIX_SIMILAR" \
  --env MIX_DIFFERENT="$MIX_DIFFERENT" \
  --env SEED="$SEED" \
  --env LR="$LR" \
  --env WARMUP="$WARMUP" \
  --env MAX_STEPS="$MAX_STEPS" \
  --env EPOCHS="$EPOCHS" \
  --env PYTHONUNBUFFERED=1 \
  --overlay "${OVERLAY_ENV}:ro" \
  "${SIF}" \
  /bin/bash -lc '
    set -euo pipefail

    source /ext3/env.sh
    conda activate mcl

    export PYTHONPATH=/scratch/xj2366/python_userbase:$PYTHONPATH
    cd "$REPO_DIR"
    export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

    python -c "import sentence_transformers; print(\"sentence-transformers OK\")"

    python -u -m open_clip_train.main \
      --semantic-sampling \
      --train-data "$COCO_IMAGES" \
      --coco-captions-json "$COCO_CAPTIONS_JSON" \
      --semantic-relation-mode "$REL_MODE" \
      --mix-similar "$MIX_SIMILAR" \
      --mix-different "$MIX_DIFFERENT" \
      --semantic-sampler-mode global \
      --semantic-sampler-compute-device cuda \
      --semantic-sampler-cache-teacher-on-device \
      --semantic-sampler-num-anchors 3 \
      --semantic-sampler-global-num-candidates 64 \
      --sampler-max-tries 3000 \
      --train-eval-type both \
      --samcl-retrieval-eval-every-n-steps 50 \
      --samcl-retrieval-eval-max-pairs 5000 \
      --seed "$SEED" \
      --batch-size 128 \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --warmup "$WARMUP" \
      --max-steps "$MAX_STEPS" \
      --workers 8 \
      --prefetch-factor 6 \
      --dataloader-worker-torch-threads 1 \
      --model ViT-B-32 \
      --pretrained "" \
      --logs "$LOG_ROOT" \
      --name "$RUN_NAME" \
      --save-most-recent \
      --save-most-recent-every-n-steps 500 \
      --resume latest \
      --cache-dir "$CACHE_DIR" \
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

echo "---- DONE: TASK $SLURM_ARRAY_TASK_ID ----"
