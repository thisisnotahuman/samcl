#!/bin/bash
# Debug B: 与主脚本相同的 **语义采样** 路径，但额外每隔 N step 打 **全局 L2 grad norm**（AMP unscale 后）。
# 目的：判断梯度是否接近 0、是否爆炸、是否在 warmup 后仍无有效更新。
# 与 slurm_coco_semantic.SBATCH 独立；默认关闭周期性 retrieval/4AFC 以省时间（可自行打开）。
#
# 依赖代码：`--log-grad-norm-every-n-steps`（见 open_clip_train/train.py）
#
# 用法：`sbatch scripts/slurm_coco_debug_B_semantic_gradnorm.SBATCH`
# 可选：`export REL_MODE=text_only` / `export MIX_SIMILAR=1` / `export MIX_DIFFERENT=0` 等与主脚本 array 对齐。

#SBATCH --job-name=oc_dbg_B
#SBATCH --output=/scratch/xj2366/samcl/SAMCL/open_clip-main/logs_debug/coco/%x_%j.out
#SBATCH --error=/scratch/xj2366/samcl/SAMCL/open_clip-main/logs_debug/coco/%x_%j.err
#SBATCH --account=torch_pr_36_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=2:00:00

set -euo pipefail

module purge
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

OVERLAY_ENV=/scratch/xj2366/mcl/mcl-env/mcl/mcl.ext3
SIF=/scratch/xj2366/containers/cuda12.1.1-cudnn8-devel-ubuntu22.04.sif
REPO_DIR=/scratch/xj2366/samcl/SAMCL/open_clip-main
COCO_IMAGES=/scratch/xj2366/samcl/data/coco2017/images/train2017
COCO_CAPTIONS_JSON=/scratch/xj2366/samcl/data/coco2017/annotations/captions_train2017.json
CACHE_DIR=/scratch/xj2366/samcl/cache/default
TEACHER_CACHE_SHARED="${TEACHER_CACHE_SHARED:-$CACHE_DIR/openclip_coco_semantic_shared_teacher}"
LOG_ROOT="${LOG_ROOT:-/scratch/xj2366/samcl/SAMCL/open_clip-main/logs_debug/coco}"

mkdir -p "$LOG_ROOT" "$CACHE_DIR" "$TEACHER_CACHE_SHARED"

# 默认对齐主脚本 array task 0（text_only + similar）；可用环境变量覆盖。
REL_MODE="${REL_MODE:-text_only}"
MIX_SIMILAR="${MIX_SIMILAR:-1.0}"
MIX_DIFFERENT="${MIX_DIFFERENT:-0.0}"

SEED=1337
LR="${LR:-5e-4}"
WARMUP="${WARMUP:-2000}"
MAX_STEPS="${MAX_STEPS:-350}"
EPOCHS="${EPOCHS:-8}"
LOG_GRAD_EVERY="${LOG_GRAD_EVERY:-25}"
RUN_NAME="${RUN_NAME:-debug_B_semantic_gradnorm_${SLURM_JOB_ID:-manual}}"

TMP_BASE="/tmp/$USER/apptainer"
mkdir -p "$TMP_BASE/tmp" "$TMP_BASE/cache"
export APPTAINER_TMPDIR="$TMP_BASE/tmp"
export APPTAINER_CACHEDIR="$TMP_BASE/cache"

export PYTHONUNBUFFERED=1

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
  --env LOG_GRAD_EVERY="$LOG_GRAD_EVERY" \
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
      --train-eval-type none \
      --log-grad-norm-every-n-steps "$LOG_GRAD_EVERY" \
      --seed "$SEED" \
      --batch-size 128 \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --warmup "$WARMUP" \
      --max-steps "$MAX_STEPS" \
      --workers 8 \
      --prefetch-factor 4 \
      --dataloader-worker-torch-threads 1 \
      --model ViT-B-32 \
      --pretrained "" \
      --logs "$LOG_ROOT" \
      --name "$RUN_NAME" \
      --cache-dir "$CACHE_DIR" \
      --teacher-cache-dir "$TEACHER_CACHE_SHARED" \
      --teacher-text-model sentence-transformers/all-MiniLM-L6-v2 \
      --teacher-image-arch resnet50 \
      --text-sim-threshold 0.30 \
      --image-sim-threshold 0.65 \
      --report-to none \
      --log-every-n-steps 10
  '

echo "---- DONE debug B: $RUN_NAME ----"
