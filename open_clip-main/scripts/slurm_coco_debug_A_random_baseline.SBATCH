#!/bin/bash
# Debug A: COCO 对比学习「最小对照」——与 slurm_coco_semantic.SBATCH 独立。
# 目的：在**同一 COCO 配对表**上，用**均匀随机 batch**（无 teacher、无语义 BatchSampler）训练若干 step，
#       看 Contrastive_loss 是否明显低于 ln(batch_size)（128 时约 4.85）。
# 若 A 正常下降而语义主脚本不降 → 重点查语义采样/teacher；若 A 也不降 → 重点查优化器/AMP/resume/WD 等。
#
# 依赖代码：`--coco-random-baseline-dataloader`（见 open_clip_train/main.py + data_semantic.py）
#
# 用法：在集群上 `sbatch scripts/slurm_coco_debug_A_random_baseline.SBATCH`
# 可选：`export MAX_STEPS=500` 等覆盖下面默认值。

#SBATCH --job-name=oc_dbg_A
#SBATCH --output=/scratch/xj2366/samcl/SAMCL/open_clip-main/logs_debug/coco/%x_%j.out
#SBATCH --error=/scratch/xj2366/samcl/SAMCL/open_clip-main/logs_debug/coco/%x_%j.err
#SBATCH --account=torch_pr_36_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=1:00:00

set -euo pipefail

module purge
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

OVERLAY_ENV=/scratch/xj2366/mcl/mcl-env/mcl/mcl.ext3
SIF=/scratch/xj2366/containers/cuda12.1.1-cudnn8-devel-ubuntu22.04.sif
REPO_DIR=/scratch/xj2366/samcl/SAMCL/open_clip-main
COCO_IMAGES=/scratch/xj2366/samcl/data/coco2017/images/train2017
COCO_CAPTIONS_JSON=/scratch/xj2366/samcl/data/coco2017/annotations/captions_train2017.json
CACHE_DIR=/scratch/xj2366/samcl/cache/default
LOG_ROOT="${LOG_ROOT:-/scratch/xj2366/samcl/SAMCL/open_clip-main/logs_debug/coco}"

mkdir -p "$LOG_ROOT"

SEED=1337
LR="${LR:-5e-4}"
WARMUP="${WARMUP:-200}"
MAX_STEPS="${MAX_STEPS:-400}"
EPOCHS="${EPOCHS:-4}"
RUN_NAME="${RUN_NAME:-debug_A_coco_random_${SLURM_JOB_ID:-manual}}"

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
  --env RUN_NAME="$RUN_NAME" \
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

    python -u -m open_clip_train.main \
      --coco-random-baseline-dataloader \
      --train-data "$COCO_IMAGES" \
      --coco-captions-json "$COCO_CAPTIONS_JSON" \
      --train-eval-type none \
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
      --report-to none \
      --log-every-n-steps 10
  '

echo "---- DONE debug A: $RUN_NAME ----"
