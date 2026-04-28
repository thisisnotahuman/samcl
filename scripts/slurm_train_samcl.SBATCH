#!/bin/bash
#SBATCH --job-name=semantic_3
#SBATCH --output=/scratch/xj2366/samcl/log_new/semantic_3/%x/%x_%A_%a.out
#SBATCH --error=/scratch/xj2366/samcl/log_new/semantic_3/%x/%x_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xj2366@nyu.edu
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=48:00:00
#SBATCH --array=0-11

set -euo pipefail

module purge

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# -------------------------- Seeds (SLURM array) --------------------------
# SEED_LIST=(1337 2002 218 3456 4567 5678 6789 7890 8901 9012 1234 2345)
SEED_LIST=(1337 2002 218 3456 4567 5678)
export SEED=${SEED_LIST[$SLURM_ARRAY_TASK_ID]}

# -------------------------- Experiment hyperparams --------------------------
# Sampling strategy: random | semantic
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-semantic}

# Semantic mix (only used when SAMPLING_STRATEGY=semantic)
MIX_R1=${MIX_R1:-0.00}
MIX_R2=${MIX_R2:-0.00}
MIX_R3=${MIX_R3:-0.00}
MIX_R4=${MIX_R4:-1.00}

# Similarity thresholds (teacher cosine)
TEXT_SIM_THRESHOLD=${TEXT_SIM_THRESHOLD:-0.30}
IMAGE_SIM_THRESHOLD=${IMAGE_SIM_THRESHOLD:-0.65}

# Semantic sampling compute device for teacher embeddings: cpu | cuda
# Setting to cuda can significantly speed up global/multi modes but uses GPU VRAM.
SEMANTIC_COMPUTE_DEVICE=${SEMANTIC_COMPUTE_DEVICE:-cpu}

# Image top-k option (set USE_IMAGE_TOPK=1 to enable)
USE_IMAGE_TOPK=${USE_IMAGE_TOPK:-0}
IMAGE_TOPK=${IMAGE_TOPK:-50}

# Semantic sampler rejection-sampling budget (higher => fewer fallbacks but slower)
SAMPLER_MAX_TRIES=${SAMPLER_MAX_TRIES:-1000}

# Semantic sampler mode:
#   - single/c: scheme C (single-anchor, original)
#   - multi/b:  scheme B (multi-anchor)
#   - global/a: scheme A (global batch-level greedy)
SEMANTIC_SAMPLER_MODE=${SEMANTIC_SAMPLER_MODE:-single}
SEMANTIC_SAMPLER_NUM_ANCHORS=${SEMANTIC_SAMPLER_NUM_ANCHORS:-3}
# Leave empty to use "majority of anchors" default
SEMANTIC_SAMPLER_MIN_ANCHOR_MATCHES=${SEMANTIC_SAMPLER_MIN_ANCHOR_MATCHES:-}
SEMANTIC_SAMPLER_GLOBAL_NUM_CANDIDATES=${SEMANTIC_SAMPLER_GLOBAL_NUM_CANDIDATES:-256}

# Train / eval
BATCH_SIZE=${BATCH_SIZE:-128}
MAX_STEPS=${MAX_STEPS:-30000}
EVAL_EVERY=${EVAL_EVERY:-300}        # default in code is 300, keep explicit here
EVAL_MAX_PAIRS=${EVAL_MAX_PAIRS:-5000}

# Main model (uni-modal backbones)
VISION_MODEL_NAME=${VISION_MODEL_NAME:-google/vit-base-patch16-224-in21k}
TEXT_MODEL_NAME=${TEXT_MODEL_NAME:-bert-base-uncased}
PROJ_DIM=${PROJ_DIM:-256}
PROJ_HIDDEN_DIM=${PROJ_HIDDEN_DIM:-512}
PROJ_LAYERS=${PROJ_LAYERS:-2}
INIT_TEMPERATURE=${INIT_TEMPERATURE:-0.07}

# Teacher (can override if you want)
TEACHER_TEXT_MODEL=${TEACHER_TEXT_MODEL:-sentence-transformers/all-MiniLM-L6-v2}
TEACHER_IMAGE_ARCH=${TEACHER_IMAGE_ARCH:-resnet50}
TEACHER_IMAGE_MODEL=${TEACHER_IMAGE_MODEL:-openai/clip-vit-base-patch32}

# Data paths (host paths; Singularity usually bind-mounts /home and /scratch by default)
COCO_IMAGES_DIR=${COCO_IMAGES_DIR:-/scratch/xj2366/samcl/data/coco2017/images/train2017}
COCO_CAPTIONS_JSON=${COCO_CAPTIONS_JSON:-/scratch/xj2366/samcl/data/coco2017/annotations/captions_train2017.json}

# Repo root on host
REPO_DIR=${REPO_DIR:-/scratch/xj2366/samcl/SAMCL}

# -------------------------- Local tmp/cache to avoid GPFS issues --------------------------
TMP_BASE="/tmp/$USER/apptainer"
mkdir -p "$TMP_BASE/tmp" "$TMP_BASE/cache"
export APPTAINER_TMPDIR="$TMP_BASE/tmp"
export APPTAINER_CACHEDIR="$TMP_BASE/cache"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"

RUN_TAG=${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

# Put model caches on node-local disk (recommended)
HF_HOME="/tmp/$USER/huggingface"
mkdir -p "$HF_HOME"

# Teacher embedding cache + run dir (prefer scratch if available)
RUNS_BASE=${RUNS_BASE:-/scratch/xj2366/samcl/runs}
CACHE_BASE=${CACHE_BASE:-/scratch/xj2366/samcl/cache}
mkdir -p "$RUNS_BASE" "$CACHE_BASE"

RUN_DIR="$RUNS_BASE/${SAMPLING_STRATEGY}_seed${SEED}_${RUN_TAG}"
CACHE_DIR="$CACHE_BASE/teacher_${SAMPLING_STRATEGY}_seed${SEED}_${RUN_TAG}"

echo "[host] Node: $(hostname)"
echo "[host] REPO_DIR=$REPO_DIR"
echo "[host] RUN_DIR=$RUN_DIR"
echo "[host] CACHE_DIR=$CACHE_DIR"
echo "[host] STRATEGY=$SAMPLING_STRATEGY seed=$SEED"

# -------------------------- Singularity / Overlay --------------------------
# 直接复用你给的路径（需要你本机存在）
OVERLAY_ENV=${OVERLAY_ENV:-/scratch/xj2366/mcl/mcl-env/mcl/mcl.ext3}
SIF=${SIF:-/scratch/xj2366/containers/cuda12.1.1-cudnn8-devel-ubuntu22.04.sif}

# -------------------------- Launch in container --------------------------
apptainer exec --nv \
  --overlay "${OVERLAY_ENV}:ro" \
  "${SIF}" \
  /bin/bash -lc 'set -euo pipefail
    echo "[container] Node: $(hostname)"
    nvidia-smi
    source /ext3/env.sh
    conda activate mcl

    export HF_HOME="'"${HF_HOME}"'"
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export HF_HUB_CACHE="$HF_HOME/hub"

    cd "'"${REPO_DIR}"'"
    export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

    # Build command
    CMD=(python -u -m samcl.train
      --coco_images_dir "'"${COCO_IMAGES_DIR}"'"
      --coco_captions_json "'"${COCO_CAPTIONS_JSON}"'"
      --sampling_strategy "'"${SAMPLING_STRATEGY}"'"
      --sampler_max_tries "'"${SAMPLER_MAX_TRIES}"'"
      --semantic_compute_device "'"${SEMANTIC_COMPUTE_DEVICE}"'"
      --seed "'"${SEED}"'"
      --batch_size "'"${BATCH_SIZE}"'"
      --max_steps "'"${MAX_STEPS}"'"
      --eval_every "'"${EVAL_EVERY}"'" --eval_max_pairs "'"${EVAL_MAX_PAIRS}"'"
      --vision_model_name "'"${VISION_MODEL_NAME}"'"
      --text_model_name "'"${TEXT_MODEL_NAME}"'"
      # --vision_from_scratch --text_from_scratch
      --proj_dim "'"${PROJ_DIM}"'" --proj_hidden_dim "'"${PROJ_HIDDEN_DIM}"'" --proj_layers "'"${PROJ_LAYERS}"'"
      --init_temperature "'"${INIT_TEMPERATURE}"'"
      --teacher_text_model "'"${TEACHER_TEXT_MODEL}"'"
      --teacher_image_arch "'"${TEACHER_IMAGE_ARCH}"'"
      --teacher_image_model "'"${TEACHER_IMAGE_MODEL}"'"
      --text_sim_threshold "'"${TEXT_SIM_THRESHOLD}"'" --image_sim_threshold "'"${IMAGE_SIM_THRESHOLD}"'"
      --cache_dir "'"${CACHE_DIR}"'"
      --run_dir "'"${RUN_DIR}"'"
      --num_workers "'"${SLURM_CPUS_PER_TASK:-8}"'" --pin_memory
    )

    if [[ "'"${SAMPLING_STRATEGY}"'" == "semantic" ]]; then
      CMD+=(--mix_r1 "'"${MIX_R1}"'" --mix_r2 "'"${MIX_R2}"'" --mix_r3 "'"${MIX_R3}"'" --mix_r4 "'"${MIX_R4}"'")
      CMD+=(--semantic_sampler_mode "'"${SEMANTIC_SAMPLER_MODE}"'")
      # multi-mode knobs (safe to pass; ignored by other modes)
      CMD+=(--semantic_sampler_num_anchors "'"${SEMANTIC_SAMPLER_NUM_ANCHORS}"'")
      if [[ -n "'"${SEMANTIC_SAMPLER_MIN_ANCHOR_MATCHES}"'" ]]; then
        CMD+=(--semantic_sampler_min_anchor_matches "'"${SEMANTIC_SAMPLER_MIN_ANCHOR_MATCHES}"'")
      fi
      # global-mode knob (safe to pass; ignored by other modes)
      CMD+=(--semantic_sampler_global_num_candidates "'"${SEMANTIC_SAMPLER_GLOBAL_NUM_CANDIDATES}"'")
      if [[ "'"${USE_IMAGE_TOPK}"'" == "1" ]]; then
        CMD+=(--use_image_topk --image_topk "'"${IMAGE_TOPK}"'")
      fi
    fi

    printf "Running: "; printf "%q " "${CMD[@]}"; printf "\n"
    "${CMD[@]}"
  '

