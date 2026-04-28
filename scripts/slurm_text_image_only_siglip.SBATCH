#!/bin/bash
#SBATCH --job-name=exp_matrix_infonce_siglip
#SBATCH --output=/scratch/xj2366/samcl/log_new/exp_matrix_infonce_siglip_pretrain/%x_%A_%a.out
#SBATCH --error=/scratch/xj2366/samcl/log_new/exp_matrix_infonce_siglip_pretrain/%x_%A_%a.err
#SBATCH --account=torch_pr_36_general
#SBATCH --mail-user=xj2366@nyu.edu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=24:00:00
#SBATCH --array=0-9

set -euo pipefail

module purge
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

SEED=1337

# =============================================================================
# Head-to-head: sampling (random vs semantic) × loss (InfoNCE vs SigLIP)
#
# 学长最小闭环里的三条线在这里都出现，并补上第 4 格做完整 2×2：
#   0  random  + InfoNCE   — baseline（无你们设计的 semantic negative）
#   1  random  + SigLIP    — SigLIP 且不依赖 semantic sampling
#   2–5  semantic + InfoNCE × 与原 exp1 相同的 4 种 relation/mix（你们方法 + softmax）
#   6–9  semantic + SigLIP × 同上（核心：SigLIP 范式里 semantic 还值不值钱）
#
# Random 分支不读 relation/mix，各跑 1 次即可；Semantic 与 slurm_text_image_only.SBATCH 对齐。
# =============================================================================

TASK=${SLURM_ARRAY_TASK_ID}

if [ "$TASK" -eq 0 ]; then
  SAMPLING_STRATEGY="random"
  PAIRWISE_LOSS="infonce"
  MATRIX_TAG="rand_infonce"
  REL_MODE="text_only"
  MIX_SIMILAR=1.00; MIX_DIFFERENT=0.00
  EXP_NAME="single_run"
elif [ "$TASK" -eq 1 ]; then
  SAMPLING_STRATEGY="random"
  PAIRWISE_LOSS="siglip"
  MATRIX_TAG="rand_siglip"
  REL_MODE="text_only"
  MIX_SIMILAR=1.00; MIX_DIFFERENT=0.00
  EXP_NAME="single_run"
elif [ "$TASK" -ge 2 ] && [ "$TASK" -le 5 ]; then
  SAMPLING_STRATEGY="semantic"
  PAIRWISE_LOSS="infonce"
  MATRIX_TAG="sem_infonce"
  REL_ID=$((TASK - 2))
  case $REL_ID in
    0)
      REL_MODE="text_only"
      MIX_SIMILAR=1.00; MIX_DIFFERENT=0.00
      EXP_NAME="text_only_similar"
      ;;
    1)
      REL_MODE="text_only"
      MIX_SIMILAR=0.00; MIX_DIFFERENT=1.00
      EXP_NAME="text_only_different"
      ;;
    2)
      REL_MODE="image_only"
      MIX_SIMILAR=1.00; MIX_DIFFERENT=0.00
      EXP_NAME="image_only_similar"
      ;;
    3)
      REL_MODE="image_only"
      MIX_SIMILAR=0.00; MIX_DIFFERENT=1.00
      EXP_NAME="image_only_different"
      ;;
  esac
elif [ "$TASK" -ge 6 ] && [ "$TASK" -le 9 ]; then
  SAMPLING_STRATEGY="semantic"
  PAIRWISE_LOSS="siglip"
  MATRIX_TAG="sem_siglip"
  REL_ID=$((TASK - 6))
  case $REL_ID in
    0)
      REL_MODE="text_only"
      MIX_SIMILAR=1.00; MIX_DIFFERENT=0.00
      EXP_NAME="text_only_similar"
      ;;
    1)
      REL_MODE="text_only"
      MIX_SIMILAR=0.00; MIX_DIFFERENT=1.00
      EXP_NAME="text_only_different"
      ;;
    2)
      REL_MODE="image_only"
      MIX_SIMILAR=1.00; MIX_DIFFERENT=0.00
      EXP_NAME="image_only_similar"
      ;;
    3)
      REL_MODE="image_only"
      MIX_SIMILAR=0.00; MIX_DIFFERENT=1.00
      EXP_NAME="image_only_different"
      ;;
  esac
else
  echo "Invalid SLURM_ARRAY_TASK_ID=$TASK (expected 0-9)" >&2
  exit 1
fi

echo "TASK=$TASK matrix=$MATRIX_TAG sampling=$SAMPLING_STRATEGY pairwise_loss=$PAIRWISE_LOSS rel=$REL_MODE exp_name=$EXP_NAME seed=$SEED"

SAMCL_SCRATCH="${SAMCL_SCRATCH:-0}"

REPO_DIR=/scratch/xj2366/samcl/SAMCL
RUN_DIR=/scratch/xj2366/samcl/runs/exp_sampling_loss_matrix/global_64_pretrain/${MATRIX_TAG}/${EXP_NAME}
CACHE_DIR=/scratch/xj2366/samcl/cache/default

# Eval: default retrieval-only. 4-AFC + retrieval: export SAMCL_EVAL_TYPE=both (requires JSON under REPO_DIR).
EVAL_TYPE="${SAMCL_EVAL_TYPE:-retrieval}"
FOUR_AFC_META="${FOUR_AFC_META:-$REPO_DIR/src/samcl/data/object_categories_eval/eval_object_categories_4150.json}"

mkdir -p "$RUN_DIR" "$CACHE_DIR"

OVERLAY_ENV=/scratch/xj2366/mcl/mcl-env/mcl/mcl.ext3
SIF=/scratch/xj2366/containers/cuda12.1.1-cudnn8-devel-ubuntu22.04.sif

apptainer exec --nv \
  --env SAMCL_SCRATCH="${SAMCL_SCRATCH}" \
  --overlay "${OVERLAY_ENV}:ro" \
  "${SIF}" \
  /bin/bash -lc "
    set -euo pipefail

    source /ext3/env.sh
    conda activate mcl

    cd $REPO_DIR
    export PYTHONPATH=\$PWD/src

    EXTRA=\"\"
    if [ \"\${SAMCL_SCRATCH:-0}\" = \"1\" ]; then
      EXTRA=\"--vision_from_scratch --text_from_scratch\"
    fi

    python -u -m samcl.train \
      --pairwise_loss $PAIRWISE_LOSS \
      --sampling_strategy $SAMPLING_STRATEGY \
      --coco_images_dir /scratch/xj2366/samcl/data/coco2017/images/train2017 \
      --coco_captions_json /scratch/xj2366/samcl/data/coco2017/annotations/captions_train2017.json \
      --semantic_relation_mode $REL_MODE \
      --mix_similar $MIX_SIMILAR --mix_different $MIX_DIFFERENT \
      --semantic_sampler_mode global \
      --semantic_sampler_compute_device cuda \
      --semantic_sampler_cache_teacher_on_device \
      --semantic_sampler_num_anchors 3 \
      --semantic_sampler_global_num_candidates 64 \
      --sampler_max_tries 3000 \
      --seed $SEED \
      --batch_size 128 \
      --max_steps 25000 \
      --eval_every 200 \
      --eval_max_pairs 5000 \
      --eval_type ${EVAL_TYPE} \
      --four_afc_metadata ${FOUR_AFC_META} \
      --vision_model_name google/vit-base-patch16-224-in21k \
      --text_model_name bert-base-uncased \
      --proj_dim 256 \
      --proj_hidden_dim 512 \
      --proj_layers 2 \
      --init_temperature 0.07 \
      --teacher_text_model sentence-transformers/all-MiniLM-L6-v2 \
      --teacher_image_arch resnet50 \
      --teacher_image_model openai/clip-vit-base-patch32 \
      --text_sim_threshold 0.30 \
      --image_sim_threshold 0.65 \
      --cache_dir $CACHE_DIR \
      --run_dir $RUN_DIR \
      --num_workers 8 \
      --pin_memory \
      --log_every 20 \
      \$EXTRA
  "
