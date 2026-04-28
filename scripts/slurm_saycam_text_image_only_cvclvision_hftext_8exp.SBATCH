#!/bin/bash
#SBATCH --job-name=saycam_relonly_cvclV_hfT_8
#SBATCH --output=/scratch/xj2366/samcl/log_new/saycam_exp1_relonly_cvclV_hfT_8/%x_%A_%a.out
#SBATCH --error=/scratch/xj2366/samcl/log_new/saycam_exp1_relonly_cvclV_hfT_8/%x_%A_%a.err
#SBATCH --account=torch_pr_36_mren
#SBATCH --partition=l40s_mren
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=48:00:00
#SBATCH --array=0-7

set -euo pipefail

module purge
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# --------------------------
# Fixed seed (per your request: 1 seed is enough)
# --------------------------
SEED=1337

# --------------------------
# 8 experiments = 4 relation conditions × 2 vision init
# init=0: CVCL vision pretrained (SayCam)  | init=1: CVCL vision from scratch
# --------------------------
REL_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))
INIT_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))

case $REL_IDX in
  0)
    REL_MODE="text_only"
    MIX_SIMILAR=1.00; MIX_DIFFERENT=0.00
    REL_NAME="text_only_similar"
    ;;
  1)
    REL_MODE="text_only"
    MIX_SIMILAR=0.00; MIX_DIFFERENT=1.00
    REL_NAME="text_only_different"
    ;;
  2)
    REL_MODE="image_only"
    MIX_SIMILAR=1.00; MIX_DIFFERENT=0.00
    REL_NAME="image_only_similar"
    ;;
  3)
    REL_MODE="image_only"
    MIX_SIMILAR=0.00; MIX_DIFFERENT=1.00
    REL_NAME="image_only_different"
    ;;
esac

if (( INIT_IDX == 0 )); then
  VISION_INIT_NAME="pretrained"
  CVCL_VISION_FROM_SCRATCH_FLAG=""
else
  VISION_INIT_NAME="scratch"
  CVCL_VISION_FROM_SCRATCH_FLAG="--cvcl_vision_from_scratch"
fi

EXP_NAME="${REL_NAME}_vision_${VISION_INIT_NAME}"
echo "Running SayCam EXP=$EXP_NAME (relation_mode=$REL_MODE) seed=$SEED | student=cvcl_vision_hf_text (text always from scratch)"

# --------------------------
# Paths (SayCam: metadata + 5fps frames)
# --------------------------
REPO_DIR=/scratch/xj2366/samcl/SAMCL
SAYCAM_ROOT=/scratch/xj2366/mcl/mcl/saycam_transcript_frames
RUN_DIR=/scratch/xj2366/samcl/runs/saycam_exp1_relonly/cvclV_hfT_single_oneframe/${EXP_NAME}
CACHE_DIR=/scratch/xj2366/samcl/cache/saycam_default

mkdir -p "$RUN_DIR" "$CACHE_DIR"

# --------------------------
# Container
# --------------------------
OVERLAY_ENV=/scratch/xj2366/mcl/mcl-env/mcl/mcl.ext3
SIF=/scratch/xj2366/containers/cuda12.1.1-cudnn8-devel-ubuntu22.04.sif
FPS_SQF=/scratch/xj2366/mcl/5fps.sqf

apptainer exec --nv \
  --overlay "${FPS_SQF}:ro" \
  --overlay "${OVERLAY_ENV}:ro" \
  "${SIF}" \
  /bin/bash -lc "
    set -euo pipefail

    source /ext3/env.sh
    conda activate mcl

    cd $REPO_DIR
    export PYTHONPATH=\$PWD/src

    # ls /5fps
    python -u -m samcl.train \
      --dataset_type saycam \
      --saycam_images_root /5fps \
      --saycam_metadata_json ${SAYCAM_ROOT}/expand_correct_seg.json \
      --saycam_mode one_frame \
      --sampling_strategy semantic \
      --semantic_relation_mode $REL_MODE \
      --mix_similar $MIX_SIMILAR --mix_different $MIX_DIFFERENT \
      --semantic_sampler_mode single \
      --semantic_sampler_compute_device cuda \
      --semantic_sampler_cache_teacher_on_device \
      --sampler_max_tries 3000 \
      --seed $SEED \
      --batch_size 128 \
      --max_steps 25000 \
      --eval_every 200 \
      --eval_max_pairs 5000 \
      --student_arch cvcl_vision_hf_text \
      --cvcl_vision_model_name dino_s_vitb16 \
      ${CVCL_VISION_FROM_SCRATCH_FLAG} \
      --cvcl_image_size 224 \
      --text_model_name bert-base-uncased \
      --text_from_scratch \
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
      --log_every 20
  "

