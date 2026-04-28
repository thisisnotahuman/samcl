#!/bin/bash
#SBATCH --job-name=simple_pretrain_same_model
#SBATCH --output=/scratch/xj2366/samcl/log_new/simple_pretrain_same_model/%x_%A_%a.out
#SBATCH --error=/scratch/xj2366/samcl/log_new/simple_pretrain_same_model/%x_%A_%a.err
#SBATCH --account=torch_pr_36_mren
#SBATCH --partition=l40s_mren
#SBATCH --mail-user=xj2366@nyu.edu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=12:00:00
#SBATCH --array=0-3

set -euo pipefail

module purge
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

SEED=1337

case $SLURM_ARRAY_TASK_ID in
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

echo "Running EXP=$EXP_NAME (relation_mode=$REL_MODE) seed=$SEED semantic_teacher_same_as_student=1"

SAMCL_SCRATCH="${SAMCL_SCRATCH:-1}"

REPO_DIR=/scratch/xj2366/samcl/SAMCL
RUN_DIR=/scratch/xj2366/samcl/runs/exp1_relonly/global_64_scratch_same_model/${EXP_NAME}
CACHE_DIR=/scratch/xj2366/samcl/cache/coco_semantic_student_mirrored

mkdir -p "$RUN_DIR" "$CACHE_DIR" "/scratch/xj2366/samcl/log_new/simple_scratch_same_model"

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
      --coco_images_dir /scratch/xj2366/samcl/data/coco2017/images/train2017 \
      --coco_captions_json /scratch/xj2366/samcl/data/coco2017/annotations/captions_train2017.json \
      --sampling_strategy semantic \
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
      --max_steps 15000 \
      --eval_every 200 \
      --eval_max_pairs 5000 \
      --vision_model_name google/vit-base-patch16-224-in21k \
      --text_model_name bert-base-uncased \
      --proj_dim 256 \
      --proj_hidden_dim 512 \
      --proj_layers 2 \
      --init_temperature 0.07 \
      --semantic_teacher_same_as_student \
      --text_sim_threshold 0.30 \
      --image_sim_threshold 0.65 \
      --cache_dir $CACHE_DIR \
      --run_dir $RUN_DIR \
      --num_workers 8 \
      --pin_memory \
      --log_every 20 \
      \$EXTRA
  "
