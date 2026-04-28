#!/bin/bash
#SBATCH --job-name=exp2_sem1_cont
#SBATCH --output=/scratch/xj2366/samcl/log_new/exp2_sem1_cont/%x_%A_%a.out
#SBATCH --error=/scratch/xj2366/samcl/log_new/exp2_sem1_cont/%x_%A_%a.err
#SBATCH --account=torch_pr_36_mren
#SBATCH --partition=l40s_mren
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=48:00:00
#SBATCH --array=0-6

set -euo pipefail

module purge
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# --------------------------
# Fixed seed
# --------------------------
SEED=1337

# --------------------------
# Experiment 2: continuum via block_global with semantic_1 (R2-only)
# K list: 2,4,8,16,32,64,128
# --------------------------
K_LIST=(2 4 8 16 32 64 128)
K=${K_LIST[$SLURM_ARRAY_TASK_ID]}

MIX_R1=0.00
MIX_R2=1.00
MIX_R3=0.00
MIX_R4=0.00

EXP_NAME="semantic_1_blockK${K}"
echo "Running $EXP_NAME (K=$K) seed=$SEED"

# --------------------------
# Paths
# --------------------------
REPO_DIR=/scratch/xj2366/samcl/SAMCL
RUNS_BASE=/scratch/xj2366/samcl/runs/exp2_continuum/semantic_1
RUN_DIR=${RUNS_BASE}/K${K}
CACHE_DIR=/scratch/xj2366/samcl/cache/default

mkdir -p "$RUN_DIR" "$CACHE_DIR"

# --------------------------
# Container
# --------------------------
OVERLAY_ENV=/scratch/xj2366/mcl/mcl-env/mcl/mcl.ext3
SIF=/scratch/xj2366/containers/cuda12.1.1-cudnn8-devel-ubuntu22.04.sif

apptainer exec --nv \
  --overlay "${OVERLAY_ENV}:ro" \
  "${SIF}" \
  /bin/bash -lc "
    set -euo pipefail

    source /ext3/env.sh
    conda activate mcl

    cd $REPO_DIR
    export PYTHONPATH=\$PWD/src

    python -u -m samcl.train \
      --coco_images_dir /scratch/xj2366/samcl/data/coco2017/images/train2017 \
      --coco_captions_json /scratch/xj2366/samcl/data/coco2017/annotations/captions_train2017.json \
      --sampling_strategy semantic \
      --semantic_relation_mode full \
      --mix_r1 $MIX_R1 --mix_r2 $MIX_R2 --mix_r3 $MIX_R3 --mix_r4 $MIX_R4 \
      --semantic_sampler_mode block_global \
      --semantic_sampler_num_blocks $K \
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
      --auto_resume \
  "

