#!/bin/bash
#SBATCH --job-name=cmp_textsim_vs_iid
#SBATCH --output=/scratch/xj2366/samcl/log_new/cmp_textsim_vs_iid/%x/%x_%A_%a.out
#SBATCH --error=/scratch/xj2366/samcl/log_new/cmp_textsim_vs_iid/%x/%x_%A_%a.err
#SBATCH --account=torch_pr_36_mren
#SBATCH --partition=l40s_mren
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=48:00:00
#
# We run 2 conditions per seed:
#   cond=0: semantic text_only + similar
#   cond=1: random (iid)
#
# Total array tasks = 2 * num_seeds.
# array size should be 2 * num_seeds.
#SBATCH --array=0-11

set -euo pipefail

module purge
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# -------------------------- Seeds --------------------------
# Keep SEED_LIST explicit; update --array accordingly: 0-(2*NUM_SEEDS-1)
SEED_LIST=(1337 2002 218 3456 4567 5678)
# SEED_LIST=(6789 7890 8901 9012 1234 2345)

NUM_SEEDS=${#SEED_LIST[@]}
COND_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))

if (( SEED_IDX < 0 || SEED_IDX >= NUM_SEEDS )); then
  echo "ERROR: SEED_IDX=$SEED_IDX out of range (NUM_SEEDS=$NUM_SEEDS). Fix --array."
  exit 1
fi

SEED=${SEED_LIST[$SEED_IDX]}

# -------------------------- Condition mapping --------------------------
if (( COND_IDX == 0 )); then
  SAMPLING_STRATEGY="semantic"
  REL_MODE="text_only"
  MIX_SIMILAR=1.00
  MIX_DIFFERENT=0.00
  EXP_NAME="text_only_similar"
else
  SAMPLING_STRATEGY="random"
  EXP_NAME="iid_random"
fi

echo "Running EXP=$EXP_NAME strategy=$SAMPLING_STRATEGY seed=$SEED (task_id=$SLURM_ARRAY_TASK_ID)"

# -------------------------- Paths --------------------------
REPO_DIR=/scratch/xj2366/samcl/SAMCL
RUNS_BASE=/scratch/xj2366/samcl/runs/cmp_textsim_vs_iid/global_64_scratch
RUN_DIR=${RUNS_BASE}/${EXP_NAME}/seed${SEED}
CACHE_DIR=/scratch/xj2366/samcl/cache/default

mkdir -p "$RUN_DIR" "$CACHE_DIR"

# -------------------------- Container --------------------------
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

    CMD=(python -u -m samcl.train
      --coco_images_dir /scratch/xj2366/samcl/data/coco2017/images/train2017
      --coco_captions_json /scratch/xj2366/samcl/data/coco2017/annotations/captions_train2017.json
      --sampling_strategy $SAMPLING_STRATEGY
      --semantic_relation_mode ${REL_MODE:-full}
      --mix_similar ${MIX_SIMILAR:-0.0} --mix_different ${MIX_DIFFERENT:-0.0}
      --semantic_sampler_mode single
      --semantic_sampler_compute_device cuda
      --semantic_sampler_cache_teacher_on_device
      --semantic_sampler_num_anchors 3
      --semantic_sampler_global_num_candidates 64
      --sampler_max_tries 3000
      --seed $SEED
      --batch_size 128
      --max_steps 25000
      --eval_every 200
      --eval_max_pairs 5000
      --vision_model_name google/vit-base-patch16-224-in21k
      --text_model_name bert-base-uncased
      --proj_dim 256
      --proj_hidden_dim 512
      --proj_layers 2
      --init_temperature 0.07
      --teacher_text_model sentence-transformers/all-MiniLM-L6-v2
      --teacher_image_arch resnet50
      --teacher_image_model openai/clip-vit-base-patch32
      --text_sim_threshold 0.30
      --image_sim_threshold 0.65
      --cache_dir $CACHE_DIR
      --run_dir $RUN_DIR
      --num_workers 8
      --pin_memory
      --log_every 20
    )

    if [[ \"$SAMPLING_STRATEGY\" == \"semantic\" ]]; then
      true
    else
      # random baseline: make sure text_only knobs do not affect behavior
      true
    fi

    printf \"Running: \"; printf \"%q \" \"\${CMD[@]}\"; printf \"\n\"
    \"\${CMD[@]}\"
  "

