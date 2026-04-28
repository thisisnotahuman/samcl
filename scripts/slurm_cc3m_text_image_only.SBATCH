#!/bin/bash
#SBATCH --job-name=cc3m_sem_infonce
#SBATCH --output=/scratch/xj2366/samcl/log_new/cc3m_text_image_only/%x_%A_%a.out
#SBATCH --error=/scratch/xj2366/samcl/log_new/cc3m_text_image_only/%x_%A_%a.err
#SBATCH --account=torch_pr_36_general
#SBATCH --mail-user=xj2366@nyu.edu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=24:00:00
#SBATCH --array=0-3

set -euo pipefail

module purge
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# CC3M WebDataset shards（无需解压 .tar；训练/索引用 tarfile 按样本读取）
CC3M_WDS="${CC3M_WDS:-/scratch/xj2366/samcl/data/cc3m_wds}"
WDS_SHARD_GLOB="${WDS_SHARD_GLOB:-cc3m-train-*.tar}"

# Teacher 缓存（student-mirrored）：使用与 student 相同的 encoder+projection 生成 teacher embedding，
# 因此不能复用 open_clip / ResNet+SBERT 的旧 teacher 缓存。
# 为了在 4 个 array task 间共享，使用 (WDS key + student_tag) -> hash 子目录。
TEACHER_CACHE_BASE="${TEACHER_CACHE_BASE:-/scratch/xj2366/samcl/cache/cc3m_semantic_student_mirrored}"
SAMCL_MAX_PAIRS="${SAMCL_MAX_PAIRS:-}"
# Must match src/samcl/train.py:student_teacher_cache_tag()（供哈希与 python 参数共用）
VISION_MODEL_NAME="${VISION_MODEL_NAME:-google/vit-base-patch16-224-in21k}"
TEXT_MODEL_NAME="${TEXT_MODEL_NAME:-bert-base-uncased}"
PROJ_DIM="${PROJ_DIM:-256}"
PROJ_HIDDEN_DIM="${PROJ_HIDDEN_DIM:-512}"
PROJ_LAYERS="${PROJ_LAYERS:-2}"
INIT_TEMPERATURE="${INIT_TEMPERATURE:-0.07}"
MAX_TEXT_LEN="${MAX_TEXT_LEN:-77}"
VISION_FROM_SCRATCH=$([ "${SAMCL_SCRATCH:-1}" = "1" ] && echo 1 || echo 0)
TEXT_FROM_SCRATCH=$([ "${SAMCL_SCRATCH:-1}" = "1" ] && echo 1 || echo 0)
STUDENT_TAG="student_mirrored|v=${VISION_MODEL_NAME}|t=${TEXT_MODEL_NAME}|pd=${PROJ_DIM}|ph=${PROJ_HIDDEN_DIM}|pl=${PROJ_LAYERS}|vt=${VISION_FROM_SCRATCH}|tt=${TEXT_FROM_SCRATCH}|it=${INIT_TEMPERATURE}|mtl=${MAX_TEXT_LEN}"
if [ -n "${TEACHER_CACHE_DIR:-}" ]; then
  CACHE_DIR="$TEACHER_CACHE_DIR"
else
  _mp_key="None"
  [ -n "$SAMCL_MAX_PAIRS" ] && _mp_key="$SAMCL_MAX_PAIRS"
  CACHE_DIR="$(
    W="$CC3M_WDS" G="$WDS_SHARD_GLOB" M="$_mp_key" T="$STUDENT_TAG" B="$TEACHER_CACHE_BASE" \
    python3 -c "import hashlib, os; p=os.path.abspath(os.environ['W']); g=os.environ['G']; m=os.environ['M']; t=os.environ['T']; k=f'wds|{p}|{g}|{m}|{t}'; h=hashlib.sha256(k.encode()).hexdigest()[:16]; print(os.path.join(os.environ['B'], h))"
  )"
fi

SEED=1337

# 与 scripts/slurm_text_image_only.SBATCH 一致：semantic + InfoNCE，4 格 relation/mix
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

echo "CC3M_WDS=$CC3M_WDS WDS_SHARD_GLOB=$WDS_SHARD_GLOB teacher_cache_dir=$CACHE_DIR EXP=$EXP_NAME rel=$REL_MODE seed=$SEED"

SAMCL_SCRATCH="${SAMCL_SCRATCH:-1}"

REPO_DIR=/scratch/xj2366/samcl/SAMCL
RUN_DIR=/scratch/xj2366/samcl/runs/cc3m_text_image_only/global_64/${EXP_NAME}

mkdir -p "$RUN_DIR" "$CACHE_DIR" "/scratch/xj2366/samcl/log_new/cc3m_text_image_only"

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
    MP_EXTRA=\"\"
    if [ -n \"${SAMCL_MAX_PAIRS:-}\" ]; then
      MP_EXTRA=\"--max_pairs ${SAMCL_MAX_PAIRS}\"
    fi

    python -u -m samcl.train \
      --dataset_type cc3m_wds \
      --wds_root \"$CC3M_WDS\" \
      --wds_shard_glob \"$WDS_SHARD_GLOB\" \
      --pairwise_loss infonce \
      --sampling_strategy semantic \
      --auto_resume \
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
      --save_latest_only \
      --eval_every 200 \
      --eval_max_pairs 5000 \
      --vision_model_name $VISION_MODEL_NAME \
      --text_model_name $TEXT_MODEL_NAME \
      --proj_dim $PROJ_DIM \
      --proj_hidden_dim $PROJ_HIDDEN_DIM \
      --proj_layers $PROJ_LAYERS \
      --init_temperature $INIT_TEMPERATURE \
      --semantic_teacher_same_as_student \
      --text_sim_threshold 0.30 \
      --image_sim_threshold 0.65 \
      --cache_dir $CACHE_DIR \
      --run_dir $RUN_DIR \
      --num_workers 8 \
      --prefetch_factor 6 \
      --pin_memory \
      --log_every 20 \
      \$MP_EXTRA \
      \$EXTRA
  "
