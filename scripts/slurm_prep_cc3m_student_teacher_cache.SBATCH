#!/bin/bash
#SBATCH --job-name=prep_cc3m_st_teacher
#SBATCH --output=/scratch/xj2366/samcl/log_new/cc3m_prep_student_teacher/%x_%j.out
#SBATCH --error=/scratch/xj2366/samcl/log_new/cc3m_prep_student_teacher/%x_%j.err
#SBATCH --account=torch_pr_36_general
#SBATCH --mail-user=xj2366@nyu.edu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --time=10:00:00

set -euo pipefail

module purge
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}

# ---------- CC3M + student-mirrored teacher（与 scripts/slurm_cc3m_text_image_only.SBATCH 对齐）----------
CC3M_WDS="${CC3M_WDS:-/scratch/xj2366/samcl/data/cc3m_wds}"
WDS_SHARD_GLOB="${WDS_SHARD_GLOB:-cc3m-train-*.tar}"
TEACHER_CACHE_BASE="${TEACHER_CACHE_BASE:-/scratch/xj2366/samcl/cache/cc3m_semantic_student_mirrored}"
SAMCL_MAX_PAIRS="${SAMCL_MAX_PAIRS:-}"

VISION_MODEL_NAME="${VISION_MODEL_NAME:-google/vit-base-patch16-224-in21k}"
TEXT_MODEL_NAME="${TEXT_MODEL_NAME:-bert-base-uncased}"
PROJ_DIM="${PROJ_DIM:-256}"
PROJ_HIDDEN_DIM="${PROJ_HIDDEN_DIM:-512}"
PROJ_LAYERS="${PROJ_LAYERS:-2}"
INIT_TEMPERATURE="${INIT_TEMPERATURE:-0.07}"
MAX_TEXT_LEN="${MAX_TEXT_LEN:-77}"
SAMCL_SCRATCH="${SAMCL_SCRATCH:-1}"
VISION_FROM_SCRATCH=$([ "${SAMCL_SCRATCH}" = "1" ] && echo 1 || echo 0)
TEXT_FROM_SCRATCH=$([ "${SAMCL_SCRATCH}" = "1" ] && echo 1 || echo 0)
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

REPO_DIR="${REPO_DIR:-/scratch/xj2366/samcl/SAMCL}"
PREP_RUN_DIR="${PREP_RUN_DIR:-/scratch/xj2366/samcl/runs/cc3m_prep_student_teacher/job_${SLURM_JOB_ID:-manual}}"

# teacher 图像编码 batch：VRAM 允许可调大以缩短 wall time、提高瞬时 SM 利用率
STUDENT_TEACHER_IMAGE_BS="${STUDENT_TEACHER_IMAGE_BS:-96}"
STUDENT_TEACHER_TEXT_BS="${STUDENT_TEACHER_TEXT_BS:-128}"

# ---------- teacher 图像：并行 JPEG 解码（显著减少 GPU 等 CPU 的空窗）----------
SAMCL_TEACHER_CACHE_DECODE_WORKERS="${SAMCL_TEACHER_CACHE_DECODE_WORKERS:-10}"

# ---------- GPU sidecar：continuous-duty 保证每个采样周期内固定比例做真实 GEMM（不靠“偶尔低于阈值”）----------
GPU_UTIL_TARGET="${GPU_UTIL_TARGET:-50}"
GPU_UTIL_WINDOW_SEC="${GPU_UTIL_WINDOW_SEC:-20}"
GPU_CONTINUOUS_DUTY="${GPU_CONTINUOUS_DUTY:-0.50}"
GPU_BURN_MATMUL_DIM="${GPU_BURN_MATMUL_DIM:-4096}"
GPU_BURN_MATMUL_ITERS="${GPU_BURN_MATMUL_ITERS:-16}"
GPU_BURN_BURST_CYCLES="${GPU_BURN_BURST_CYCLES:-6}"

mkdir -p "$CACHE_DIR" "$PREP_RUN_DIR" "/scratch/xj2366/samcl/log_new/cc3m_prep_student_teacher"

echo "PREP_RUN_DIR=$PREP_RUN_DIR"
echo "CACHE_DIR=$CACHE_DIR"
echo "CC3M_WDS=$CC3M_WDS SAMCL_SCRATCH=$SAMCL_SCRATCH decode_workers=$SAMCL_TEACHER_CACHE_DECODE_WORKERS img_teacher_bs=$STUDENT_TEACHER_IMAGE_BS continuous_duty=$GPU_CONTINUOUS_DUTY"

OVERLAY_ENV="${OVERLAY_ENV:-/scratch/xj2366/mcl/mcl-env/mcl/mcl.ext3}"
SIF="${SIF:-/scratch/xj2366/containers/cuda12.1.1-cudnn8-devel-ubuntu22.04.sif}"

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
    export SAMCL_WDS_TAR_CACHE_MAX_OPEN=\"\${SAMCL_WDS_TAR_CACHE_MAX_OPEN:-16}\"
    export SAMCL_TEACHER_CACHE_DECODE_WORKERS=\"${SAMCL_TEACHER_CACHE_DECODE_WORKERS}\"

    python -u -m samcl.tools.gpu_util_sidecar \
      --device cuda:0 \
      --nvidia-smi-gpu-index 0 \
      --threshold ${GPU_UTIL_TARGET} \
      --window-sec ${GPU_UTIL_WINDOW_SEC} \
      --poll-interval 1.0 \
      --continuous-duty ${GPU_CONTINUOUS_DUTY} \
      --matmul-dim ${GPU_BURN_MATMUL_DIM} \
      --matmul-iters ${GPU_BURN_MATMUL_ITERS} \
      --burst-cycles ${GPU_BURN_BURST_CYCLES} \
      &
    SIDE_PID=\$!
    trap 'kill \$SIDE_PID 2>/dev/null || true' EXIT INT TERM

    EXTRA=\"\"
    if [ \"\${SAMCL_SCRATCH:-0}\" = \"1\" ]; then
      EXTRA=\"--vision_from_scratch --text_from_scratch\"
    fi
    MP_EXTRA=\"\"
    if [ -n \"${SAMCL_MAX_PAIRS:-}\" ]; then
      MP_EXTRA=\"--max_pairs ${SAMCL_MAX_PAIRS}\"
    fi

    python -u -m samcl.train \
      --prep_teacher_cache_only \
      --dataset_type cc3m_wds \
      --wds_root \"$CC3M_WDS\" \
      --wds_shard_glob \"$WDS_SHARD_GLOB\" \
      --seed 1337 \
      --vision_model_name $VISION_MODEL_NAME \
      --text_model_name $TEXT_MODEL_NAME \
      --proj_dim $PROJ_DIM \
      --proj_hidden_dim $PROJ_HIDDEN_DIM \
      --proj_layers $PROJ_LAYERS \
      --init_temperature $INIT_TEMPERATURE \
      --semantic_teacher_same_as_student \
      --student_teacher_image_batch_size $STUDENT_TEACHER_IMAGE_BS \
      --student_teacher_text_batch_size $STUDENT_TEACHER_TEXT_BS \
      --cache_dir $CACHE_DIR \
      --run_dir $PREP_RUN_DIR \
      \$MP_EXTRA \
      \$EXTRA
  "

echo "DONE prep teacher cache -> $CACHE_DIR"
