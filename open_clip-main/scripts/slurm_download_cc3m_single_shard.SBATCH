#!/bin/bash
#SBATCH --job-name=hf_cc3m_one_shard
#SBATCH --output=/scratch/xj2366/samcl/logs/%x_%j.out
#SBATCH --error=/scratch/xj2366/samcl/logs/%x_%j.err
#SBATCH --account=torch_pr_36_mren
#SBATCH --partition=l40s_mren
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xj2366@nyu.edu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#
# 只下载 CC3M WebDataset 中的单个 .tar 分片（默认补 cc3m-train-0078.tar）。
# huggingface-cli 会在 --local-dir 下写入/续传该文件，与全量下载共用同一目录即可。
#
# 纯下载不需要 GPU：可把分区改成 cpu、去掉 #SBATCH --gres=gpu:1，并 apptainer exec 不用 --nv（与全量脚本一致）。
#
# 换其它分片：提交前 export HF_CC3M_SHARD=cc3m-train-0123.tar
# 数据集：https://huggingface.co/datasets/pixparse/cc3m-wds

set -euo pipefail

module purge

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

OVERLAY_ENV=/scratch/xj2366/mcl/mcl-env/mcl/mcl.ext3
SIF=/scratch/xj2366/containers/cuda12.1.1-cudnn8-devel-ubuntu22.04.sif

TMP_BASE="/tmp/$USER/apptainer"
mkdir -p "$TMP_BASE/tmp" "$TMP_BASE/cache"
export APPTAINER_TMPDIR="$TMP_BASE/tmp"
export APPTAINER_CACHEDIR="$TMP_BASE/cache"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"

export HF_CC3M_REPO="${HF_CC3M_REPO:-pixparse/cc3m-wds}"
export HF_CC3M_DIR="${HF_CC3M_DIR:-/scratch/xj2366/samcl/data/cc3m_wds}"
export HF_CC3M_SHARD="${HF_CC3M_SHARD:-cc3m-train-0078.tar}"
export PYTHONUNBUFFERED=1

mkdir -p /scratch/xj2366/samcl/logs
mkdir -p "$HF_CC3M_DIR"

echo "SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname)  DATE=$(date -Is)"
echo "HF_CC3M_REPO=$HF_CC3M_REPO"
echo "HF_CC3M_DIR=$HF_CC3M_DIR"
echo "HF_CC3M_SHARD=$HF_CC3M_SHARD"

if [[ "$HF_CC3M_SHARD" != *.tar ]]; then
  echo "ERROR: HF_CC3M_SHARD should be a .tar filename (e.g. cc3m-train-0078.tar)." >&2
  exit 1
fi

echo "---- connectivity (host) ----"
if ! curl -sS --max-time 30 -o /dev/null -w "huggingface.co HTTP %{http_code}\n" https://huggingface.co; then
  echo "WARNING: curl to huggingface.co failed; download may still work."
fi

apptainer exec \
  --env HF_CC3M_REPO="$HF_CC3M_REPO" \
  --env HF_CC3M_DIR="$HF_CC3M_DIR" \
  --env HF_CC3M_SHARD="$HF_CC3M_SHARD" \
  --env PYTHONUNBUFFERED=1 \
  --overlay "${OVERLAY_ENV}:ro" \
  "${SIF}" \
  /bin/bash -lc '
    set -euo pipefail
    echo "[container] $(hostname) python: $(which python)"
    source /ext3/env.sh
    conda activate mcl
    python -m pip install -U pip
    python -m pip install -U "huggingface_hub[cli]>=0.23"
    huggingface-cli download "${HF_CC3M_REPO}" "${HF_CC3M_SHARD}" \
      --repo-type dataset \
      --local-dir "${HF_CC3M_DIR}"
  '

echo "---- done (host) ----"
TARGET="${HF_CC3M_DIR}/${HF_CC3M_SHARD}"
if [[ -f "$TARGET" ]]; then
  ls -lh "$TARGET"
else
  echo "WARNING: expected file missing: $TARGET" >&2
  exit 1
fi
