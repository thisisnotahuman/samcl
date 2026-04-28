#!/bin/bash
#SBATCH --job-name=debug_gpu
#SBATCH --account=torch_pr_36_general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xj2366@nyu.edu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

echo "Job started on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 防止 job 立刻退出
sleep infinity
EOF


# 跑上后attach进去开interactive模式：srun --jobid=1868xxx --pty /bin/bash