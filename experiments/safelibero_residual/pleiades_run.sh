#!/bin/bash
# Run one command for experiments/safelibero_residual inside the slres env on a pleiades GPU node.
#   sbatch -J NAME -p pleiades --time=HH:MM:SS -c 8 --mem=48G --gres=gpu:1 \
#          -o /data/asingal2/slurm_logs/%x_%j.out experiments/safelibero_residual/pleiades_run.sh python eval_sl.py ...
set -euo pipefail
REPO=/home/asingal2/safety/situational_safety_for_embodied_agents
cd "$REPO/experiments/safelibero_residual"
export PATH=/data/asingal2/envs/slres/bin:$PATH
export LD_LIBRARY_PATH=/data/asingal2/envs/slres/lib:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 PYTHONPATH=$PWD
export LIBERO_CONFIG_PATH=$HOME/.safelibero HF_HOME=/data/asingal2/hf_cache TORCH_HOME=/data/asingal2/hf_cache/torch
echo "host $(hostname)  gpus ${CUDA_VISIBLE_DEVICES:-none}  start $(date)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
"$@"
echo "EXIT_OK $(date)"
