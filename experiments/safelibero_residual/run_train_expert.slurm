#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH --exclude=pleiades-1-3,pleiades-1-10
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# Train the residual action expert (pi05_base expert + frozen pi05_base PaliGemma; train_expert.py).
#   sbatch -J ex1 --export=ALL,OUT=runs_v2/expert/ex1,ARGS="--steps 30000" run_train_expert.slurm
#   multi-GPU: add --gres=gpu:N to sbatch (data parallel; the global batch is split over the GPUs)
set -euo pipefail
: "${OUT:?}"
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
OP=/home/asingal2/safety/situational_safety_for_embodied_agents/vlsa-aegis/openpi
export PYTHONPATH=$OP/src:$OP/packages/openpi-client/src:$PWD OPENPI_DATA_HOME=/data/asingal2/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=1
echo "host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-none} start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
/data/asingal2/envs/openpi/bin/python train_expert.py --out $OUT ${ARGS:-}
echo "EXIT_OK $(date)"
