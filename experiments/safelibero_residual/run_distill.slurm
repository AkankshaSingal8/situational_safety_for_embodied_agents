#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH --exclude=pleiades-1-3,pleiades-1-10
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# Distil pi0.5 + shield into pi0.5 (distill_pi05.py). Arguments go through the environment, not --export,
# because sbatch --export splits on commas:
#   export OUT=/data/asingal2/openpi_cache/openpi-assets/checkpoints/pi05_distill1 ARGS="--steps 10000"
#   sbatch -J distill1 --gres=gpu:4 --export=ALL run_distill.slurm
set -euo pipefail
: "${OUT:?}"
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
OP=/home/asingal2/safety/situational_safety_for_embodied_agents/vlsa-aegis/openpi
export PYTHONPATH=$OP/src:$OP/packages/openpi-client/src:$PWD OPENPI_DATA_HOME=/data/asingal2/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=1
echo "host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-none} start $(date)"
/data/asingal2/envs/openpi/bin/python distill_pi05.py --out $OUT ${ARGS:-}
echo "EXIT_OK $(date)"
