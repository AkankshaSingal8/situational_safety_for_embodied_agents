#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=160G
#SBATCH -t 0-12:00:00
#SBATCH --exclude=pleiades-1-3,pleiades-1-10
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# Distilled-pi0.5 chunks for the residual training episodes (relabel_distill.py). Args via the environment:
#   export ARGS="--ckpt pi05_distill1/step_6000 --out /data/asingal2/relabel/d6000 --shard 0/4"
#   sbatch -J rl0 --export=ALL run_relabel.slurm
set -euo pipefail
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
OP=/home/asingal2/safety/situational_safety_for_embodied_agents/vlsa-aegis/openpi
export PYTHONPATH=$OP/src:$OP/packages/openpi-client/src:$PWD OPENPI_DATA_HOME=/data/asingal2/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=1
echo "host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-none} start $(date)"
/data/asingal2/envs/openpi/bin/python relabel_distill.py ${ARGS:?}
echo "EXIT_OK $(date)"
