#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 1-12:00:00
#SBATCH --exclude=pleiades-1-3
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# Frozen pi05_base PaliGemma features (openpi env).
#   sbatch -J pg_pca --export=ALL,MODE=pca run_pg_feats.slurm
#   sbatch -J pg_spatial --export=ALL,S=spatial,ROUNDS=r0 run_pg_feats.slurm
set -euo pipefail
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
OP=/home/asingal2/safety/situational_safety_for_embodied_agents/vlsa-aegis/openpi
export PYTHONPATH=$OP/src:$OP/packages/openpi-client/src:$PWD
export OPENPI_DATA_HOME=/data/asingal2/openpi_cache XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PY=/data/asingal2/envs/openpi/bin/python
echo "host $(hostname) start $(date)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
if [ "${MODE:-}" = pca ]; then
  $PY pg_feats.py --fit_pca --out ${OUT:-data_v2/feats_pg}
else
  : "${S:?}"
  $PY pg_feats.py --suite $S --rounds ${ROUNDS:-r0} --out ${OUT:-data_v2/feats_pg} --name ${NAME:-$S}
fi
echo ALL_DONE $(date)
