#!/bin/bash
#SBATCH --job-name=adjoint_probe
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Enforcement candidate A GO/NO-GO: reverse-mode grad of terminal safety
# margin through the 10-step denoise loop w.r.t. initial noise (frozen pi0.5,
# prefix KV constant). Mechanism metrics only — no performance claims.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="$WT/openpi_guided/src"
python "$WT/openpi_guided/scripts/adjoint_probe.py" \
  --config pi05_libero --dir "$ROOT/vlsa-aegis/checkpoints/pi05_libero"
echo "ADJOINT_PROBE_DONE"
