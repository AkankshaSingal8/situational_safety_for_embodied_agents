#!/bin/bash
#SBATCH --job-name=entity_mv
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=02:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Offline entity multi-view probe: does camera fusion (agentview+birdview /
# all static views) beat the board's single-agentview entity localization?
# Pure perception — no policy server, no rollouts.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"

conda activate openvla_libero_merged
export MUJOCO_GL=egl
PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/vlm_pipeline" \
python "$WT/vlm_pipeline/entity_multiview_probe.py" \
  --levels II,I --episodes_per_task 10 \
  --out "$WT/entity_mv_probe/probe.json"
echo "ENTITY_MV_COMPLETE"
