#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# v2 training scenes: official layouts 0-39 (episodes 40-49 are the test set and are never used), 6 obstacle draws
# per layout, task and level -> 1,920 scenes per suite.
set -euo pipefail
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
for S in spatial object goal long; do
  [ -f data_v2/$S/scenes_train.npz ] || ./pleiades_run.sh python scenes.py --suite safelibero_$S \
      --out data_v2/$S/scenes_train.npz --layouts 0-39 --per_layout 6 --seed 10 &
done
wait
echo ALL_DONE
