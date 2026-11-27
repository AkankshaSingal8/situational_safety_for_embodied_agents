#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH --exclude=pleiades-1-3,pleiades-1-10
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# Closed-loop validation scenes for checkpoint selection (never the test tables):
#   data_gso/<S>/scenes_val.npz   48 held-out validation GSO objects (gso_val.json) x 1 per level, layouts 0-39
#   data_v2/<S>/scenes_val_sl.npz SafeLIBERO obstacles, fresh placements (seed 77), layouts 0-11, 1 per layout/task/level
set -euo pipefail
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
export GSO_LOCAL=/tmp/gso_$USER
for S in spatial object goal long; do
  ( [ -f data_gso/$S/scenes_val.npz ] || ./pleiades_run.sh python collect_gso.py --suite safelibero_$S --split train \
      --layouts 0-39 --scenes_only --workers 2 --seed 31 --balanced 1 --objects_file runs_v2/gso_val.json \
      --out data_gso/$S/scenes_val.npz
    [ -f data_v2/$S/scenes_val_sl.npz ] || ./pleiades_run.sh python scenes.py --suite safelibero_$S \
      --out data_v2/$S/scenes_val_sl.npz --layouts 0-11 --per_layout 1 --seed 77 ) &
done
wait
echo ALL_DONE
