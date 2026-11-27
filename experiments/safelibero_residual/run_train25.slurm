#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH --exclude=pleiades-1-3,pleiades-1-10
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# Diagnostic twin of Table 2 with 25 TRAINING GSO objects (runs_v2/gso_train25.json) x 4 per level, new placements:
# separates "never learned GSO avoidance" from "learned it but does not transfer to unseen objects".
set -euo pipefail
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
export GSO_LOCAL=/tmp/gso_$USER
./pleiades_run.sh python collect_gso.py --suite safelibero_$S --split train --layouts 0-39 --scenes_only --workers 2 \
    --seed 61 --balanced 4 --objects_file runs_v2/gso_train25.json --out data_gso/$S/scenes_train25.npz
echo ALL_DONE
