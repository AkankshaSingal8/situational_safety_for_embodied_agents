#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH --exclude=pleiades-1-3
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# Paired seen/novel test scenes on test layouts 40-49 (one obstacle per scene, types cycled).
#   sbatch -J nsc_spatial --export=ALL,S=spatial run_novel_scenes.slurm
set -euo pipefail
: "${S:?}"
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
R=./pleiades_run.sh
[ -n "${SKIP_GEN:-}" ] || $R python scenes.py --suite safelibero_$S --out data_v2/$S/scenes_pair.npz --layouts 40-49 --per_layout ${PER:-3} \
    --n_obs 1 --seed 49 --cycle_types milk_obstacle_1,wine_bottle_obstacle_1,white_storage_box_obstacle_1,red_coffee_mug_obstacle_1,yellow_book_obstacle_1
$R python novel_scenes.py --suite safelibero_$S --scenes data_v2/$S/scenes_pair.npz \
    --out_seen data_v2/$S/scenes_pair_seen.npz --out_novel data_v2/$S/scenes_pair_novel.npz
echo ALL_DONE
