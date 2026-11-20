#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 08:00:00
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# Goal only got 714 of 1,920 scenes (placement rejections hit the 60-try cap per layout). Two more seeds.
set -euo pipefail
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
./pleiades_run.sh python scenes.py --suite safelibero_goal --out data_v2/goal/scenes_train_b.npz --layouts 0-39 --per_layout 6 --seed 11 &
./pleiades_run.sh python scenes.py --suite safelibero_goal --out data_v2/goal/scenes_train_c.npz --layouts 0-39 --per_layout 6 --seed 12 &
wait
echo ALL_DONE
