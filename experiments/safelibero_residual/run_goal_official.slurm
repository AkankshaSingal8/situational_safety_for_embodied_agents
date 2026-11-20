#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH --exclude=pleiades-1-3,pleiades-1-10
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# Goal cells that scenes.py could not fill: official layouts 0-39 (x3 copies), then verified collection into r0_official.
set -euo pipefail
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
./pleiades_run.sh python scenes_official.py --suite safelibero_goal --cells I/0,I/1,II/0,II/1,II/2 --layouts 0-39 --copies 3 \
    --out data_v2/goal/scenes_official.npz
PI_PORT=$((8000 + RANDOM % 1000))
source ./run_pi05_server.sh pi05_libero $PI_PORT
trap 'kill $PI_SERVER_PID 2>/dev/null || true' EXIT
./pleiades_run.sh python collect_v2.py --suite safelibero_goal --ckpt runs_scale/goal/base/ema_last.pt --base pi05 \
    --pi_port $PI_PORT --scenes data_v2/goal/scenes_official.npz --cell_cfgs runs_v2/goal/cell_cfgs.json --tries 3 \
    --workers 6 --out data_v2/goal/r0_official
echo ALL_DONE
