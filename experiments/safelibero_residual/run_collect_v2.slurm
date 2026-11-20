#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 3-00:00:00
#SBATCH --exclude=pleiades-1-3,pleiades-1-10
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# v2 verified residual data for one suite: pi0.5 LIBERO + lookahead shield, kept only on safe success.
#   sbatch -J col_spatial --export=ALL,S=spatial run_collect_v2.slurm
#   optional: IDS=lo-hi (scene subset), ROUND=r0 (output dir name), WORKERS=6, SCENES=<npz>,
#             CFGS=<cell_cfgs json>, TRIES=3, EXTRA="--p_drive 1.0"
# Resumable: scenes with a final log line are skipped on resubmission.
set -euo pipefail
: "${S:?}"
ROUND=${ROUND:-r0}
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
PI_PORT=$((8000 + RANDOM % 1000))
source ./run_pi05_server.sh pi05_libero $PI_PORT
trap 'kill $PI_SERVER_PID 2>/dev/null || true' EXIT
./pleiades_run.sh python collect_v2.py --suite safelibero_$S --ckpt runs_scale/$S/base/ema_last.pt --base pi05 \
    --pi_port $PI_PORT --scenes ${SCENES:-data_v2/$S/scenes_train.npz} --cell_cfgs ${CFGS:-runs_v2/$S/cell_cfgs.json} --tries ${TRIES:-3} \
    --workers ${WORKERS:-6} --out data_v2/$S/$ROUND ${IDS:+--scene_ids $IDS} ${EXTRA:-}
echo ALL_DONE
