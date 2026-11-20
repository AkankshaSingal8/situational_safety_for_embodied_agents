#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --exclude=pleiades-1-3,pleiades-1-10
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# pi0.5 base rows under the scale-up protocol (env seed 0, 20 settle steps, 256 px, 5 of 10 steps executed):
# all 50 official episodes x 4 tasks x 2 levels for one suite and one checkpoint (pi05_libero | pi05_base).
#   sbatch -J pi05_object --export=ALL,SUITE=safelibero_object,PI=pi05_libero run_pi05_base_eval.slurm
set -euo pipefail
: "${SUITE:?}" "${PI:?}"
S=${SUITE#safelibero_}
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
PORT=$((8000 + RANDOM % 1000))
source ./run_pi05_server.sh $PI $PORT
trap 'kill $PI_SERVER_PID 2>/dev/null || true' EXIT
./pleiades_run.sh python eval_sl.py --suite $SUITE --ckpt runs_scale/$S/base/ema_last.pt --base pi05 --pi_port $PORT \
    --arm base --env_seed 0 --settle 20 --res 256 --exec_steps 5 --workers 4 --out runs_scale/$S/$PI/eval_official
echo ALL_DONE
