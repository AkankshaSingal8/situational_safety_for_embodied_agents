#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH --exclude=pleiades-1-3,pleiades-1-10
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# New Table 3: training layouts 0-39, task objects untouched, obstacle at a new position (newpos_scenes.py).
#   STEP=gen   S=spatial                 candidates -> data_v2/<S>/scenes_newpos_cand.npz
#   STEP=check S=spatial IDS=0-479       shield run (pi0.5 + lookahead shield) -> data_v2/<S>/newpos_check/<IDS>
#   STEP=pick  S=spatial                 shield safe successes, 200 per level -> data_v2/<S>/scenes_newpos.npz
# newpos_check is a test-set check: never add it to training --rounds.
set -euo pipefail
: "${S:?}" "${STEP:?}"
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
R=./pleiades_run.sh
C=data_v2/$S/scenes_newpos_cand.npz
if [ "$STEP" = gen ]; then
  $R python newpos_scenes.py gen --suite safelibero_$S --out $C
elif [ "$STEP" = check ]; then
  : "${IDS:?}"
  PI_PORT=$((8000 + RANDOM % 1000))
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
  source ./run_pi05_server.sh pi05_libero $PI_PORT
  trap 'kill $PI_SERVER_PID 2>/dev/null || true' EXIT
  $R python collect_v2.py --suite safelibero_$S --ckpt runs_scale/$S/base/ema_last.pt --base pi05 --pi_port $PI_PORT \
      --cell_cfgs runs_v2/$S/cell_cfgs.json --tries 3 --workers ${WORKERS:-6} --scenes $C --p_drive 1.0 \
      --seed 51 --scene_ids $IDS --out data_v2/$S/newpos_check/$IDS
else
  mkdir -p data_v2/$S/newpos_check/all
  cat data_v2/$S/newpos_check/[0-9]*/log_*.jsonl > data_v2/$S/newpos_check/all/log_all.jsonl
  $R python newpos_scenes.py pick --suite safelibero_$S --cand $C --logs data_v2/$S/newpos_check/all \
      --out data_v2/$S/scenes_newpos.npz
fi
echo ALL_DONE
