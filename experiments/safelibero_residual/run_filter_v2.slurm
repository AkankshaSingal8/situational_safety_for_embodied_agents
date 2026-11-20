#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --exclude=pleiades-1-3,pleiades-1-10
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# pi0.5 LIBERO + safety_filter.py (v2 labels) on official episodes, one or more filter configs.
#   sbatch -J filt_smoke --export=ALL,SUITES="spatial object goal long",EPS=0-4,CFGS=default run_filter_v2.slurm
# CFGS: space-separated names from filter_grid.json (or "default"); outputs runs_v2/<suite>/filter/<TAG>/<cfg>/eps<EPS>
set -euo pipefail
: "${SUITES:?}" "${EPS:?}" "${CFGS:?}"
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
R=./pleiades_run.sh
PI_PORT=$((8000 + RANDOM % 1000))
source ./run_pi05_server.sh pi05_libero $PI_PORT
trap 'kill $PI_SERVER_PID 2>/dev/null || true' EXIT
for S in $SUITES; do
  for C in $CFGS; do
    if [ "$C" = base ]; then
      $R python eval_sl.py --suite safelibero_$S --ckpt runs_scale/$S/base/ema_last.pt --base pi05 --pi_port $PI_PORT \
        --env_seed 0 --settle 20 --exec_steps 5 --res 256 --workers 6 --arm base --eps $EPS --out runs_v2/$S/base/eps$EPS
      continue
    fi
    J=$(python3 -c "import json,sys; g=json.load(open('filter_grid.json')); print(json.dumps(g['$C']))")
    ARM=$(python3 -c "import json,sys; g=json.load(open('filter_grid.json')); print('shield' if g['$C'].get('type')=='shield' else 'filter')")
    $R python eval_sl.py --suite safelibero_$S --ckpt runs_scale/$S/base/ema_last.pt --base pi05 --pi_port $PI_PORT \
      --env_seed 0 --settle 20 --exec_steps 5 --res 256 --workers 6 --arm $ARM --text_mode true \
      --oracle_cfg "$J" --eps $EPS --out runs_v2/$S/filter/${TAG:-b}/$C/eps$EPS
  done
done
echo ALL_DONE
