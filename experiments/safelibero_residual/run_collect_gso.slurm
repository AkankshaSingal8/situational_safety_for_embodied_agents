#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH --exclude=pleiades-1-3,pleiades-1-10
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# GSO-obstacle residual data (collect_gso.py): pi0.5 LIBERO + lookahead shield, kept only on safe success.
#   sbatch -J cg_spatial --export=ALL,S=spatial run_collect_gso.slurm            (training data, layouts 0-39)
#   optional: ITEMS=lo-hi (subset of the suite's items), PER=4 (objects per layout), WORKERS=6,
#             OUT=data_gso_r2 SEED=2 CFGS=runs_v2/$S/cell_cfgs_r1.json TRIES=6 (r2: new placements, 6 tries)
#   test scenes (no rollouts, no pi0.5 server): SCENES_ONLY=1 -> data_gso/<S>/scenes_test.npz (held-out objects, 40-49)
#   new-object test (balanced): SCENES_ONLY=1 SNAME=scenes_newobj SPLIT=test LAYOUTS=0-39 SEED=3
#             EXTRA="--balanced 4 --objects_file runs_v2/gso_test_clean.json"
#   retry of items without a safe success: OUT=data_gso_retry SEED=<as original> EXTRA="--retry_from data_gso/$S"
# Resumable: items with a final log line are skipped on resubmission.
set -euo pipefail
: "${S:?}"
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
export GSO_LOCAL=/tmp/gso_$USER
if [ -n "${SCENES_ONLY:-}" ]; then
  ./pleiades_run.sh python collect_gso.py --suite safelibero_$S --split ${SPLIT:-test} --layouts ${LAYOUTS:-40-49} \
      --per_layout ${PER:-3} --scenes_only --workers ${WORKERS:-6} --seed ${SEED:-1} \
      --out data_gso/$S/${SNAME:-scenes_test}.npz ${EXTRA:-}
  echo ALL_DONE; exit 0
fi
PI_PORT=$((8000 + RANDOM % 1000))
source ./run_pi05_server.sh pi05_libero $PI_PORT
trap 'kill $PI_SERVER_PID 2>/dev/null || true' EXIT
./pleiades_run.sh python collect_gso.py --suite safelibero_$S --split train --layouts 0-39 --per_layout ${PER:-4} \
    --ckpt runs_scale/$S/base/ema_last.pt --base pi05 --pi_port $PI_PORT --cell_cfgs ${CFGS:-runs_v2/$S/cell_cfgs.json} \
    --tries ${TRIES:-3} --workers ${WORKERS:-6} --seed ${SEED:-0} --out ${OUT:-data_gso}/$S ${ITEMS:+--item_ids $ITEMS} ${EXTRA:-}
echo ALL_DONE
