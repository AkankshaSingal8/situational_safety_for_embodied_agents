#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH --exclude=pleiades-1-3,pleiades-1-10
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# Test eval (official episodes 40-49) of pi0.5 alone and pi0.5 + v2 residual, frozen protocol.
#   sbatch -J ev2_spatial --export=ALL,S=spatial,RES=runs_v2/train/joint_r0_nz50,ARMS="base true" run_eval_v2.slurm
# ARMS: base | true | none | wrong | held  (text modes for residual_v2). Out: runs_v2/<S>/test/<tag>/<arm>
set -euo pipefail
: "${S:?}" "${ARMS:?}"
EPS=${EPS:-40-49}
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
R=./pleiades_run.sh
PI_PORT=$((8000 + RANDOM % 1000))
source ./run_pi05_server.sh pi05_libero $PI_PORT
trap 'kill $PI_SERVER_PID 2>/dev/null || true' EXIT
COMMON="--suite safelibero_$S --ckpt runs_scale/$S/base/ema_last.pt --base pi05 --pi_port $PI_PORT --env_seed 0 --settle 20 --exec_steps 5 --res 256 --workers ${WORKERS:-6} --eps $EPS"
for A in $ARMS; do
  if [ "$A" = base ]; then
    $R python eval_sl.py $COMMON --arm base --out runs_v2/$S/test/pi05/eps$EPS
  else
    : "${RES:?}"
    $R python eval_sl.py $COMMON --arm residual_v2 --residual $RES/residual.pt --text_mode $A \
        --out runs_v2/$S/test/$(basename $RES)/$A/eps$EPS
  fi
done
echo ALL_DONE
