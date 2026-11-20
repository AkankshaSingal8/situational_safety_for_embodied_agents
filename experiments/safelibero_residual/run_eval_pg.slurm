#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH --exclude=pleiades-1-3,pleiades-1-10
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# Test eval (official episodes 40-49) of pi0.5 alone and pi0.5 + frozen-backbone residual (train_pg.py).
#   sbatch -J epg_spatial --export=ALL,S=spatial,RES=runs_v2/pg/final_pg,ARMS="base true" run_eval_pg.slurm
#   NOVEL=1 swaps every obstacle for an unseen look-alike (novel_objects.py); outputs go under .../novel/
#   SCENES=scenes_pair_seen | SCENES=scenes_pair_novel NOVEL=1 evaluates the paired test scenes (data_v2/<S>/<SCENES>.npz)
# EX=<runs_v2/expert/NAME/best.pkl> evaluates the residual action expert (res_server.py, arm residual_ex) instead of
#   RES; outputs go to .../<NAME>/<arm>/...   SCENES=<file> without NOVEL also works for data_gso test scenes
#   (SCENES path may contain a directory: SCENES=../../data_gso/spatial/scenes_test -> TAG=scenes_test).
# ARMS: base | true | none | wrong | held.  Out: runs_v2/<S>/test[_novel]/<tag>/<arm>/eps<EPS>
# pleiades-1-10 is excluded: its 24 GB GPUs run out of memory with the pi0.5 + PaliGemma servers and 6 workers.
set -euo pipefail
: "${S:?}" "${ARMS:?}"
EPS=${EPS:-40-49}
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
R=./pleiades_run.sh
OP=/home/asingal2/safety/situational_safety_for_embodied_agents/vlsa-aegis/openpi
PI_PORT=$((8000 + RANDOM % 1000)); PG_PORT=$((9000 + RANDOM % 1000))
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
source ./run_pi05_server.sh pi05_libero $PI_PORT
PG_PID=""
trap 'kill $PI_SERVER_PID $PG_PID 2>/dev/null || true' EXIT
if [ -n "${EX:-}" ]; then
  EX_PORT=$((10000 + RANDOM % 1000))
  (PYTHONPATH=$OP/src:$OP/packages/openpi-client/src:$PWD OPENPI_DATA_HOME=/data/asingal2/openpi_cache \
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 /data/asingal2/envs/openpi/bin/python res_server.py --port $EX_PORT --ckpt $EX \
   > /data/asingal2/slurm_logs/res_server_${SLURM_JOB_ID}.log 2>&1) &
  PG_PID=$!
  for i in $(seq 1 240); do (echo > /dev/tcp/127.0.0.1/$EX_PORT) 2>/dev/null && break
    kill -0 $PG_PID 2>/dev/null || { tail -30 /data/asingal2/slurm_logs/res_server_${SLURM_JOB_ID}.log; exit 1; }; sleep 5; done
  echo "res server up on $EX_PORT"
fi
BB=$( [ -n "${RES:-}" ] && /data/asingal2/envs/slres/bin/python -c "import torch;print(torch.load('$RES/residual.pt',map_location='cpu',weights_only=False)['backbone'])" || echo none)
if [ "$BB" = pg ] || [ "$BB" = sig ]; then
  (PYTHONPATH=$OP/src:$OP/packages/openpi-client/src:$PWD OPENPI_DATA_HOME=/data/asingal2/openpi_cache \
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 /data/asingal2/envs/openpi/bin/python pg_server.py --port $PG_PORT \
   > /data/asingal2/slurm_logs/pg_server_${SLURM_JOB_ID}.log 2>&1) &
  PG_PID=$!
  for i in $(seq 1 180); do (echo > /dev/tcp/127.0.0.1/$PG_PORT) 2>/dev/null && break
    kill -0 $PG_PID 2>/dev/null || { tail -30 /data/asingal2/slurm_logs/pg_server_${SLURM_JOB_ID}.log; exit 1; }; sleep 5; done
  echo "pg server up on $PG_PORT"
fi
TAG=test${NOVEL:+_novel}; SUB=eps$EPS; SC=""
if [ -n "${SCENES:-}" ]; then TAG=$(basename $SCENES); SUB=all; SC="--scenes data_v2/$S/$SCENES.npz"; fi
COMMON="--suite safelibero_$S --ckpt runs_scale/$S/base/ema_last.pt --base pi05 --pi_port $PI_PORT --env_seed 0 --settle 20 --exec_steps 5 --res 256 --workers ${WORKERS:-6} --eps $EPS $SC ${NOVEL:+--novel}"
for A in $ARMS; do
  if [ "$A" = base ]; then
    $R python eval_sl.py $COMMON --arm base --out runs_v2/$S/$TAG/pi05/$SUB
  elif [ -n "${EX:-}" ]; then
    $R python eval_sl.py $COMMON --arm residual_ex --ex_port $EX_PORT --text_mode $A \
        --out runs_v2/$S/$TAG/$(basename $(dirname $EX))/$A/$SUB
  else
    : "${RES:?}"
    $R python eval_sl.py $COMMON --arm residual_pg --residual $RES/residual.pt --pg_port $PG_PORT --text_mode $A \
        --out runs_v2/$S/$TAG/$(basename $RES)/$A/$SUB
  fi
done
echo ALL_DONE
