#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH --exclude=pleiades-1-3,pleiades-1-10
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# ex2 DAgger round: pi0.5 + ex2 residual drives (each chunk checked by the shield's lookahead; the shield's plan runs
# instead when ex2's chunk would collide), the shield labels every visited state with plan - pi0.5. Episodes without
# a collision are kept whatever the task outcome (meta: dagger, success, ex_frac).
#   GSO obstacles, training objects, layouts 0-39:
#     sbatch -J dag_gso_long --export=ALL,S=long,MODE=gso,EX=runs_v2/expert/ex2_1gpu_bs64/best.pkl run_dagger_ex.slurm
#   SafeLIBERO obstacles, training scenes (layouts 0-39), IDS=0-959 (level I) or 960-1919 (level II):
#     sbatch -J dag_v2_long_II --export=ALL,S=long,MODE=v2,IDS=960-1919,EX=... run_dagger_ex.slurm
#   optional: ROUND=dag1 (output name), SEED, PER (GSO objects per layout, default 2), ITEMS, WORKERS=6,
#   OBJS (gso mode: json list of model names, e.g. runs_v2/gso_rs_objects.json with GSO_RESCALED=1),
#   SCENES (v2 mode scene file, default data_v2/<S>/scenes_train.npz; round 2 uses scenes_train_d.npz)
# Outputs: data_gso_<ROUND>/<S>/  or  data_v2/<S>/<ROUND>_<IDS>/   (resumable)
set -euo pipefail
: "${S:?}" "${MODE:?}" "${EX:?}"
ROUND=${ROUND:-dag1}
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
OP=/home/asingal2/safety/situational_safety_for_embodied_agents/vlsa-aegis/openpi
PI_PORT=$((8000 + RANDOM % 1000)); EX_PORT=$((10000 + RANDOM % 1000))
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
source ./run_pi05_server.sh pi05_libero $PI_PORT
(PYTHONPATH=$OP/src:$OP/packages/openpi-client/src:$PWD OPENPI_DATA_HOME=/data/asingal2/openpi_cache \
 XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 /data/asingal2/envs/openpi/bin/python res_server.py --port $EX_PORT --ckpt $EX \
 > /data/asingal2/slurm_logs/res_server_${SLURM_JOB_ID}.log 2>&1) &
EX_PID=$!
trap 'kill $PI_SERVER_PID $EX_PID 2>/dev/null || true' EXIT
for i in $(seq 1 240); do (echo > /dev/tcp/127.0.0.1/$EX_PORT) 2>/dev/null && break
  kill -0 $EX_PID 2>/dev/null || { tail -30 /data/asingal2/slurm_logs/res_server_${SLURM_JOB_ID}.log; exit 1; }; sleep 5; done
echo "res server up on $EX_PORT"
export GSO_LOCAL=/tmp/gso_$USER
COMMON="--suite safelibero_$S --ckpt runs_scale/$S/base/ema_last.pt --base pi05 --pi_port $PI_PORT --ex_port $EX_PORT \
  --cell_cfgs runs_v2/$S/cell_cfgs.json --tries 1 --workers ${WORKERS:-6}"
if [ "$MODE" = gso ]; then
  ./pleiades_run.sh python collect_gso.py $COMMON --split train --layouts 0-39 --per_layout ${PER:-2} \
      --seed ${SEED:-21} --out data_gso_$ROUND/$S ${ITEMS:+--item_ids $ITEMS} ${OBJS:+--objects_file $OBJS}
else
  : "${IDS:?}"
  ./pleiades_run.sh python collect_v2.py $COMMON --scenes ${SCENES:-data_v2/$S/scenes_train.npz} --p_drive 1.0 \
      --seed ${SEED:-21} --scene_ids $IDS --out data_v2/$S/${ROUND}_${IDS}
fi
echo ALL_DONE
