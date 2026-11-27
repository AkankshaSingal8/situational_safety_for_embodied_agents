#!/bin/bash
# GR00T-N1.6-3B on NVIDIA's stock RoboCasa tasks (reproduces the paper's GR00T completion) + OopsieVerse
# DamageSim for safe completion / health. See ../eval/run_groot_stock_damagesim.py.
#   TASK=open_single_door sbatch -J ov_grootdmg_open_single_door -p pleiades -x pleiades-1-3,pleiades-1-10 \
#     --time=05:00:00 -c 8 --mem=64G --gres=gpu:1 -o /data/asingal2/slurm_logs/%x_%j.out eval_groot_stock_damagesim.slurm
set -uo pipefail
TASK=${TASK:?set TASK}
NUM_TRIALS=${NUM_TRIALS:-30}
G=/data/asingal2/oopsieverse/externals/Isaac-GR00T
GNV=/data/asingal2/oopsieverse/externals/Isaac-GR00T-nv
CLI=$G/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python
OV=/home/asingal2/safety/ext/oopsieverse
EVAL=/home/asingal2/safety/wt-oopsieverse-eval/oopsieverse_eval/eval
export DMG_STOCK_ROBOT=${DMG_STOCK_ROBOT:-0} HF_HOME=/data/asingal2/hf_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PATH=/home/asingal2/.local/bin:$PATH
PORT=$((6100 + SLURM_JOB_ID % 300))
OUT=${OUT:-/data/asingal2/oopsieverse/results/groot_stock_damagesim/$TASK}
mkdir -p $OUT
echo "host $(hostname) task=$TASK trials=$NUM_TRIALS port=$PORT"
$CLI -c "import damagesim.robosuite.damageable_env" 2>/dev/null || \
  uv pip install -q --python $CLI -e $OV --no-deps numba==0.62.1 scipy matplotlib 2>&1 | tail -3

if [ -n "${RES_CKPT:-}" ]; then  # GR00T base + residual expert (residual/residual_server.py)
/data/asingal2/envs/apptainer/bin/apptainer exec --nv --bind /data --bind /home/asingal2 --env HF_HOME=$HF_HOME \
  --pwd $GNV /data/asingal2/oopsieverse/b1k_sandbox /data/asingal2/envs/groot_srv_u24/bin/python \
  /home/asingal2/safety/wt-oopsieverse-eval/oopsieverse_eval/residual/residual_server.py --ckpt $RES_CKPT \
  --task $TASK --text_mode ${TEXT_MODE:-true} --n ${RES_N:-1} --gate_thr ${GATE_THR:-0} --port $PORT > $OUT/server_$SLURM_JOB_ID.log 2>&1 &
else
/data/asingal2/envs/apptainer/bin/apptainer exec --nv --bind /data --bind /home/asingal2 --env HF_HOME=$HF_HOME \
  --pwd $GNV /data/asingal2/oopsieverse/b1k_sandbox /data/asingal2/envs/groot_srv_u24/bin/python \
  $GNV/gr00t/eval/run_gr00t_server.py --model-path nvidia/GR00T-N1.6-3B --embodiment-tag ROBOCASA_PANDA_OMRON \
  --use-sim-policy-wrapper --port $PORT > $OUT/server_$SLURM_JOB_ID.log 2>&1 &
fi
SP=$!
trap 'kill $SP 2>/dev/null' EXIT
for i in $(seq 1 90); do
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -30 $OUT/server_$SLURM_JOB_ID.log; exit 1; }
  $CLI -c "
import sys; from gr00t.policy.server_client import PolicyClient
sys.exit(0 if PolicyClient(host='127.0.0.1', port=$PORT, timeout_ms=5000).ping() else 1)" 2>/dev/null && { echo "server up"; break; }
  sleep 10
  [ $i -eq 90 ] && { echo "server never answered"; exit 1; }
done
cd $G
$CLI $EVAL/run_groot_stock_damagesim.py --task $TASK --num_trials $NUM_TRIALS --port $PORT --results_dir $OUT --seed ${SEED:-7} ${SHIELD:+--shield} ${SHIELD_CFG:+--shield_cfg "$SHIELD_CFG"} ${SAVE_DATA:+--save_data $SAVE_DATA} ${STOP_ON_SUCCESS:+--stop_on_success}
RC_=$?; echo "EXIT=$RC_"; exit $RC_
