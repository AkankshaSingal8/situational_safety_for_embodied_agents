#!/bin/bash
# OopsieVerse Table I GR00T on RoboCasa: off-the-shelf nvidia/GR00T-N1.6-3B, ROBOCASA_PANDA_OMRON
# embodiment (pretrained in the base checkpoint), paper protocol (full-length rollouts, no early stop).
#   TASKS=open_single_door MAX_STEPS=720 sbatch -J ov_groot_open_single_door -p pleiades -x pleiades-1-3,pleiades-1-10 \
#        --time=05:00:00 -c 8 --mem=64G --gres=gpu:1 -o /data/asingal2/slurm_logs/%x_%j.out eval_robocasa_groot.slurm
set -uo pipefail
TASKS=${TASKS:?set TASKS}
NUM_TRIALS=${NUM_TRIALS:-30}
MAX_STEPS=${MAX_STEPS:-720}   # NVIDIA rollout_policy.py default and examples/robocasa README
N_ACTION_STEPS=${N_ACTION_STEPS:-8}   # NVIDIA gymnasium_groot.py --n-action-steps default
MODEL=${MODEL:-nvidia/GR00T-N1.6-3B}
EVAL_DIR=/home/asingal2/safety/wt-oopsieverse-eval/oopsieverse_eval
RC=/data/asingal2/envs/oopsieverse_robocasa
SRV=/data/asingal2/envs/oopsieverse_groot_srv
export HF_HOME=/data/asingal2/hf_cache
PORT=$((5600 + SLURM_JOB_ID % 300))
OUT=${OUT:-/data/asingal2/oopsieverse/results/robocasa_papermatch/groot_n16_full/$TASKS}
mkdir -p "$OUT"
echo "host $(hostname) model=$MODEL tasks=[$TASKS] trials=$NUM_TRIALS steps=$MAX_STEPS port=$PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -e $EVAL_DIR/external/oopsieverse/oopsiebench ] || { rmdir $EVAL_DIR/external/oopsieverse 2>/dev/null; ln -sfn /home/asingal2/safety/ext/oopsieverse $EVAL_DIR/external/oopsieverse; }
$RC/bin/python -c "import zmq" 2>/dev/null || $RC/bin/pip install -q pyzmq

# Server: unpatched NVIDIA Isaac-GR00T n1.6.1-release env with flash-attn, run in the Ubuntu 24.04 sandbox
# (glibc 2.39). The host env (glibc 2.28) needed flash-attn removed + forced eager attention, which degraded
# the model (open-loop OpenSingleDoor position MAE 0.176 vs 0.076 unpatched; ref/groot_openloop_check.py).
export PATH=/data/asingal2/envs/apptainer/bin:$PATH
apptainer exec --nv --bind /data --bind /home/asingal2 --env HF_HOME=$HF_HOME /data/asingal2/oopsieverse/b1k_sandbox \
  /data/asingal2/envs/groot_srv_u24/bin/python $EVAL_DIR/servers/groot_server.py --model_path $MODEL --port $PORT > $OUT/server_${SLURM_JOB_ID}.log 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null' EXIT
for i in $(seq 1 120); do
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -40 $OUT/server_${SLURM_JOB_ID}.log; exit 1; }
  (cd $EVAL_DIR/eval && $RC/bin/python -c "
import sys; from groot_zmq_client import GrootZmqClient
sys.exit(0 if GrootZmqClient('127.0.0.1', $PORT, timeout_ms=5000).ping() else 1)") 2>/dev/null && { echo "server up after ~$((i*15))s"; break; }
  sleep 10
  [ $i -eq 90 ] && { echo "server never answered"; exit 1; }
done

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
$RC/bin/python $EVAL_DIR/eval/run_robocasa_groot_eval.py --tasks $TASKS --num_trials $NUM_TRIALS \
   --server_host 127.0.0.1 --server_port $PORT --results_dir $OUT --seed 7 \
   --max_episode_steps $MAX_STEPS --n_action_steps $N_ACTION_STEPS --paper_protocol
RC_=$?; echo "EXIT=$RC_"; exit $RC_
