#!/bin/bash
# Paper-matched RoboCasa eval on pleiades (OopsieVerse Table I / Appendix E protocol):
# Open Single Door, Turn On Microwave, PnP Counter->Microwave, Turn On Stove, 30 trials each.
#   CKPT=pi0_libero|pi0_base|pi05_base|pi05_libero sbatch -J ov_rc_$CKPT -p pleiades -x pleiades-1-3 \
#        --time=08:00:00 -c 8 --mem=48G --gres=gpu:1 -o /data/asingal2/slurm_logs/%x_%j.out \
#        oopsieverse_eval/pleiades/eval_robocasa_papermatch.slurm
# *_base checkpoints are served with the matching *_libero config and LIBERO norm stats (linked into
# their assets dir), the same convention as pi05_base on SafeLIBERO. Client is unchanged from the
# PSC paper-match job (run_robocasa_pi05_eval.py; its LIBERO obs schema is shared by pi0 and pi0.5).
set -uo pipefail
CKPT=${CKPT:?set CKPT}
TASKS=${TASKS:-"open_single_door turn_on_microwave counter_to_microwave turn_on_stove"}
NUM_TRIALS=${NUM_TRIALS:-30}
SCHEMA=libero; REPLAN=5
case $CKPT in
  *_droid) CFG=$CKPT; SCHEMA=droid; REPLAN=8 ;;   # openpi examples/droid open_loop_horizon=8
  pi0_robocasa365) CFG=robocasa365; SCHEMA=robocasa365; REPLAN=5 ;;
  pi05_robocasa365) CFG=robocasa365_pi05; SCHEMA=robocasa365; REPLAN=5 ;;  # openpi@ca4c6d7 pi05_pretrain_human300  # robocasa-benchmark/openpi examples/robocasa
  pi0_*) CFG=pi0_libero ;; pi05_*) CFG=pi05_libero ;; *) echo "bad CKPT"; exit 2 ;; esac
EVAL_DIR=/home/asingal2/safety/wt-oopsieverse-eval/oopsieverse_eval
OP=/home/asingal2/safety/situational_safety_for_embodied_agents/vlsa-aegis/openpi
RC=/data/asingal2/envs/oopsieverse_robocasa
export OPENPI_DATA_HOME=/data/asingal2/openpi_cache
PORT=$((8300 + SLURM_JOB_ID % 500))
OUT=${OUT:-/data/asingal2/oopsieverse/results/robocasa_papermatch/$CKPT}
mkdir -p "$OUT"
echo "host $(hostname) ckpt=$CKPT cfg=$CFG port=$PORT tasks=[$TASKS] trials=$NUM_TRIALS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# client deps (idempotent)
# openpi-client pins numpy<2, but robocasa asserts numpy==2.2.5; the client only uses basic numpy.
# Install once only: concurrent jobs racing on the same env's editable .pth broke pip.
if ! $RC/bin/python -c "import openpi_client, tree, websockets" 2>/dev/null; then
  $RC/bin/pip install -q --no-deps -e $OP/packages/openpi-client 2>&1 | tail -2
  $RC/bin/pip install -q "numpy==2.2.5" dm-tree msgpack websockets tree 2>&1 | tail -2
fi
export PYTHONUNBUFFERED=1
[ -e $EVAL_DIR/external/oopsieverse/oopsiebench ] || { rmdir $EVAL_DIR/external/oopsieverse 2>/dev/null; ln -sfn /home/asingal2/safety/ext/oopsieverse $EVAL_DIR/external/oopsieverse; }

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
if [ "$CFG" = robocasa365_pi05 ]; then
  (cd $OP && /data/asingal2/envs/openpi/bin/python $EVAL_DIR/servers/serve_pi0_robocasa365.py --pi05 --port $PORT \
     --ckpt $OPENPI_DATA_HOME/robocasa365/pi05_pretrain_human300/multitask_learning/75000) > $OUT/server_${SLURM_JOB_ID}.log 2>&1 &
elif [ "$CFG" = robocasa365 ]; then  # official RoboCasa365 pi0 (HF robocasa/robocasa365_checkpoints)
  (cd $OP && /data/asingal2/envs/openpi/bin/python $EVAL_DIR/servers/serve_pi0_robocasa365.py --port $PORT \
     --ckpt $OPENPI_DATA_HOME/robocasa365/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000) > $OUT/server_${SLURM_JOB_ID}.log 2>&1 &
else
(cd $OP && /data/asingal2/envs/openpi/bin/python scripts/serve_policy.py --port $PORT policy:checkpoint \
   --policy.config=$CFG --policy.dir=$OPENPI_DATA_HOME/openpi-assets/checkpoints/$CKPT) > $OUT/server_${SLURM_JOB_ID}.log 2>&1 &
fi
SP=$!
trap 'kill $SP 2>/dev/null' EXIT
for i in $(seq 1 360); do
  (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && { echo "server up after $((i*5))s"; break; }
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -40 $OUT/server_${SLURM_JOB_ID}.log; exit 1; }
  sleep 5
done

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
$RC/bin/python $EVAL_DIR/eval/run_robocasa_pi05_eval.py --tasks $TASKS --num_trials $NUM_TRIALS \
   --host localhost --port $PORT --results_dir $OUT --seed 7 --schema $SCHEMA --replan_steps $REPLAN ${STOP_ON_SUCCESS:+--stop_on_success} --max_episode_steps ${MAX_STEPS:-400}
RC=$?; echo "EXIT=$RC"; exit $RC
