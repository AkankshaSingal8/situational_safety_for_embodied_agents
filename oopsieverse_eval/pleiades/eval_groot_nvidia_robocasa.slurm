#!/bin/bash
# NVIDIA's own GR00T-N1.6-3B RoboCasa eval (Isaac-GR00T examples/robocasa/README.md, verbatim commands),
# stock RoboCasa (squarefk/robocasa@d89d481), to check our server against NVIDIA's published numbers.
#   ENV_NAME=robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env N_EPISODES=20 sbatch -J ov_groot_nv \
#     -p pleiades -x pleiades-1-3,pleiades-1-10 --time=03:00:00 -c 8 --mem=64G --gres=gpu:1 \
#     -o /data/asingal2/slurm_logs/%x_%j.out eval_groot_nvidia_robocasa.slurm
set -uo pipefail
ENV_NAME=${ENV_NAME:-robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env}
N_EPISODES=${N_EPISODES:-20}
N_ENVS=${N_ENVS:-5}
G=/data/asingal2/oopsieverse/externals/Isaac-GR00T
SRV=/data/asingal2/envs/oopsieverse_groot_srv
CLI=$G/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python
export HF_HOME=/data/asingal2/hf_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
PORT=$((5900 + SLURM_JOB_ID % 90))
OUT=/data/asingal2/oopsieverse/results/groot_nvidia_robocasa/$(basename $ENV_NAME)
mkdir -p $OUT
cd $G
echo "host $(hostname) env=$ENV_NAME episodes=$N_EPISODES port=$PORT"
/data/asingal2/envs/apptainer/bin/apptainer exec --nv --bind /data --bind /home/asingal2 --env HF_HOME=$HF_HOME \
  --pwd /data/asingal2/oopsieverse/externals/Isaac-GR00T-nv /data/asingal2/oopsieverse/b1k_sandbox /data/asingal2/envs/groot_srv_u24/bin/python /data/asingal2/oopsieverse/externals/Isaac-GR00T-nv/gr00t/eval/run_gr00t_server.py --model-path nvidia/GR00T-N1.6-3B \
  --embodiment-tag ROBOCASA_PANDA_OMRON --use-sim-policy-wrapper --port $PORT > $OUT/server_$SLURM_JOB_ID.log 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null' EXIT
for i in $(seq 1 90); do
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -30 $OUT/server_$SLURM_JOB_ID.log; exit 1; }
  (cd /home/asingal2/safety/wt-oopsieverse-eval/oopsieverse_eval/eval && /data/asingal2/envs/oopsieverse_robocasa/bin/python -c "
import sys; from groot_zmq_client import GrootZmqClient
sys.exit(0 if GrootZmqClient('127.0.0.1', $PORT, timeout_ms=5000).ping() else 1)") 2>/dev/null && { echo "server up"; break; }
  sleep 10
done
$CLI gr00t/eval/rollout_policy.py --n_episodes $N_EPISODES --policy_client_host 127.0.0.1 \
  --policy_client_port $PORT --max_episode_steps=720 --env_name $ENV_NAME --n_action_steps 8 --n_envs $N_ENVS 2>&1 | tee $OUT/client_$SLURM_JOB_ID.log | grep -v "^\s*$" | tail -40
RC_=${PIPESTATUS[0]}; echo "EXIT=$RC_"; exit $RC_
