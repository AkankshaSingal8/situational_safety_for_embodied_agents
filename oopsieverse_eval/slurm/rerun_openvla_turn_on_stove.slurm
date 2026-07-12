#!/bin/bash
# Standalone rerun of OpenVLA full-eval for turn_on_stove only (array shard
# 5 failed before the get_ep_meta()==None fix -- see
# RoboCasaAdapter.get_instruction()). Not a full array, just this one task.
#SBATCH --job-name=oopsieverse_openvla_rerun_stove
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=63000M
#SBATCH --time=01:30:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail
REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/oopsieverse-eval
EVAL_DIR="$REPO/oopsieverse_eval"
PORT=8199
TASK_NAME="turn_on_stove"
SERVER_LOG=/ocean/projects/cis250185p/asingal/slurm_logs/openvla_server_stove_${SLURM_JOB_ID}.log

echo "=== OpenVLA rerun: task=$TASK_NAME, port=$PORT ==="
mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

module load anaconda3

conda activate /ocean/projects/cis250185p/asingal/envs/oopsieverse_openvla_srv
export HF_HOME=/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/hf
export TRANSFORMERS_CACHE="$HF_HOME"

python "$EVAL_DIR/servers/openvla_server.py" \
    --checkpoint openvla/openvla-7b-finetuned-libero-10 \
    --unnorm_key libero_10 \
    --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "Waiting for server on port $PORT..."
MAX_WAIT=900
ELAPSED=0
until curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Server did not start. Log:"; cat "$SERVER_LOG"; exit 1
    fi
done
echo "Server ready!"

conda activate /ocean/projects/cis250185p/asingal/envs/oopsieverse_robocasa
export MUJOCO_GL=egl

python "$EVAL_DIR/eval/run_robocasa_openvla_eval.py" \
    --tasks "$TASK_NAME" \
    --num_trials 10 \
    --server_url "http://localhost:${PORT}" \
    --results_dir "$EVAL_DIR/results/openvla_robocasa_full/$TASK_NAME" \
    --seed 7

EXIT=$?
echo "=== $TASK_NAME done, exit=$EXIT ==="
exit $EXIT
