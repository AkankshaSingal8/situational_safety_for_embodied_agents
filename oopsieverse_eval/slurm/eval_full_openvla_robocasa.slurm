#!/bin/bash
# Full OpenVLA RoboCasa evaluation: all 16 tasks x 10 trials.
#
# SLURM ARRAY (one array index per task), not one monolithic job: OpenVLA's
# smoke test measured ~326s/episode (single-step autoregressive decoding,
# no OFT-style action chunking) -- 16 tasks x 10 trials serially would be
# ~14.5 hours in one job. Splitting into 16 independent ~1hr shards lets
# them run concurrently (subject to cluster availability) instead of
# serially, and a failure in one task's shard doesn't lose the other 15.
#SBATCH --job-name=oopsieverse_openvla_full
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=63000M
#SBATCH --time=01:30:00
#SBATCH --array=0-15
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/oopsieverse-eval
EVAL_DIR="$REPO/oopsieverse_eval"
# +10000 per array index so ports never collide if shards land on the same node.
PORT=$((8100 + SLURM_ARRAY_TASK_ID + 100))
SERVER_LOG=/ocean/projects/cis250185p/asingal/slurm_logs/openvla_server_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log

TASKS=(pick_egg serve_pastry open_single_door turn_on_faucet turn_on_microwave turn_on_stove
       open_drawer close_drawer place_plate counter_to_microwave prepare_coffee shelve_item
       prepare_breakfast dishes_to_sink nav_lift_bowl wipe_counter)
TASK_NAME="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "=== OpenVLA full eval: task=$TASK_NAME (array idx $SLURM_ARRAY_TASK_ID), port=$PORT ==="

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
