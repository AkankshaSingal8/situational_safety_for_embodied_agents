#!/bin/bash
# Standalone rerun of Fast-WAM full-eval for turn_on_stove only (the main
# full-suite run predates the get_ep_meta()==None fix -- see
# RoboCasaAdapter.get_instruction() -- and failed all 10 trials for this
# one task; the other 14 tasks x 10 trials all ran cleanly).
#SBATCH --job-name=oopsieverse_fastwam_rerun_stove
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=01:00:00
#SBATCH --exclude=w006
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail
REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/oopsieverse-eval
EVAL_DIR="$REPO/oopsieverse_eval"
MAIN_REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
SERVER_LOG=/ocean/projects/cis250185p/asingal/slurm_logs/fastwam_server_stove_${SLURM_JOB_ID}.log
PORT=8399

echo "=== Fast-WAM rerun: task=turn_on_stove, port=$PORT ==="
mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

module load anaconda3

conda activate /ocean/projects/cis250185p/asingal/envs/oopsieverse_fastwam_srv
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export DIFFSYNTH_MODEL_BASE_PATH="$MAIN_REPO/fast-wam/checkpoints"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

python "$EVAL_DIR/servers/fastwam_server.py" \
    --ckpt_path "$MAIN_REPO/fast-wam/checkpoints/libero_uncond_2cam224.pt" \
    --stats_path "$MAIN_REPO/fast-wam/checkpoints/libero_uncond_2cam224_dataset_stats.json" \
    --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "Waiting for server on port $PORT..."
MAX_WAIT=1200
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

python "$EVAL_DIR/eval/run_robocasa_fastwam_eval.py" \
    --tasks turn_on_stove \
    --num_trials 10 \
    --server_url "http://localhost:${PORT}" \
    --results_dir "$EVAL_DIR/results/fastwam_robocasa_full/turn_on_stove" \
    --seed 7

EXIT=$?
echo "=== turn_on_stove done, exit=$EXIT ==="
exit $EXIT
