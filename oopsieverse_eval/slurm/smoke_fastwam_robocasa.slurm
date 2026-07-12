#!/bin/bash
# Fast-WAM (off-the-shelf, libero_uncond_2cam224) smoke test on OopsieVerse
# RoboCasa: 3 episodes each on 2 representative tasks. Uses a fresh, isolated
# oopsieverse_fastwam_srv env (server) + the already-built oopsieverse_robocasa
# env (sim client) -- neither touches the existing fastwam_env or its SafeLIBERO
# coupling. Reuses the already-downloaded Wan2.2-TI2V-5B + Fast-WAM checkpoint
# read-only. H100 for the video-diffusion model's VRAM needs.
#SBATCH --job-name=oopsieverse_fastwam_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=01:30:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#SBATCH --mail-user=asingal2@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail
echo "=== OopsieVerse RoboCasa x Fast-WAM smoke test ==="

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/oopsieverse-eval
EVAL_DIR="$REPO/oopsieverse_eval"
MAIN_REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
SERVER_LOG=/ocean/projects/cis250185p/asingal/slurm_logs/fastwam_server_${SLURM_JOB_ID}.log
PORT=8300

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

module load anaconda3

# ── Start Fast-WAM REST server ───────────────────────────────────────────
conda activate /ocean/projects/cis250185p/asingal/envs/oopsieverse_fastwam_srv
export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
export DIFFSYNTH_MODEL_BASE_PATH="$MAIN_REPO/fast-wam/checkpoints"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

echo "Starting Fast-WAM server..."
python "$EVAL_DIR/servers/fastwam_server.py" \
    --ckpt_path "$MAIN_REPO/fast-wam/checkpoints/libero_uncond_2cam224.pt" \
    --stats_path "$MAIN_REPO/fast-wam/checkpoints/libero_uncond_2cam224_dataset_stats.json" \
    --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

cleanup() {
    echo "Stopping server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for server on port $PORT..."
MAX_WAIT=1200
ELAPSED=0
until curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    echo "  ...${ELAPSED}s"
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Server did not start. Log:"
        cat "$SERVER_LOG"
        exit 1
    fi
done
echo "Server ready!"

# ── Run eval client (sim-host env) ───────────────────────────────────────
conda activate /ocean/projects/cis250185p/asingal/envs/oopsieverse_robocasa
export MUJOCO_GL=egl

python "$EVAL_DIR/eval/run_robocasa_fastwam_eval.py" \
    --tasks pick_egg turn_on_faucet \
    --num_trials 3 \
    --server_url "http://localhost:${PORT}" \
    --results_dir "$EVAL_DIR/results/fastwam_robocasa_smoke" \
    --seed 7

EVAL_EXIT=$?
echo "=== Smoke test done, exit=$EVAL_EXIT ==="
echo "Check results: $EVAL_DIR/results/fastwam_robocasa_smoke/"
echo "Server log: $SERVER_LOG"
exit $EVAL_EXIT
