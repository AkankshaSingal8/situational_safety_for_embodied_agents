#!/bin/bash
# GR00T-N1.6 RoboCasa smoke test (risk-gate spike, task step 6): 3 episodes
# on pick_egg via the real GR00T-driven client, PLUS a matched random-action
# baseline (same task/seeds), to distinguish "the policy is doing something
# reactive" from "the policy is emitting noise" -- see motion_stats.py.
#
# Server runs inside the uv-managed `oopsieverse_groot_srv` venv (needs the
# h100-80 GPU -- see spike_groot_risk_gate.slurm's postmortem on why v100-32
# fails with "no kernel image available"); client runs in the already-built
# `oopsieverse_robocasa` conda env via a lightweight ZMQ reimplementation
# (groot_zmq_client.py), NOT by importing the real gr00t package into that
# env (same house style as launch_cosmos_server.sh's Apptainer isolation).
#SBATCH --job-name=oopsieverse_groot_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=63000M
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

set -uo pipefail
echo "=== OopsieVerse RoboCasa x GR00T-N1.6 smoke test ==="

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
EVAL_DIR="$REPO/.worktrees/oopsieverse-eval/oopsieverse_eval"
GROOT_ENV=/ocean/projects/cis250185p/asingal/envs/oopsieverse_groot_srv
SERVER_LOG=/ocean/projects/cis250185p/asingal/slurm_logs/groot_server_${SLURM_JOB_ID}.log
PORT=5555

mkdir -p /ocean/projects/cis250185p/asingal/slurm_logs

module load anaconda3
module load cuda/12.6.1
export HF_HOME=/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/hf

# ── Start GR00T server (uv-managed venv) ─────────────────────────────────
"$GROOT_ENV/bin/python" "$EVAL_DIR/servers/groot_server.py" --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

cleanup() {
    echo "Stopping server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for server on port $PORT..."
MAX_WAIT=900
ELAPSED=0
until nc -z localhost "$PORT" 2>/dev/null; do
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

# ── Run real GR00T-driven episodes (sim-host env) ────────────────────────
conda activate /ocean/projects/cis250185p/asingal/envs/oopsieverse_robocasa
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

python -u "$EVAL_DIR/eval/run_robocasa_groot_eval.py" \
    --tasks pick_egg \
    --num_trials 3 \
    --server_host 127.0.0.1 --server_port "$PORT" \
    --results_dir "$EVAL_DIR/results/groot_robocasa_smoke" \
    --seed 7
GROOT_EXIT=$?
echo "=== GR00T client done, exit=$GROOT_EXIT ==="

# ── Run matched random-action baseline (no server needed) ────────────────
python -u "$EVAL_DIR/eval/groot_random_baseline_eval.py" \
    --tasks pick_egg \
    --num_trials 3 \
    --results_dir "$EVAL_DIR/results/groot_robocasa_random_baseline" \
    --seed 7
RANDOM_EXIT=$?
echo "=== Random baseline done, exit=$RANDOM_EXIT ==="

# ── Compare motion stats ──────────────────────────────────────────────────
python -u - <<'PYEOF'
import json

groot = json.load(open("/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/oopsieverse-eval/oopsieverse_eval/results/groot_robocasa_smoke/episodes.json"))
random_ = json.load(open("/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/oopsieverse-eval/oopsieverse_eval/results/groot_robocasa_random_baseline/episodes.json"))

def summarize(records, label):
    disp = [r["eef_total_displacement"] for r in records if r.get("error") is None]
    diff = [r["mean_frame_diff"] for r in records if r.get("error") is None]
    print(f"{label}: n={len(disp)}")
    print(f"  eef_total_displacement: {disp}")
    print(f"  mean_frame_diff:        {diff}")
    return disp, diff

print("=== MOTION-DEGENERACY COMPARISON ===")
groot_disp, groot_diff = summarize(groot, "GR00T-driven")
rand_disp, rand_diff = summarize(random_, "Random baseline")

import statistics
if groot_disp and rand_disp:
    print(f"\nMean eef displacement -- GR00T: {statistics.mean(groot_disp):.4f}  Random: {statistics.mean(rand_disp):.4f}")
if groot_diff and rand_diff:
    print(f"Mean frame diff        -- GR00T: {statistics.mean(groot_diff):.4f}  Random: {statistics.mean(rand_diff):.4f}")
PYEOF

echo "Check results: $EVAL_DIR/results/groot_robocasa_smoke/ and groot_robocasa_random_baseline/"
echo "Server log: $SERVER_LOG"
exit $GROOT_EXIT
