#!/bin/bash
#SBATCH --job-name=fgd_pi05_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w008,w009
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

# Tier-GT in-denoising DCBF flow guidance on pi0.5 — SMOKE (Spatial L1+L2, n=5/task)
# Gate: aggregate CAR >= ~2x baseline (14/12) at TSR within n=5 noise of 67/55.5;
# videos must show detouring, not stalling.

set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"        # shared checkpoint dir (read-only use)
TRIALS=${TRIALS:-5}
OUT=${OUT:-$WT/fgd_smoke}
PORT=${PORT:-8123}

# ---------------- server (guided, private openpi copy) ----------------
conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="$WT/openpi_guided/src"
python "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" \
  --policy.config=pi05_libero \
  --policy.dir="$AEGIS/checkpoints/pi05_libero" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null; then
    break
  fi
  sleep 5
done
kill -0 "$SERVER_PID"

# ---------------- client (openvla_libero_merged: the env with months of
# validated batch LIBERO+EGL runs; py3.8 'libero' env core-dumped in batch
# and was never batch-validated — see job 42157846) ----------------
conda activate openvla_libero_merged
export MUJOCO_GL=egl
export PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

# Phase 0: disabled-guidance sanity (task 0, L1, n=2) — same server, enabled=0
python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_spatial --safety_level I \
  --num_trials_per_task 2 --task_indices 0 \
  --host 127.0.0.1 --port "$PORT" \
  --disable_guidance \
  --results_output_dir "$OUT/results_disabled_sanity"

# Phase 1: guided smoke, both levels, videos on
for LEVEL in I II; do
  python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
    --task_suite_name safelibero_spatial --safety_level "$LEVEL" \
    --num_trials_per_task "$TRIALS" \
    --host 127.0.0.1 --port "$PORT" \
    --save_videos \
    --results_output_dir "$OUT/results" \
    --video_output_dir "$OUT/videos"
done

echo "SMOKE_COMPLETE"
