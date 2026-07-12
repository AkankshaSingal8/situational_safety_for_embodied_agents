#!/bin/bash
#SBATCH --job-name=ctx_pi05_pilot
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --time=20:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WORKTREE="$ROOT/.worktrees/contextual-predictive-safety-filter"
AEGIS="$ROOT/vlsa-aegis"
TRIALS=${TRIALS:-50}
OUT=${OUT:-$WORKTREE/contextual_pilot/pi05}
PORT=${PORT:-8000}

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="$AEGIS/openpi/src:${PYTHONPATH:-}"
python "$AEGIS/openpi/scripts/serve_policy.py" \
  --port "$PORT" policy:checkpoint --policy.config=pi05_libero \
  --policy.dir="$AEGIS/checkpoints/pi05_libero" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  if python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null; then
    break
  fi
  sleep 5
done
kill -0 "$SERVER_PID"

conda activate libero
export MUJOCO_GL=egl
export PYTHONPATH="$AEGIS/openpi/src:$AEGIS/openpi/packages/openpi-client/src:$ROOT/SafeLIBERO/safelibero:$WORKTREE/vlm_pipeline:${PYTHONPATH:-}"
for LEVEL in I II; do
  python "$WORKTREE/vlm_pipeline/run_safelibero_pi05_eval.py" \
    --task_suite_name safelibero_spatial \
    --safety_level "$LEVEL" \
    --num_trials_per_task "$TRIALS" \
    --host 127.0.0.1 --port "$PORT" \
    --use_contextual_filter \
    --results_output_dir "$OUT/results" \
    --video_output_dir "$OUT/videos"
done
