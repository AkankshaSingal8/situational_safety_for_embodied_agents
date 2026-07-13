#!/bin/bash
#SBATCH --job-name=fgd_pi05_suite_n50
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w008,w009
#SBATCH --time=20:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

# Tier-GT guidance on pi0.5 — full-suite n=50 evaluation (Phase 1).
# Usage: sbatch -J fgd_pi05_goal_n50 --export=ALL,SUITE=safelibero_goal,PORT=8125 slurm/fgd_pi05_suite_n50.slurm
# Config r3: eef_radius=0.09, inflation=0.004/step, gamma=0.9, d_safe=0.01, companion points.

set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/fgd_n50_$TAG"
SUITE=${SUITE:-safelibero_goal}
PORT=${PORT:-8125}
K=${K:-1}
SCHEDULE=${SCHEDULE:-uniform}
EEFR=${EEFR:-0.09}
CORRIDOR=${CORRIDOR:-}
COMPANIONS=${COMPANIONS:-True}
TAG=${TAG:-r3}

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="$WT/openpi_guided/src"
python "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule "$SCHEDULE" --num_candidates "$K" --eef_radius "$EEFR" --use_companions "$COMPANIONS" \
  --policy.config=pi05_libero \
  --policy.dir="$AEGIS/checkpoints/pi05_libero" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 240); do
  if python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null; then
    break
  fi
  sleep 5
done
kill -0 "$SERVER_PID"

conda activate openvla_libero_merged
export MUJOCO_GL=egl
export PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

for LEVEL in I II; do
python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name "$SUITE" --safety_level "$LEVEL" \
  --num_trials_per_task 50 \
  --host 127.0.0.1 --port "$PORT" $CORRIDOR \
  --results_output_dir "$OUT/results"
done

echo "SUITE_N50_COMPLETE $SUITE"
