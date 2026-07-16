#!/bin/bash
#SBATCH --job-name=fgd_devcells_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w005,w006,w007,w008,w009
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

# Track-A schedule A/B: ramp repair schedule on dev cells (n=5).
# References (uniform r3): Spatial L1 smoke 60/35, n50 58.5/44.5; Object L1 t1 18/0, t2 48/0.

set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/fgd_devcells"
PORT=${PORT:-8128}
SCHEDULE=${SCHEDULE:-ramp}
K=${K:-1}
TAG=${TAG:-ramp}
EEFR=${EEFR:-0.09}
CORRIDOR=${CORRIDOR:-}

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="$WT/openpi_guided/src"
python "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule "$SCHEDULE" --num_candidates "$K" --eef_radius "$EEFR" \
  --policy.config=pi05_libero \
  --policy.dir="$AEGIS/checkpoints/pi05_libero" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 240); do
  if python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null; then break; fi
  sleep 5
done
kill -0 "$SERVER_PID"

conda activate openvla_libero_merged
export MUJOCO_GL=egl
export PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_spatial --safety_level I \
  --num_trials_per_task 5 --host 127.0.0.1 --port "$PORT" $CORRIDOR \
  --results_output_dir "$OUT/$TAG"

python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_object --safety_level I \
  --num_trials_per_task 5 --task_indices 1 2 \
  --host 127.0.0.1 --port "$PORT" $CORRIDOR \
  --results_output_dir "$OUT/$TAG"

python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_long --safety_level I \
  --num_trials_per_task 5 --task_indices 0 1 \
  --host 127.0.0.1 --port "$PORT" $CORRIDOR \
  --results_output_dir "$OUT/$TAG"

echo "DEVCELLS_SMOKE_COMPLETE"
