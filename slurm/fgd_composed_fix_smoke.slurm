#!/bin/bash
#SBATCH --job-name=fgd_composed_fix_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w005,w006,w007,w008,w009
#SBATCH --time=03:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

# n=5 smoke to ROUTE the fixture-dest + stale-target corridor fix
# (commit cc3f766) before committing to n=50 reruns. Cells: Goal L2 task 1
# (cabinet, the fixture-dest bug: composed vs r3 TSR 76->28, CAR 82->60) and
# Long L2 tasks 0,1 (basket, the stale-target bug: TSR 84->50/82->38).
# Composed config: K=8, ramp schedule, corridor, eef_r=0.12, companions.

set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/fgd_composed_fix_smoke"
PORT=${PORT:-8129}

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="$WT/openpi_guided/src"
python "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule ramp --num_candidates 8 --eef_radius 0.12 \
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
  --task_suite_name safelibero_goal --safety_level II \
  --num_trials_per_task 5 --task_indices 1 \
  --host 127.0.0.1 --port "$PORT" --corridor \
  --results_output_dir "$OUT/results"

python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_long --safety_level II \
  --num_trials_per_task 5 --task_indices 0 1 \
  --host 127.0.0.1 --port "$PORT" --corridor \
  --results_output_dir "$OUT/results"

echo "COMPOSED_FIX_SMOKE_COMPLETE"
