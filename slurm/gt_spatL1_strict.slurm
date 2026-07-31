#!/bin/bash
#SBATCH --job-name=gt_spatL1_strict
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

# GT Spatial L1 strict-SR fill: reproduce the Table-1 board cell (71.0/23.5,
# r2 config, job 42165451 era) WITH per-episode logging so strict SR
# (success AND zero-collision) can be stated exactly. Config r2 pinned
# explicitly against server default drift: sphere, eef_radius 0.09,
# inflation 0.004/step, gamma 0.9, d_safe 0.01, EEF-only (NO companions),
# uniform repair schedule, K=1, no corridor, GT obstacle positions.

set -euo pipefail
module load anaconda3
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh

ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/gt_spatL1_strict"
PORT=${PORT:-8285}
N=${N:?export N explicitly}

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="$WT/openpi_guided/src"
python "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" \
  --policy.config=pi05_libero \
  --policy.dir="$AEGIS/checkpoints/pi05_libero" \
  --gamma 0.9 --d_safe 0.01 --eef_radius 0.09 --inflation_slope 0.004 \
  --repair_schedule uniform --num_candidates 1 --no-use_companions &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null; then
    break
  fi
  sleep 5
done
kill -0 "$SERVER_PID"

conda activate openvla_libero_merged
export MUJOCO_GL=egl
export PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_spatial --safety_level I \
  --num_trials_per_task "$N" \
  --host 127.0.0.1 --port "$PORT" \
  --results_output_dir "$OUT/results"

echo "GT_SPATL1_STRICT_COMPLETE"
