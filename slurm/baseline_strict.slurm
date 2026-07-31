#!/bin/bash
#SBATCH --job-name=baseline_strict
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.err

# Baseline strict-SR fill for Table 1b: rerun the unguided pi0.5 baseline
# column (all 4 suites x L1/L2, n=50/task, seed 7 protocol) WITH per-episode
# logging so strict SR (success AND zero-collision) is exact instead of the
# min(TSR,CAR) upper bound. Uses the guided client's sanity-baseline path
# (--disable_guidance -> enabled=0 through the identical code path/metrics).

set -euo pipefail
module load anaconda3
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh

ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/baseline_strict"
N=${N:?export N explicitly}

SUITES=(safelibero_spatial safelibero_goal safelibero_object safelibero_long)
SUITE="${SUITES[$SLURM_ARRAY_TASK_ID]}"
PORT=$((8290 + SLURM_ARRAY_TASK_ID))

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="$WT/openpi_guided/src"
python "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" \
  --policy.config=pi05_libero \
  --policy.dir="$AEGIS/checkpoints/pi05_libero" \
  --repair_schedule uniform --num_candidates 1 &
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

for LEVEL in I II; do
python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name "$SUITE" --safety_level "$LEVEL" \
  --num_trials_per_task "$N" \
  --host 127.0.0.1 --port "$PORT" --disable_guidance \
  --results_output_dir "$OUT/results"
done

echo "BASELINE_STRICT_COMPLETE_${SUITE}"
