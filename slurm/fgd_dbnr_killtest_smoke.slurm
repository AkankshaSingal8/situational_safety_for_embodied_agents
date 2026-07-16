#!/bin/bash
#SBATCH --job-name=fgd_dbnr_killtest
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w005,w006,w007,w008,w009
#SBATCH --time=02:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

# DBNR kill-test (CONTEXT_HANDOFF.md #4.5), pre-registered dev cells:
# Spatial L1 t1 (TSR stuck <=28 across every config so far) + Long L2 t2
# (near-floor TSR). n=5 routing only. Composed config (K=8/ramp/corridor/
# eef_r=0.12/companions) so the repair actually fires under realistic
# margins. Reads dbnr_tau_mean / dbnr_headroom_mean from the results JSON
# guidance block (commit 519e3b9). Kill DBNR if tau ~= 0 on both cells;
# proceed to full restitution implementation if tau > 0 and headroom > 0.

set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/fgd_dbnr_killtest"
PORT=${PORT:-8132}

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
  --task_suite_name safelibero_spatial --safety_level I \
  --num_trials_per_task 5 --task_indices 1 \
  --host 127.0.0.1 --port "$PORT" --corridor \
  --results_output_dir "$OUT/results"

python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_long --safety_level II \
  --num_trials_per_task 5 --task_indices 2 \
  --host 127.0.0.1 --port "$PORT" --corridor \
  --results_output_dir "$OUT/results"

echo "DBNR_KILLTEST_SMOKE_COMPLETE"
