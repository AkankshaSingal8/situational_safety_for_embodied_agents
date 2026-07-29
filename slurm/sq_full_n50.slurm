#!/bin/bash
#SBATCH --job-name=sq_full_n50
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# SUPERQUADRIC FULL EVAL (systematic-debugging phase 3/4, user request):
# best SQ config — boxy(sq_eps 0.4) + corridor + shape_scale fix — at BOARD
# SCALE (Spatial L1 n=50/task = 200 eps, GT extents), vs the stored sphere
# r3@0.09 row 71.0/23.5 (n=200) on the SAME r3 base server config.
# Hypothesis (pre-registered from n=10/n=40 evidence): SQ lands ~62 TSR /
# ~45 CAR — CAR up, TSR down, driven by t1 (obstacle-on-grasp) smothering;
# t2 holds ~100 TSR. i.e. a frontier point, NOT a dominator.
#   sbatch --export=ALL,N=50 slurm/sq_full_n50.slurm
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/sq_full_n50"
PORT=${PORT:-8262}
N=${N:-50}

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule uniform --num_candidates 1 \
  --eef_radius 0.09 --sq_eps 0.4 \
  --policy.config=pi05_libero --policy.dir="$AEGIS/checkpoints/pi05_libero" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 240); do
  python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
  sleep 5
done
kill -0 "$SERVER_PID"

conda activate openvla_libero_merged
export MUJOCO_GL=egl
PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline" \
python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_spatial --safety_level I \
  --num_trials_per_task "$N" --host 127.0.0.1 --port "$PORT" \
  --obstacle_shape superquadric --sq_fit sym --corridor \
  --results_output_dir "$OUT/boxy_corridor_n50/results"
echo "SQ_FULL_N50_COMPLETE"
