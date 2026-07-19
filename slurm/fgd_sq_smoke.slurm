#!/bin/bash
#SBATCH --job-name=fgd_sqsmk
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w005,w006,w007,w008,w009,w010
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Superquadric obstacle-shape smoke (Spatial L1, n=10, GT tier): anisotropic
# barrier from GT AABB half-extents instead of the scalar sphere radius.
# Arm A: ellipsoid (sq_eps 1.0). Arm B: boxy superquadric (sq_eps 0.4).
# Base config = r3 (K=1, uniform, eef_r 0.09, companions, no corridor) whose
# sphere reference is 71.0/23.5 at n=50. Hypothesis: taller barrier kills the
# low-flyover CAR hits, thinner sides preserve TSR.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/fgd_sq_smoke"

run_arm () {  # port eps outdir
  local port=$1 eps=$2 outdir=$3
  conda activate aegis
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
  PYTHONPATH="$WT/openpi_guided/src" python \
    "$WT/openpi_guided/scripts/serve_policy_guided.py" \
    --port "$port" --repair_schedule uniform --num_candidates 1 \
    --eef_radius 0.09 --sq_eps "$eps" \
    --policy.config=pi05_libero --policy.dir="$AEGIS/checkpoints/pi05_libero" &
  SERVER_PID=$!
  for _ in $(seq 1 240); do
    python -c "import socket; s=socket.create_connection(('127.0.0.1',$port),1); s.close()" 2>/dev/null && break
    sleep 5
  done
  kill -0 "$SERVER_PID"
  conda activate openvla_libero_merged
  export MUJOCO_GL=egl
  PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline" \
  python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
    --task_suite_name safelibero_spatial --safety_level I \
    --num_trials_per_task 10 --host 127.0.0.1 --port "$port" \
    --obstacle_shape superquadric \
    --results_output_dir "$outdir/results"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

run_arm 8181 1.0 "$OUT/ellipsoid"
run_arm 8182 0.4 "$OUT/boxy"
echo "SQ_SMOKE_COMPLETE"
