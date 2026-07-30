#!/bin/bash
#SBATCH --job-name=sq_ellip_c2
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=05:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# SUPERQUADRIC C2/C3-FIXED DEV-CELL ARM (audit 2026-07-29, fix 2026-07-30):
# ellipsoid keep-out (sq_eps 1.0, was boxy 0.4) with the support-plane
# true-distance barrier (server C2 fix — conservative, no oblique
# under-enforcement) and --sq_fit main (tight AABB of the LARGEST geom part
# only; drops the moka handle the n=200 arm's triply-inflated fit included).
# Offline verifier (vlm_pipeline/verify_sq_barrier.py --form support) already
# confirmed min true clearance = 100.0 mm on the enforced boundary for all
# moka pot / wine bottle fits — enforcement is conservative before rollouts.
# Dev cell: Spatial L1, n=20/task, vs sphere r3@0.09 + sq_full_n50 rows.
#   sbatch slurm/sq_ellip_c2.slurm
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/sq_ellip_c2"
PORT=${PORT:-8276}
N=${N:-20}

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule uniform --num_candidates 1 \
  --eef_radius 0.09 --sq_eps 1.0 \
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
  --obstacle_shape superquadric --sq_fit main --corridor \
  --results_output_dir "$OUT/ellip_main_corridor_n20/results"
echo "SQ_ELLIP_C2_COMPLETE"
