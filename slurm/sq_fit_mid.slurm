#!/bin/bash
#SBATCH --job-name=sq_fit_mid
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=10:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Revision STAGE 2 (Stage-1 verdicts 2026-07-27): B_dual survived with
# +25 TSR / +5 CAR on the spatial dev cells at K=8 (beat the K16 static
# compute control's +5/0). Escalate to Spatial L1 FULL (t0-t3) + Object t1
# at n=20/task, paired vs a same-job production control. Arms:
#   B_dual       — adaptive eta as smoked
#   B_dual_gate  — + corridor-gated lambda (eta=0 within 0.12 m of target;
#                  the Object-t1 fix — eta must not fight the grasp)
#   B_dual_AR    — + adaptive replan cadence (the other survivor; additivity)
#   CTRL_prod    — production composed repair, same seeds (reference)
# Survive to Stage 3 (n=50 board): beats CTRL on BOTH axes, McNemar p<0.1.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/sq_fit_mid"
PORT=${PORT:-8252}
N=${N:-20}

run_clients () {  # $1 = results subdir, $2... = extra client args
  local sub="$1"; shift
  conda activate openvla_libero_merged
  export MUJOCO_GL=egl
  export PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
  python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
    --task_suite_name safelibero_spatial --safety_level I \
    --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --corridor --log_trajectories "$@" \
    --results_output_dir "$OUT/$sub"
  python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
    --task_suite_name safelibero_object --safety_level I \
    --task_indices 1 --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --corridor --log_trajectories "$@" \
    --results_output_dir "$OUT/$sub"
}

start_server () {  # $@ = extra server args
  conda activate aegis
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
  export PYTHONPATH="$WT/openpi_guided/src"
  python "$WT/openpi_guided/scripts/serve_policy_guided.py" \
    --port "$PORT" --policy.config=pi05_libero \
    --policy.dir="$AEGIS/checkpoints/pi05_libero" \
    --num_candidates 8 --eef_radius 0.09 "$@" &
  SERVER_PID=$!
  for _ in $(seq 1 300); do
    python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
    sleep 5
  done
  kill -0 "$SERVER_PID"
}
stop_server () { kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true; sleep 10; }
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# SUPERQUADRIC CERTIFICATE-FIX RE-SMOKE (root cause 2026-07-28): corridor
# exemption never reached the shape term in the ACCEPTANCE certificate (only
# the repair sweep) — grasp chunks certificate-vetoed on t1 (handle-inflated
# AABB puts the grasp target 8-15mm inside the unrelaxed keep-out; verified
# offline vs the moka pot collision mesh). Reference pre-fix (42431256):
# t1 0.10/0.40, t2 1.00/0.60. Hypothesis: t1 rescued, t2 holds.
conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="$WT/openpi_guided/src"
python "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --policy.config=pi05_libero \
  --policy.dir="$AEGIS/checkpoints/pi05_libero" \
  --num_candidates 8 --eef_radius 0.09 --repair_schedule ramp --sq_eps 0.4 &
SERVER_PID=$!
for _ in $(seq 1 300); do
  python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
  sleep 5
done
kill -0 "$SERVER_PID"
conda activate openvla_libero_merged
export MUJOCO_GL=egl
export PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_spatial --safety_level I \
  --task_indices 1 2 --num_trials_per_task 10 \
  --host 127.0.0.1 --port "$PORT" --corridor --obstacle_shape superquadric --sq_fit mid \
  --results_output_dir "$OUT/sq_certfix"
echo "SQ_CERT_FIX_COMPLETE"
