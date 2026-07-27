#!/bin/bash
#SBATCH --job-name=revision_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Runtime steering revision Stage-1 smoke (spec 2026-07-27): dev cells
# Spatial L1 t0+t1 and Object L1 t1, n=10/task, seed-paired with the stored
# production rows (seed 7 => first 10 episodes identical). Arms:
#   A2_stall  — production server, client stall-recovery ladder
#   A1C       — production server + progress/lookahead selection terms
#   A12C     — both of the above
#   B_dual    — repair off, K=8, renoise backstop; client dual-adaptive eta
# Survive rule: no regression on either axis AND >= +5pp on the target cell.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/revision_smoke"
PORT=${PORT:-8218}
N=${N:-10}

run_clients () {  # $1 = results subdir, $2... = extra client args
  local sub="$1"; shift
  conda activate openvla_libero_merged
  export MUJOCO_GL=egl
  export PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
  python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
    --task_suite_name safelibero_spatial --safety_level I \
    --task_indices 0 1 --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --corridor "$@" \
    --results_output_dir "$OUT/$sub"
  python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
    --task_suite_name safelibero_object --safety_level I \
    --task_indices 1 --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --corridor "$@" \
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

# Group 1: production server (composed ramp repair) — A2 client controller.
start_server --repair_schedule ramp
run_clients A2_stall --stall_recovery
stop_server

# Group 2: production server + A1/C selection terms — A1C, then A12C.
start_server --repair_schedule ramp --progress_weight 5.0 --lookahead_weight 3.0
run_clients A1C
run_clients A12C --stall_recovery
stop_server

# Group 3: B — repair off, renoise certificate backstop, client dual eta.
start_server --repair_schedule none --renoise_attempts 2
run_clients B_dual --dual_eta
echo "REVISION_SMOKE_COMPLETE"
