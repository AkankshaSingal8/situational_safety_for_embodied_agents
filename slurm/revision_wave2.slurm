#!/bin/bash
#SBATCH --job-name=revision_wave2
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Revision wave 2 (compute-scaling arms, spec 2026-07-27): same dev cells as
# revision_smoke (Spatial L1 t0+t1, Object L1 t1, n=10, seed-paired).
#   K16_A1C — candidate scaling 8->16 with progress/lookahead selection
#   AR      — adaptive replanning (2-step cadence inside 0.12 m clearance)
# K=32 gated on K16 showing a monotone gain (dose-response first).
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/revision_smoke"
PORT=${PORT:-8220}
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
    --repair_schedule ramp --eef_radius 0.09 \
    --progress_weight 5.0 --lookahead_weight 3.0 "$@" &
  SERVER_PID=$!
  for _ in $(seq 1 300); do
    python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
    sleep 5
  done
  kill -0 "$SERVER_PID"
}

trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

start_server --num_candidates 16
run_clients K16_A1C
kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true; sleep 10
start_server --num_candidates 8
run_clients AR --adaptive_replan
echo "REVISION_WAVE2_COMPLETE"
