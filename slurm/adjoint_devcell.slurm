#!/bin/bash
#SBATCH --job-name=adjoint_devcell
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=10:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Enforcement candidate A stage 2 (probe was GO): adjoint noise-space arm vs
# composed-repair CONTROL on the pre-registered dev cells, n=20, same seeds
# (paired; identity=GT; certificate identical). Two server phases on one GPU.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/adjoint_devcell"
PORT=${PORT:-8175}
N=${N:-20}

run_clients () {  # $1 = results subdir
  conda activate openvla_libero_merged
  export MUJOCO_GL=egl
  export PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
  python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
    --task_suite_name safelibero_spatial --safety_level I \
    --task_indices 0 1 --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --corridor \
    --results_output_dir "$OUT/$1"
  python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
    --task_suite_name safelibero_object --safety_level I \
    --task_indices 1 --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --corridor \
    --results_output_dir "$OUT/$1"
}

start_server () {  # $@ = extra server args
  conda activate aegis
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
  export PYTHONPATH="$WT/openpi_guided/src"
  python "$WT/openpi_guided/scripts/serve_policy_guided.py" \
    --port "$PORT" --policy.config=pi05_libero \
    --policy.dir="$AEGIS/checkpoints/pi05_libero" "$@" &
  SERVER_PID=$!
  for _ in $(seq 1 300); do
    python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
    sleep 5
  done
  kill -0 "$SERVER_PID"
}

trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# Phase 1: control — composed in-denoising repair (current method)
start_server --repair_schedule ramp --num_candidates 8 --eef_radius 0.09
run_clients control_repair
kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true; sleep 10

# Phase 2: candidate A — adjoint noise-space (repair path replaced)
start_server --num_candidates 8 --eef_radius 0.09 \
  --adjoint_steps 5 --adjoint_lr 0.05 --adjoint_tau 0.02
run_clients adjoint
echo "ADJOINT_DEVCELL_COMPLETE"
