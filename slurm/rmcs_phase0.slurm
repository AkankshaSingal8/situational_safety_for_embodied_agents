#!/bin/bash
#SBATCH --job-name=rmcs_phase0
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# RMCS Phase 0: diversity + dither measurement (diagnostics only, ZERO
# behavior change). Production composed config (ramp K8 r0.09 + corridor),
# --log_k_diag records per replan: K candidate prefix displacements,
# per-candidate min-margins, selected index, EEF position, obstacle centers.
# Cells: Long L1 t1/t3, Long L2 t2/t3, Goal L2 t0; n=5 eps/task, seed 7.
# Then rmcs_phase0_analyze.py applies the pre-registered gate
# (diversity_in_stall >= 0.25 AND revisit_rate >= 0.50; INVALID if
# pooled n_stalled < 20).
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/rmcs_phase0"
PORT=${PORT:-8275}
CPP="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule ramp --num_candidates 8 --eef_radius 0.09 \
  --policy.config=pi05_libero --policy.dir="$AEGIS/checkpoints/pi05_libero" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 240); do
  python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
  sleep 5
done
kill -0 "$SERVER_PID"

run_cell () {  # $1 subdir, rest = suite/level/task args
  local sub="$1"; shift
  conda activate openvla_libero_merged
  export MUJOCO_GL=egl
  PYTHONPATH="$CPP" python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
    --num_trials_per_task 5 --seed 7 --host 127.0.0.1 --port "$PORT" \
    --corridor --log_k_diag \
    "$@" --results_output_dir "$OUT/$sub"
}

run_cell long_L1 --task_suite_name safelibero_long --safety_level I --task_indices 1 3
run_cell long_L2 --task_suite_name safelibero_long --safety_level II --task_indices 2 3
run_cell goal_L2 --task_suite_name safelibero_goal --safety_level II --task_indices 0

# results_dir = <results_output_dir>/<suite>/<level>/, so k_diag files land at
# $OUT/<cell>/<suite>/<level>/k_diag_*.jsonl — three wildcard levels.
python "$WT/vlm_pipeline/rmcs_phase0_analyze.py" \
  --diag_glob "$OUT/*/*/*/k_diag_*.jsonl" \
  --out "$OUT/gate.json"
echo "RMCS_PHASE0_COMPLETE"
