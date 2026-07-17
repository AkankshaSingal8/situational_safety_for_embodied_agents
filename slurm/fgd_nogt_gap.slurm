#!/bin/bash
#SBATCH --job-name=fgd_ngtgap
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w005,w006,w007,w008,w009
#SBATCH --time=12:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Gap-cell iteration 2 (n=20), two sequential server configs:
# Arm A — Spatial L1: r3-thin + corridor (K=1, uniform, eef_r 0.09, companions).
#   composed-thin got CAR 35 (clears 28.5) but TSR 60 < 67; r3-thin held TSR 71.
#   Corridor lifts CAR at zero TSR cost (McNemar, Spatial L2) -> should clear both.
# Arm B — Goal L2: composed-thin at eef_r 0.10 (K=8, ramp, corridor).
#   composed-thin@0.09 hit exactly 75.0/48.75 vs reimpl bar 75.0/49.5 — one
#   radius notch for the missing CAR sliver.
set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/fgd_nogt_gap"
PORT_A=${PORT_A:-8166}
PORT_B=${PORT_B:-8167}

run_server () {  # port, extra server args...
  local port=$1; shift
  conda activate aegis
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
  PYTHONPATH="$WT/openpi_guided/src" python \
    "$WT/openpi_guided/scripts/serve_policy_guided.py" \
    --port "$port" "$@" \
    --policy.config=pi05_libero --policy.dir="$AEGIS/checkpoints/pi05_libero" &
  SERVER_PID=$!
  for _ in $(seq 1 240); do
    python -c "import socket; s=socket.create_connection(('127.0.0.1',$port),1); s.close()" 2>/dev/null && break
    sleep 5
  done
  kill -0 "$SERVER_PID"
}
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

CLIENT_PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

# Arm A: Spatial L1, r3-thin + corridor
run_server "$PORT_A" --repair_schedule uniform --num_candidates 1 --eef_radius 0.09
conda activate openvla_libero_merged
export MUJOCO_GL=egl
PYTHONPATH="$CLIENT_PYTHONPATH" python \
  "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_spatial --safety_level I \
  --num_trials_per_task 20 --host 127.0.0.1 --port "$PORT_A" --corridor --obstacle_pos_source percep --obstacle_id_source symbolic \
  --results_output_dir "$OUT/r3thin_corridor/results"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

# Arm B: Goal L2, composed-thin at eef_r 0.10
run_server "$PORT_B" --repair_schedule ramp --num_candidates 8 --eef_radius 0.10
conda activate openvla_libero_merged
export MUJOCO_GL=egl
PYTHONPATH="$CLIENT_PYTHONPATH" python \
  "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_goal --safety_level II \
  --num_trials_per_task 20 --host 127.0.0.1 --port "$PORT_B" --corridor --obstacle_pos_source percep --obstacle_id_source symbolic \
  --results_output_dir "$OUT/composed_r10/results"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

# Arm C: Object L1, composed-thin (K=8, ramp, eef_r 0.09) — GT-tier winner
run_server "${PORT_C:-8168}" --repair_schedule ramp --num_candidates 8 --eef_radius 0.09
conda activate openvla_libero_merged
export MUJOCO_GL=egl
PYTHONPATH="$CLIENT_PYTHONPATH" python \
  "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_object --safety_level I \
  --num_trials_per_task 20 --host 127.0.0.1 --port "${PORT_C:-8168}" --corridor --obstacle_pos_source percep --obstacle_id_source symbolic \
  --results_output_dir "$OUT/composed_thin_obj/results"
echo "NOGT_GAP_COMPLETE"
