#!/bin/bash
#SBATCH --job-name=tilt_spatial_n50
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=12:00:00
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
OUT="$WT/tilt_spatial_n50"
PORT=${PORT:-8226}
N=${N:-50}

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

# DERIVED-OPERATOR DECISIVE EVAL (user rule 2026-07-28): barrier-gradient
# tilt guidance (constrained-sampling derivation) + composed repair, Spatial
# L1 FULL n=50/task (n=200). Acceptance: beats baseline 67.0/14.0 AND
# SOTA-reimpl 62.0/28.5 on BOTH axes -> becomes the paper's method.
#   CD_tilt      — ramp repair + dual-controller lambda on the DERIVED operator
#   CD_tilt_gate — same + 0.12 m client gate (insurance; corridor exemption
#                  is already inside the danger function)
start_server --repair_schedule ramp
run_clients CD_tilt      --dual_eta --dual_tilt
run_clients CD_tilt_gate --dual_eta --dual_tilt --dual_gate_dist 0.12
echo "TILT_SPATIAL_N50_COMPLETE"
