#!/bin/bash
#SBATCH --job-name=ls_shld_par
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=16:00:00
#SBATCH --array=0-3
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.err
# Shield arms RERUN with execution parity (--exec_parity) for the fair Table 6:
# the 2026-08-01 forensics showed the shared client ran ALL filtered arms
# 2.2-2.5x slower than the authors' harness (stock-vs-vendored robosuite OSC
# gains). Rerunning posthoc + reimpl under parity keeps the filtered-vs-
# baseline comparison honest in BOTH directions. Server actuation model
# retuned for the vendored controller (~2x motion/unit action):
# --translation_scale 2.0 --cmd_clip 0.025. Everything else mirrors
# slurm/ls_shields.slurm exactly.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_shields_parity"
PORT=$((8380 + SLURM_ARRAY_TASK_ID))
SUITES=(obstacle_avoidance human_safety obstacle_avoidance_human affordance)
SUITE=${SUITES[$SLURM_ARRAY_TASK_ID]}
N=${N:-10}
CPP="$LS:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

start_server () {  # $1 = repair schedule (posthoc|uniform)
  conda activate aegis
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
  PYTHONPATH="$WT/openpi_guided/src" python \
    "$WT/openpi_guided/scripts/serve_policy_guided.py" \
    --port "$PORT" --repair_schedule "$1" --num_candidates 1 \
    --eef_radius 0.10 --no-use-companions \
    --translation_scale 2.0 --cmd_clip 0.025 \
    --policy.config=pi05_libero --policy.dir="$LS_WEIGHTS" &
  SERVER_PID=$!
  for _ in $(seq 1 240); do
    python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
    sleep 5
  done
  kill -0 "$SERVER_PID"
}
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

run_levels () {  # $1 = arm name
  conda activate openvla_libero_merged
  export MUJOCO_GL=egl
  export LIBERO_CONFIG_PATH="$LS/.libero_config"
  export PYTHONPATH="$CPP"
  for LVL in 0 1 2; do
    python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
      --suite "$SUITE" --level "$LVL" --num_trials_per_task "$N" \
      --host 127.0.0.1 --port "$PORT" --obstacle_id_source gt \
      --exec_parity \
      --results_output_dir "$OUT/$1"
  done
}

start_server posthoc
run_levels posthoc
kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true

start_server uniform
run_levels reimpl
echo "LS_SHLD_PAR_${SUITE}_COMPLETE"
