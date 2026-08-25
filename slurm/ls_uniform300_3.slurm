#!/bin/bash
#SBATCH --job-name=ls_uniform300_3
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# UNIFORM n=300 rerun, script 3/6 (2026-08-13): single-protocol paper table.
# This script: ARM 3 -- reimpl shield (SOTA-reimpl), all 4 suites, all
# levels. Flags/server mirror slurm/ls_shields_parity.slurm's reimpl arm
# exactly: GT id source, exec parity, shield server (num_candidates 1,
# no-use-companions, eef_radius 0.10, translation_scale 2.0, cmd_clip
# 0.025), repair_schedule=uniform (this is what ls_shields_parity.slurm
# calls the "reimpl" arm -- start_server uniform / run_levels reimpl).
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_uniform300/reimpl"
PORT=8427
N=${N:-20}
CPP="$LS:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
SERVER_PID=""

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule uniform --num_candidates 1 \
  --eef_radius 0.10 --no-use-companions \
  --translation_scale 2.0 --cmd_clip 0.025 \
  --policy.config=pi05_libero --policy.dir="$LS_WEIGHTS" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 240); do
  python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
  sleep 5
done
kill -0 "$SERVER_PID"

conda activate openvla_libero_merged
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$LS/.libero_config"
export PYTHONPATH="$CPP"
for SUITE in obstacle_avoidance human_safety obstacle_avoidance_human affordance; do
  for LVL in 0 1 2; do
    python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
      --suite "$SUITE" --level "$LVL" --num_trials_per_task "$N" \
      --host 127.0.0.1 --port "$PORT" --obstacle_id_source gt \
      --exec_parity \
      --results_output_dir "$OUT"
  done
done
echo "LS_UNIFORM300_3_COMPLETE"
