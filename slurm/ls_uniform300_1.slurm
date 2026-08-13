#!/bin/bash
#SBATCH --job-name=ls_uniform300_1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# UNIFORM n=300 rerun, script 1/6 (2026-08-13): single-protocol paper table
# (all 5 method arms x 4 suites x L0-2 x N=20/task = 300/cell). This script:
# ARM 1 -- baseline (unfiltered pi0.5-safety), all 4 suites, all levels.
# NOTE: the task brief pointed at slurm/ls_base_rows.slurm for this arm, but
# that script (a) uses the OFF-THE-SHELF pi0.5 checkpoint
# (vlsa-aegis/checkpoints/pi05_libero, not the safety-finetuned
# LIBERO-Safety checkpoint every other main-table arm uses) and (b) omits
# --exec_parity. Both would break the "single protocol" premise of this
# table (ls_ours_parity_full.slurm's own comment: exec parity lifts the
# unfiltered baseline itself, e.g. oa 68/78/80 vs non-parity 42/38/40).
# Resolved: this script instead mirrors ls_ours_parity_full.slurm's
# "base_par" arm -- safety-finetuned checkpoint, --disable_guidance
# --exec_parity, run across all 4 suites. See task-4-uniform300-report.md.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_uniform300/base"
PORT=8425
N=${N:-20}
CPP="$LS:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
SERVER_PID=""

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule ramp --num_candidates 8 --eef_radius 0.09 \
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
      --host 127.0.0.1 --port "$PORT" \
      --disable_guidance --exec_parity \
      --results_output_dir "$OUT"
  done
done
echo "LS_UNIFORM300_1_COMPLETE"
