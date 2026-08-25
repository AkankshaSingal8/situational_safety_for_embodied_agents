#!/bin/bash
#SBATCH --job-name=ls_uniform300_5
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --time=09:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# UNIFORM n=300 rerun, script 5/6 (2026-08-13): single-protocol paper table.
# This script: ARM 5 -- ours-noGT, oa + oah only (all levels). noGT is the
# slowest arm (~60 min/suite-level, 12 cells total across 4 suites -> ~12h,
# over the 11h budget), so it is split into two scripts by whole suite:
# this one (oa, oah) and script 6 (hs, aff). Config is
# slurm/ls_fol_ablate_1.slurm's common routing WITHOUT --fol_ablate
# (incumbent, non-ablated cell): oa = fol@eng0.30 (no cone), oah =
# fol@eng0.30+cone60. Common flags: --entity_pos_source percep
# --duality_disengage --guard_topk 2 --exec_parity.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_uniform300/nogt"
PORT=8429
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
NOGT="--entity_pos_source percep --duality_disengage --guard_topk 2 --exec_parity"

# oa: fol, eng0.30, no cone
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --obstacle_id_source fol $NOGT \
    --engage_radius 0.30 \
    --results_output_dir "$OUT"
done
# oah: fol, eng0.30, cone60
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance_human --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --obstacle_id_source fol $NOGT \
    --engage_radius 0.30 --engage_cone 60 \
    --results_output_dir "$OUT"
done
echo "LS_UNIFORM300_5_COMPLETE"
