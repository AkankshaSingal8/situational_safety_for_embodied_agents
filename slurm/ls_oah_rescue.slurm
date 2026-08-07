#!/bin/bash
#SBATCH --job-name=ls_oahres
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# OAH no-GT residual rescue (cone-confirm forensics 2026-08-07): after cone60,
# remaining failures are 600-step freezes with CORRECT guards and ZERO
# violations -- the hand-held hazard sits inside the engagement core near the
# goal (L0 t4 0/50 vs base 5/10; L1 t9 14/50 vs base 8/10; mild tax t5/t6).
# Baseline wins these cells only by colliding (base t4 3 viol/10). Two arms
# on the confirmed F3 cone config (n=10/task, L0+L1):
#   R1: +holding_disengage (client): drop guards while gripper holds -- cures
#       carry-phase blocking if freezes are post-grasp.
#   R2: server eef_radius 0.09 -> 0.06 (barrier calibration): thread past a
#       goal-adjacent hazard; accept iff viol stays < base 5.3%.
# Promotion rule: fix t4/t9 without regressing t0-t3/t5-t8 or violations.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_oah_rescue"
PORT=8401
N=${N:-10}
CPP="$LS:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

start_server () {  # $1 = eef_radius
  conda activate aegis
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
  PYTHONPATH="$WT/openpi_guided/src" python \
    "$WT/openpi_guided/scripts/serve_policy_guided.py" \
    --port "$PORT" --repair_schedule ramp --num_candidates 8 --eef_radius "$1" \
    --translation_scale 2.0 --cmd_clip 0.025 \
    --policy.config=pi05_libero --policy.dir="$LS_WEIGHTS" &
  SERVER_PID=$!
  for _ in $(seq 1 240); do
    python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
    sleep 5
  done
  kill -0 "$SERVER_PID"
  conda activate openvla_libero_merged
  export MUJOCO_GL=egl
  export LIBERO_CONFIG_PATH="$LS/.libero_config"
  export PYTHONPATH="$CPP"
}
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
NOGT="--obstacle_id_source fol --entity_pos_source percep --duality_disengage --guard_topk 2 --exec_parity"

# R1: confirmed cone config + holding_disengage (server eef_radius 0.09)
start_server 0.09
for LVL in 0 1; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance_human --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" $NOGT \
    --engage_radius 0.30 --engage_cone 60 --holding_disengage \
    --results_output_dir "$OUT/R1_holding"
done
kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true; sleep 10

# R2: confirmed cone config, server eef_radius 0.06
start_server 0.06
for LVL in 0 1; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance_human --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" $NOGT \
    --engage_radius 0.30 --engage_cone 60 \
    --results_output_dir "$OUT/R2_eef006"
done
echo "LS_OAH_RESCUE_COMPLETE"
