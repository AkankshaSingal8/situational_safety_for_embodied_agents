#!/bin/bash
#SBATCH --job-name=ls_wrist
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Arm F3 (user-directed): per-replan mover tracking from the WRIST camera
# (robot0_eye_in_hand primary, agentview refine-only via consistency tau).
# F closed raw agentview refresh (jitter), F2 closed EMA agentview track
# (lag) — both diagnosed as DETECTOR-VIEW quality on a moving hand. F3
# changes the VIEW: at close range (where all freezes/violations happen)
# the wrist camera sees the hand large and unoccluded. Light EMA (0.6) +
# innovation gate 0.20 keeps outliers out without F2's lag.
# Refs: F3-cone static snapshot L0 53.6/2.8, L1 44.4/0 (n=50/task).
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_oah_wrist"
PORT=8407
N=${N:-10}
CPP="$LS:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

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

for LVL in 0 1; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance_human --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source fol --entity_pos_source percep \
    --duality_disengage --guard_topk 2 --exec_parity \
    --engage_radius 0.30 --engage_cone 60 \
    --percep_refresh_movers \
    --percep_refresh_cameras robot0_eye_in_hand,agentview \
    --entity_view_tau 0.08 \
    --percep_refresh_alpha 0.6 --percep_refresh_gate 0.20 \
    --results_output_dir "$OUT/F3_wrist_track"
done
echo "LS_WRIST_TRACK_COMPLETE"
