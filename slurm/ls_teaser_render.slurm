#!/bin/bash
#SBATCH --job-name=ls_teaser
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Paper Fig 1 (teaser) re-render: paired episodes where base violated and
# ours-noGT succeeded clean (paper/figs/teaser_episode_candidates.json):
#   oa  L2 t11 ep1/ep4  (base clips bottle_of_shampoo__2_1)
#   oah L2 t14 ep7      (base violates bottle_of_alcohol_1_with_hand_1)
# Configs copied verbatim from the board incumbents (ls_ours_parity_full /
# ls_cone_confirm F3); only --task_filter + --save_videos added. Episode
# indices are deterministic per (suite, level, task, ep), so re-running
# eps 0..N-1 reproduces the flagged episodes.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_teaser_render"
PORT=8399
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
NOGT="--obstacle_id_source fol --entity_pos_source percep --duality_disengage --guard_topk 2 --exec_parity"

# oa L2 task 11: eps 0-4 covers ep1 + ep4
python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
  --suite obstacle_avoidance --level 2 --num_trials_per_task 5 \
  --task_filter 11 --save_videos --log_transitions \
  --host 127.0.0.1 --port "$PORT" \
  --disable_guidance --exec_parity \
  --results_output_dir "$OUT/base_oa_L2"
python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
  --suite obstacle_avoidance --level 2 --num_trials_per_task 5 \
  --task_filter 11 --save_videos --log_transitions \
  --host 127.0.0.1 --port "$PORT" $NOGT \
  --engage_radius 0.30 \
  --results_output_dir "$OUT/nogt_oa_L2"
# oah L2 task 14: eps 0-7 covers ep7
python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
  --suite obstacle_avoidance_human --level 2 --num_trials_per_task 8 \
  --task_filter 14 --save_videos --log_transitions \
  --host 127.0.0.1 --port "$PORT" \
  --disable_guidance --exec_parity \
  --results_output_dir "$OUT/base_oah_L2"
python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
  --suite obstacle_avoidance_human --level 2 --num_trials_per_task 8 \
  --task_filter 14 --save_videos --log_transitions \
  --host 127.0.0.1 --port "$PORT" $NOGT \
  --engage_radius 0.30 --engage_cone 60 \
  --results_output_dir "$OUT/nogt_oah_L2"
echo "LS_TEASER_RENDER_COMPLETE"
