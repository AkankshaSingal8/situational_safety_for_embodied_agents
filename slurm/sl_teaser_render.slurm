#!/bin/bash
#SBATCH --job-name=sl_teaser
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# SafeLIBERO teaser render (paper fig, SafeLIBERO analogue of Fig 1):
# Spatial L2, tasks 0+1, eps 0-19, paired init states. Board mining
# (paper/figs/sl_teaser_candidates.json, from baseline_strict vs
# nogt_uniform_board episodes): within eps 0-19, task 0 has 5 and task 1 has
# 6 "hard" pairs (base collides AND times out at 300 steps; ours no-GT clean
# success in ~85-160 steps). Renders are not bit-reproducible, so the final
# pair is selected from THIS render's own logs, as for the LS teaser.
# Configs verbatim from the frozen board: server = nogt_uniform_board cell 1
# (Spatial L2: ramp K8 eef 0.12, gamma 0.9 d_safe 0.01 inflation_slope
# 0.004); ours client = uniform no-GT stack (percep pos + symbolic id +
# percep entities 2-view + VLM priors, corridor); base = --disable_guidance
# (baseline_strict code path). Only --task_indices/--save_videos/
# --log_transitions added.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/sl_teaser_render"
PRIORS="$WT/results_tables/vlm_hazard_priors.json"
test -s "$PRIORS"
PORT=8437
CPP="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule ramp --num_candidates 8 --eef_radius 0.12 \
  --gamma 0.9 --d_safe 0.01 --inflation_slope 0.004 \
  --policy.config=pi05_libero --policy.dir="$AEGIS/checkpoints/pi05_libero" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 240); do
  python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
  sleep 5
done
kill -0 "$SERVER_PID"

conda activate openvla_libero_merged
export MUJOCO_GL=egl
export PYTHONPATH="$CPP"

# base arm: unguided pi0.5 (identical code path, enabled=0)
python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_spatial --safety_level II \
  --num_trials_per_task 20 --task_indices 0 1 \
  --host 127.0.0.1 --port "$PORT" --disable_guidance \
  --save_videos --video_output_dir "$OUT/base/videos" --log_transitions \
  --results_output_dir "$OUT/base"

# ours arm: frozen no-GT board Spatial L2 config
python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_spatial --safety_level II \
  --num_trials_per_task 20 --task_indices 0 1 \
  --host 127.0.0.1 --port "$PORT" --corridor \
  --obstacle_pos_source percep --obstacle_id_source symbolic \
  --entity_pos_source percep --entity_cameras agentview,birdview \
  --hazard_prior_source vlm --vlm_prior_json "$PRIORS" \
  --save_videos --video_output_dir "$OUT/nogt/videos" --log_transitions \
  --results_output_dir "$OUT/nogt"

echo "SL_TEASER_RENDER_COMPLETE"
