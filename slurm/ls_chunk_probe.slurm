#!/bin/bash
#SBATCH --job-name=ls_chunk_probe
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# LS finetune-gap forensics (2026-07-27): all input conventions verified
# against the released training parquet (metric deltas, 20 Hz, 8-dim state,
# camera, prompts, init states, settle wait). Demos finish the L0 tasks in
# ~200 steps; our rollouts wander 600 and time out. Remaining suspect:
# CHUNK EXECUTION CADENCE — cutting the 10-step flow chunk at 5 steps may
# cause dithering. Probe: baseline (no guidance) obstacle_avoidance L0+L1,
# n=10, replan_steps 10 vs 5 (same-day control), plus videos on rs10.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_chunk_probe"
PORT=${PORT:-8221}
N=${N:-10}

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="$WT/openpi_guided/src"
python "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule ramp --num_candidates 1 \
  --translation_scale 1.0 --cmd_clip 0.05 \
  --policy.config=pi05_libero \
  --policy.dir="$LS_WEIGHTS" &
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
export PYTHONPATH="$LS:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
for LVL in 0 1; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --disable_guidance \
    --replan_steps 10 --save_videos \
    --results_output_dir "$OUT/rs10"
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --disable_guidance \
    --replan_steps 5 \
    --results_output_dir "$OUT/rs5"
done
echo "LS_CHUNK_PROBE_COMPLETE"
