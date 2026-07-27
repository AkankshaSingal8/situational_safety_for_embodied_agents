#!/bin/bash
#SBATCH --job-name=ls_revision_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Revision Stage-1 smoke on the LS human_safety dev suite (spec 2026-07-27):
# the cell where composed repair crushes TSR to 0-26 vs baseline 54/56/26
# with safe-rate already 100 — pure steering tax. Arms (guided_gt id, n=10):
#   ssm        — A3 speed-and-separation dynamic margins only
#   ssm_stall  — A3 + A2 stall-recovery ladder
# Reference rows: stored baseline + guided_gt (seed-paired first 10 eps).
# Survive rule: TSR toward baseline (>= +10pp vs guided_gt) with safe-rate
# within 2pp of 100.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_revision_smoke"
PORT=${PORT:-8219}
N=${N:-10}

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="$WT/openpi_guided/src"
python "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule ramp --num_candidates 8 --eef_radius 0.09 \
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
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite human_safety --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --obstacle_id_source gt \
    --ssm_margins \
    --results_output_dir "$OUT/ssm"
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite human_safety --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --obstacle_id_source gt \
    --ssm_margins --stall_recovery \
    --results_output_dir "$OUT/ssm_stall"
done
echo "LS_REVISION_SMOKE_COMPLETE"
