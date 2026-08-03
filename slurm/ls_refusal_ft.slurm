#!/bin/bash
#SBATCH --job-name=ls_refusal_ft
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=05:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#
# Table 5 refresh (user directive 2026-08-02): rerun the reasoning_safety
# refusal rows with the SAFETY-FINETUNED pi0.5 (pi05_libero_safety) as base
# policy, replacing the off-the-shelf-checkpoint baseline column so Table 5
# is consistent with the finetuned-only LS story (Tables 4/4b removed).
# Protocol mirrors slurm/ls_base_rows.slurm refusal block exactly
# (--raw_actions, n=10/task x 5 tasks = 50 eps/level), only the checkpoint
# differs. Both arms rerun: baseline (--disable_guidance, executes unsafe
# instructions) and ours (--refuse_unsafe, E4 governor declines pre-motion).
# NOTE: some reasoning_safety L2 bddls are absent from the release; the
# client skips absent tasks (footnote in table if L2 n < 50).
set -euo pipefail
module load anaconda3
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh

ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_refusal_ft"
PORT=8375
CPP="$LS:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule ramp --num_candidates 8 --eef_radius 0.09 \
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

for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite reasoning_safety --level "$LVL" --num_trials_per_task 10 \
    --host 127.0.0.1 --port "$PORT" --raw_actions --disable_guidance \
    --results_output_dir "$OUT/refusal_baseline"
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite reasoning_safety --level "$LVL" --num_trials_per_task 10 \
    --host 127.0.0.1 --port "$PORT" --raw_actions --refuse_unsafe \
    --results_output_dir "$OUT/refusal"
done
echo "LS_REFUSAL_FT_COMPLETE"
