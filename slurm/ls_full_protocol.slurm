#!/bin/bash
#SBATCH --job-name=ls_full
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=16:00:00
#SBATCH --array=0-3
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.err
# LIBERO-Safety FULL benchmark protocol on the authors' FINETUNED checkpoint:
# 4 suites x 3 levels x 5 tasks x 10 trials (authors' pre-recorded init
# states, first 10 of 50) = 600 rollouts per arm. One array task per suite;
# arms guided_gt + guided_nogt_fol (baseline runs in a separate session).
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_full_ft"
PORT=$((8300 + SLURM_ARRAY_TASK_ID))
SUITES=(obstacle_avoidance human_safety obstacle_avoidance_human affordance)
SUITE=${SUITES[$SLURM_ARRAY_TASK_ID]}
ARMS=${ARMS:-"guided_gt guided_nogt_fol"}
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
  for ARM in $ARMS; do
    case "$ARM" in
      guided_gt)       EXTRA="--obstacle_id_source gt";;
      guided_symbolic) EXTRA="--obstacle_id_source symbolic";;
      guided_fol)      EXTRA="--obstacle_id_source fol";;
      guided_nogt_fol) EXTRA="--obstacle_id_source fol --entity_pos_source percep";;
      guided_nogt_sym) EXTRA="--obstacle_id_source symbolic --entity_pos_source percep";;
    esac
    python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
      --suite "$SUITE" --level "$LVL" --num_trials_per_task "$N" \
      --host 127.0.0.1 --port "$PORT" $EXTRA \
      --results_output_dir "$OUT/$ARM"
  done
done
echo "LS_FULL_${SUITE}_COMPLETE"
