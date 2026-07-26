#!/bin/bash
#SBATCH --job-name=ls_rows
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w005,w006,w007,w008,w009
#SBATCH --time=16:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# LIBERO-Safety first method rows: baseline / guided-GT / guided-symbolic arms
# on obstacle_avoidance + human_safety, plus refusal arm on reasoning_safety.
# Suite/level/trials overridable: --export=ALL,SUITES=...,LEVELS=...,N=...
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_rows"
PORT=${PORT:-8159}
SUITES=${SUITES:-"obstacle_avoidance human_safety"}
LEVELS=${LEVELS:-"0 1 2"}
N=${N:-10}
ARMS=${ARMS:-"baseline guided_gt guided_symbolic"}
REFUSAL=${REFUSAL:-1}

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
export LIBERO_CONFIG_PATH="$LS/.libero_config"  # DIR containing config.yaml (smoke finding)
export PYTHONPATH="$LS:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
for SUITE in $SUITES; do
  for LVL in $LEVELS; do
    for ARM in $ARMS; do
      case "$ARM" in
        baseline)        EXTRA="--disable_guidance";;
        guided_gt)       EXTRA="--obstacle_id_source gt";;
        guided_symbolic) EXTRA="--obstacle_id_source symbolic";;
        guided_fol)      EXTRA="--obstacle_id_source fol";;
        guided_fol_dual) EXTRA="--obstacle_id_source fol --duality_disengage";;
        guided_nogt_fol) EXTRA="--obstacle_id_source fol --entity_pos_source percep";;
        guided_nogt_sym) EXTRA="--obstacle_id_source symbolic --entity_pos_source percep";;
        prompt_baseline) EXTRA="--disable_guidance --obstacle_id_source fol --safety_prompt";;
        prompt_guided)   EXTRA="--obstacle_id_source fol --safety_prompt";;
      esac
      python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
        --suite "$SUITE" --level "$LVL" --num_trials_per_task "$N" \
        --host 127.0.0.1 --port "$PORT" $EXTRA \
        --results_output_dir "$OUT/$ARM"
    done
  done
done
if [ "$REFUSAL" = "1" ]; then
  for LVL in $LEVELS; do
    python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
      --suite reasoning_safety --level "$LVL" --num_trials_per_task "$N" \
      --host 127.0.0.1 --port "$PORT" --disable_guidance \
      --results_output_dir "$OUT/refusal_baseline"
    python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
      --suite reasoning_safety --level "$LVL" --num_trials_per_task "$N" \
      --host 127.0.0.1 --port "$PORT" --refuse_unsafe \
      --results_output_dir "$OUT/refusal"
  done
fi
echo "LS_ROWS_COMPLETE"
