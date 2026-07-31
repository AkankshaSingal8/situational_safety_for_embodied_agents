#!/bin/bash
#SBATCH --job-name=ls_impr_dev
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.err
# Ours-improved DEV arms for Table 6 (finetuned ckpt). Corridor is already
# always-on in the LS runner and GT guards all hazards, so the remaining
# steering-tax levers are: duality disengage (v3: hs 14->40 TSR on this ckpt)
# and --guard_topk 2 for the no-GT identity sources. Dev cells = the tax
# suites (obstacle_avoidance + human_safety), L0-2, n=4/task (n=20/cell),
# paired against ls_full_ft cells. Gate: pooled TSR +5pp, violations ~0.
#   arms: gt_dual      = guided_gt + --duality_disengage
#         nogt_dual_k2 = fol percep + --duality_disengage --guard_topk 2
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_improved_dev"
PORT=$((8330 + SLURM_ARRAY_TASK_ID))
SUITES=(obstacle_avoidance human_safety)
SUITE=${SUITES[$SLURM_ARRAY_TASK_ID]}
N=${N:-4}
CPP="$LS:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule ramp --num_candidates 8 --eef_radius 0.09 \
  --translation_scale 1.0 --cmd_clip 0.05 \
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
    --suite "$SUITE" --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source gt --duality_disengage \
    --results_output_dir "$OUT/gt_dual"
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite "$SUITE" --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source fol --entity_pos_source percep \
    --duality_disengage --guard_topk 2 \
    --results_output_dir "$OUT/nogt_dual_k2"
done
echo "LS_IMPR_DEV_${SUITE}_COMPLETE"
