#!/bin/bash
#SBATCH --job-name=ls_ours_par
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=16:00:00
#SBATCH --array=0-3
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.err
# OURS full arms under execution parity for Table 6 (finetuned ckpt), full
# protocol: authors' init states, 4 suites x L0/1/2 x 5 tasks x N=10.
# Config carried from the ls_engage_dev gate winner; override the engagement
# radius at submit time with --export=ALL,ENG=<r> (default 0.30; ENG=0
# disables gating = parity alone).
#   gt_par_full   = GT identity + duality disengage + exec parity + gating
#   nogt_par_full = fol identity + percep + topk2 + dual + parity + gating
# Server = guided board config with parity actuation model
# (--translation_scale 2.0 --cmd_clip 0.025).
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_ours_parity_full"
PORT=$((8390 + SLURM_ARRAY_TASK_ID))
SUITES=(obstacle_avoidance human_safety obstacle_avoidance_human affordance)
SUITE=${SUITES[$SLURM_ARRAY_TASK_ID]}
N=${N:-10}
# Dev gate 42899554 (2026-08-03): all arms passed; gt_par_eng22 best pooled
# (65.0), nogt tested at 0.30 -> per-arm radii. ENG_* override at submit.
ENG_GT=${ENG_GT:-0.22}
ENG_NOGT=${ENG_NOGT:-0.30}
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
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite "$SUITE" --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source gt --duality_disengage --exec_parity \
    --engage_radius "$ENG_GT" \
    --results_output_dir "$OUT/gt_par_full"
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite "$SUITE" --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source fol --entity_pos_source percep \
    --duality_disengage --guard_topk 2 --exec_parity \
    --engage_radius "$ENG_NOGT" \
    --results_output_dir "$OUT/nogt_par_full"
  # Baseline AT PARITY: conseq-data cross-check (2026-08-03) showed exec
  # parity lifts the unfiltered baseline itself (oa 68/78/80 vs authors'
  # harness 42/38/40) -> Table 6 comparator must be self-run at parity.
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite "$SUITE" --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --disable_guidance --exec_parity \
    --results_output_dir "$OUT/base_par"
done
echo "LS_OURS_PAR_${SUITE}_COMPLETE"
