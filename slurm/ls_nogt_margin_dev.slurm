#!/bin/bash
#SBATCH --job-name=ls_margin
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# No-GT margin push (user directive 2026-08-07: no-GT must beat baseline+SOTA
# on SR by a strong margin). Deficit decomposition (ledger 2026-08-04/06/07):
#   oa:  t7-class paralysis (guard engaged in work zone @eng0.30; eng0.22
#        fails only via 1-view percep noise -- gt@0.22 9/10 vs nogt 2/10).
#   aff: folprop guards property-positive objects the eef never approaches
#        (L0 60 vs base 76, L2 30 vs 46) -- same paralysis regime the cone
#        cured on OAH (+16.7 noGT pooled, 43073780).
# Arms (n=10/task, promotion rule: fix target cells w/o regressing siblings):
#   A oa  L0-2 nogt fol      eng0.30 + cone60            (paralysis rescue)
#   B oa  L0-2 nogt fol      eng0.22 + 2-view entities   (percep-noise fix)
#   C aff L0-2 nogt folprop  eng0.30 + cone60            (paralysis rescue)
# Same server/protocol as ls_ours_parity_full campaign.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_nogt_margin_dev"
PORT=8399
N=${N:-10}
CONE=${CONE:-60}
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
NOGT="--entity_pos_source percep --duality_disengage --guard_topk 2 --exec_parity"

# A: oa nogt fol + cone60 @ eng0.30
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --obstacle_id_source fol $NOGT \
    --engage_radius 0.30 --engage_cone "$CONE" \
    --results_output_dir "$OUT/A_oa_nogt_cone"
done
# B: oa nogt fol + 2-view entities @ eng0.22
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --obstacle_id_source fol $NOGT \
    --engage_radius 0.22 --entity_cameras agentview,birdview \
    --results_output_dir "$OUT/B_oa_nogt_2view"
done
# C: aff nogt folprop + cone60 @ eng0.30
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite affordance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --obstacle_id_source folprop $NOGT \
    --engage_radius 0.30 --engage_cone "$CONE" \
    --results_output_dir "$OUT/C_aff_folprop_cone"
done
echo "LS_NOGT_MARGIN_DEV_COMPLETE"
