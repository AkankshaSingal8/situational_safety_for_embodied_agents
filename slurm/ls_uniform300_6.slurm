#!/bin/bash
#SBATCH --job-name=ls_uniform300_6
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=09:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# UNIFORM n=300 rerun, script 6/6 (2026-08-13): single-protocol paper table.
# This script: ARM 5 -- ours-noGT, hs + aff only (all levels; see script 5
# for the oa/oah half of this arm and the split rationale). Config is
# slurm/ls_fol_ablate_2.slurm and slurm/ls_fol_ablate_3.slurm's common
# routing WITHOUT --fol_ablate (incumbent, non-ablated cell): hs =
# folprop@eng0.30 (no cone), aff = folpropvlm@eng0.30+cone60. Common flags:
# --entity_pos_source percep --duality_disengage --guard_topk 2
# --exec_parity.
# NOTE (concern, see task-4-uniform300-report.md): slurm/ls_vanilla_tier.slurm's
# V1 aff noGT arm runs folpropvlm@eng0.30 WITHOUT --engage_cone 60, which
# disagrees with ls_fol_ablate_3.slurm. This script follows the task's named
# reference (ls_fol_ablate_*) and keeps cone60 on aff; flagging the
# disagreement rather than resolving it.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_uniform300/nogt"
PORT=8430
N=${N:-20}
CPP="$LS:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
SERVER_PID=""

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

# hs: folprop, eng0.30, no cone
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite human_safety --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --obstacle_id_source folprop $NOGT \
    --engage_radius 0.30 \
    --results_output_dir "$OUT"
done
# aff: folpropvlm, eng0.30, cone60
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite affordance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --obstacle_id_source folpropvlm $NOGT \
    --engage_radius 0.30 --engage_cone 60 \
    --results_output_dir "$OUT"
done
echo "LS_UNIFORM300_6_COMPLETE"
