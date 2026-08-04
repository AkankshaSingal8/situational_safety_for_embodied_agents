#!/bin/bash
#SBATCH --job-name=aff_folprop
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Affordance FOL property-predicate smoke: does folprop (HAZARD := PROTECTED
# | MOVING | PROP & ~MENTIONED & ~TARGET, possibly-EMPTY guard set) fix the
# no-GT affordance dip (fol/sym guarded an on-path non-hazard at L1, -28 TSR
# vs GT arm, 42631100)? Two arms under the EXACT Table-6 nogt campaign
# config (mirrors slurm/ls_ours_parity_full.slurm nogt_par_full: percep
# entities, duality, topk2, exec parity, engage 0.30) so the control arm
# also cross-checks the campaign's affordance cell:
#   ctrl_fol     = --obstacle_id_source fol      (current campaign arm)
#   folprop      = --obstacle_id_source folprop  (property-gated, may guard nothing)
# affordance x L0/1/2 x 5 tasks x N=10 x 2 arms = 300 eps.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_aff_folprop"
PORT=8388
N=${N:-10}
ENG=${ENG:-0.30}
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
    --suite affordance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source fol --entity_pos_source percep \
    --duality_disengage --guard_topk 2 --exec_parity \
    --engage_radius "$ENG" \
    --results_output_dir "$OUT/ctrl_fol"
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite affordance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source folprop --entity_pos_source percep \
    --duality_disengage --guard_topk 2 --exec_parity \
    --engage_radius "$ENG" \
    --results_output_dir "$OUT/folprop"
done
echo "AFF_FOLPROP_SMOKE_COMPLETE"
