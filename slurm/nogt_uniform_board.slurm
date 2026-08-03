#!/bin/bash
#SBATCH --job-name=nogt_uniform
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=10:00:00
#SBATCH --array=0-7
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.err
# no-GT UNIFORM-STACK board rerun (8 cells, n=50/task = 200 eps/cell) at HEAD.
# Perception stack is UNIFORM across all cells: percep obstacle pos + symbolic
# id (identity recalibration in code), percep entities fused over
# agentview+birdview (2-view), VLM hazard priors (haiku anchored, floor 0.5).
# Each cell keeps its PROMOTED enforcement config (verified provenance):
#   0 Spatial L1 : uniform K1 r0.09, NO corridor  (fgd_nogt_spatial_vlm, 70.0/46.5)
#   1 Spatial L2 : ramp K8 r0.12, corridor        (fgd_vlmfin_r12,        80.5/57.5)
#   2 Goal    L1 : ramp K8 r0.12, corridor        (fgd_vlmfin_r12,        73.5/60.5)
#   3 Goal    L2 : ramp K8 r0.10, corridor        (fgd_goalII_calib,      73.5/59.5)
#   4 Object  L1 : ramp K8 r0.09, corridor        (f3_board_n50/L1,       47.0/58.5)
#   5 Object  L2 : ramp K8 r0.09, corridor        (f3_board_n50/L2,       80.0/66.0)
#   6 Long    L1 : ramp K8 r0.09, corridor        (fgd_vlmfin_long,       47.5/56.5)
#   7 Long    L2 : ramp K8 r0.09, corridor        (fgd_nogt_long,         44.0/79.5)
# Core barrier params pinned against server-default drift: gamma 0.9,
# d_safe 0.01, inflation_slope 0.004, companions ON (HEAD default).
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
PRIORS="$WT/results_tables/vlm_hazard_priors.json"
test -s "$PRIORS"
ID=${SLURM_ARRAY_TASK_ID:?}
PORT=$((8390 + ID))
N=${N:-50}
CPP="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

case "$ID" in
  0) SUITE=safelibero_spatial; LEVEL=I;  LTAG=L1
     SERVER_ARGS="--repair_schedule uniform --num_candidates 1 --eef_radius 0.09"
     CORRIDOR="" ;;
  1) SUITE=safelibero_spatial; LEVEL=II; LTAG=L2
     SERVER_ARGS="--repair_schedule ramp --num_candidates 8 --eef_radius 0.12"
     CORRIDOR="--corridor" ;;
  2) SUITE=safelibero_goal;    LEVEL=I;  LTAG=L1
     SERVER_ARGS="--repair_schedule ramp --num_candidates 8 --eef_radius 0.12"
     CORRIDOR="--corridor" ;;
  3) SUITE=safelibero_goal;    LEVEL=II; LTAG=L2
     SERVER_ARGS="--repair_schedule ramp --num_candidates 8 --eef_radius 0.10"
     CORRIDOR="--corridor" ;;
  4) SUITE=safelibero_object;  LEVEL=I;  LTAG=L1
     SERVER_ARGS="--repair_schedule ramp --num_candidates 8 --eef_radius 0.09"
     CORRIDOR="--corridor" ;;
  5) SUITE=safelibero_object;  LEVEL=II; LTAG=L2
     SERVER_ARGS="--repair_schedule ramp --num_candidates 8 --eef_radius 0.09"
     CORRIDOR="--corridor" ;;
  6) SUITE=safelibero_long;    LEVEL=I;  LTAG=L1
     SERVER_ARGS="--repair_schedule ramp --num_candidates 8 --eef_radius 0.09"
     CORRIDOR="--corridor" ;;
  7) SUITE=safelibero_long;    LEVEL=II; LTAG=L2
     SERVER_ARGS="--repair_schedule ramp --num_candidates 8 --eef_radius 0.09"
     CORRIDOR="--corridor" ;;
  *) echo "bad array id $ID" >&2; exit 1 ;;
esac
OUT="$WT/nogt_uniform_board/${SUITE}_${LTAG}"

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
# shellcheck disable=SC2086
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" $SERVER_ARGS \
  --gamma 0.9 --d_safe 0.01 --inflation_slope 0.004 \
  --policy.config=pi05_libero --policy.dir="$AEGIS/checkpoints/pi05_libero" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 240); do
  python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
  sleep 5
done
kill -0 "$SERVER_PID"

conda activate openvla_libero_merged
export MUJOCO_GL=egl
# shellcheck disable=SC2086
PYTHONPATH="$CPP" python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name "$SUITE" --safety_level "$LEVEL" \
  --num_trials_per_task "$N" --host 127.0.0.1 --port "$PORT" $CORRIDOR \
  --obstacle_pos_source percep --obstacle_id_source symbolic \
  --entity_pos_source percep --entity_cameras agentview,birdview \
  --hazard_prior_source vlm --vlm_prior_json "$PRIORS" \
  --results_output_dir "$OUT"
echo "NOGT_UNIFORM_${SLURM_ARRAY_TASK_ID}_COMPLETE"
