#!/bin/bash
#SBATCH --job-name=f0_obj_attrib
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=10:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# F0: attribute the no-GT Object L1 TSR regression (41.5 vs GT 53.5) to a
# perception component. Board no-GT config (vlmfin: composed r0.09 K8 ramp +
# corridor + percep pos/entity + symbolic id + VLM priors), one component
# swapped to GT per arm, n=20/task = 80 eps, seed-paired (seed 7):
#   full_percep — control (expect ~41.5-level)
#   gt_obspos   — GT obstacle position, rest percep (localization error)
#   gt_entity   — GT entity/corridor anchors, rest percep (corridor placement)
#   gt_id       — GT identity + table priors, positions percep (mis-ranking)
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/f0_obj_attrib"
PRIORS="$WT/results_tables/vlm_hazard_priors.json"
test -s "$PRIORS"
PORT=${PORT:-8272}
N=${N:-20}
CPP="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule ramp --num_candidates 8 --eef_radius 0.09 \
  --policy.config=pi05_libero --policy.dir="$AEGIS/checkpoints/pi05_libero" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 240); do
  python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
  sleep 5
done
kill -0 "$SERVER_PID"

run_arm () {  # $1 subdir, rest = overrides
  local sub="$1"; shift
  conda activate openvla_libero_merged
  export MUJOCO_GL=egl
  PYTHONPATH="$CPP" python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
    --task_suite_name safelibero_object --safety_level I \
    --num_trials_per_task "$N" --host 127.0.0.1 --port "$PORT" --corridor \
    --obstacle_pos_source percep --obstacle_id_source symbolic \
    --entity_pos_source percep \
    --hazard_prior_source vlm --vlm_prior_json "$PRIORS" \
    "$@" --results_output_dir "$OUT/$sub"
}

run_arm full_percep
run_arm gt_obspos --obstacle_pos_source gt
run_arm gt_entity --entity_pos_source gt
run_arm gt_id --obstacle_id_source gt --hazard_prior_source table
echo "F0_OBJ_ATTRIB_COMPLETE"
