#!/bin/bash
#SBATCH --job-name=f3_entity_mv
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=05:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# F3: multi-view entity localization arm. Board no-GT Object L1 config,
# identical to f0_obj_attrib/full_percep EXCEPT entity localization fuses
# agentview+birdview (probe 42844384: med_xy 19->9 mm, p90 49->21 mm on
# corridor entities). Paired control = f0_obj_attrib/full_percep (n=20).
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/f3_entity_mv"
PRIORS="$WT/results_tables/vlm_hazard_priors.json"
test -s "$PRIORS"
PORT=${PORT:-8277}
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

conda activate openvla_libero_merged
export MUJOCO_GL=egl
PYTHONPATH="$CPP" python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_object --safety_level I \
  --num_trials_per_task "$N" --host 127.0.0.1 --port "$PORT" --corridor \
  --obstacle_pos_source percep --obstacle_id_source symbolic \
  --entity_pos_source percep --entity_cameras agentview,birdview \
  --hazard_prior_source vlm --vlm_prior_json "$PRIORS" \
  --results_output_dir "$OUT/mv_2view"
echo "F3_ENTITY_MV_COMPLETE"
