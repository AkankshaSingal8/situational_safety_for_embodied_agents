#!/bin/bash
#SBATCH --job-name=fgd_vlmf109
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=10:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# VLM-general no-GT finals: Goal L2 (composed r0.10) + Object L1 (composed-thin
# r0.09), full Tier-Percep + VLM priors. Board refs (table): 74.0/68.0, 56.5/80.5.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/fgd_vlmfin"
PRIORS="$WT/results_tables/vlm_hazard_priors.json"
test -s "$PRIORS"
CPP="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

run_server () {  # port radius
  conda activate aegis
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
  PYTHONPATH="$WT/openpi_guided/src" python \
    "$WT/openpi_guided/scripts/serve_policy_guided.py" \
    --port "$1" --repair_schedule ramp --num_candidates 8 --eef_radius "$2" \
    --policy.config=pi05_libero --policy.dir="$AEGIS/checkpoints/pi05_libero" &
  SERVER_PID=$!
  for _ in $(seq 1 240); do
    python -c "import socket; s=socket.create_connection(('127.0.0.1',$1),1); s.close()" 2>/dev/null && break
    sleep 5
  done
  kill -0 "$SERVER_PID"
}
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
run_cell () {  # port suite level
  conda activate openvla_libero_merged
  export MUJOCO_GL=egl
  PYTHONPATH="$CPP" python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
    --task_suite_name "$2" --safety_level "$3" \
    --num_trials_per_task 50 --host 127.0.0.1 --port "$1" --corridor \
    --obstacle_pos_source percep --obstacle_id_source symbolic \
    --entity_pos_source percep \
    --hazard_prior_source vlm --vlm_prior_json "$PRIORS" \
    --results_output_dir "$OUT/$2_$3/results"
}

run_server 8191 0.10
run_cell 8191 safelibero_goal II
kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true

run_server 8192 0.09
run_cell 8192 safelibero_object I
echo "VLMFIN_R10_09_COMPLETE"
