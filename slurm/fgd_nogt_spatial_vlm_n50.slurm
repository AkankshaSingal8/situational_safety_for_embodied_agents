#!/bin/bash
#SBATCH --job-name=fgd_ngtsvlm
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# No-GT Spatial L1 n=50 with VLM priors (haiku anchored + floor 0.5) — n=50
# confirm of the prior-swap smoke (vlm arm 72.5/37.5 vs table 60.0/42.5 at
# n=40). If it holds, this closes the last open no-GT cell AND does it with
# the fully general identification stack. Server = r3 (K1/uniform/0.09).
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/fgd_nogt_spatial_vlm"
PORT=8186
PRIORS="$WT/results_tables/vlm_hazard_priors.json"
test -s "$PRIORS"

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule uniform --num_candidates 1 --eef_radius 0.09 \
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
CPP="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
PYTHONPATH="$CPP" python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_spatial --safety_level I \
  --num_trials_per_task 50 --host 127.0.0.1 --port "$PORT" \
  --obstacle_pos_source percep --obstacle_id_source symbolic \
  --entity_pos_source percep \
  --hazard_prior_source vlm --vlm_prior_json "$PRIORS" \
  --results_output_dir "$OUT/results"
echo "NOGT_SPATIAL_VLM_N50_COMPLETE"
