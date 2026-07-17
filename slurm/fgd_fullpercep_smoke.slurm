#!/bin/bash
#SBATCH --job-name=fgd_fpercep
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w005,w006,w007,w008,w009
#SBATCH --time=03:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# FULL Tier-Percep smoke (Spatial L1+L2, n=5): identity=symbolic, obstacle
# pos=percep, AND entity positions (identity candidates + corridor
# target/dest) = percep. The method reads no obs['*_pos'] at all.
# Compare against fgd_nogt arms (entity=gt) to isolate the entity-position
# privilege tax.
set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/fgd_fullpercep_smoke"
PORT=${PORT:-8160}
conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="$WT/openpi_guided/src"
python "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule ramp --num_candidates 8 --eef_radius 0.12 \
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
export PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
for LVL in I II; do
  python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
    --task_suite_name safelibero_spatial --safety_level "$LVL" \
    --num_trials_per_task 5 --host 127.0.0.1 --port "$PORT" --corridor \
    --obstacle_pos_source percep --obstacle_id_source symbolic \
    --entity_pos_source percep \
    --results_output_dir "$OUT/results"
done
echo "FULLPERCEP_SMOKE_COMPLETE"
