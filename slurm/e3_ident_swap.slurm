#!/bin/bash
#SBATCH --job-name=e3_ident_swap
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=10:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# E3 identification-swap (the core thesis experiment): enforcement FIXED
# (composed GT-geometry config), ONLY the identity source varies:
#   gt / symbolic(VLM-prior, floor 0.5) / fol(hard rules, no priors)
# Spatial L1+L2, n=10/task, same --seed => same init states across arms
# (paired design; McNemar on per-episode outcomes).
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/e3_ident_swap"
PORT=${PORT:-8171}
N=${N:-10}
conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="$WT/openpi_guided/src"
python "$WT/openpi_guided/scripts/serve_policy_guided.py" \
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
export PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
for LVL in I II; do
  for ARM in gt symbolic fol; do
    EXTRA=""
    [ "$ARM" = "symbolic" ] && EXTRA="--hazard_prior_source vlm --vlm_prior_json $WT/results_tables/vlm_hazard_priors.json"
    python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
      --task_suite_name safelibero_spatial --safety_level "$LVL" \
      --num_trials_per_task "$N" --host 127.0.0.1 --port "$PORT" --corridor \
      --obstacle_id_source "$ARM" $EXTRA \
      --results_output_dir "$OUT/$ARM"
  done
done
echo "E3_IDENT_SWAP_COMPLETE"
