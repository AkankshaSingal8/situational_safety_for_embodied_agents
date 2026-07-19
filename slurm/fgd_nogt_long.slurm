#!/bin/bash
#SBATCH --job-name=fgd_ngtlong
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w005,w006,w007,w008,w009,w010
#SBATCH --time=16:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# No-GT Long L1+L2 n=50 — the two missing no-GT scoreboard cells.
# Config: composed-thin (K=8, ramp, eef_r 0.09, corridor, companions) — the
# no-GT Object L1 winner. Full Tier-Percep (pos/id/entity all percep).
# Bars: baseline 58.0/15.0 (L1), 51.0/16.5 (L2).
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/fgd_nogt_long"
PORT=8184

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
CPP="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
for LEVEL in I II; do
  PYTHONPATH="$CPP" python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
    --task_suite_name safelibero_long --safety_level "$LEVEL" \
    --num_trials_per_task 50 --host 127.0.0.1 --port "$PORT" --corridor \
    --obstacle_pos_source percep --obstacle_id_source symbolic \
    --entity_pos_source percep \
    --results_output_dir "$OUT/results"
done
echo "NOGT_LONG_COMPLETE"
