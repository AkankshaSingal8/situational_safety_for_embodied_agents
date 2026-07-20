#!/bin/bash
#SBATCH --job-name=fgd_posthoc
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=16:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Post-hoc shield row (AEGIS architecture class, run correctly): unguided
# denoising + one exact projection sweep on the final chunk. Same GT barrier
# params as the SOTA-reimpl referee (r0.10, K=1, no companions, no corridor)
# so the delta vs reimpl isolates post-hoc vs in-denoising ARCHITECTURE.
# Parameterized: --export=ALL,SUITES="safelibero_spatial safelibero_goal",PORT=8210
set -euo pipefail
: "${SUITES:?ERROR: SUITES env var not set}"
: "${PORT:=8210}"
: "${NUM:=50}"
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/fgd_posthoc_row"

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule posthoc --num_candidates 1 \
  --eef_radius 0.10 --no-use-companions \
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
for SUITE in $SUITES; do
  for LEVEL in I II; do
    PYTHONPATH="$CPP" python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
      --task_suite_name "$SUITE" --safety_level "$LEVEL" \
      --num_trials_per_task "$NUM" --host 127.0.0.1 --port "$PORT" \
      --results_output_dir "$OUT/${SUITE}_${LEVEL}/results"
  done
done
echo "POSTHOC_ROW_COMPLETE $SUITES"
