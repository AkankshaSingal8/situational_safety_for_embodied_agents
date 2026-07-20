#!/bin/bash
#SBATCH --job-name=fgd_aegis
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --exclude=w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=16:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# AEGIS self-run baseline row: post-hoc CBF-QP shield (vlsa-aegis main_aegis.py,
# GroundingDINO perception + cvxpy QP) on plain pi0.5, SafeLIBERO n=50.
# Same >1mm displacement collision criterion as every other table row.
# Parameterized: --export=ALL,SUITES="safelibero_spatial safelibero_goal",PORT=8200
set -euo pipefail
: "${SUITES:?ERROR: SUITES env var not set}"
: "${PORT:=8200}"
: "${NUM:=50}"
: "${LEVELS:=I II}"
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="${OUTDIR:-$WT/fgd_aegis_selfrun}"
mkdir -p "$OUT"

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$AEGIS/openpi/src" python "$AEGIS/openpi/scripts/serve_policy.py" \
  --port "$PORT" policy:checkpoint \
  --policy.config=pi05_libero --policy.dir="$AEGIS/checkpoints/pi05_libero" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 240); do
  python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
  sleep 5
done
kill -0 "$SERVER_PID"

conda activate libero
export MUJOCO_GL=egl
export LD_PRELOAD=/ocean/projects/cis250185p/asingal/envs/libero/lib/libstdc++.so.6
export PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$AEGIS/main:${PYTHONPATH:-}"
cd "$AEGIS"  # GroundingDINO config/weights are repo-relative

for SUITE in $SUITES; do
  for LEVEL in $LEVELS; do
    python "$AEGIS/main/main_aegis.py" \
      --task_suite_name "$SUITE" --safety_level "$LEVEL" \
      --num_trials_per_task "$NUM" --host 127.0.0.1 --port "$PORT" \
      --results_output_dir "$OUT" \
      --video_out_path "$OUT/videos"
  done
done
echo "AEGIS_SELFRUN_COMPLETE $SUITES"
