#!/bin/bash
#SBATCH --job-name=cvs_tournament
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w008,w009
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err

set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/viability-steering-tournament-20260713"
AEGIS="$ROOT/vlsa-aegis"
MODE=${MODE:-phase_adaptive}
TRIALS=${TRIALS:-2}
K=${K:-4}
PORT=${PORT:-8130}
TRANSLATION_SCALE=${TRANSLATION_SCALE:-0.05}
OUT="$WT/viability_tournament/${MODE}_scale-${TRANSLATION_SCALE}_k${K}"
mkdir -p "$OUT"

echo "mode=$MODE trials=$TRIALS k=$K scale=$TRANSLATION_SCALE out=$OUT"
conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="$WT/openpi_guided/src"
python "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --policy.config=pi05_libero \
  --policy.dir="$AEGIS/checkpoints/pi05_libero" \
  --translation-scale "$TRANSLATION_SCALE" >"$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 240); do
  if python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null; then break; fi
  sleep 5
done
kill -0 "$SERVER_PID"

conda activate openvla_libero_merged
export MUJOCO_GL=egl
export PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
for LEVEL in I II; do
  python -u "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
    --task_suite_name safelibero_spatial --safety_level "$LEVEL" \
    --num_trials_per_task "$TRIALS" --host 127.0.0.1 --port "$PORT" \
    --steering_mode "$MODE" --num_candidates "$K" \
    --results_output_dir "$OUT/results"
done
echo "TOURNAMENT_COMPLETE mode=$MODE scale=$TRANSLATION_SCALE"
