#!/bin/bash
#SBATCH --job-name=aegis_repl_base
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --exclude=w002
#SBATCH --time=10:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# AEGIS-paper replication (arXiv 2512.11891), arm (a): BASELINE pi0.5.
# One suite per job, BOTH safety levels, n=50/task, seed 7 — paper protocol.
# PYTHONPATH deliberately uses vlsa-aegis/safelibero (NOT $REPO/SafeLIBERO):
# the AEGIS arm requires the eef_marker scene body that only exists there, so
# both arms run identical scenes. Usage: sbatch --export=ALL,SUITE=safelibero_spatial ...
: "${SUITE:?set SUITE}"
set -uo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
AEGIS=$REPO/vlsa-aegis
PORT=${PORT:-8241}
OUT=$REPO/aegis_repl

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
PYTHONPATH="$AEGIS/openpi/src" python "$AEGIS/openpi/scripts/serve_policy.py" \
  policy:checkpoint --policy.config=pi05_libero \
  --policy.dir="$AEGIS/checkpoints/pi05_libero" --port "$PORT" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
  sleep 5
done
kill -0 "$SERVER_PID"

conda activate libero
export MUJOCO_GL=egl
export LD_PRELOAD=/ocean/projects/cis250185p/asingal/envs/libero/lib/libstdc++.so.6
export PYTHONPATH="$AEGIS/openpi/src:$AEGIS/safelibero:$REPO/vlm_pipeline"
for LEVEL in I II; do
  python "$REPO/vlm_pipeline/run_safelibero_pi05_eval.py" \
    --task_suite_name "$SUITE" --safety_level "$LEVEL" \
    --num_trials_per_task 50 --host localhost --port "$PORT" \
    --results_output_dir "$OUT/baseline" \
    --video_output_dir "$OUT/baseline_video" --seed 7
done
echo "AEGIS_REPL_BASELINE_${SUITE}_COMPLETE"
