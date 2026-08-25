#!/bin/bash
#SBATCH --job-name=aegis_repl_gt
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --exclude=w002
#SBATCH --time=12:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# AEGIS-paper replication (arXiv 2512.11891), arm (b): AEGIS CBF-QP layer,
# GT-localization variant (perception stack replaced by simulator obstacle
# pose; the paper-exact GLM-4.5V+GroundingDINO path needs ZHIPUAI_API_KEY —
# run aegis_repl_paper.slurm once the key is available). main_aegis_gt.py =
# repo HEAD method with the eef_marker regression reverted (upstream body).
# One suite per job, BOTH levels, n=50/task, seed 7.
: "${SUITE:?set SUITE}"
set -uo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
AEGIS=$REPO/vlsa-aegis
PORT=${PORT:-8242}
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
export PYTHONPATH="$AEGIS/openpi/src:$AEGIS/safelibero:$AEGIS/main"
for LEVEL in I II; do
  python "$AEGIS/main/main_aegis_gt.py" \
    --task-suite-name "$SUITE" --safety-level "$LEVEL" \
    --num-trials-per-task 50 --host localhost --port "$PORT" \
    --video-out-path "$OUT/aegis_gt_video" \
    --results-output-dir "$OUT/aegis_gt"
done
echo "AEGIS_REPL_GT_${SUITE}_COMPLETE"
