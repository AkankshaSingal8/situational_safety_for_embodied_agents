#!/bin/bash
#SBATCH --job-name=nogt_long_diag
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=03:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# No-GT Long weak-cell VIDEO forensics (2026-07-28): 100% of failures are
# timeouts; parser binds correctly; obstacle sits INSIDE the sphere keep-out
# of a stage-2 entity (soup 0.118 m / pudding 0.151 m < r_eff 0.165).
# Hypothesis: stage-2 grasp must thread the relaxed corridor window (0.066 m)
# and the repair pressure stalls it. n=3 videos on the two worst cells:
# Long L1 t1 (18 TSR) and Long L2 t2 (10 TSR), exact no-GT board config.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/nogt_long_diag"
PORT=${PORT:-8253}

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
PYTHONPATH="$CPP" python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_long --safety_level I \
  --task_indices 1 --num_trials_per_task 3 --host 127.0.0.1 --port "$PORT" \
  --corridor --obstacle_pos_source percep --obstacle_id_source symbolic \
  --entity_pos_source percep --save_videos \
  --results_output_dir "$OUT/L1_t1" --video_output_dir "$OUT/L1_t1_video"
PYTHONPATH="$CPP" python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_long --safety_level II \
  --task_indices 2 --num_trials_per_task 3 --host 127.0.0.1 --port "$PORT" \
  --corridor --obstacle_pos_source percep --obstacle_id_source symbolic \
  --entity_pos_source percep --save_videos \
  --results_output_dir "$OUT/L2_t2" --video_output_dir "$OUT/L2_t2_video"
echo "NOGT_LONG_DIAG_COMPLETE"
