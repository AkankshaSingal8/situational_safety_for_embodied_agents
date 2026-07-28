#!/bin/bash
#SBATCH --job-name=visovr_pilot
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# pi0.7-style VISUAL conditioning pilot (image channel, not text): keep-out
# disk + target ring drawn into the policy's agentview input. Two arms on
# Spatial L1, paired vs the stored baseline 67.0/14.0 (n=200):
#   overlay_only    — guidance disabled; does the image channel ALONE move CAR?
#   overlay_repair  — overlay + composed in-denoising repair (additivity read)
# GO signal: overlay_only CAR > baseline CAR by >=8pp at n=20/task.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/fgd_visovr_pilot"
PORT=${PORT:-8172}
NUM=${NUM:-20}

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
  --task_suite_name safelibero_spatial --safety_level I \
  --num_trials_per_task "$NUM" --host 127.0.0.1 --port "$PORT" \
  --visual_overlay --disable_guidance --save_videos \
  --results_output_dir "$OUT/overlay_only/results"
PYTHONPATH="$CPP" python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_spatial --safety_level I \
  --num_trials_per_task "$NUM" --host 127.0.0.1 --port "$PORT" \
  --visual_overlay \
  --results_output_dir "$OUT/overlay_repair/results"
echo "VISOVR_PILOT_COMPLETE"
