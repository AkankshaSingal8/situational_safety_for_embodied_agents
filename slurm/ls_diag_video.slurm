#!/bin/bash
#SBATCH --job-name=ls_diag
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=01:30:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# LIBERO-Safety TSR=0 forensics: baseline episodes with saved videos, eef path
# telemetry, and a 520-step cap on the easiest level (L0) — is the policy
# moving sensibly (protocol OK, tasks hard) or flailing (protocol broken)?
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_diag"
PORT=${PORT:-8163}

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="$WT/openpi_guided/src"
python "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule ramp --num_candidates 1 \
  --policy.config=pi05_libero \
  --policy.dir="$LS_WEIGHTS" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 240); do
  python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
  sleep 5
done
kill -0 "$SERVER_PID"

conda activate openvla_libero_merged
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$LS/.libero_config"
export PYTHONPATH="$LS:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
  --suite obstacle_avoidance --level 0 --num_trials_per_task 2 \
  --host 127.0.0.1 --port "$PORT" --disable_guidance \
  --save_videos --max_steps 520 \
  --results_output_dir "$OUT/baseline_L0"
python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
  --suite human_safety --level 0 --num_trials_per_task 1 \
  --host 127.0.0.1 --port "$PORT" --disable_guidance \
  --save_videos --max_steps 520 \
  --results_output_dir "$OUT/baseline_hs_L0"
echo "LS_DIAG_COMPLETE"
