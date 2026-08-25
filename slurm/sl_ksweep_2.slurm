#!/bin/bash
#SBATCH --job-name=sl_ksweep_2
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --time=10:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#
# K-sweep GO/NO-GO arm 2/4: ours-guided, K=8, SafeLIBERO Spatial L2, GT tier,
# full suite (all tasks), n=50/task. Cached board incumbent for ours-guided
# on Spatial is K=1; this arm probes K=8 to see whether more candidates
# help. Server config otherwise matches the ours-current guided Spatial
# board config (gamma 0.9, d_safe 0.01, r 0.09, inflation_slope 0.004,
# --no-use_companions), minus any consequence-select machinery.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
OUT="$WT/sl_ksweep/K8/L2"
PORT=8437
SRVLOG=/ocean/projects/cis250185p/asingal/slurm_logs/sl_ksweep_2_${SLURM_JOB_ID}_server.log
CPP="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" \
  --policy.config=pi05_libero --policy.dir="$AEGIS/checkpoints/pi05_libero" \
  --gamma 0.9 --d_safe 0.01 --eef_radius 0.09 --inflation_slope 0.004 \
  --repair_schedule uniform --num_candidates 8 --no-use_companions \
  > "$SRVLOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 240); do
  python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
  sleep 5
done
kill -0 "$SERVER_PID"

conda activate openvla_libero_merged
export MUJOCO_GL=egl
export PYTHONPATH="$CPP"

# ---- Full K=8, L2 arm (all tasks, n=50) ------------------------------------
python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_spatial --safety_level II \
  --num_trials_per_task ${N:-50} \
  --host 127.0.0.1 --port "$PORT" \
  --results_output_dir "$OUT"

echo "SL_KSWEEP_2_COMPLETE"
