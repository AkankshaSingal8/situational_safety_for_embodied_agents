#!/bin/bash
#SBATCH --job-name=conseq_e2
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=10:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#
# E2 (task #43): consequence-select arm on SafeLIBERO Spatial L1, GT tier,
# standard pi0.5 ckpt, n=50/task -- compared offline against the four cached
# arms (baseline / AEGIS post-hoc / SOTA-reimpl / ours-current-guided).
# Gated by G1[SL] PASS (2026-08-03: err_ratio 0.039, AUC 0.932, rho 0.714).
# E1 (LS) is NOT run: G1[LS] failed its progress-Spearman bar (0.5974 vs 0.6).
#
# Server = ours-current guided Spatial-L1 board config (gamma 0.9, d_safe
# 0.01, r 0.09, inflation_slope 0.004, --no-use_companions) EXCEPT
# --num_candidates 8 (selector needs K>1; board cell is K=1 -- disclosed
# departure) + the pinned consequence pairing:
#   --consequence_domain SL --consequence_action_scale 20.0
#   --consequence_action_clip 0
#
# Stage 1 SMOKE: task 0 only, n=10, then gate on the server's
# consequence_select log lines (fallback fraction). Abort (exit 3) if the
# selector never fires or falls back >=90% -- tune pessimism/threshold
# before spending the full 200 episodes.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
AEGIS="$ROOT/vlsa-aegis"
MODEL="$WT/consequence_train/sl_model"
OUT="$WT/consequence_e2"
PORT=8380
SRVLOG=/ocean/projects/cis250185p/asingal/slurm_logs/conseq_e2_${SLURM_JOB_ID}_server.log
CPP="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
test -f "$MODEL/consequence_model_member0.pt"

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" \
  --policy.config=pi05_libero --policy.dir="$AEGIS/checkpoints/pi05_libero" \
  --gamma 0.9 --d_safe 0.01 --eef_radius 0.09 --inflation_slope 0.004 \
  --repair_schedule uniform --num_candidates 8 --no-use_companions \
  --consequence_select "$MODEL" \
  --consequence_domain SL --consequence_action_scale 20.0 \
  --consequence_action_clip 0 \
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

# ---- Stage 1: smoke (task 0, n=10) + fallback-rate gate -------------------
python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_spatial --safety_level I \
  --task_indices 0 --num_trials_per_task 10 \
  --host 127.0.0.1 --port "$PORT" \
  --results_output_dir "$OUT/smoke"

TOTAL=$(grep -c "consequence_select: {" "$SRVLOG" || true)
FB=$(grep -c "'fallback': True" "$SRVLOG" || true)
echo "SMOKE consequence_select lines: total=$TOTAL fallback=$FB"
if [ "$TOTAL" -eq 0 ]; then
  echo "SMOKE_GATE_FAIL: selector never fired (wiring problem)"; exit 3
fi
if [ $((FB * 10)) -ge $((TOTAL * 9)) ]; then
  echo "SMOKE_GATE_FAIL: fallback rate >= 90% ($FB/$TOTAL) -- tune pessimism/threshold"
  exit 3
fi
echo "SMOKE_GATE_PASS ($FB/$TOTAL fallback)"

# ---- Stage 2: full E2 (all tasks, n=50) -----------------------------------
python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
  --task_suite_name safelibero_spatial --safety_level I \
  --num_trials_per_task 50 \
  --host 127.0.0.1 --port "$PORT" \
  --results_output_dir "$OUT/full"

echo "CONSEQ_E2_COMPLETE"
