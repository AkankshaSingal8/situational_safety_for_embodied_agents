#!/bin/bash
#SBATCH --job-name=ls_eng_dev
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=16:00:00
#SBATCH --array=0-2
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.err
# Execution-parity + engagement-gating DEV gate for Table 6 (finetuned ckpt).
# Root causes (2026-08-01 forensics):
#  (1) DOMINANT: robosuite OSC gains mismatch — authors' harness uses vendored
#      robosuite (output_max=2, kp=750); our client env loads stock 1.4.1
#      (output_max=0.05, kp=150) -> same policy runs 2.2-2.5x slower -> mass
#      600-step timeouts across ALL filtered arms. Fix: client --exec_parity
#      (gains monkeypatch + raw action passthrough + resize_with_pad 224).
#  (2) Steering tax where guards are active (oa suites, hs t13). Fix:
#      --engage_radius R (guard sent only when dist(eef,guard)<=R).
# Server for guided arms under parity: --translation_scale 2.0 --cmd_clip 0.025
# (barrier actuation model must track the vendored controller's ~2x motion).
# Dev cells = 3 tax suites, L0-2, n=4/task (20/cell). Gate: pooled SR >=
# baseline-2pp AND viol <= baseline (watch oa L2 t11/t14 added violations).
#   arms: gt_par        = gt_dual + --exec_parity            (parity alone)
#         gt_par_eng22  = gt_par + --engage_radius 0.22
#         gt_par_eng30  = gt_par + --engage_radius 0.30
#         nogt_par_eng30= nogt_dual_k2 + parity + eng 0.30
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_engage_dev"
PORT=$((8360 + SLURM_ARRAY_TASK_ID))
SUITES=(obstacle_avoidance obstacle_avoidance_human human_safety)
SUITE=${SUITES[$SLURM_ARRAY_TASK_ID]}
N=${N:-4}
CPP="$LS:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule ramp --num_candidates 8 --eef_radius 0.09 \
  --translation_scale 2.0 --cmd_clip 0.025 \
  --policy.config=pi05_libero --policy.dir="$LS_WEIGHTS" &
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
export PYTHONPATH="$CPP"
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite "$SUITE" --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source gt --duality_disengage --exec_parity \
    --results_output_dir "$OUT/gt_par"
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite "$SUITE" --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source gt --duality_disengage --exec_parity \
    --engage_radius 0.22 \
    --results_output_dir "$OUT/gt_par_eng22"
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite "$SUITE" --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source gt --duality_disengage --exec_parity \
    --engage_radius 0.30 \
    --results_output_dir "$OUT/gt_par_eng30"
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite "$SUITE" --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source fol --entity_pos_source percep \
    --duality_disengage --guard_topk 2 --exec_parity \
    --engage_radius 0.30 \
    --results_output_dir "$OUT/nogt_par_eng30"
done
echo "LS_ENG_DEV_${SUITE}_COMPLETE"
