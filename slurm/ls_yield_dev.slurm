#!/bin/bash
#SBATCH --job-name=ls_yielddev
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=03:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# G-Y2 dev smoke (spec 2026-08-11-yield-regime-design.md): Yield regime on
# oah, both tiers, n=10/task L0-2, promoted cone configs + --yield_regime.
# Freeze pool targets (F3 forensics): t3 ~2/10, t4 0/10, t9 ~3/10.
# PASS iff >=3 previously-frozen episodes now succeed AND zero new
# violations vs paired reference cells (ls_cone_confirm F2/F3, first 10 eps).
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_yield_dev"
PORT=8407
N=${N:-10}
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
NOGT="--obstacle_id_source fol --entity_pos_source percep --duality_disengage --guard_topk 2 --exec_parity"

# Y-GT: oah GT tier, promoted cone config + yield
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance_human --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source gt --duality_disengage --exec_parity \
    --engage_radius 0.22 --engage_cone 60 --yield_regime \
    --results_output_dir "$OUT/Y_oah_gt"
done
# Y-noGT: oah noGT tier, promoted cone config + yield
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance_human --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" $NOGT \
    --engage_radius 0.30 --engage_cone 60 --yield_regime \
    --results_output_dir "$OUT/Y_oah_nogt"
done
echo "LS_YIELD_DEV_COMPLETE"
