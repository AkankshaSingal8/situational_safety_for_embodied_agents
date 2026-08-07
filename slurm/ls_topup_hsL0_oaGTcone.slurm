#!/bin/bash
#SBATCH --job-name=ls_topup
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=03:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Table 6 provenance top-ups:
# 1. hs L0 nogt FOLPROP (gapfix CD arm covered L1/L2 only; offline check
#    2026-08-07: 3/5 hs-L0 scenes elect different guards under folprop --
#    the pooled 87.3 suite figure currently borrows the fol L0 cell).
#    folpropvlm == folprop on all cached hs scenes -> cell double-counts.
# 2. oa L0-2 GT + eng0.22 + cone60: symmetric GT arm for the margin-dev
#    oa-nogt cone test (43152008 arm A) so a promotion applies to both tiers.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_topup"
PORT=8402
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

# 1: hs L0 nogt folprop (final no-GT hs config)
python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
  --suite human_safety --level 0 --num_trials_per_task "$N" \
  --host 127.0.0.1 --port "$PORT" \
  --obstacle_id_source folprop --entity_pos_source percep \
  --duality_disengage --guard_topk 2 --exec_parity \
  --engage_radius 0.30 \
  --results_output_dir "$OUT/hsL0_nogt_folprop"
# 2: oa L0-2 GT + cone60 @ eng0.22
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source gt --duality_disengage --exec_parity \
    --engage_radius 0.22 --engage_cone 60 \
    --results_output_dir "$OUT/oa_gt_cone"
done
echo "LS_TOPUP_COMPLETE"
