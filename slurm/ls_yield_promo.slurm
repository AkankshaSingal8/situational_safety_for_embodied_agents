#!/bin/bash
#SBATCH --job-name=ls_yieldpromo
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=12:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# G-Y3 promotion (spec 2026-08-11): Yield regime oah BOTH tiers at N=50/task
# L0-2 (750 eps/arm), paired 1:1 vs ls_cone_confirm F2/F3 (same N, same
# seeds). SUBMIT ONLY AFTER G-Y2 ROUND-2 PASS. Promotion rule: TSR strictly
# up, violations not up (exact McNemar, paired). hs tier is gated separately
# after oah promotes.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_yield_promo"
PORT=8413
N=${N:-50}
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

# P1: oah GT tier + yield (paired vs F2_oah_gt_cone)
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance_human --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source gt --duality_disengage --exec_parity \
    --engage_radius 0.22 --engage_cone 60 --yield_regime \
    --results_output_dir "$OUT/P_oah_gt_yield"
done
# P2: oah noGT tier + yield + F3 refresh (paired vs F3_oah_nogt_cone)
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance_human --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" $NOGT \
    --engage_radius 0.30 --engage_cone 60 --yield_regime \
    --percep_refresh_movers \
    --percep_refresh_cameras robot0_eye_in_hand,agentview \
    --entity_view_tau 0.08 \
    --percep_refresh_alpha 0.6 --percep_refresh_gate 0.20 \
    --results_output_dir "$OUT/P_oah_nogt_yield"
done
echo "LS_YIELD_PROMO_COMPLETE"
