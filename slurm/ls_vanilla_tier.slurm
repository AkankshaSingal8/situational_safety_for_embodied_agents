#!/bin/bash
#SBATCH --job-name=ls_vanilla
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=14:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# VANILLA TIER (2026-08-12): the big-margin table. Vanilla pi05_libero (never
# safety-trained) baseline vs vanilla + ours-noGT, all 4 suites L0-2,
# n=10/task (150/cell). Claim under test: training-free FOL-grounded steering
# recovers (or exceeds) what dedicated safety finetuning buys, on a policy
# that has never seen safety data. Margin vs ITS OWN baseline = vanilla's
# violation rate. noGT configs = same class-driven routing as the frozen
# board noGT row (all eng 0.30; oa/oah fol [+cone60 oah], hs folprop,
# aff folpropvlm w/ cached VLM tables; percep positions; API-free).
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
VAN_WEIGHTS="$ROOT/vlsa-aegis/checkpoints/pi05_libero"
OUT="$WT/ls_vanilla_tier"
PORT=8417
N=${N:-10}
CPP="$LS:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule ramp --num_candidates 8 --eef_radius 0.09 \
  --translation_scale 2.0 --cmd_clip 0.025 \
  --policy.config=pi05_libero --policy.dir="$VAN_WEIGHTS" &
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
NOGT="--entity_pos_source percep --duality_disengage --guard_topk 2 --exec_parity"

# Arm V0: vanilla baseline, all suites
for SUITE in obstacle_avoidance human_safety obstacle_avoidance_human affordance; do
  for LVL in 0 1 2; do
    python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
      --suite "$SUITE" --level "$LVL" --num_trials_per_task "$N" \
      --host 127.0.0.1 --port "$PORT" --disable_guidance --exec_parity \
      --results_output_dir "$OUT/V0_vanilla_base"
  done
done
# Arm V1: vanilla + ours noGT (class-routed configs)
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --obstacle_id_source fol $NOGT \
    --engage_radius 0.30 \
    --results_output_dir "$OUT/V1_ours_nogt"
done
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite human_safety --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --obstacle_id_source folprop $NOGT \
    --engage_radius 0.30 \
    --results_output_dir "$OUT/V1_ours_nogt"
done
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance_human --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --obstacle_id_source fol $NOGT \
    --engage_radius 0.30 --engage_cone 60 \
    --results_output_dir "$OUT/V1_ours_nogt"
done
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite affordance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --obstacle_id_source folpropvlm $NOGT \
    --engage_radius 0.30 \
    --results_output_dir "$OUT/V1_ours_nogt"
  done
echo "LS_VANILLA_TIER_COMPLETE"
