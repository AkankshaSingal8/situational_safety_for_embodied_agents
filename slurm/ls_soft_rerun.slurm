#!/bin/bash
#SBATCH --job-name=ls_soft_rerun
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=24:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# LIBERO-Safety rerun with the FINALIZED noise-space enforcement config
# (acceptance-box winner from enforce_mid_n50 / job 42662560): server runs
# --repair_schedule none + FK tilt + soft repulsor instead of composed repair.
# Hypothesis: fixes the human_safety steering tax (safe-rate 100 but TSR
# crushed to single digits under composed repair — over-conservative near
# the human), while keeping the violations->0 result on the oa suites.
# Submit AFTER the box verdict with the winning eta:
#   sbatch --export=ALL,ETA=0.0045,N=50 slurm/ls_soft_rerun.slurm
# Add suites for the consistency pass:
#   --export=ALL,ETA=...,N=50,SUITES="human_safety obstacle_avoidance obstacle_avoidance_h"
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_rows_soft"
PORT=${PORT:-8217}
SUITES=${SUITES:-"human_safety"}
LEVELS=${LEVELS:-"0 1 2"}
N=${N:-50}
ARMS=${ARMS:-"guided_gt guided_fol guided_nogt_fol"}
ETA=${ETA:?pass the acceptance-box winning repulsor_eta, e.g. ETA=0.0045}
FK_BETA=${FK_BETA:-40.0}

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="$WT/openpi_guided/src"
python "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule none --num_candidates 8 --eef_radius 0.09 \
  --fk_beta "$FK_BETA" --repulsor_eta "$ETA" \
  --translation_scale 1.0 --cmd_clip 0.05 \
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
for SUITE in $SUITES; do
  for LVL in $LEVELS; do
    for ARM in $ARMS; do
      case "$ARM" in
        guided_gt)       EXTRA="--obstacle_id_source gt";;
        guided_symbolic) EXTRA="--obstacle_id_source symbolic";;
        guided_fol)      EXTRA="--obstacle_id_source fol";;
        guided_nogt_fol) EXTRA="--obstacle_id_source fol --entity_pos_source percep";;
        guided_nogt_sym) EXTRA="--obstacle_id_source symbolic --entity_pos_source percep";;
      esac
      python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
        --suite "$SUITE" --level "$LVL" --num_trials_per_task "$N" \
        --host 127.0.0.1 --port "$PORT" $EXTRA \
        --results_output_dir "$OUT/$ARM"
    done
  done
done
echo "LS_SOFT_RERUN_COMPLETE"
