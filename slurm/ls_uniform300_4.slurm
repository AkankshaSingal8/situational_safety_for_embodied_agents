#!/bin/bash
#SBATCH --job-name=ls_uniform300_4
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=10:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# UNIFORM n=300 rerun, script 4/6 (2026-08-13): single-protocol paper table.
# This script: ARM 4 -- ours-GT, all 4 suites, all levels. Per-suite config
# taken from slurm/ls_ours_parity_full.slurm's gt_par_full arm (GT id
# source, duality_disengage, exec_parity, engage_radius 0.22), EXCEPT oah
# which adds --engage_cone 60 per the F2 cone promotion (see
# slurm/ls_yield_dev.slurm's non-yield GT arm and slurm/ls_cone_confirm.slurm
# F2). CONCERN (see task-4-uniform300-report.md): ls_ours_parity_full.slurm
# runs oa/hs/aff GT with NO cone, but slurm/ls_oa_gtcone_confirm.slurm
# (n=50 confirm) and slurm/ls_topup_hsL0_oaGTcone.slurm both separately run
# an oa GT+cone60 config that appears to have been under promotion review.
# Per task instructions this script uses ls_ours_parity_full.slurm's
# no-cone oa/hs/aff config and flags the asymmetry (oah cone60, others not)
# for the parent to rule on rather than resolving it here.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_uniform300/gt"
PORT=8428
N=${N:-20}
CPP="$LS:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
SERVER_PID=""

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
GT_COMMON="--obstacle_id_source gt --duality_disengage --exec_parity --engage_radius 0.22"

# oa: gt, eng0.22, no cone (ls_ours_parity_full.slurm)
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" $GT_COMMON \
    --results_output_dir "$OUT"
done
# oah: gt, eng0.22, cone60 (F2 cone promotion)
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance_human --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" $GT_COMMON --engage_cone 60 \
    --results_output_dir "$OUT"
done
# hs: gt, eng0.22, no cone (ls_ours_parity_full.slurm)
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite human_safety --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" $GT_COMMON \
    --results_output_dir "$OUT"
done
# aff: gt, eng0.22, no cone (ls_ours_parity_full.slurm)
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite affordance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" $GT_COMMON \
    --results_output_dir "$OUT"
done
echo "LS_UNIFORM300_4_COMPLETE"
