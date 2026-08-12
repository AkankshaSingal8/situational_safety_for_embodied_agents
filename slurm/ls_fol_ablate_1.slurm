#!/bin/bash
#SBATCH --job-name=ls_fol_ablate_1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=11:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# FOL ABLATION TIER, script 1/3 (2026-08-12): binding-set ablations at the
# frozen noGT configs, paired 1:1 with the incumbent board cells (see
# results_tables/fol_ablation_binding.json). Controller-fixed 9-arm cell
# list, split into three whole-arm scripts so each stays under 11h; the
# three oah arms (slowest suite) are spread one-per-script. This script:
# oa geom_only, oa no_exempt, oah geom_only.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_fol_ablate"
PORT=8419
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
NOGT="--entity_pos_source percep --duality_disengage --guard_topk 2 --exec_parity"

# A_oa_geom_only: obstacle_avoidance, fol, geom_only
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --obstacle_id_source fol $NOGT \
    --engage_radius 0.30 --fol_ablate geom_only \
    --results_output_dir "$OUT/A_oa_geom_only"
done
# A_oa_no_exempt: obstacle_avoidance, fol, no_exempt
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --obstacle_id_source fol $NOGT \
    --engage_radius 0.30 --fol_ablate no_exempt \
    --results_output_dir "$OUT/A_oa_no_exempt"
done
# A_oah_geom_only: obstacle_avoidance_human, fol, geom_only
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance_human --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --obstacle_id_source fol $NOGT \
    --engage_radius 0.30 --engage_cone 60 --fol_ablate geom_only \
    --results_output_dir "$OUT/A_oah_geom_only"
done
echo "LS_FOL_ABLATE_1_COMPLETE"
