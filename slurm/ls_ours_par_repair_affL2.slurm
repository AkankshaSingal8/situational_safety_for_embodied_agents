#!/bin/bash
#SBATCH --job-name=ls_par_rep
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=02:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Repair for 42943477_3 (affordance) server crash at L2: nogt arm died at
# 32/50 eps (partial quarantined to _partial_quarantine/), base arm L2 never
# ran. Reruns exactly those two arm-levels under the identical
# ls_ours_parity_full.slurm server/client config.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_ours_parity_full"
PORT=8394
N=${N:-10}
ENG_NOGT=${ENG_NOGT:-0.30}
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
python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
  --suite affordance --level 2 --num_trials_per_task "$N" \
  --host 127.0.0.1 --port "$PORT" \
  --obstacle_id_source fol --entity_pos_source percep \
  --duality_disengage --guard_topk 2 --exec_parity \
  --engage_radius "$ENG_NOGT" \
  --results_output_dir "$OUT/nogt_par_full"
python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
  --suite affordance --level 2 --num_trials_per_task "$N" \
  --host 127.0.0.1 --port "$PORT" \
  --disable_guidance --exec_parity \
  --results_output_dir "$OUT/base_par"
echo "LS_PAR_REPAIR_AFF_L2_COMPLETE"
