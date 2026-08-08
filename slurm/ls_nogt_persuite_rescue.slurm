#!/bin/bash
#SBATCH --job-name=ls_persuite
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Per-suite no-GT rescue (user directive 2026-08-08: no-GT must beat baseline
# per-suite). Bottleneck decomposition (ledger):
#  oa  deficit = t7-class PERCEP NOISE at engagement (gt@0.22 9/10 vs nogt
#      2/10; cone no-transfer; blind 2-view REJECTED for importing birdview
#      failures t0/t10). Arm D = consistency-GATED 2-view fusion @eng0.22:
#      birdview refines when views agree within tau=0.06, never overrides.
#  oah deficit = HARD-REPAIR FREEZES on goal-adjacent hand-held hazards (t4
#      0/50, t9 14/50, correct guards, zero violations). Arm E = confirmed
#      cone config + server --repulsor_eta 0.005: the soft velocity channel
#      that un-sticks obstacle-on-grasp geometry on SafeLIBERO (55% where
#      hard repair scores 0-10%), never tried on LS.
# Promotion rule unchanged: fix target cells w/o regressing siblings/viol.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_persuite_rescue"
PORT=8405
N=${N:-10}
CPP="$LS:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

start_server () {  # $1 = repulsor_eta
  conda activate aegis
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
  PYTHONPATH="$WT/openpi_guided/src" python \
    "$WT/openpi_guided/scripts/serve_policy_guided.py" \
    --port "$PORT" --repair_schedule ramp --num_candidates 8 --eef_radius 0.09 \
    --translation_scale 2.0 --cmd_clip 0.025 --repulsor_eta "$1" \
    --policy.config=pi05_libero --policy.dir="$LS_WEIGHTS" &
  SERVER_PID=$!
  for _ in $(seq 1 240); do
    python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
    sleep 5
  done
  kill -0 "$SERVER_PID"
  conda activate openvla_libero_merged
  export MUJOCO_GL=egl
  export LIBERO_CONFIG_PATH="$LS/.libero_config"
  export PYTHONPATH="$CPP"
}
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
NOGT="--obstacle_id_source fol --entity_pos_source percep --duality_disengage --guard_topk 2 --exec_parity"

# Arm D: oa no-GT, gated 2-view @ eng0.22 (percep-noise fix, gt-matched radius)
start_server 0.0
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" $NOGT \
    --engage_radius 0.22 \
    --entity_cameras agentview,birdview --entity_view_tau 0.06 \
    --results_output_dir "$OUT/D_oa_gated2view"
done
kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true; sleep 10

# Arm E: oah no-GT cone + soft repulsor (freeze levels L0+L1)
start_server 0.005
for LVL in 0 1; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance_human --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" $NOGT \
    --engage_radius 0.30 --engage_cone 60 \
    --results_output_dir "$OUT/E_oah_repulsor"
done
echo "LS_PERSUITE_RESCUE_COMPLETE"
