#!/bin/bash
#SBATCH --job-name=ls_cone50
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=07:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Paralysis-class rescue (gapfix 43010653 follow-up): stall_recovery FAILED
# on hs-L2-t13 (1->0/10) -- the failure is keep-out blocking the workspace,
# not a stall attractor. Test APPROACHING gating (--engage_cone 60):
# guard engaged iff dist<=0.14m OR eef closing within 60 deg; a protected
# intruder BESIDE the path no longer walls off the work zone.
#   F1 hs  L2   gt   eng0.22 + cone60   (t13 target; siblings t10-14 hold)
#   F2 oah L0-2 gt   eng0.22 + cone60   (GT arm -26 succ paralysis, all
#                                        guards = OBJ_with_hand intruders)
#   F3 oah L0-2 nogt eng0.30 + cone60   (nogt L1 18/50 target)
# Same server/protocol as ls_ours_parity_full campaign. Promotion rule:
# an arm promotes only if it fixes its target cells without regressing
# siblings (paired vs campaign cells).
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_cone_confirm"
PORT=8396
N=${N:-50}
CONE=${CONE:-60}
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

# F2: oah L0-2 gt + cone
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance_human --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source gt --duality_disengage --exec_parity \
    --engage_radius 0.22 --engage_cone "$CONE" \
    --results_output_dir "$OUT/F2_oah_gt_cone"
done
# F3: oah L0-2 nogt + cone
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance_human --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" $NOGT \
    --engage_radius 0.30 --engage_cone "$CONE" \
    --results_output_dir "$OUT/F3_oah_nogt_cone"
done
echo "LS_CONE_DEV_COMPLETE"
