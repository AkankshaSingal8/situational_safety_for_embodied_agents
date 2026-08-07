#!/bin/bash
#SBATCH --job-name=conseq_e1c
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=03:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# E1-COMPOSE companion to 43155835 (parallel session's E1 pinned run; my
# duplicate 43159334 cancelled). Single arm: threshold 1.0 / pessimism 0 --
# feasibility never empties, so the critic ALWAYS ranks the K
# repair-certified candidates by predicted progress (compose-not-compete:
# certification stays with the in-denoising repair; the critic only breaks
# freeze ties). Targets the paralysis mechanism behind the LS SR deficit.
# oa L0-2 GT parity, n=10/task; score vs 43155835 full/ (pinned),
# gt_par_full oa 70.7/4.0, base 74.0/4.0; paired t7-cell check.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
MODEL="$WT/consequence_train/ls_model"
OUT="$WT/consequence_e1"
PORT=8404
N=${N:-10}
CPP="$LS:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
test -f "$MODEL/consequence_model_member0.pt"

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule ramp --num_candidates 8 --eef_radius 0.09 \
  --translation_scale 2.0 --cmd_clip 0.025 \
  --consequence_select "$MODEL" \
  --consequence_domain LS --consequence_action_scale 0.5 \
  --consequence_action_clip 0 \
  --consequence_threshold 1.0 --consequence_pessimism 0.0 \
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
GTFLAGS="--obstacle_id_source gt --duality_disengage --exec_parity --engage_radius 0.22"
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" $GTFLAGS \
    --results_output_dir "$OUT/full_compose"
done
echo "CONSEQ_E1_COMPOSE_COMPLETE"
