#!/bin/bash
#SBATCH --job-name=conseq_v2
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Critic v2 full eval (user-approved 2026-08-08): composite-distance
# discounted progress labels + percep-noise-injection training + stall-gated
# selection (--consequence_stall_gate 0.01: critic overrides the legacy
# winner ONLY when that winner's predicted progress < 1 cm discounted
# reduction — breaks stalls, leaves healthy replans untouched).
# Arms: oa L0-2 n=10/task, GT tier (vs cone cell 79.3/4.7/75.3, rank-v1
# compose 77.3/3.3, incumbent 70.7/4.0) and no-GT tier (vs incumbent
# 68.7/0.7; the noise-injection training is aimed exactly here).
# Self-gating: waits up to 2h for ls_model_v2/g1_report.json, aborts
# (exit 3) unless v2.gate_pass is true (clean rho>=0.70, noisy rho>=0.60).
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
MODEL="$WT/consequence_train/ls_model_v2"
OUT="$WT/conseq_v2_eval"
PORT=8407
N=${N:-10}
CPP="$LS:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

for _ in $(seq 1 24); do
  [ -f "$MODEL/g1_report.json" ] && break
  echo "waiting for v2 gate report..."; sleep 300
done
GATE=$(python -c "import json;print(json.load(open('$MODEL/g1_report.json'))['v2']['gate_pass'])" 2>/dev/null || echo missing)
if [ "$GATE" != "True" ]; then echo "V2_GATE_NOT_PASSED ($GATE) — aborting"; exit 3; fi
test -f "$MODEL/consequence_model_member1.pt"

conda activate aegis
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PYTHONPATH="$WT/openpi_guided/src" python \
  "$WT/openpi_guided/scripts/serve_policy_guided.py" \
  --port "$PORT" --repair_schedule ramp --num_candidates 8 --eef_radius 0.09 \
  --translation_scale 2.0 --cmd_clip 0.025 \
  --consequence_select "$MODEL" --consequence_n_members 2 \
  --consequence_domain LS --consequence_action_scale 0.5 \
  --consequence_action_clip 0 \
  --consequence_threshold 1.0 --consequence_pessimism 0.0 \
  --consequence_stall_gate 0.01 \
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

# Arm 1: GT tier
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source gt --duality_disengage --exec_parity \
    --engage_radius 0.22 \
    --results_output_dir "$OUT/gt_v2"
done
# Arm 2: no-GT tier (incumbent config)
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source fol --entity_pos_source percep \
    --duality_disengage --guard_topk 2 --exec_parity \
    --engage_radius 0.30 \
    --results_output_dir "$OUT/nogt_v2"
done
echo "CONSEQ_V2_EVAL_COMPLETE"
