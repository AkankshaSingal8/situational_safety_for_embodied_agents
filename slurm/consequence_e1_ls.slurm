#!/bin/bash
#SBATCH --job-name=conseq_e1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
#
# E1 (task #43 extension, user-requested 2026-08-07): learned-critic
# (consequence-model) candidate selection on LIBERO-Safety obstacle_avoidance,
# GT tier, finetuned LS ckpt, exec-parity board config.
#
# History: E1 was blocked at G1 because the LS model missed ONLY the
# progress-Spearman bar by 0.0026 (0.5974 vs 0.60) while crushing the other
# two gates (path err_ratio 0.127 vs <=0.5, contact AUC 0.9988 vs >=0.90).
# User now explicitly asked to try critic steering on LS -- this run is that
# trial, with the marginal-G1 caveat disclosed at scoring.
#
# Hypothesis being tested (freeze forensics 2026-08-06/07): the no-GT/GT SR
# deficit is task-blind paralysis -- repair blocks progress when hazard
# overlaps the work zone (oa t7 = canonical cell). Critic selection picks the
# max-predicted-progress candidate among safety-feasible ones, repair only as
# fallback -- a learned, task-aware alternative to the hand gates (cone,
# holding_disengage).
#
# Server = exec-parity LS board config + PINNED E1 consequence pairing
# (2026-08-02 final review): --consequence_domain LS
# --consequence_action_scale 0.5 --consequence_action_clip 0 (TS=2.0).
#
# Stage 1 SMOKE: oa L1 task 7 (t7 is an L1 task; 43155835 failed on level 0) only (the freeze cell), n=10; gate on server
# consequence_select lines -- abort (exit 3) if selector never fires.
# Stage 2a E1-PINNED: oa L0-2, n=10/task, pre-registered rule (threshold
#   0.10 / pessimism 1.0). E2 evidence: this fell back 74.5% of replans.
# Stage 2b E1-COMPOSE: same cells, threshold 1.0 / pessimism 0 -- the
#   feasibility gate never empties, so the critic ALWAYS ranks the K
#   repair-certified candidates by predicted progress (compose-not-compete;
#   safety stays with the in-denoising repair, the critic only breaks
#   freeze ties). Server restart between arms (threshold is server-side).
# Score vs gt_par_full oa (70.7 SR / 4.0 viol) + baseline-at-parity
# (74.0 / 4.0); t7-cell paired check.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
MODEL="$WT/consequence_train/ls_model"
OUT="$WT/consequence_e1"
PORT=8403
N=${N:-10}
SRVLOG=/ocean/projects/cis250185p/asingal/slurm_logs/conseq_e1_${SLURM_JOB_ID}_server.log
CPP="$LS:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
test -f "$MODEL/consequence_model_member0.pt"

start_server () {  # $1 = threshold, $2 = pessimism
  conda activate aegis
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
  PYTHONPATH="$WT/openpi_guided/src" python \
    "$WT/openpi_guided/scripts/serve_policy_guided.py" \
    --port "$PORT" --repair_schedule ramp --num_candidates 8 --eef_radius 0.09 \
    --translation_scale 2.0 --cmd_clip 0.025 \
    --consequence_select "$MODEL" \
    --consequence_domain LS --consequence_action_scale 0.5 \
    --consequence_action_clip 0 \
    --consequence_threshold "$1" --consequence_pessimism "$2" \
    --policy.config=pi05_libero --policy.dir="$LS_WEIGHTS" \
    >> "$SRVLOG" 2>&1 &
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
GTFLAGS="--obstacle_id_source gt --duality_disengage --exec_parity --engage_radius 0.22"

# ---- Stage 1: smoke (oa L0 t7 = freeze cell, n=10) + wiring gate ----------
start_server 0.10 1.0
python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
  --suite obstacle_avoidance --level 1 --task_filter 7 \
  --num_trials_per_task "$N" \
  --host 127.0.0.1 --port "$PORT" $GTFLAGS \
  --results_output_dir "$OUT/smoke_t7"

TOTAL=$(grep -c "consequence_select: {" "$SRVLOG" || true)
FB=$(grep -c "'fallback': True" "$SRVLOG" || true)
echo "SMOKE consequence_select lines: total=$TOTAL fallback=$FB"
if [ "$TOTAL" -eq 0 ]; then
  echo "SMOKE_GATE_FAIL: selector never fired (wiring problem)"; exit 3
fi
echo "SMOKE_GATE_PASS ($FB/$TOTAL fallback)"

# ---- Stage 2a: E1-pinned (oa L0-2, n=10/task, threshold .10/pess 1.0) -----
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" $GTFLAGS \
    --results_output_dir "$OUT/full_pinned"
done
kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true; sleep 10

# ---- Stage 2b: E1-compose (threshold 1.0 / pessimism 0 = always rank) -----
start_server 1.0 0.0
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" $GTFLAGS \
    --results_output_dir "$OUT/full_compose"
done
echo "CONSEQ_E1_COMPLETE"
