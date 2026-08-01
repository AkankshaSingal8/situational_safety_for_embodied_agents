#!/bin/bash
#SBATCH --job-name=conseq_data
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=10:00:00
#SBATCH --array=0-1
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%A_%a.err
#
# Collect consequence-model training transitions (--log_transitions) from
# two domains, at execution-parity / deployment conditions, for task #43
# (see docs/superpowers/specs/2026-08-01-consequence-steering-design.md).
#
# ARRAY 0 -- LIBERO-Safety obstacle_avoidance, GT tier, safety-finetuned
#   checkpoint, execution-parity conditions (mirrors slurm/ls_engage_dev.slurm
#   server/client scaffolding exactly). Baseline arm (no guidance) and guided
#   arm (GT obstacle id + duality_disengage), both under --exec_parity, for
#   levels 0/1/2.
#
#   NOTE: run_guided_libero_safety_eval.py DOES implement --disable_guidance
#   natively (verified via grep), so the baseline arm below uses it directly
#   -- no substitution needed.
#
# ARRAY 1 -- SafeLIBERO Spatial L1, GT tier, standard (non-finetuned) pi0.5
#   checkpoint. Baseline arm mirrors slurm/baseline_strict.slurm's server
#   block (AEGIS checkpoint $ROOT/vlsa-aegis/checkpoints/pi05_libero,
#   --repair_schedule uniform --num_candidates 1, --disable_guidance client
#   flag). Guided ("ours-current") arm mirrors slurm/gt_spatL1_strict.slurm
#   (2026-07-30 19:19, most recently modified GT-tier guided Spatial-L1
#   script; reproduces the Table-1 board cell r2 config: --gamma 0.9
#   --d_safe 0.01 --eef_radius 0.09 --inflation_slope 0.004
#   --repair_schedule uniform --num_candidates 1 --no-use_companions,
#   client defaults obstacle_id_source=gt / entity_pos_source=gt). Because
#   the guided server's --no-use_companions differs from the baseline
#   server's defaults (use_companions=True), the guided arm needs its own
#   second server instance on PORT2=8372.
set -euo pipefail
module load anaconda3
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh

ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
OUT="$WT/consequence_data"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  # ---------------------------------------------------------------------
  # ARRAY 0: LIBERO-Safety obstacle_avoidance, GT tier, finetuned ckpt,
  # execution parity (mirrors slurm/ls_engage_dev.slurm).
  # ---------------------------------------------------------------------
  LS="$ROOT/LIBERO-Safety"
  LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
  PORT=8370
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

  for LVL in 0 1 2; do
    python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
      --suite obstacle_avoidance --level "$LVL" --num_trials_per_task 10 \
      --host 127.0.0.1 --port "$PORT" \
      --disable_guidance --exec_parity --log_transitions \
      --results_output_dir "$OUT/ls_baseline"
    python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
      --suite obstacle_avoidance --level "$LVL" --num_trials_per_task 4 \
      --host 127.0.0.1 --port "$PORT" \
      --obstacle_id_source gt --duality_disengage --exec_parity --log_transitions \
      --results_output_dir "$OUT/ls_guided"
  done

  echo "CONSEQ_DATA_0_COMPLETE"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
  # ---------------------------------------------------------------------
  # ARRAY 1: SafeLIBERO Spatial L1, GT tier, standard pi0.5 checkpoint.
  # Baseline server mirrors slurm/baseline_strict.slurm; guided server
  # mirrors slurm/gt_spatL1_strict.slurm (see header note above).
  # ---------------------------------------------------------------------
  AEGIS="$ROOT/vlsa-aegis"
  PORT=8371
  PORT2=8372
  CPP="$ROOT/SafeLIBERO/safelibero:$WT/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"

  conda activate aegis
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
  export PYTHONPATH="$WT/openpi_guided/src"

  python "$WT/openpi_guided/scripts/serve_policy_guided.py" \
    --port "$PORT" \
    --policy.config=pi05_libero \
    --policy.dir="$AEGIS/checkpoints/pi05_libero" \
    --repair_schedule uniform --num_candidates 1 &
  BASELINE_SERVER_PID=$!

  python "$WT/openpi_guided/scripts/serve_policy_guided.py" \
    --port "$PORT2" \
    --policy.config=pi05_libero \
    --policy.dir="$AEGIS/checkpoints/pi05_libero" \
    --gamma 0.9 --d_safe 0.01 --eef_radius 0.09 --inflation_slope 0.004 \
    --repair_schedule uniform --num_candidates 1 --no-use_companions &
  GUIDED_SERVER_PID=$!

  trap 'kill "$BASELINE_SERVER_PID" "$GUIDED_SERVER_PID" 2>/dev/null || true' EXIT

  for _ in $(seq 1 180); do
    python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),1); s.close()" 2>/dev/null && break
    sleep 5
  done
  kill -0 "$BASELINE_SERVER_PID"

  for _ in $(seq 1 180); do
    python -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT2),1); s.close()" 2>/dev/null && break
    sleep 5
  done
  kill -0 "$GUIDED_SERVER_PID"

  conda activate openvla_libero_merged
  export MUJOCO_GL=egl
  export PYTHONPATH="$CPP"

  python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
    --task_suite_name safelibero_spatial --safety_level I \
    --num_trials_per_task 20 \
    --host 127.0.0.1 --port "$PORT" --disable_guidance --log_transitions \
    --results_output_dir "$OUT/sl_baseline"

  python "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
    --task_suite_name safelibero_spatial --safety_level I \
    --num_trials_per_task 10 \
    --host 127.0.0.1 --port "$PORT2" --log_transitions \
    --results_output_dir "$OUT/sl_guided"

  echo "CONSEQ_DATA_1_COMPLETE"
fi
