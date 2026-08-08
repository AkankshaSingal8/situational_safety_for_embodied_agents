#!/bin/bash
#SBATCH --job-name=ls_nboost
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --exclude=w002,w004,w005,w006,w007,w008,w009,w010
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# N-boost confirms (significance pass 2026-08-08): the two beat-baseline
# cells are favorable trends at n=150 (GT-oa cone 17:9 discordant p=0.17;
# noGT-hs 17:10 p=0.25; violation wins p=.06-.07). +20 eps/task on the SAME
# promoted configs pools to n=450/cell — at current discordant ratios that
# crosses p<0.05 on both success claims and the violation claims. ALSO
# +20/task baseline arms on the same suites so pairing stays 1:1 at n=450.
set -euo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
ROOT=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WT="$ROOT/.worktrees/flow-guidance-tier-gt"
LS="$ROOT/LIBERO-Safety"
LS_WEIGHTS=$(ls -d /ocean/projects/cis250185p/asingal/.hf_cache/hub/models--LIBERO-Safety--pi05_libero_safety/snapshots/*/)
OUT="$WT/ls_nboost"
PORT=8409
N=${N:-20}
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

# 1: oa GT cone +20/task (promoted board config)
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source gt --duality_disengage --exec_parity \
    --engage_radius 0.22 --engage_cone 60 \
    --results_output_dir "$OUT/oa_gt_cone_n20"
done
# 2: hs no-GT folprop +20/task (promoted board config)
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite human_safety --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" \
    --obstacle_id_source folprop --entity_pos_source percep \
    --duality_disengage --guard_topk 2 --exec_parity \
    --engage_radius 0.30 \
    --results_output_dir "$OUT/hs_nogt_folprop_n20"
done
# 3+4: baseline +20/task on both suites (guidance off) for 1:1 pairing
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite obstacle_avoidance --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --disable_guidance --exec_parity \
    --results_output_dir "$OUT/base_oa_n20"
done
for LVL in 0 1 2; do
  python "$WT/vlm_pipeline/run_guided_libero_safety_eval.py" \
    --suite human_safety --level "$LVL" --num_trials_per_task "$N" \
    --host 127.0.0.1 --port "$PORT" --disable_guidance --exec_parity \
    --results_output_dir "$OUT/base_hs_n20"
done
echo "LS_NBOOST_COMPLETE"
