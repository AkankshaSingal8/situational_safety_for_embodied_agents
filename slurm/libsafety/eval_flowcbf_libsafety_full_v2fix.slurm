#!/bin/bash
# Full flow-CBF (English et al., arXiv:2607.01378) re-eval on LIBERO-Safety
# using the FIXED flow_cbf_sample.py (full-trajectory joint correction,
# matching Algorithm 1 exactly -- see plan doc / flow_cbf_sample.py module
# docstring for what was wrong in the earlier single-step version). Uses a
# distinct checkpoint_name (flowcbf_pi05_libero_safety_v2fix) so these
# results do not mix with or get clobbered by the earlier, algorithmically
# deviated run's files.
#
# Same per-task retry pattern as the original eval_flowcbf_libsafety_full.slurm
# (in-process driver, no server/client split -- correction runs inside the
# denoising loop itself).
#SBATCH --job-name=full_flowcbf_v2fix
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=10:00:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/slurm_logs/%x_%j.err

set -uo pipefail
: "${SUITE:?ERROR: SUITE env var not set (export via --export=ALL,SUITE=...)}"

REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WORKTREE=$REPO/.worktrees/libero-safety-benchmark
LIBSAFETY_DIR=$WORKTREE/LIBERO-Safety
MAX_RETRIES=3
NUM_TASKS=15
CKPT_NAME=flowcbf_pi05_libero_safety_v2fix

echo "=== [job ${SLURM_JOB_ID}] flow-CBF (FIXED, full-trajectory) suite=${SUITE} (per-task retry) ==="

source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="${LIBSAFETY_DIR}/.libero_config"

cd "$WORKTREE"

for TASK_IDX in $(seq 0 $((NUM_TASKS - 1))); do
    EXISTING="${LIBSAFETY_DIR}/results/${SUITE}_task${TASK_IDX}_${CKPT_NAME}.json"
    if [ -f "$EXISTING" ]; then
        echo "=== Suite $SUITE  Task $TASK_IDX  already done, skipping ==="
        continue
    fi
    SUCCESS=0
    for ATTEMPT in $(seq 1 $MAX_RETRIES); do
        echo "=== Suite $SUITE  Task $TASK_IDX  Attempt $ATTEMPT/$MAX_RETRIES ==="
        python run_libsafety_eval_flowcbf.py \
            --task_suite_name "$SUITE" \
            --task_index "$TASK_IDX" \
            --num_trials_per_task 3 \
            --checkpoint_name "$CKPT_NAME" \
            --results_out_path "${LIBSAFETY_DIR}/results" \
            --video_out_path "${LIBSAFETY_DIR}/rollouts" \
            --seed 7
        EXIT=$?
        if [ $EXIT -eq 0 ]; then
            SUCCESS=1
            break
        else
            echo "Attempt $ATTEMPT failed (exit=$EXIT), retrying in 10s..."
            sleep 10
        fi
    done
    if [ $SUCCESS -eq 0 ]; then
        echo "GAVE UP: suite=$SUITE task=$TASK_IDX after $MAX_RETRIES attempts" | tee -a "${LIBSAFETY_DIR}/results/failed_tasks_flowcbf_v2fix.log"
    fi
done

echo "=== Suite $SUITE complete, aggregating ==="
python3 - "$SUITE" "${LIBSAFETY_DIR}/results" "$CKPT_NAME" << 'PYEOF'
import json, glob, sys
suite, results_dir, ckpt_name = sys.argv[1], sys.argv[2], sys.argv[3]
merged = []
for fn in sorted(glob.glob(f"{results_dir}/{suite}_task*_{ckpt_name}.json")):
    if "_all_tasks_" in fn:
        continue
    merged.extend(json.load(open(fn)))
out_fn = f"{results_dir}/{suite}_all_tasks_{ckpt_name}.json"
json.dump(merged, open(out_fn, "w"), indent=2)
print(f"Wrote {len(merged)} episodes to {out_fn}")
PYEOF
