#!/bin/bash
# Regression check (disabled-correction eager vs traced pi0.5 must still
# match, since that path is untouched by the trajectory-correction fix) +
# smoke test (2 episodes) of the FIXED flow_cbf_sample.py (full-trajectory
# joint correction, replacing the earlier single-step version) before
# submitting the full multi-suite re-run.
#SBATCH --job-name=validate_smoke_flowcbf_fix
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=00:45:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/slurm_logs/%x_%j.err

set -uo pipefail
REPO=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
WORKTREE=$REPO/.worktrees/libero-safety-benchmark
LIBSAFETY_DIR=$WORKTREE/LIBERO-Safety

source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="${LIBSAFETY_DIR}/.libero_config"

cd "$WORKTREE"

echo "=== [1/2] Regression: validate_flow_cbf.py (disabled-correction path vs traced pi0.5) ==="
python3 validate_flow_cbf.py
VALIDATE_EXIT=$?
echo "VALIDATE_EXIT=$VALIDATE_EXIT"
if [ $VALIDATE_EXIT -ne 0 ]; then
    echo "REGRESSION_FAILED -- aborting, do not proceed to smoke/full run"
    exit 1
fi

echo "=== [2/2] Smoke test: 2 episodes on obstacle_avoidance task 2 (fixed full-trajectory correction) ==="
python3 run_libsafety_eval_flowcbf.py \
    --task_suite_name obstacle_avoidance \
    --task_index 2 \
    --num_trials_per_task 2 \
    --checkpoint_name flowcbf_pi05_libero_safety_v2fix_smoke \
    --results_out_path "${LIBSAFETY_DIR}/results" \
    --video_out_path "${LIBSAFETY_DIR}/rollouts" \
    --seed 7
SMOKE_EXIT=$?
echo "SMOKE_EXIT=$SMOKE_EXIT"

echo "=== Smoke results ==="
python3 -c "
import json, glob
for fn in sorted(glob.glob('${LIBSAFETY_DIR}/results/obstacle_avoidance_task2_flowcbf_pi05_libero_safety_v2fix_smoke.json')):
    for r in json.load(open(fn)):
        print(fn, '-> success=', r['success'], 'collision=', r.get('collision', 'MISSING'), 'steps=', r['num_steps'])
"
