#!/bin/bash
#SBATCH --job-name=debug_moka_crash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=00:10:00
#SBATCH --output=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/slurm_logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/slurm_logs/%x_%j.err

set -uo pipefail
source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_client
export MUJOCO_GL=egl
export LD_PRELOAD=/ocean/projects/cis250185p/asingal/envs/libsafety_client/lib/libstdc++.so.6
export LIBERO_CONFIG_PATH=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/LIBERO-Safety/.libero_config
export PYTHONFAULTHANDLER=1

cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark
python3 -X faulthandler -c "
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict['obstacle_avoidance']()
task = task_suite.get_task(2)
print('TASK:', task.language, flush=True)
task_bddl_file = task_suite.get_task_bddl_file_path_by_level_id(task.level, task.level_id)
print('BDDL:', task_bddl_file, flush=True)
env = OffScreenRenderEnv(bddl_file_name=str(task_bddl_file), camera_heights=256, camera_widths=256)
print('ENV CREATED OK', flush=True)
env.seed(7)
env.reset()
print('RESET OK', flush=True)
obs = env.step([0.0]*6+[-1.0])
print('STEP OK', flush=True)
"
echo "PYTHON_EXIT=$?"
