#!/bin/bash
#SBATCH -J slres_baseproto
#SBATCH -A cis240028p
#SBATCH -p GPU-small
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH -t 01:30:00
#SBATCH -o /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH -e /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Base flow policy on all 400 official SafeLIBERO-Spatial episodes under the pi0.5 driver's protocol
# (vlm_pipeline/run_safelibero_pi05_eval.py: env.seed(7), 20 dummy settle steps, 300 max steps, >1 mm collision).
set -euo pipefail
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/safelibero_residual
PY=/ocean/projects/cis250185p/asingal/envs/openvla_libero_merged/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=$PWD
$PY eval_sl.py --ckpt runs/base/ema_last.pt --workers 5 --out runs/base/eval_official_pi05proto --eps 0-49 --env_seed 7 --settle 20
echo ALL_DONE
