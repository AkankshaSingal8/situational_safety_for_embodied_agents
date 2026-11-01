#!/bin/bash
#SBATCH -J rflow_poc
#SBATCH -A cis240028p
#SBATCH -p GPU-small
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH -t 04:00:00
#SBATCH -o /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH -e /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Train main / notext / shuffled residual flows, then run every evaluation rollout.
set -euo pipefail
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/residual_flow_poc
PY=/ocean/projects/cis250185p/asingal/envs/flow_policy/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1
STEPS=${STEPS:-30000}
# the three small models share the one GPU
for V in main notext shuffled; do
  $PY train.py --variant $V --out runs/$V --steps $STEPS > logs/train_$V.log 2>&1 &
done
wait
tail -n 2 logs/train_*.log
$PY evaluate.py --runs runs --out runs/eval --workers 5 ${EVAL_EXTRA:-} > logs/eval.log 2>&1
tail -n 3 logs/eval.log
