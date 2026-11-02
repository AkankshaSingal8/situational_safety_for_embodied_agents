#!/bin/bash
#SBATCH -J slres_gate
#SBATCH -A cis240028p
#SBATCH -p GPU-small
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH -t 02:30:00
#SBATCH -o /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.out
#SBATCH -e /ocean/projects/cis250185p/asingal/slurm_logs/%x_%j.err
# Oracle gate + scene generation.
#  1. generate training scenes (layouts 0-24) and two-obstacle test scenes (layouts 25-49)
#  2. oracle configs on official episodes 0-9 (80 episodes each, tuning), paired against base
#  (the chosen config is then extended to episodes 0-24 in the residual job: same out dir, resumes)
set -euo pipefail
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/experiments/safelibero_residual
PY=/ocean/projects/cis250185p/asingal/envs/openvla_libero_merged/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=$PWD
CK=runs/base/ema_last.pt; G=runs/gate; mkdir -p $G data
declare -A CFG=(
  [c0]='{}'
  [c1]='{"mode": "side"}'
  [c2]='{"m_xy": 0.04, "m_z": 0.04}'
  [c3]='{"use_arm": false}'
)
$PY eval_sl.py --ckpt $CK --out $G/base --eps 0-9 --workers 5
for C in ${CONFIGS:-c0 c1 c2 c3}; do
  $PY eval_sl.py --ckpt $CK --out $G/oracle_$C --arm oracle --text_mode true --oracle_cfg "${CFG[$C]}" --eps 0-9 --workers 5
done
$PY compare.py $G/base $($PY eval_sl.py --ckpt $CK --out $G/base --eps 0-9 --workers 5
for C in ${CONFIGS:-c0 c1 c2 c3}; do echo $G/oracle_$C; done) --eps 0-9 --json $G/tuning.json
$PY eval_sl.py --ckpt $CK --out runs/base/eval_noobs_v2 --eps 0-9 --workers 5 --no_obstacle
[ -f data/scenes_train.npz ] || $PY scenes.py --out data/scenes_train.npz --layouts 0-24 --per_layout 6 --seed 0
[ -f data/scenes_test2.npz ] || $PY scenes.py --out data/scenes_test2.npz --layouts 25-49 --per_layout 1 --n_obs 2 --seed 1
echo ALL_DONE
