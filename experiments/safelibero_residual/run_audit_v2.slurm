#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --exclude=pleiades-1-3
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# Audit of the v2 residual data: label check, open-loop replay, pi0.5-alone counterfactual (audit_v2.py).
#   sbatch -J audit_spatial --export=ALL,S=spatial run_audit_v2.slurm
set -euo pipefail
: "${S:?}"
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
PI_PORT=$((8000 + RANDOM % 1000))
source ./run_pi05_server.sh pi05_libero $PI_PORT
trap 'kill $PI_SERVER_PID 2>/dev/null || true' EXIT
./pleiades_run.sh python audit_v2.py --suite safelibero_$S --data data_v2/$S/${ROUND:-r0} --scenes ${SCENES:-data_v2/$S/scenes_train.npz} \
    --ckpt runs_scale/$S/base/ema_last.pt --pi_port $PI_PORT --n_acted ${NA:-30} --n_quiet ${NQ:-10} --workers 6 \
    --out runs_v2/audit/$S${ROUND:+_$ROUND}
echo ALL_DONE
