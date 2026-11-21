#!/bin/bash
#SBATCH -p pleiades
#SBATCH --exclude=pleiades-1-3,pleiades-1-10
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
# Residual action expert training with a batch-size fallback: tries batch $BS, and if that runs out of GPU memory
# retries with batch $BS_FALLBACK (output dir <OUT>_bs<N>). ARGS must reach the job through the environment
# (export ARGS=...; sbatch --export=ALL ...): sbatch --export=VAR=value splits values at commas.
#   export OUT=runs_v2/expert/ex2_1gpu BS=64 BS_FALLBACK=32 ARGS="..."; sbatch -J ex2_1gpu --gres=gpu:1 ... run_train_fallback.slurm
set -uo pipefail
: "${OUT:?}" "${BS:?}" "${BS_FALLBACK:?}" "${ARGS:?}"
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
OP=/home/asingal2/safety/situational_safety_for_embodied_agents/vlsa-aegis/openpi
export PYTHONPATH=$OP/src:$OP/packages/openpi-client/src:$PWD OPENPI_DATA_HOME=/data/asingal2/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=1
echo "host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-none} start $(date)"
echo "ARGS: $ARGS"
for B in $BS $BS_FALLBACK; do
  O=${OUT}_bs$B
  LOG=/data/asingal2/slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_bs$B.log
  /data/asingal2/envs/openpi/bin/python train_expert.py --out $O --bs $B $ARGS 2>&1 | tee $LOG
  rc=${PIPESTATUS[0]}
  if [ $rc -eq 0 ]; then echo "EXIT_OK batch $B $(date)"; exit 0; fi
  if grep -qiE "RESOURCE_EXHAUSTED|out of memory" $LOG; then echo "batch $B out of memory, trying next"; continue; fi
  echo "failed (rc $rc), not an OOM"; exit $rc
done
exit 1
