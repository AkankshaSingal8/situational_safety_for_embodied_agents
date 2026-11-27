#!/bin/bash
# GR00T-N1.6-BEHAVIOR1k policy server (Isaac-GR00T examples/BEHAVIOR/README.md) on an Ada node (flash-attn needs
# Ampere+), for a BEHAVIOR client on pleiades-1-10. Writes "<host> <port>" to $HOSTFILE, serves until $STOPFILE exists.
#   HOSTFILE=... STOPFILE=... sbatch -J ov_b1k_srv -p pleiades -x pleiades-1-3,pleiades-1-10 --time=24:00:00 -c 4 \
#     --mem=48G --gres=gpu:1 -o /data/asingal2/slurm_logs/%x_%j.out groot_b1k_server.slurm
set -uo pipefail
HOSTFILE=${HOSTFILE:?}; STOPFILE=${STOPFILE:?}; N_CLIENTS=${N_CLIENTS:-1}  # stop when N_CLIENTS files ${STOPFILE}* exist
MODEL=${MODEL:-nvidia/GR00T-N1.6-BEHAVIOR1k}
GNV=/data/asingal2/oopsieverse/externals/Isaac-GR00T-nv
PORT=$((7100 + SLURM_JOB_ID % 300))
rm -f $HOSTFILE ${STOPFILE}*
/data/asingal2/envs/apptainer/bin/apptainer exec --nv --bind /data --bind /home/asingal2 --env HF_HOME=/data/asingal2/hf_cache \
  --pwd $GNV /data/asingal2/oopsieverse/b1k_sandbox /data/asingal2/envs/groot_srv_u24/bin/python \
  $GNV/gr00t/eval/run_gr00t_server.py --model-path $MODEL --embodiment-tag BEHAVIOR_R1_PRO \
  --use-sim-policy-wrapper --host 0.0.0.0 --port $PORT > ${HOSTFILE}.log 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null' EXIT
for i in $(seq 1 120); do
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -30 ${HOSTFILE}.log; exit 1; }
  (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break
  sleep 10
done
sleep 60   # model load after bind
echo "$(hostname) $PORT" > $HOSTFILE; echo "serving at $(hostname):$PORT"
while [ $(ls ${STOPFILE}* 2>/dev/null | wc -l) -lt $N_CLIENTS ] && kill -0 $SP 2>/dev/null; do sleep 30; done
echo "SERVER_DONE"
