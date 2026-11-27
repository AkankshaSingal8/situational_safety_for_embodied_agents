#!/bin/bash
# GR00T-N1.6-BEHAVIOR1k on one OopsieVerse-paper B1K task (eval/run_groot_b1k_oopsie.py) on pleiades-1-10 in the
# Ubuntu 24.04 sandbox, against a server from groot_b1k_server.slurm ($HOSTFILE). Touches ${STOPFILE}.$TASK on exit.
#   HOSTFILE=... STOPFILE=... TASK=attach_camera NUM_TRIALS=30 sbatch -J ov_b1kg_attach_camera -p pleiades \
#     -w pleiades-1-10 --dependency=after:<server_job> --time=2-00:00:00 -c 8 --mem=64G --gres=gpu:1 \
#     -o /data/asingal2/slurm_logs/%x_%j.out groot_b1k_oopsie_client.slurm
set -uo pipefail
HOSTFILE=${HOSTFILE:?}; STOPFILE=${STOPFILE:?}; TASK=${TASK:?}
NUM_TRIALS=${NUM_TRIALS:-30}; MAX_STEPS=${MAX_STEPS:-0}
TRIAL_START=${TRIAL_START:-0}
OUT=${OUT:-/data/asingal2/oopsieverse/results/groot_b1k_oopsie/$TASK/part$TRIAL_START}
GNV=/data/asingal2/oopsieverse/externals/Isaac-GR00T-nv
EVAL=/home/asingal2/safety/wt-oopsieverse-eval/oopsieverse_eval/eval
E=/data/asingal2/envs/groot_b1k
ICD=/data/asingal2/oopsieverse/tools/nvidia_icd.json
mkdir -p $OUT
trap "touch ${STOPFILE}.$TASK.$TRIAL_START" EXIT
for i in $(seq 1 720); do [ -s $HOSTFILE ] && break; sleep 20; done
[ -s $HOSTFILE ] || { echo "no server hostfile after 4h"; exit 1; }
read SHOST SPORT < $HOSTFILE
sleep ${STAGGER:-0}   # concurrent Kit startups on 1-10 race on the shared extension cache
echo "client $(hostname) -> $SHOST:$SPORT task=$TASK trials=$NUM_TRIALS max_steps=$MAX_STEPS"
/data/asingal2/envs/apptainer/bin/apptainer exec --nv --bind /data --bind /home/asingal2 \
  --env OMNIGIBSON_HEADLESS=1 --env OMNI_KIT_ACCEPT_EULA=YES --env VK_ICD_FILENAMES=$ICD \
  --env OMNIGIBSON_DATA_PATH=/data/asingal2/oopsieverse/b1k_tp_data --env PYTHONUNBUFFERED=1 \
  --env HF_HOME=/data/asingal2/hf_cache --pwd $GNV /data/asingal2/oopsieverse/b1k_sandbox bash -c "
  LDP=\$(find $E/lib/python3.10/site-packages/isaacsim -maxdepth 3 -type d -name bin 2>/dev/null | tr '\n' ':')
  export LD_LIBRARY_PATH=\"\$LDP:\${LD_LIBRARY_PATH:-}\"
  $E/bin/python $EVAL/run_groot_b1k_oopsie.py --task $TASK --num_trials $NUM_TRIALS --max_steps $MAX_STEPS --trial_start $TRIAL_START \
    --host $SHOST --port $SPORT --results_dir $OUT"
RC=$?; echo "EXIT=$RC"; exit $RC
