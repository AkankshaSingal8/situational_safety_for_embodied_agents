#!/bin/bash
# NVIDIA's BEHAVIOR rollout client (gr00t/eval/rollout_policy.py, verbatim README flags) on pleiades-1-10 inside the
# Ubuntu 24.04 sandbox, against a GR00T server started by groot_b1k_server.slurm ($HOSTFILE).
#   HOSTFILE=... STOPFILE=... ENV_NAME=sim_behavior_r1_pro/attach_a_camera_to_a_tripod N_EPISODES=10 \
#   sbatch -J ov_b1k_cli -p pleiades -w pleiades-1-10 --time=24:00:00 -c 8 --mem=64G --gres=gpu:1 \
#     -o /data/asingal2/slurm_logs/%x_%j.out groot_b1k_client.slurm
set -uo pipefail
HOSTFILE=${HOSTFILE:?}; STOPFILE=${STOPFILE:?}
ENV_NAME=${ENV_NAME:?}; N_EPISODES=${N_EPISODES:-10}; MAX_STEPS=${MAX_STEPS:-999999999}
GNV=/data/asingal2/oopsieverse/externals/Isaac-GR00T-nv
E=/data/asingal2/envs/groot_b1k
ICD=/data/asingal2/oopsieverse/tools/nvidia_icd.json
for i in $(seq 1 360); do [ -s $HOSTFILE ] && break; sleep 20; done
[ -s $HOSTFILE ] || { echo "no server hostfile"; exit 1; }
read SHOST SPORT < $HOSTFILE
echo "client $(hostname) -> $SHOST:$SPORT env=$ENV_NAME episodes=$N_EPISODES"
trap 'touch $STOPFILE' EXIT
/data/asingal2/envs/apptainer/bin/apptainer exec --nv --bind /data --bind /home/asingal2 \
  --env OMNIGIBSON_HEADLESS=1 --env OMNI_KIT_ACCEPT_EULA=YES --env VK_ICD_FILENAMES=$ICD \
  --env OMNIGIBSON_DATA_PATH=/data/asingal2/oopsieverse/b1k_tp_data --env PYTHONUNBUFFERED=1 \
  --env HF_HOME=/data/asingal2/hf_cache --pwd $GNV /data/asingal2/oopsieverse/b1k_sandbox bash -c "
  LDP=\$(find $E/lib/python3.10/site-packages/isaacsim -maxdepth 3 -type d -name bin 2>/dev/null | tr '\n' ':')
  export LD_LIBRARY_PATH=\"\$LDP:\${LD_LIBRARY_PATH:-}\"
  $E/bin/python gr00t/eval/rollout_policy.py --n_episodes $N_EPISODES --policy_client_host $SHOST \
    --policy_client_port $SPORT --max_episode_steps=$MAX_STEPS --env_name $ENV_NAME --n_action_steps 8 --n_envs 1"
RC=$?; echo "EXIT=$RC"; exit $RC
