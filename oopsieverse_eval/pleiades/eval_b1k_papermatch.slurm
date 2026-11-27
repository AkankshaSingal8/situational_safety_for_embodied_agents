#!/bin/bash
# Paper-matched BEHAVIOR-1K eval on pleiades: the 3 Table-I B1K tasks present in the
# proj/safemanibench registry (add_firewood, wipe_counter, food_in_microwave; "attach camera"
# has no equivalent), 30 trials each, one array index per task (one Isaac Sim boot per task).
# MUST run on pleiades-1-10 (Quadro RTX 6000, driver 580): Isaac Sim 4.5 segfaults in
# librtx.scenedb on the Ada nodes' driver 610.43 (plain SimulationApp too; verified 2026-11-15).
#   CKPT=pi0_libero|pi0_base|pi05_base sbatch -J ov_b1k_$CKPT -p pleiades -w pleiades-1-10 \
#        --array=0-2 --time=12:00:00 -c 16 --mem=96G --gres=gpu:2 \
#        -o /data/asingal2/slurm_logs/%x_%A_%a.out oopsieverse_eval/pleiades/eval_b1k_papermatch.slurm
# GPU 0: openpi server. GPU 1: Isaac Sim client inside the Ubuntu 24.04 sandbox (glibc 2.39).
set -uo pipefail
CKPT=${CKPT:?set CKPT}
NUM_TRIALS=${NUM_TRIALS:-30}
TASKS=(add_firewood wipe_counter food_in_microwave)
TASK=${TASK:-${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}}
case $CKPT in pi0_*) CFG=pi0_libero ;; pi05_*) CFG=pi05_libero ;; *) echo "bad CKPT"; exit 2 ;; esac
EVAL_DIR=/home/asingal2/safety/wt-oopsieverse-eval/oopsieverse_eval
OV=/home/asingal2/safety/ext/oopsieverse
OP=/home/asingal2/safety/situational_safety_for_embodied_agents/vlsa-aegis/openpi
SB=/data/asingal2/oopsieverse/b1k_sandbox
E=/data/asingal2/envs/oopsieverse_b1k
export PATH=/data/asingal2/envs/apptainer/bin:$PATH
export OPENPI_DATA_HOME=/data/asingal2/openpi_cache
PORT=$((8800 + SLURM_JOB_ID % 500))
OUT=${OUT:-/data/asingal2/oopsieverse/results/b1k_papermatch/$CKPT/$TASK}
mkdir -p "$OUT"
echo "host $(hostname) ckpt=$CKPT cfg=$CFG task=$TASK port=$PORT trials=$NUM_TRIALS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
[ -e $EVAL_DIR/external/oopsieverse/oopsiebench ] || { rmdir $EVAL_DIR/external/oopsieverse 2>/dev/null; ln -sfn $OV $EVAL_DIR/external/oopsieverse; }

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6  # 24 GB cards on 1-10
(cd $OP && CUDA_VISIBLE_DEVICES=0 /data/asingal2/envs/openpi/bin/python scripts/serve_policy.py --port $PORT policy:checkpoint \
   --policy.config=$CFG --policy.dir=$OPENPI_DATA_HOME/openpi-assets/checkpoints/$CKPT) > $OUT/server_${SLURM_JOB_ID}.log 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null' EXIT
for i in $(seq 1 360); do
  (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && { echo "server up after $((i*5))s"; break; }
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -40 $OUT/server_${SLURM_JOB_ID}.log; exit 1; }
  sleep 5
done

# Vulkan ICD that resolves libGLX_nvidia from the libs `apptainer --nv` mounts (no host-path binds).
BINDS="--bind /data --bind /home/asingal2"
ICD=/data/asingal2/oopsieverse/tools/nvidia_icd.json
echo "ICD=$ICD"
LDCMD='LDP=$(find '$E'/lib/python3.10/site-packages/isaacsim -maxdepth 3 -type d -name bin 2>/dev/null | tr "\n" ":"); export LD_LIBRARY_PATH="$LDP:${LD_LIBRARY_PATH:-}"'
# cwd = oopsieverse root: some B1K task reset hooks open init-state pickles by relative path.
apptainer exec --nv $BINDS --env OMNIGIBSON_HEADLESS=1 --env OMNI_KIT_ACCEPT_EULA=YES \
   ${ICD:+--env VK_ICD_FILENAMES=$ICD} --env CUDA_VISIBLE_DEVICES=1 --env PYTHONUNBUFFERED=1 $SB \
   bash -c "$LDCMD; source /opt/miniconda3/etc/profile.d/conda.sh && conda activate $E && cd $OV && \
     python -u $EVAL_DIR/eval/run_b1k_pi05_eval.py --tasks $TASK --num_trials $NUM_TRIALS \
       --host localhost --port $PORT --results_dir $OUT --seed 7"
RC=$?; echo "EXIT=$RC"; exit $RC
