#!/bin/bash
# Start a pi0.5 policy server (vlsa-aegis/openpi serve_policy, LIBERO I/O config) and wait until it listens.
#   source run_pi05_server.sh pi05_libero 8000      # or pi05_base
# pi05_base is served with the pi05_libero config and the LIBERO norm stats, as in
# slurm/safelibero/eval_pi05base_safelibero.slurm.  Sets PI_SERVER_PID; the caller kills it on exit.
#   ./run_pi05_server.sh download    # fetch both checkpoints into $OPENPI_DATA_HOME (no GPU needed)
CKPT_NAME=${1:-pi05_libero}
PORT=${2:-8000}
OP=/home/asingal2/safety/situational_safety_for_embodied_agents/vlsa-aegis/openpi
PYO=/data/asingal2/envs/openpi/bin/python
export OPENPI_DATA_HOME=/data/asingal2/openpi_cache
PI_CKPTS=$OPENPI_DATA_HOME/openpi-assets/checkpoints

if [ "$CKPT_NAME" = download ]; then
  for c in pi05_libero pi05_base; do
    $PYO -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/$c'))"
  done
  # pi05_base has no LIBERO norm stats: link pi05_libero's (same transforms as the reference eval)
  mkdir -p $PI_CKPTS/pi05_base/assets
  [ -e $PI_CKPTS/pi05_base/assets/physical-intelligence ] || ln -s $PI_CKPTS/pi05_libero/assets/physical-intelligence $PI_CKPTS/pi05_base/assets/physical-intelligence
  ls -la $PI_CKPTS/*/assets
  exit 0
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}
LOG=/data/asingal2/slurm_logs/pi05_server_${SLURM_JOB_ID:-local}_$PORT.log
(cd $OP && $PYO scripts/serve_policy.py --port $PORT policy:checkpoint --policy.config=pi05_libero \
    --policy.dir=$PI_CKPTS/$CKPT_NAME > $LOG 2>&1) &
PI_SERVER_PID=$!
for i in $(seq 1 180); do
  if (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then echo "pi0.5 server ($CKPT_NAME) up on $PORT after $((i * 5)) s"; break; fi
  if ! kill -0 $PI_SERVER_PID 2>/dev/null; then echo "pi0.5 server died; log:"; tail -30 $LOG; return 1 2>/dev/null || exit 1; fi
  sleep 5
done
