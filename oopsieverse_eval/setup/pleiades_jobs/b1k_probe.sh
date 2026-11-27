#!/bin/bash
/data/asingal2/envs/apptainer/bin/apptainer exec --nv --bind /data --bind /home/asingal2   --env OMNIGIBSON_HEADLESS=1 --env OMNI_KIT_ACCEPT_EULA=YES --env VK_ICD_FILENAMES=/data/asingal2/oopsieverse/tools/nvidia_icd.json   --env OMNIGIBSON_DATA_PATH=/data/asingal2/oopsieverse/b1k_tp_data --env PYTHONUNBUFFERED=1   --env HF_HOME=/data/asingal2/hf_cache --pwd /data/asingal2/oopsieverse/externals/Isaac-GR00T-nv /data/asingal2/oopsieverse/b1k_sandbox bash -c "
  LDP=\$(find /data/asingal2/envs/groot_b1k/lib/python3.10/site-packages/isaacsim -maxdepth 3 -type d -name bin 2>/dev/null | tr '\n' ':')
  export LD_LIBRARY_PATH=\"\$LDP:\${LD_LIBRARY_PATH:-}\"
  /data/asingal2/envs/groot_b1k/bin/python /home/asingal2/safety/wt-oopsieverse-eval/oopsieverse_eval/eval/run_groot_b1k_oopsie.py --task $1 --probe --host x --port 1 --results_dir /tmp/probe"
