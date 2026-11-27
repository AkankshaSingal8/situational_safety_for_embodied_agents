#!/bin/bash
# Official RoboCasa365 pi0 (robocasa-benchmark/openpi fork, 75k steps on pretrain_human300).
set -euo pipefail
D=/data/asingal2/openpi_cache/robocasa365
mkdir -p $D
/data/asingal2/envs/openpi/bin/python -c "
from huggingface_hub import snapshot_download
p=snapshot_download('robocasa/robocasa365_checkpoints', allow_patterns=['pi0/*'], local_dir='$D', max_workers=8)
print(p)"
du -sh $D/pi0/*/*/*
echo DL_DONE
