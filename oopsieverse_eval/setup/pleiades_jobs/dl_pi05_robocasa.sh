#!/bin/bash
# Official RoboCasa365 pi0.5 (robocasa-benchmark/openpi@ca4c6d7, config pi05_pretrain_human300, 75k steps). Weights + norm stats only.
set -euo pipefail
D=/data/asingal2/openpi_cache/robocasa365
/data/asingal2/envs/openpi/bin/python -c "
from huggingface_hub import snapshot_download
P='pi05_pretrain_human300/multitask_learning/75000'
print(snapshot_download('robocasa/robocasa365_checkpoints', allow_patterns=[P+'/params/*',P+'/assets/*',P+'/_CHECKPOINT_METADATA'], local_dir='$D', max_workers=8))"
du -sh $D/pi05_pretrain_human300/multitask_learning/75000/*
echo DL_DONE
