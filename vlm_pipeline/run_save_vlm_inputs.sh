#!/usr/bin/env bash
# Required for headless rendering on HPC (Bridges2)
export MUJOCO_GL=egl

python vlm_pipeline/save_vlm_inputs.py \
    `# output root; suite/level/task/episode folders are created inside` \
    --output_dir vlm_inputs/safelibero_spatial \
    `# allowed: safelibero_spatial` \
    --task_suite safelibero_spatial \
    `# allowed: I  II  (space-separated for multiple, e.g. --safety_levels I II)` \
    --safety_levels I \
    `# allowed: 0 1 2 3  (space-separated; omit flag entirely to run all 4 tasks)` \
    --task_ids 1 2 3 \
    `# allowed: 1-50  (collects episodes 0..N-1; max 50 pre-computed states per task)` \
    --num_episodes 10 \
    `# allowed: any positive int; typically 256 or 512` \
    --resolution 512
