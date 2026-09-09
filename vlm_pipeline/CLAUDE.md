# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VLM-based Semantic Safety Filter for SafeLIBERO benchmark - a research project implementing Control Barrier Functions (CBF) for safe robot manipulation using Vision-Language Models (Qwen VLM) to extract semantic safety constraints.

Three architectural methods are implemented:
- **M1 (Seg+VLM)**: Segmentation (GroundingDINO+SAM2) → VLM reasoning → CBF
- **M2 (VLM-only)**: Direct VLM perception and reasoning → CBF
- **M3 (3D+VLM)**: 3D reconstruction → VLM with spatial context → CBF

## Core Pipeline

```
1. save_vlm_inputs.py → Episode observations (RGB, depth, segmentation, metadata)
2. qwen_vlm_worker.py → VLM semantic predicates (spatial, behavioral, pose constraints)
3. cbf_construction.py → Superquadric safe sets from predicates + 3D data
4. semantic_cbf_filter.py → QP-based action certification at runtime
5. run_libero_eval.py → Full evaluation on SafeLIBERO benchmark
```

## Key Subdirectories

- `openvla-oft/`: OpenVLA policy fine-tuned for LIBERO tasks (external repo)
- `SafeLIBERO/`: SafeLIBERO benchmark suite (external repo)
- `vlsa-aegis/`: VLSA-Aegis project with perception tools (external repo)
- `vlm_inputs/`: Episode observations for VLM processing
- `results/`: VLM inference outputs and evaluation metrics
- `cbf_outputs/`: CBF parameters and visualizations
- `rollouts/`: Episode rollout videos
- `prompt_tuning_benchmark_set/`: Ground-truth constraint annotations

## Environment Setup

The project uses multiple conda environments:

### LIBERO Environment (Python 3.8)
```bash
# Setup script
bash libero_env_setup.sh

# Activate
conda activate libero_env

# Key dependencies: robosuite 1.4.1, libero, mujoco 3.2.3, numpy 1.22.4
```

### OpenVLA-LIBERO Environment (Python 3.10)
```bash
# Setup script
bash openvla_safelibero_setup.sh

# Activate
conda activate openvla_libero_merged

# Key dependencies: torch 2.2.0, transformers, prismatic-vlm
```

### Qwen VLM Environment (separate from main execution)
The VLM runs in a separate conda environment and is called via subprocess from the main pipeline.

## Common Commands

### 1. Capture VLM Inputs from SafeLIBERO Episodes
```bash
export MUJOCO_GL=egl  # Required for headless rendering on HPC
python save_vlm_inputs.py \
    --output_dir vlm_inputs/safelibero_spatial \
    --resolution 512
```
Captures 400 episodes (4 tasks × 2 safety levels × 50 episodes) with RGB, depth, segmentation, and metadata.

### 2. Run VLM Inference for Constraint Extraction

Single episode:
```bash
python qwen_vlm_worker.py \
    --method m1 \
    --input_folder vlm_inputs/safelibero_spatial/level_I/task_0/episode_00 \
    --output_json results/m1_task0_ep00.json
```

Batch processing:
```bash
python qwen_vlm_worker.py \
    --method m1 \
    --input_dir vlm_inputs/safelibero_spatial \
    --output_json results/m1_spatial_all.json
```

Available methods: `m1`, `m2`, `m3`


### 4. Run SafeLIBERO Evaluation

```bash
python run_libero_eval.py \
    --task_suite safelibero_spatial \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --num_trials_per_task 20 \
    --use_safety_filter \
    --cbf_method m1
```

## Architecture Details

### VLM Constraint Types
The VLM extracts three types of semantic safety constraints:

1. **Spatial Relationships** (`S_r`): Which spatial configurations are unsafe
   - Relationships: `"above"`, `"below"`, `"around in front of"`, `"around behind"`
   - Example: `["moka_pot_obstacle", ["above", "around in front of"]]`

2. **Behavioral Constraints** (`S_b`): Objects requiring cautious approach
   - Example: `["end_effector", ["caution"]]`
   - Implemented via class-K∞ function modulation: `α_cautious(h) = 0.25 * h²` vs default `α(h) = h²`

3. **Pose Constraints** (`S_T`): Orientation restrictions
   - Example: `["end_effector", ["rotation lock"]]`
   - Prevents spilling when holding liquids


Runs at 20 Hz during episode rollouts. VLM inference runs only once at episode start to extract constraints.

## Important Design Decisions

### Image Resolution
- Policy input: 256×256 (standard for OpenVLA-OFT)
- VLM perception input: 512×512 (higher quality for constraint extraction)
- These are kept separate to avoid conflating policy and safety filter perception needs

### Multi-Prompt Strategy
Following Brunke et al. (RA-L 2025), the VLM is queried separately for each constraint type with N=5 repetitions and majority voting. This improves precision (60% vs 29% single-prompt) and recall (99% vs 78%).

### Ground Truth Fallback
When segmentation fails, the system can fall back to ground-truth object positions from MuJoCo `sim.data` to isolate VLM reasoning quality from perception errors. Use `--use_gt_perception` flag.

### Workspace Bounds
```python
WORKSPACE_BOUNDS = {
    "x_min": -0.5, "x_max": 0.5,
    "y_min": -0.3, "y_max": 0.6,
    "z_table": 0.81,  # Table surface height
    "z_max": 1.4,     # Ceiling for "above" extension
}
```

## File Formats

### VLM Output JSON
```json
{
  "level_I/task_0/episode_00": {
    "description": "Pick up the black bowl between the plate and the ramekin",
    "end_object": "black bowl",
    "objects": [
      ["moka_pot_obstacle", ["above", "around in front of"]],
      ["plate", []],
      ["end_effector", ["caution", "rotation lock"]]
    ]
  }
}
```

### CBF Parameters JSON
```json
{
  "constraints": [
    {
      "object": "moka_pot_obstacle",
      "relationship": "above",
      "type": "ellipsoid",
      "params": {
        "center": [0.15, 0.08, 0.82],
        "scales": [0.06, 0.06, 0.35],
        "epsilon1": 1.0,
        "epsilon2": 1.0
      },
      "h_at_eef": 2.31
    }
  ],
  "behavioral": {"caution": true},
  "pose": {"rotation_lock": false}
}
```

## Debugging and Validation

### Check VLM Outputs
Compare against ground truth in `prompt_tuning_benchmark_set/`:
```bash
python benchmark_prompts.py \
    --vlm_json results/m1_spatial_all.json \
    --gt_dir prompt_tuning_benchmark_set
```

### Visualize CBF Constraints
Output includes:
- `vis_3d.html`: Interactive 3D view of superquadric + point cloud
- `vis_slice_z*.png`: 2D heatmap at fixed z-height (red=unsafe, green=safe)
- `vis_rgb_overlay.png`: h=0 boundary projected onto camera image
- `vis_dashboard.png`: Multi-constraint summary figure

### Numerical Validation
For correct CBF:
- All extended point cloud points should have `h ≤ 0.1`
- Current end-effector should have `h > 0` (starts safe)
- Gradient `∇h` should be nonzero at boundary

### Episode Observation Keys
After `env.step()`, observations include:
- `agentview_image`: (512, 512, 3) uint8
- `agentview_depth`: (512, 512, 1) float32 (z-buffer)
- `agentview_segmentation_instance`: (512, 512, 1) int32
- `robot0_eef_pos`: (3,) float64 - end-effector position
- `{object_name}_pos`: (3,) float64 - object positions

Camera parameters extracted via:
```python
from robosuite.utils.camera_utils import (
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix,
    get_real_depth_map,
)
```

## Running on Bridges2 HPC

Always set `MUJOCO_GL=egl` for headless rendering:
```bash
export MUJOCO_GL=egl
```

Typical SLURM job:
```bash
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --mem=32GB

module load anaconda3
conda activate openvla_libero_merged
export MUJOCO_GL=egl

python run_libero_eval.py [args]
```

## Evaluation Metrics

- **Violation Rate (VR)**: % of timesteps with safety violations
- **Task Success Rate (SR)**: % of episodes completing task goal
- **Filter Activation Rate (FAR)**: % of timesteps where CBF modifies action
- **Action Deviation (AD)**: Mean `||u_cert - u_cmd||` when filter is active
- **Predicate Accuracy (PA)**: Precision/recall vs ground truth benchmark

## Related Documentation

- `readme.md`: Implementation plan for `save_vlm_inputs.py`
- `vlm_pipeline_readme.md`: Full design document for VLM safety filter
- `CBF_construction.md`: Detailed CBF construction and visualization guide
- `openvla-oft/LIBERO.md`: OpenVLA fine-tuning on LIBERO
- `SafeLIBERO/README.md`: SafeLIBERO benchmark documentation
