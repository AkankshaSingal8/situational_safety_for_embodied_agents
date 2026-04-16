# Minimal Semantic CBF Demo

A clean, educational implementation of VLM-based semantic safety filters using Control Barrier Functions (CBFs) with ellipsoid safe sets, following Brunke et al.'s paper "Semantically Safe Robot Manipulation."

## Overview

This minimal implementation demonstrates the core pipeline:

```
VLM JSON → GT Positions → Point Cloud → Extension → Ellipsoid Fit → CBF → Visualization
```

## Features

- **Minimal dependencies**: Only numpy and matplotlib
- **Educational**: Clear, commented code (~350 lines)
- **Fast**: No optimization loops, runs in <1 second
- **Standalone**: No dependencies on existing CBF code

## Usage

```bash
# Activate the conda environment with matplotlib
conda activate openvla_libero_merged

# Run the demo
python minimal_cbf_demo.py
```

## Output

The script produces:

1. **Console output**: CBF safety report for all constraints
2. **Visualization**: `cbf_demo_output.png` showing 3D scene with ellipsoid safe sets

### Example Output

```
======================================================================
MINIMAL SEMANTIC CBF DEMO
======================================================================

[1/4] Loading VLM predictions and ground-truth metadata...
  Loaded 6 in-workspace objects
  Robot EEF at: [-0.211, -0.011, 1.174]

[2/4] Generating point clouds and fitting ellipsoids...
  Processing 24 spatial constraints...
  Built 24 CBF constraints

[3/4] CBF Safety Report
======================================================================
Object                         | Relationship         | h(x_ee)    | Status
----------------------------------------------------------------------
moka_pot_obstacle_1            | above                |  +12.385   | ✓ SAFE
...
----------------------------------------------------------------------
Overall Safety: SAFE ✓
  h_min = 12.385  |  h_max = 64.263

[4/4] Generating 3D visualization...
Visualization saved to: cbf_demo_output.png
```

## Implementation Details

### Extension Strategies

Following the paper's approach:

| Relationship | Geometric Meaning | Extension Method |
|--------------|-------------------|------------------|
| `above` | Unsafe above object | Extend point cloud from object z to `z_max=1.4m` (workspace ceiling) |
| `below` | Unsafe below object | Extend from `z_table=0.81m` to object z |
| `around in front of` | Unsafe in front (-y) | Add points shifted by `[0, -0.35, 0]` m |
| `around behind` | Unsafe behind (+y) | Add points shifted by `[0, +0.35, 0]` m |

### Ellipsoid Fitting

Simple statistics-based approach (no optimization):

```python
center = mean(extended_points)
semi_axes = std(extended_points) * 2.5  # 2.5σ coverage
```

### CBF Formula

```python
h(x) = Σᵢ ((xᵢ - cᵢ) / sᵢ)² - 1
```

- **h > 0**: Safe (outside ellipsoid)
- **h = 0**: Boundary
- **h < 0**: Violated (inside unsafe region)

## Code Structure

```python
# 1. Constants
WORKSPACE_BOUNDS = {...}

# 2. Data Loading
load_data(vlm_json_path, metadata_path)

# 3. Point Cloud Generation
generate_sphere_pointcloud(center, radius, n_points)

# 4. Spatial Extension
extend_pointcloud(points, relationship)

# 5. Ellipsoid Fitting
fit_ellipsoid(points)

# 6. CBF Evaluation
evaluate_cbf(x_ee, center, semi_axes)

# 7. Visualization
visualize_3d_scene(constraints, objects, eef_pos, task_desc, output_path)
```

## Input Files

- **VLM JSON**: `results/m1_task0_ep00.json`
  - VLM-predicted spatial relationships
  - Format: `{"single": {"objects": [[obj_name, [relationships]]}}`

- **Metadata**: `vlm_inputs/safelibero_spatial/level_I/task_0/episode_00/metadata.json`
  - Ground-truth object positions
  - Robot end-effector state
  - Task description

## Visualization

The 3D visualization shows:

- **Green ellipsoid wireframes**: Safe sets (h > 0.1)
- **Yellow wireframes**: Marginal constraints (-0.1 < h < 0.1)
- **Red wireframes**: Violated constraints (h < -0.1)
- **Gray diamonds**: Ground-truth object positions
- **Cyan star**: Robot end-effector

## Extensions

This minimal implementation can be extended with:

1. **Depth-based point clouds**: Replace sphere sampling with actual depth reconstruction
2. **Superquadric shapes**: Add ε₁, ε₂ parameters for tighter fits
3. **QP-based safety filter**: Add `certify_action(u_cmd)` function
4. **Runtime integration**: Embed in robot control loop at 20 Hz

## Comparison with Production Code

| Feature | Minimal (this) | Production (cbf_superquadric.py) |
|---------|----------------|----------------------------------|
| Lines of code | ~350 | ~800 |
| Fitting method | 2.5σ statistics | scipy.optimize.minimize |
| Shape | Ellipsoid only | Full superquadric (ε₁, ε₂) |
| Visualization | matplotlib PNG | plotly interactive HTML |
| Runtime | <1 second | ~3 seconds |
| Purpose | Educational | Research deployment |

## Verification

Expected results for `episode_00`:

- **Objects**: 6 in-workspace objects
- **Constraints**: 24 spatial (6 objects × 4 relationships)
- **Safety status**: All SAFE (robot starts in safe configuration)
- **h_min**: ~12-15 (moka_pot_obstacle has smallest margin)

## References

- Brunke et al., "Semantically Safe Robot Manipulation", IEEE RA-L 2025
- Paper PDF: `Semantically_Safe_Robot_Manipulation_From_Semantic_Scene_Understanding_to_Motion_Safeguards.pdf`
- CLAUDE.md: Project documentation
