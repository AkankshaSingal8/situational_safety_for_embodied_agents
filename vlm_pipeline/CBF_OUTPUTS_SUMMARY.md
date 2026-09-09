# CBF Demo Outputs Summary

## Generated Files

### 1. `minimal_cbf_demo.py` (Original - Matplotlib)
**Output**: `cbf_demo_output.png` (709 KB)
- Static 3D visualization
- Shows superquadric ellipsoids only
- Fast generation (<1 second)
- Good for quick verification

**To run**:
```bash
conda activate openvla_libero_merged
python minimal_cbf_demo.py
```

### 2. `minimal_cbf_demo_interactive.py` ⭐ **RECOMMENDED**
**Output**: `cbf_demo_interactive.html` (6.5 MB)
- **Interactive 3D visualization** (rotate, zoom, pan)
- **Point cloud** from depth map (21,593 points)
- **Superquadric ellipsoids** overlaid (24 constraints)
- **Ground-truth object markers**
- **Toggle layers** via legend
- Dark theme matching build_3d_map.py

**To run**:
```bash
python minimal_cbf_demo_interactive.py
```

**To view**:
```bash
# Option 1: Direct browser
firefox cbf_demo_interactive.html

# Option 2: Python HTTP server
python -m http.server 8000
# Then open: http://localhost:8000/cbf_demo_interactive.html
```

## Visualization Features

### Interactive Controls (HTML version)
- **Rotate**: Click and drag
- **Zoom**: Scroll wheel
- **Pan**: Right-click and drag
- **Toggle layers**: Click legend items to show/hide
  - Point Cloud (gray, semi-transparent)
  - Ground-truth objects (colored diamonds)
  - CBF ellipsoids (green = safe, yellow = marginal, red = violated)
  - Robot EEF (cyan cross)

### Color Coding

| Element | Color | Meaning |
|---------|-------|---------|
| Point Cloud | Gray | Depth-reconstructed scene |
| Ellipsoids (Green) | rgba(50,200,50,0.25) | h > 0.1 (SAFE) |
| Ellipsoids (Yellow) | rgba(255,200,50,0.35) | -0.1 < h < 0.1 (marginal) |
| Ellipsoids (Red) | rgba(255,50,50,0.45) | h < -0.1 (VIOLATED) |
| Robot EEF | Cyan | End-effector position |
| Objects | Various | Object-specific colors |

## Visualization Comparison

| Feature | PNG (matplotlib) | HTML (plotly) |
|---------|------------------|---------------|
| **Interactive** | ✗ | ✓ |
| **Point cloud** | ✗ | ✓ (21k points) |
| **Ellipsoids** | ✓ (wireframe) | ✓ (surface) |
| **File size** | 709 KB | 6.5 MB |
| **Generation time** | <1 sec | ~3 sec |
| **Dependencies** | matplotlib | plotly |
| **Best for** | Quick checks | Detailed analysis |

## Data Summary (Episode 00)

### Point Cloud Statistics
- **Total points**: 21,593
- **Downsampling**: 3 (from 512×512 depth map)
- **Workspace**: x ∈ [-0.5, 0.5], y ∈ [-0.3, 0.6], z ∈ [0.81, 1.4]

### CBF Constraints
- **Objects**: 6 in-workspace
  - akita_black_bowl_1
  - akita_black_bowl_2
  - cookies_1
  - glazed_rim_porcelain_ramekin_1
  - plate_1
  - moka_pot_obstacle_1

- **Relationships per object**: 4
  - above
  - below
  - around in front of
  - around behind

- **Total constraints**: 24 (6 objects × 4 relationships)

### Safety Status
- **Overall**: SAFE ✓
- **h_min**: 9.878 (moka_pot_obstacle)
- **h_max**: 63.928 (akita_black_bowl_2)
- **All constraints**: h > 0 (safe)

## Technical Details

### Extension Strategies Implemented

```python
# Above: extend to workspace ceiling
if relationship == "above":
    z_range = np.linspace(z_obj, 1.4, 10)
    extended = np.vstack([points + [0,0,dz] for dz in ...])

# Below: extend to table surface  
elif relationship == "below":
    z_range = np.linspace(0.81, z_obj, 10)
    extended = np.vstack([points + [0,0,dz] for dz in ...])

# Around front/behind: horizontal extension
elif relationship == "around in front of":
    extended = np.vstack([points, points + [0, -0.35, 0]])
```

### Ellipsoid Fitting

```python
center = np.mean(extended_points, axis=0)
semi_axes = np.std(extended_points, axis=0) * 2.5  # 2.5σ
semi_axes = np.maximum(semi_axes, 0.03)  # min 3cm
```

### CBF Evaluation

```python
h(x) = Σᵢ ((xᵢ - cᵢ) / sᵢ)² - 1

where:
  x = end-effector position
  c = ellipsoid center
  s = semi-axes lengths
  h > 0: safe (outside unsafe region)
  h < 0: violated (inside unsafe region)
```

## Usage Examples

### Example 1: Verify Safety at Current Pose
```bash
python minimal_cbf_demo_interactive.py
# Check console for h_min value
# Open HTML to visually inspect clearances
```

### Example 2: Analyze Specific Constraint
```bash
python minimal_cbf_demo_interactive.py
# Open HTML
# Click on "moka_pot_obstacle - above" in legend
# Observe ellipsoid shape and EEF clearance
```

### Example 3: Compare Extension Strategies
```bash
# Open HTML
# Toggle different relationships for same object
# Compare "above" vs "below" ellipsoid shapes
```

## Files in This Directory

```
minimal_cbf_demo.py                  # Matplotlib version
minimal_cbf_demo_interactive.py      # Plotly interactive version
cbf_demo_output.png                  # Static visualization
cbf_demo_interactive.html            # Interactive visualization ⭐
MINIMAL_CBF_README.md                # Original implementation guide
CBF_OUTPUTS_SUMMARY.md               # This file
```

## Next Steps

### For Research
1. Test on more episodes: modify paths in `main()` to load different episodes
2. Integrate with policy: call `evaluate_cbf()` during rollouts
3. Add QP filter: implement action certification `certify_action(u_cmd)`

### For Deployment
1. Switch to production code: `cbf_superquadric.py` for tighter fits
2. Enable depth-based point clouds: replace sphere sampling
3. Add runtime optimization: use scipy.optimize for ellipsoid fitting

## References

- Paper: Brunke et al., "Semantically Safe Robot Manipulation", IEEE RA-L 2025
- build_3d_map.py: Original point cloud visualization
- cbf_construction.py: Production CBF implementation
- CLAUDE.md: Full project documentation
