# Visualization Tuning Guide

## Quick Parameter Reference

You can easily adjust the visualization by changing constants at the top of `minimal_cbf_demo_interactive.py`:

```python
# Lines 32-38 in minimal_cbf_demo_interactive.py
EXTENSION_DISTANCE = 0.15  # ← ADJUST THIS (meters)
DEFAULT_RADIUS = 0.05

# Visualization parameters
ELLIPSOID_OPACITY = 0.15   # ← ADJUST THIS (0-1)
ELLIPSOID_MODE = "surface" # ← ADJUST THIS ("surface" or "wireframe")
POINT_CLOUD_OPACITY = 0.6  # ← ADJUST THIS (0-1)
```

## Parameter Guide

### 1. Extension Distance (Around Margins)

**Parameter**: `EXTENSION_DISTANCE`  
**Default**: 0.15 m (reduced from 0.35 m)  
**Range**: 0.05 - 0.50 m

Controls how far the "around in front of" and "around behind" constraints extend horizontally.

```python
EXTENSION_DISTANCE = 0.10  # Tight fit - smaller safe zones
EXTENSION_DISTANCE = 0.15  # Balanced (current)
EXTENSION_DISTANCE = 0.25  # Conservative - larger margins
```

**Effect**:
- **Lower** (0.05-0.10): Tighter ellipsoids, less conservative, smaller safe zones
- **Higher** (0.30-0.50): Larger ellipsoids, more conservative, bigger safe zones

### 2. Ellipsoid Opacity

**Parameter**: `ELLIPSOID_OPACITY`  
**Default**: 0.15 (reduced from 0.35)  
**Range**: 0.05 - 0.50

Controls transparency of ellipsoid surfaces.

```python
ELLIPSOID_OPACITY = 0.05  # Nearly invisible - best point cloud visibility
ELLIPSOID_OPACITY = 0.15  # Light overlay (current) - good balance
ELLIPSOID_OPACITY = 0.30  # Solid - emphasizes constraints
```

**Effect**:
- **Lower** (0.05-0.10): More transparent, point cloud very visible, ellipsoids subtle
- **Higher** (0.30-0.50): More opaque, ellipsoids prominent, point cloud harder to see

**Recommended**: 0.10-0.20 for best balance

### 3. Ellipsoid Mode

**Parameter**: `ELLIPSOID_MODE`  
**Options**: `"surface"` or `"wireframe"`

Changes rendering style of ellipsoids.

```python
ELLIPSOID_MODE = "surface"    # Smooth surfaces (current)
ELLIPSOID_MODE = "wireframe"  # Mesh edges only - better visibility
```

**Surface Mode**:
- ✓ Smooth, professional appearance
- ✓ Clear safe/unsafe boundaries
- ✗ Can obscure point cloud even with low opacity

**Wireframe Mode**:
- ✓ Excellent point cloud visibility
- ✓ Shows ellipsoid structure clearly
- ✓ Less file size
- ✗ Can look cluttered with many constraints

### 4. Point Cloud Opacity

**Parameter**: `POINT_CLOUD_OPACITY`  
**Default**: 0.6  
**Range**: 0.3 - 1.0

Controls visibility of depth-reconstructed point cloud.

```python
POINT_CLOUD_OPACITY = 0.4  # Subtle background
POINT_CLOUD_OPACITY = 0.6  # Visible but not dominant (current)
POINT_CLOUD_OPACITY = 0.9  # Prominent scene geometry
```

**Effect**:
- **Lower** (0.3-0.4): Point cloud fades into background, emphasizes ellipsoids
- **Higher** (0.7-1.0): Point cloud very visible, shows detailed scene structure

## Recommended Configurations

### Configuration 1: Balanced (Current Default)
**Best for**: General analysis, presentations

```python
EXTENSION_DISTANCE = 0.15
ELLIPSOID_OPACITY = 0.15
ELLIPSOID_MODE = "surface"
POINT_CLOUD_OPACITY = 0.6
```

**Result**: Clear point cloud, subtle ellipsoids, good overall view

---

### Configuration 2: Emphasize Point Cloud
**Best for**: Analyzing scene geometry, debugging depth reconstruction

```python
EXTENSION_DISTANCE = 0.15
ELLIPSOID_OPACITY = 0.08     # Very transparent
ELLIPSOID_MODE = "wireframe" # Edges only
POINT_CLOUD_OPACITY = 0.85   # Very visible
```

**Result**: Point cloud dominates, ellipsoids as light overlay

---

### Configuration 3: Emphasize Constraints
**Best for**: CBF analysis, safety verification, understanding unsafe regions

```python
EXTENSION_DISTANCE = 0.20
ELLIPSOID_OPACITY = 0.30     # More opaque
ELLIPSOID_MODE = "surface"
POINT_CLOUD_OPACITY = 0.35   # Subtle background
```

**Result**: Ellipsoids prominent, clear safe/unsafe boundaries

---

### Configuration 4: Wireframe Style
**Best for**: Technical illustrations, maximum visibility of all layers

```python
EXTENSION_DISTANCE = 0.15
ELLIPSOID_OPACITY = 0.20
ELLIPSOID_MODE = "wireframe"
POINT_CLOUD_OPACITY = 0.7
```

**Result**: Clean wireframe ellipsoids, clear point cloud

---

## Quick Comparison

| Setting | Point Cloud Visibility | Ellipsoid Clarity | Best Use Case |
|---------|------------------------|-------------------|---------------|
| Config 1 (Current) | ★★★★☆ | ★★★★☆ | General purpose |
| Config 2 | ★★★★★ | ★★☆☆☆ | Depth analysis |
| Config 3 | ★★☆☆☆ | ★★★★★ | Safety analysis |
| Config 4 | ★★★★★ | ★★★★☆ | Technical docs |

## How to Apply Changes

1. **Edit the file**:
   ```bash
   nano minimal_cbf_demo_interactive.py
   # or
   vim minimal_cbf_demo_interactive.py
   ```

2. **Modify lines 32-38** with desired values

3. **Re-run**:
   ```bash
   python minimal_cbf_demo_interactive.py
   ```

4. **Refresh browser** to view new `cbf_demo_interactive.html`

## Visual Effects Summary

### Extension Distance Impact
```
0.10m: Tight ellipsoids (smaller unsafe regions)
       [●]  small margin around object

0.15m: Balanced (current)
       [ ● ] medium margin

0.25m: Conservative
       [  ●  ] large margin
```

### Opacity Impact
```
Ellipsoid Opacity 0.05:  ░░░  (barely visible)
                  0.15:  ▒▒▒  (current - subtle)
                  0.30:  ▓▓▓  (prominent)

Point Cloud   0.4:      ░░░  (background)
              0.6:      ▒▒▒  (current - visible)
              0.9:      ▓▓▓  (dominant)
```

## Interactive Controls in Browser

Once the HTML is open, you can also:

1. **Toggle layers**: Click legend items to show/hide
   - Hide "Point Cloud (scene)" to see just ellipsoids
   - Hide individual constraints to reduce clutter
   - Hide all ellipsoids to see just the point cloud

2. **Camera controls**:
   - **Rotate**: Click + drag
   - **Zoom**: Scroll wheel
   - **Pan**: Right-click + drag
   - **Reset**: Double-click

3. **Screenshot**: Click camera icon in top-right of plot

## Troubleshooting

### "Ellipsoids too dominant, can't see objects"
→ Reduce `ELLIPSOID_OPACITY` to 0.10 or lower  
→ Try `ELLIPSOID_MODE = "wireframe"`

### "Point cloud too dim"
→ Increase `POINT_CLOUD_OPACITY` to 0.8-0.9

### "Too many overlapping ellipsoids"
→ Open HTML and click legend to hide some constraints  
→ Focus on one object at a time

### "Extension margins too large/small"
→ Adjust `EXTENSION_DISTANCE`:
  - Too large: reduce to 0.10-0.12
  - Too small: increase to 0.20-0.25

## Advanced: Dynamic Parameter Selection

You can also modify the script to accept command-line arguments:

```python
# Add at top of main():
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--opacity", type=float, default=0.15)
parser.add_argument("--extension", type=float, default=0.15)
args = parser.parse_args()

ELLIPSOID_OPACITY = args.opacity
EXTENSION_DISTANCE = args.extension
```

Then run:
```bash
python minimal_cbf_demo_interactive.py --opacity 0.10 --extension 0.20
```

## Results

### Before (v1 - original)
- Extension: 0.35 m
- Ellipsoid opacity: 0.35
- Point cloud opacity: 0.4
- **Issue**: Ellipsoids too dominant, objects hard to see

### After (v2 - current)
- Extension: 0.15 m ✓ (reduced margins)
- Ellipsoid opacity: 0.15 ✓ (more transparent)
- Point cloud opacity: 0.6 ✓ (more visible)
- **Result**: Clear view of both point cloud AND ellipsoids

## Performance Notes

- **File size**: ~6.5 MB regardless of opacity settings
- **Generation time**: ~2-3 seconds
- **Browser performance**: Smooth even with 21k points + 24 ellipsoids
- **Wireframe mode**: Slightly faster rendering in browser
