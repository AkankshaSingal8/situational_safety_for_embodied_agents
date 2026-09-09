# From VLM Predicates to Control Barrier Functions: Construction and Visualization Plan

## 0. Where We Are in the Pipeline

```
[DONE]  save_vlm_inputs.py    → episode folders with RGB, depth, seg, metadata
[DONE]  qwen_vlm_worker.py    → VLM predicate JSON (M1/M2/M3)
[THIS]  cbf_construction.py   → superquadric safe sets from predicates + 3D data
[NEXT]  safety_filter.py      → QP that certifies actions at runtime
```

The VLM JSON gives us the *what* — which (object, relationship) pairs are unsafe, which objects require caution, whether rotation is locked. This document covers turning that into the *where* — differentiable mathematical functions h(x) that define the boundary between safe and unsafe end-effector positions.

---

## 1. Inputs to CBF Construction

For each episode, we have:

| Source | Data | Used for |
|--------|------|----------|
| VLM JSON | `objects: [["plate_1", ["above"]], ...]` | Which constraints to build |
| VLM JSON | `end_effector: ["caution", "rotation lock"]` | Behavioral / pose modulation |
| `metadata.json` | `objects.plate_1.position: [0.3, 0.1, 0.82]` | Object centroid in world frame |
| `agentview_depth.npy` | 512×512 float32 metric depth | Point cloud generation |
| `agentview_seg.npy` | 512×512 int32 instance IDs | Per-object point cloud extraction |
| `camera_params.json` | Intrinsic (3×3), extrinsic (4×4) | Depth → 3D back-projection |
| `metadata.json` | `robot_state.eef_pos` | Current end-effector position |
| `metadata.json` | `geom_id_to_name` | Mapping seg IDs → object names |

---

## 2. Step-by-Step CBF Construction

### Step 1: Build Per-Object Point Clouds

For each object that has at least one spatial constraint in the VLM output:

```python
def build_object_point_cloud(obs_folder, object_name, geom_id_to_name, camera_params):
    """
    1. Load agentview_seg.npy and agentview_depth.npy
    2. Find all pixel (u, v) where seg[u,v] maps to object_name via geom_id_to_name
    3. For each such pixel, back-project to 3D:
         z = depth[u, v]
         x = (u - cx) * z / fx
         y = (v - cy) * z / fy
    4. Transform from camera frame to world frame using extrinsic matrix
    5. Return Nx3 point cloud in world coordinates
    """
```

**Fallback (M3-GT):** If segmentation is unreliable, use the ground-truth object position from `metadata.json` and approximate the object as a sphere or box with a default radius (e.g., 5cm). This isolates CBF correctness from perception errors.

### Step 2: Extend Point Cloud by Spatial Relationship

Following Brunke et al. Section IV-B.1, the spatial relationship determines *which direction* to extend the unsafe region beyond the physical object:

| Relationship | Extension | Implementation |
|---|---|---|
| `above` | Extend upward (+z) to workspace ceiling | Duplicate point cloud, set z_copy = z_max_workspace |
| `below` | Extend downward (-z) to table surface | Duplicate point cloud, set z_copy = z_table |
| `around in front of` | Extend in -y (toward robot base) | Duplicate, offset y by -extension_distance |
| `around behind` | Extend in +y (away from robot) | Duplicate, offset y by +extension_distance |
| `around` | Extend radially in xy-plane | Duplicate with offsets in ±x and ±y |

```python
def extend_point_cloud(points, relationship, workspace_bounds):
    """
    points: Nx3 original object point cloud
    relationship: string from VLM output
    workspace_bounds: dict with z_max, z_table, etc.
    
    Returns: Mx3 extended point cloud (M > N)
    """
    extended = points.copy()
    
    if relationship == "above":
        upper = points.copy()
        upper[:, 2] = workspace_bounds["z_max"]
        extended = np.vstack([points, upper])
    
    elif relationship == "below":
        lower = points.copy()
        lower[:, 2] = workspace_bounds["z_table"]
        extended = np.vstack([points, lower])
    
    # ... similar for around in front of, around behind
    
    return extended
```

### Step 3: Fit Superquadric to Extended Point Cloud

A superquadric is a smooth, differentiable shape defined by:

```
g(x; θ) = ( ((x-cx)/ax)^(2/ε2) + ((y-cy)/ay)^(2/ε2) )^(ε2/ε1) + ((z-cz)/az)^(2/ε1)
```

Where:
- `(cx, cy, cz)` = center of the superquadric
- `(ax, ay, az)` = scale (half-extents) along each axis
- `(ε1, ε2)` = shape parameters (1.0 = ellipsoid, 0.1 = box-like)

**Fitting procedure:**

```python
from scipy.optimize import minimize

def fit_superquadric(points):
    """
    Fit a superquadric to a point cloud.
    
    Parameters to optimize: cx, cy, cz, ax, ay, az, ε1, ε2
    (and optionally a rotation quaternion for oriented objects)
    
    Objective: minimize sum of (g(p_i; θ) - 1)^2 over all points p_i
    The surface of the superquadric is where g = 1.
    Points on the surface should satisfy g ≈ 1.
    
    Returns: SuperquadricParams dataclass
    """
    # Initial guess: centroid = mean, scales = std, ε = 1.0 (ellipsoid)
    center = points.mean(axis=0)
    scales = points.std(axis=0) * 2.0  # rough bounding box half-widths
    x0 = [*center, *scales, 1.0, 1.0]
    
    def residuals(params):
        cx, cy, cz, ax, ay, az, e1, e2 = params
        # Shift to local frame
        local = points - np.array([cx, cy, cz])
        # Evaluate superquadric
        term_xy = (np.abs(local[:,0]/ax)**(2/e2) + 
                   np.abs(local[:,1]/ay)**(2/e2))**(e2/e1)
        term_z = np.abs(local[:,2]/az)**(2/e1)
        g = term_xy + term_z
        return np.sum((g - 1.0)**2)
    
    bounds = [(None,None)]*3 + [(0.01,None)]*3 + [(0.05,2.0)]*2
    result = minimize(residuals, x0, bounds=bounds, method='L-BFGS-B')
    return result.x
```

**Simpler alternative for prototyping:** Skip superquadric fitting entirely. Use axis-aligned ellipsoids, which are superquadrics with ε1 = ε2 = 1.0. The center is the point cloud centroid and the scales are the standard deviations × a safety multiplier:

```python
def fit_ellipsoid(points, safety_margin=1.3):
    center = points.mean(axis=0)
    scales = points.std(axis=0) * safety_margin
    scales = np.maximum(scales, 0.03)  # minimum 3cm in any dimension
    return center, scales  # ε1 = ε2 = 1.0 implicit
```

### Step 4: Define the CBF

For each spatial constraint `i`, the CBF is:

```
h_sem,i(x_ee) = g_i(x_ee; θ_i) - 1
```

- `h > 0` → end-effector is outside the unsafe region (SAFE)
- `h = 0` → end-effector is on the boundary
- `h < 0` → end-effector is inside the unsafe region (VIOLATED)

```python
def evaluate_cbf(x_ee, sq_params):
    """
    Evaluate h_sem = g(x_ee; θ) - 1 for a single superquadric constraint.
    
    x_ee: (3,) end-effector position
    sq_params: (cx, cy, cz, ax, ay, az, e1, e2)
    
    Returns: scalar h value
    """
    cx, cy, cz, ax, ay, az, e1, e2 = sq_params
    local = x_ee - np.array([cx, cy, cz])
    
    term_xy = (np.abs(local[0]/ax)**(2/e2) + 
               np.abs(local[1]/ay)**(2/e2))**(e2/e1)
    term_z = np.abs(local[2]/az)**(2/e1)
    g = term_xy + term_z
    
    return g - 1.0
```

### Step 5: Compute CBF Gradient (for QP constraints)

The CBF-QP constraint is:

```
ḣ(q, u) = ∂h/∂x_ee · J(q) · u ≥ -α(h(q))
```

The gradient ∂h/∂x_ee is needed at runtime. For superquadrics it's analytically differentiable:

```python
def cbf_gradient(x_ee, sq_params):
    """
    Compute ∂h/∂x_ee (3-vector) analytically.
    This is the key ingredient for the QP constraint.
    """
    cx, cy, cz, ax, ay, az, e1, e2 = sq_params
    dx, dy, dz = x_ee - np.array([cx, cy, cz])
    
    # Partial derivatives of g w.r.t. x, y, z
    # (chain rule through the superquadric formula)
    # ... (see Brunke et al. supplementary for full derivation)
```

For the ellipsoid special case (ε1 = ε2 = 1.0), the gradient simplifies to:

```python
def ellipsoid_cbf_gradient(x_ee, center, scales):
    """Gradient of h = (x-cx)²/ax² + (y-cy)²/ay² + (z-cz)²/az² - 1"""
    diff = x_ee - center
    grad = 2.0 * diff / (scales ** 2)
    return grad
```

---

## 3. How to Validate CBF Correctness

The CBF is a mathematical function. "Correct" means:

1. **The unsafe region is inside the h < 0 volume** — the superquadric should tightly enclose the extended point cloud.
2. **The safe region is outside** — the current end-effector position should have h > 0 (otherwise the constraint is already violated before the filter starts).
3. **The boundary (h = 0) is smooth** — no discontinuities that would make the QP ill-conditioned.
4. **The gradient points away from the unsafe region** — so the QP knows which direction is "safer."

**Numerical checks (no visualization needed):**

```python
# Check 1: All extended point cloud points should be inside (h ≤ 0 or h ≈ 0)
for p in extended_points:
    h = evaluate_cbf(p, sq_params)
    assert h <= 0.1, f"Surface point has h={h:.3f}, should be ≤ 0"

# Check 2: Current EEF should be safe
h_eef = evaluate_cbf(eef_pos, sq_params)
assert h_eef > 0, f"EEF starts inside unsafe region! h={h_eef:.3f}"

# Check 3: Gradient should be nonzero at the boundary
grad = cbf_gradient(eef_pos, sq_params)
assert np.linalg.norm(grad) > 1e-6, "Zero gradient — QP will be degenerate"
```

---

## 4. Visualization Plan

### 4.1 Visualization 1: 3D Superquadric + Point Cloud (Primary Validation)

The most informative view. Shows whether the fitted superquadric actually encloses the right region.

**What to render:**
- Object point cloud (colored dots)
- Extended point cloud (lighter-colored dots)
- Superquadric surface (semi-transparent mesh, h = 0 isosurface)
- End-effector position (large marker)
- Color: green if h > 0 (safe), red if h < 0 (violated)

**Tool:** `matplotlib` 3D or `open3d` or `plotly`

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_cbf_3d(points, extended_points, sq_params, eef_pos):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Object point cloud
    ax.scatter(*points.T, c='blue', s=1, alpha=0.3, label='Object')
    # Extended region
    ax.scatter(*extended_points.T, c='cyan', s=1, alpha=0.1, label='Extended')
    
    # Superquadric surface (h=0 isosurface via marching cubes or parametric)
    u = np.linspace(-np.pi, np.pi, 50)
    v = np.linspace(-np.pi/2, np.pi/2, 50)
    U, V = np.meshgrid(u, v)
    cx, cy, cz, ax_, ay_, az_, e1, e2 = sq_params
    # Parametric superquadric surface
    X = cx + ax_ * np.sign(np.cos(V)) * np.abs(np.cos(V))**e1 * np.sign(np.cos(U)) * np.abs(np.cos(U))**e2
    Y = cy + ay_ * np.sign(np.cos(V)) * np.abs(np.cos(V))**e1 * np.sign(np.sin(U)) * np.abs(np.sin(U))**e2
    Z = cz + az_ * np.sign(np.sin(V)) * np.abs(np.sin(V))**e1
    ax.plot_surface(X, Y, Z, alpha=0.15, color='red')
    
    # End-effector
    h_val = evaluate_cbf(eef_pos, sq_params)
    color = 'green' if h_val > 0 else 'red'
    ax.scatter(*eef_pos, c=color, s=200, marker='*', label=f'EEF (h={h_val:.2f})')
    
    ax.set_title(f'CBF Constraint: h(x_ee) = {h_val:.3f}')
    ax.legend()
```

### 4.2 Visualization 2: 2D Slice Heatmap (Quantitative Understanding)

Fix one coordinate (e.g., z = table height + 0.15m) and plot h(x, y, z_fixed) as a 2D heatmap. This shows the safe/unsafe boundary as a contour line.

**What to render:**
- Color map: h values over a grid of (x, y) at fixed z
- Contour line at h = 0 (the CBF boundary)
- Object centroid marker
- End-effector position marker
- Green region (h > 0, safe), red region (h < 0, unsafe)

```python
def visualize_cbf_slice(sq_params, eef_pos, z_slice, workspace_bounds):
    x_range = np.linspace(workspace_bounds["x_min"], workspace_bounds["x_max"], 200)
    y_range = np.linspace(workspace_bounds["y_min"], workspace_bounds["y_max"], 200)
    X, Y = np.meshgrid(x_range, y_range)
    
    H = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            H[i, j] = evaluate_cbf(np.array([X[i,j], Y[i,j], z_slice]), sq_params)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, H, levels=50, cmap='RdYlGn')  # red=unsafe, green=safe
    plt.colorbar(label='h(x, y, z_fixed)')
    plt.contour(X, Y, H, levels=[0.0], colors='black', linewidths=2)  # boundary
    plt.scatter(eef_pos[0], eef_pos[1], c='blue', s=100, marker='*', label='EEF')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'CBF slice at z = {z_slice:.2f}m')
    plt.legend()
```

**Why this matters:** The 2D slice immediately reveals if the unsafe region shape makes sense. For an "above" constraint on a laptop, the slice at z = laptop_height + 0.1 should show a red oval over the laptop footprint, and green everywhere else.

### 4.3 Visualization 3: CBF Overlay on RGB Image (Paper-Ready)

Project the h = 0 contour from 3D back onto the 2D camera image. This is the visualization Brunke et al. use in their Fig. 3 — it shows the constraint envelopes overlaid on the actual scene.

**Steps:**
1. Generate a 3D grid of points where h ≈ 0 (isosurface extraction)
2. Project each 3D point to pixel coordinates using camera intrinsics + extrinsics
3. Draw the projected contour on the RGB image

```python
def project_cbf_on_image(rgb_image, sq_params, camera_intrinsic, camera_extrinsic):
    """
    Project the h=0 isosurface onto the camera image.
    """
    from skimage import measure
    
    # 1. Build 3D volume of h values
    x = np.linspace(-0.5, 0.5, 100)
    y = np.linspace(-0.5, 0.5, 100)
    z = np.linspace(0.7, 1.5, 60)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    H = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                H[i,j,k] = evaluate_cbf(np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]]), sq_params)
    
    # 2. Extract h=0 isosurface vertices
    verts, faces, _, _ = measure.marching_cubes(H, level=0.0)
    # Map verts from grid indices back to world coordinates
    verts_world = verts * np.array([x[1]-x[0], y[1]-y[0], z[1]-z[0]]) + np.array([x[0], y[0], z[0]])
    
    # 3. Project to image
    T_cam = camera_extrinsic  # 4x4 world-to-camera
    K = camera_intrinsic      # 3x3
    
    ones = np.ones((verts_world.shape[0], 1))
    verts_h = np.hstack([verts_world, ones])  # Nx4
    verts_cam = (T_cam @ verts_h.T).T[:, :3]  # Nx3 in camera frame
    
    # Only keep points in front of camera
    mask = verts_cam[:, 2] > 0
    verts_cam = verts_cam[mask]
    
    pixels = (K @ verts_cam.T).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    
    # 4. Draw on image
    overlay = rgb_image.copy()
    for px, py in pixels.astype(int):
        if 0 <= px < overlay.shape[1] and 0 <= py < overlay.shape[0]:
            overlay[py, px] = [255, 0, 0]  # red boundary
    
    return overlay
```

### 4.4 Visualization 4: Multi-Constraint Summary (Per-Episode Dashboard)

For each episode, produce a single figure showing all constraints at once:

```
┌──────────────────────────────────────────────────┐
│  Episode: level_I / task_0 / episode_00          │
│  Task: Pick up the black bowl ...                │
│  Held object: black bowl                         │
│  Constraints: 3 spatial, caution=yes, rot=no     │
├──────────────┬───────────────────────────────────┤
│              │  3D view with all superquadrics   │
│  RGB image   │  (colored by constraint type)     │
│  with CBF    │  + EEF position                   │
│  contours    │                                   │
├──────────────┼───────────────────────────────────┤
│  2D slice    │  h-value bar chart for each       │
│  z = 0.9m   │  constraint at current EEF pos    │
└──────────────┴───────────────────────────────────┘
```

---

## 5. Behavioral and Pose Constraints (Non-Geometric)

These don't produce superquadrics — they modify the QP parameters:

### 5.1 Caution

If `caution` is in the `end_effector` constraints:
- Default class-K∞: `α(h) = h²`
- Cautious class-K∞: `α_c(h) = 0.25 * h²`

This is a scalar parameter change, not a new CBF. **Visualize by plotting both α curves** and showing that the cautious one is flatter (allows less aggressive approach to the boundary).

```python
h = np.linspace(0, 2, 100)
alpha_default = h**2
alpha_cautious = 0.25 * h**2

plt.plot(h, alpha_default, label='α(h) = h² (default)')
plt.plot(h, alpha_cautious, label='α_c(h) = h²/4 (caution)', linestyle='--')
plt.xlabel('h (distance to boundary)')
plt.ylabel('α(h) (max approach rate)')
plt.title('Behavioral constraint: caution slows approach')
plt.legend()
```

### 5.2 Rotation Lock

If `rotation lock` is in the `end_effector` constraints:
- Set `w_rot > 0` in the QP objective
- The QP penalizes `||log(R_des · R_cur^T)^∨ - ψ||²`

**Visualize as a time series** during rollout: plot the orientation error `||log(R_des · R_cur^T)^∨||` over time, comparing with and without the rotation lock. When active, this value should stay small.

---

## 6. Implementation Steps

### Step 1: Implement `build_object_point_cloud()`

Read depth + seg + camera params from the episode folder. Back-project masked depth pixels to 3D. Test on one episode — verify the point cloud matches the object's known position in `metadata.json`.

### Step 2: Implement `extend_point_cloud()`

For each (object, relationship) pair from the VLM JSON, extend in the correct direction. Verify visually: for "above", the extended cloud should form a tall column above the object.

### Step 3: Implement `fit_superquadric()` (or `fit_ellipsoid()`)

Start with the ellipsoid shortcut. Run the numerical checks from §3. If all extended points have h ≤ 0.1 and the EEF has h > 0, the fit is good.

### Step 4: Implement `evaluate_cbf()` and `cbf_gradient()`

Pure math — no dependencies on perception. Unit-test with known shapes (sphere of radius r centered at origin: h(x) = |x|²/r² - 1).

### Step 5: Build Visualization 2 (2D slice)

This is the fastest way to sanity-check. For each spatial constraint, produce one slice plot. If the red region covers the right area, the CBF is correct.

### Step 6: Build Visualization 3 (RGB overlay)

Project the h = 0 surface onto the agentview image. Compare visually with Brunke et al. Fig. 3. This is the paper-ready figure.

### Step 7: Build Visualization 4 (dashboard)

Combine all constraints for one episode into a single summary figure. Run for all 400 episodes, save as PNGs for quick scanning.

---

## 7. What "Correct" Looks Like (Expected Outputs)

### Example: Task 0, Level I — "Pick up the black bowl between the plate and the ramekin"

If the VLM output says `["obstacle_moka_pot_1", ["above", "around in front of"]]`:

**2D slice at z = table + 0.15m** should show:
- A red region directly above the moka pot footprint (from "above" constraint)
- A red region extending toward the robot from the moka pot (from "around in front of")
- Green everywhere else
- The black bowl's start position and the plate's position should both be green

**RGB overlay** should show:
- A semi-transparent red envelope above the moka pot
- No envelope over the plate (no constraint)
- The robot arm should be outside all envelopes

**h-value check:**
- `h(eef_initial_pos)` should be positive (robot starts safe)
- `h(moka_pot_centroid + [0, 0, 0.1])` should be negative (point directly above moka pot is unsafe)
- `h(plate_centroid + [0, 0, 0.1])` should be positive (point above plate is fine)

### Example: No spatial constraints (sponge-like object)

If the VLM output says all objects have empty constraint lists:
- No superquadrics are constructed
- Only geometric collision avoidance CBFs remain
- Visualization should show no red regions except object bounding volumes

---

## 8. Differences Across Methods at the CBF Stage

| Aspect | M1 | M2 | M3 |
|--------|----|----|-----|
| Point cloud source | Seg mask + depth | VLM bounding box + depth | DBSCAN clusters or GT positions |
| Object identity | From segmentation labels | From VLM free-text (fuzzy matching needed) | From metadata labels |
| Superquadric quality | Best (clean per-object points) | Worst (bbox crops are noisy) | Medium (clusters may merge objects) |
| Fallback if perception fails | Use GT positions from metadata | Use GT positions from metadata | Already using GT positions |

For **comparing methods**, produce the same visualization for all three on the same episode. The CBF boundary shapes will differ because the underlying point clouds differ, even when the VLM predicates are identical. This is one of the key results to show in the paper.

---

## 9. Output File Structure

```
cbf_outputs/
└── safelibero_spatial/
    └── m1/                            # or m2, m3
        ├── level_I/
        │   └── task_0/
        │       └── episode_00/
        │           ├── cbf_params.json         # superquadric params per constraint
        │           ├── cbf_values.json          # h(eef) for each constraint
        │           ├── vis_3d.html              # interactive 3D (plotly)
        │           ├── vis_slice_z0.90.png      # 2D slice
        │           ├── vis_rgb_overlay.png      # projected on camera image
        │           └── vis_dashboard.png        # summary figure
        └── level_II/
            └── ...
```

### `cbf_params.json` schema

```json
{
  "constraints": [
    {
      "object": "obstacle_moka_pot_1",
      "relationship": "above",
      "type": "superquadric",
      "params": {
        "center": [0.15, 0.08, 0.82],
        "scales": [0.06, 0.06, 0.35],
        "epsilon1": 1.0,
        "epsilon2": 1.0
      },
      "h_at_eef": 2.31,
      "gradient_at_eef": [0.12, -0.03, 0.45]
    }
  ],
  "behavioral": {
    "caution": true,
    "alpha_scale": 0.25
  },
  "pose": {
    "rotation_lock": false,
    "w_rot": 0.0
  }
}
```