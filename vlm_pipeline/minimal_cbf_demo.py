#!/usr/bin/env python3
"""
Minimal Semantic CBF Demo

A clean, educational implementation of VLM-based semantic safety filters
using Control Barrier Functions with ellipsoid safe sets.

Pipeline: VLM JSON → GT Positions → Point Cloud → Extension → Ellipsoid Fit → CBF → Viz

Usage:
    python minimal_cbf_demo.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for HPC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================================
# CONSTANTS
# ============================================================================

WORKSPACE_BOUNDS = {
    "x_min": -0.5,
    "x_max": 0.5,
    "y_min": -0.3,
    "y_max": 0.6,
    "z_table": 0.81,   # Table surface height
    "z_max": 1.4,      # Workspace ceiling
}

EXTENSION_DISTANCE = 0.35  # meters for "around" relationships
DEFAULT_RADIUS = 0.05      # meters - object sphere approximation


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(vlm_json_path, metadata_path):
    """
    Load VLM predictions and ground-truth metadata.

    Returns:
        tuple: (objects_dict, eef_pos, task_description)
            objects_dict: {obj_name: {"position": [x,y,z], "relationships": [...]}}
            eef_pos: [x, y, z] robot end-effector position
            task_description: str
    """
    # Load VLM JSON
    with open(vlm_json_path, 'r') as f:
        vlm_data = json.load(f)

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Extract VLM predictions
    vlm_single = vlm_data.get("single", {})
    task_desc = vlm_single.get("description", "")
    vlm_objects = vlm_single.get("objects", [])

    # Extract ground-truth positions
    gt_objects = metadata.get("objects", {})
    eef_pos = np.array(metadata["robot_state"]["eef_pos"])

    # Build combined data structure
    objects_dict = {}
    for obj_name, relationships in vlm_objects:
        # Skip end-effector (behavioral constraints)
        if obj_name == "end_effector":
            continue

        # Get ground-truth position
        if obj_name not in gt_objects:
            continue

        position = np.array(gt_objects[obj_name]["position"])

        # Filter off-scene objects (placed at x=±10, y=±10, etc.)
        if abs(position[0]) > 5 or abs(position[1]) > 5:
            continue

        # Filter out non-spatial relationships
        spatial_rels = [r for r in relationships
                       if r not in ["caution", "rotation lock"]]

        if spatial_rels:
            objects_dict[obj_name] = {
                "position": position,
                "relationships": spatial_rels
            }

    return objects_dict, eef_pos, task_desc


# ============================================================================
# POINT CLOUD GENERATION
# ============================================================================

def generate_sphere_pointcloud(center, radius=DEFAULT_RADIUS, n_points=100):
    """
    Generate a spherical point cloud around a center position.

    Uses uniform spherical sampling with radius variation for volume filling.

    Args:
        center: [x, y, z] center position
        radius: sphere radius in meters
        n_points: number of points to generate

    Returns:
        np.ndarray: (n_points, 3) point cloud in world coordinates
    """
    center = np.array(center)

    # Uniform spherical sampling
    theta = np.random.uniform(0, 2 * np.pi, n_points)  # azimuthal angle
    phi = np.random.uniform(0, np.pi, n_points)        # polar angle
    r = radius * np.cbrt(np.random.uniform(0, 1, n_points))  # radial (cube root for uniform volume)

    # Convert to Cartesian
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    points = np.stack([x, y, z], axis=1) + center
    return points


# ============================================================================
# POINT CLOUD EXTENSION (Spatial Semantics)
# ============================================================================

def extend_pointcloud(points, relationship):
    """
    Extend point cloud based on spatial relationship semantics.

    Implements Brunke et al.'s extension strategies:
    - "above": extend to workspace ceiling (z_max)
    - "below": extend to table surface (z_table)
    - "around in front of": extend horizontally in -y direction
    - "around behind": extend horizontally in +y direction

    Args:
        points: (N, 3) numpy array of 3D points
        relationship: spatial relationship string

    Returns:
        np.ndarray: Extended point cloud
    """
    z_obj = points[:, 2].mean()

    if relationship == "above":
        # Extend vertically from object to workspace ceiling
        z_range = np.linspace(z_obj, WORKSPACE_BOUNDS["z_max"], 10)
        layers = [points + [0, 0, dz] for dz in (z_range - z_obj)]
        extended = np.vstack(layers)

    elif relationship == "below":
        # Extend vertically from table surface to object
        z_range = np.linspace(WORKSPACE_BOUNDS["z_table"], z_obj, 10)
        layers = [points + [0, 0, dz] for dz in (z_range - z_obj)]
        extended = np.vstack(layers)

    elif relationship == "around in front of":
        # Extend horizontally in -y direction (robot's front)
        extended = np.vstack([
            points,
            points + [0, -EXTENSION_DISTANCE, 0]
        ])

    elif relationship == "around behind":
        # Extend horizontally in +y direction (robot's behind)
        extended = np.vstack([
            points,
            points + [0, +EXTENSION_DISTANCE, 0]
        ])

    else:
        # Unknown relationship - return original points
        extended = points

    return extended


# ============================================================================
# ELLIPSOID FITTING
# ============================================================================

def fit_ellipsoid(points):
    """
    Fit an ellipsoid to a point cloud using simple statistics.

    Uses mean and standard deviation (2.5σ coverage for safety margin).
    No optimization - prioritizes clarity and speed.

    Args:
        points: (N, 3) numpy array

    Returns:
        tuple: (center, semi_axes)
            center: [cx, cy, cz]
            semi_axes: [sx, sy, sz]
    """
    center = np.mean(points, axis=0)
    semi_axes = np.std(points, axis=0) * 2.5  # 2.5σ coverage

    # Enforce minimum size (3cm per axis)
    semi_axes = np.maximum(semi_axes, 0.03)

    return center, semi_axes


# ============================================================================
# CBF EVALUATION
# ============================================================================

def evaluate_cbf(x_ee, center, semi_axes):
    """
    Evaluate ellipsoid-based Control Barrier Function.

    h(x) = Σᵢ ((xᵢ - cᵢ) / sᵢ)² - 1

    - h > 0: safe (outside ellipsoid)
    - h = 0: boundary
    - h < 0: violated (inside unsafe region)

    Args:
        x_ee: [x, y, z] end-effector position
        center: [cx, cy, cz] ellipsoid center
        semi_axes: [sx, sy, sz] ellipsoid semi-axes

    Returns:
        float: CBF value h(x)
    """
    diff = x_ee - center
    normalized_sq = (diff / semi_axes) ** 2
    h = np.sum(normalized_sq) - 1.0
    return h


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def create_ellipsoid_mesh(center, semi_axes, n_grid=30):
    """
    Generate ellipsoid surface mesh for plotly visualization.

    Parametric ellipsoid:
        x = cx + sx * cos(u) * sin(v)
        y = cy + sy * sin(u) * sin(v)
        z = cz + sz * cos(v)

    Args:
        center: [cx, cy, cz]
        semi_axes: [sx, sy, sz]
        n_grid: grid resolution

    Returns:
        tuple: (X, Y, Z) mesh grids
    """
    u = np.linspace(0, 2 * np.pi, n_grid)
    v = np.linspace(0, np.pi, n_grid)
    U, V = np.meshgrid(u, v)

    X = center[0] + semi_axes[0] * np.cos(U) * np.sin(V)
    Y = center[1] + semi_axes[1] * np.sin(U) * np.sin(V)
    Z = center[2] + semi_axes[2] * np.cos(V)

    return X, Y, Z


def get_cbf_color(h_value):
    """
    Map CBF value to color (traffic light scheme).

    Returns:
        str: CSS color string
    """
    if h_value > 0.1:
        return "rgba(50, 200, 50, 0.4)"   # Green - safe
    elif h_value > -0.1:
        return "rgba(255, 200, 50, 0.5)"  # Yellow - marginal
    else:
        return "rgba(255, 50, 50, 0.6)"   # Red - violated


def get_object_marker_color(obj_name):
    """Assign color to object markers based on name."""
    if "bowl" in obj_name.lower():
        return "#1f1f1f"
    elif "plate" in obj_name.lower():
        return "#d4d4d4"
    elif "ramekin" in obj_name.lower():
        return "#e8dcc8"
    elif "cookies" in obj_name.lower():
        return "#c8a25c"
    elif "moka_pot" in obj_name.lower():
        return "#708090"
    else:
        return "#888888"


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_3d_scene(constraints, objects, eef_pos, task_desc, output_path):
    """
    Create 3D visualization with matplotlib.

    Displays:
    - Ground-truth object positions (markers)
    - Fitted ellipsoid safe sets (wireframes)
    - Robot end-effector (cyan marker)
    - CBF safety status (color-coded by h-value)

    Args:
        constraints: list of constraint dicts
        objects: dict of {obj_name: {"position": ...}}
        eef_pos: [x, y, z] end-effector position
        task_desc: task description string
        output_path: where to save PNG
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set dark background
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0f0f0f')
    ax.xaxis.pane.set_facecolor('#141414')
    ax.yaxis.pane.set_facecolor('#141414')
    ax.zaxis.pane.set_facecolor('#141414')
    ax.grid(color='gray', alpha=0.3)

    # 1. Plot ground-truth object positions
    for obj_name, obj_data in objects.items():
        pos = obj_data["position"]
        label = obj_name.replace("_", " ").title()

        # Plot object marker
        ax.scatter(*pos, s=100, marker='D', c='#888888', edgecolors='white',
                  linewidths=1, label=label, alpha=0.9)

    # 2. Plot ellipsoid safe sets
    colors_cm = plt.cm.tab20(np.linspace(0, 1, len(constraints)))

    for idx, c in enumerate(constraints):
        center = c["center"]
        semi_axes = c["semi_axes"]
        h_value = c["h_value"]

        # Generate ellipsoid mesh
        X, Y, Z = create_ellipsoid_mesh(center, semi_axes, n_grid=20)

        # Color based on safety
        if h_value > 0.1:
            color = '#32c832'  # Green - safe
            alpha = 0.15
        elif h_value > -0.1:
            color = '#ffc832'  # Yellow - marginal
            alpha = 0.25
        else:
            color = '#ff3232'  # Red - violated
            alpha = 0.35

        # Plot wireframe
        ax.plot_wireframe(X, Y, Z, color=color, alpha=alpha, linewidth=0.5)

        # Plot center
        ax.scatter(*center, s=30, c=color, alpha=0.8)

    # 3. Plot robot end-effector
    ax.scatter(*eef_pos, s=300, c='cyan', marker='*', edgecolors='white',
              linewidths=2, label='Robot EEF', zorder=10)

    # Calculate overall safety status
    h_min = min(c["h_value"] for c in constraints) if constraints else 0
    safety_status = "SAFE ✓" if h_min > 0 else "UNSAFE ✗"

    # Labels and title
    ax.set_xlabel('X (m)', color='white', fontsize=10)
    ax.set_ylabel('Y (m)', color='white', fontsize=10)
    ax.set_zlabel('Z (m)', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)

    title = f"Semantic CBF Safety Filter\n{task_desc}\nStatus: {safety_status} (h_min={h_min:.3f})"
    ax.set_title(title, color='white', fontsize=12, pad=20)

    # Legend
    ax.legend(loc='upper left', fontsize=7, framealpha=0.8, ncol=2)

    # Equal aspect ratio
    max_range = 0.5
    mid_x = (WORKSPACE_BOUNDS["x_min"] + WORKSPACE_BOUNDS["x_max"]) / 2
    mid_y = (WORKSPACE_BOUNDS["y_min"] + WORKSPACE_BOUNDS["y_max"]) / 2
    mid_z = (WORKSPACE_BOUNDS["z_table"] + WORKSPACE_BOUNDS["z_max"]) / 2
    ax.set_xlim([mid_x - max_range, mid_x + max_range])
    ax.set_ylim([mid_y - max_range, mid_y + max_range])
    ax.set_zlim([mid_z - max_range, mid_z + max_range])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0a0a0a')
    plt.close()
    print(f"\nVisualization saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Orchestrate the minimal CBF pipeline."""

    print("=" * 70)
    print("MINIMAL SEMANTIC CBF DEMO")
    print("=" * 70)

    # File paths
    vlm_json = "results/m1_task0_ep00.json"
    metadata = "vlm_inputs/safelibero_spatial/level_I/task_0/episode_00/metadata.json"

    # 1. Load data
    print("\n[1/4] Loading VLM predictions and ground-truth metadata...")
    objects, eef_pos, task_desc = load_data(vlm_json, metadata)
    print(f"  Loaded {len(objects)} in-workspace objects")
    print(f"  Robot EEF at: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}]")
    print(f"  Task: {task_desc}")

    # 2. Build constraints
    print("\n[2/4] Generating point clouds and fitting ellipsoids...")
    constraints = []
    total_relationships = sum(len(o["relationships"]) for o in objects.values())
    print(f"  Processing {total_relationships} spatial constraints...")

    for obj_name, obj_data in objects.items():
        position = obj_data["position"]

        for relationship in obj_data["relationships"]:
            # Generate sphere point cloud around object
            pc = generate_sphere_pointcloud(position, radius=DEFAULT_RADIUS, n_points=100)

            # Extend based on spatial relationship
            pc_ext = extend_pointcloud(pc, relationship)

            # Fit ellipsoid
            center, semi_axes = fit_ellipsoid(pc_ext)

            # Evaluate CBF at end-effector
            h_value = evaluate_cbf(eef_pos, center, semi_axes)

            constraints.append({
                "object": obj_name,
                "relationship": relationship,
                "center": center,
                "semi_axes": semi_axes,
                "h_value": h_value,
                "extended_points": pc_ext
            })

    print(f"  Built {len(constraints)} CBF constraints")

    # 3. Print safety report
    print("\n[3/4] CBF Safety Report")
    print("=" * 70)
    print(f"{'Object':<30} | {'Relationship':<20} | {'h(x_ee)':<10} | Status")
    print("-" * 70)

    for c in constraints:
        status = "✓ SAFE" if c["h_value"] > 0 else "✗ VIOLATED"
        print(f"{c['object']:<30} | {c['relationship']:<20} | {c['h_value']:+8.3f}   | {status}")

    h_min = min(c["h_value"] for c in constraints)
    h_max = max(c["h_value"] for c in constraints)
    overall = "SAFE ✓" if h_min > 0 else "UNSAFE ✗"

    print("-" * 70)
    print(f"Overall Safety: {overall}")
    print(f"  h_min = {h_min:.3f}  |  h_max = {h_max:.3f}")

    # 4. Visualize
    print("\n[4/4] Generating 3D visualization...")
    output_png = "cbf_demo_output.png"
    visualize_3d_scene(constraints, objects, eef_pos, task_desc, output_png)

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nView {output_png} to see the 3D visualization.")


if __name__ == "__main__":
    main()
