#!/usr/bin/env python3
"""
build_cbf_ellipsoids.py

Offline CBF Ellipsoid Construction for SafeLIBERO

Builds ellipsoidal barrier functions from VLM semantic constraints + ground-truth
object positions. Ellipsoids are computed once per task (scene is static) and saved
to JSON for fast runtime loading.

Usage:
    python build_cbf_ellipsoids.py \
        --vlm_json results/m1_task0_ep00.json \
        --metadata vlm_inputs/safelibero_spatial/level_I/task_0/episode_00/metadata.json \
        --output cbf_outputs/task_0_ellipsoids.json
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import plotly.graph_objects as go


# ============================================================================
# WORKSPACE BOUNDS (from minimal_cbf_demo_interactive.py)
# ============================================================================

WORKSPACE_BOUNDS = {
    "x_min": -0.5,
    "x_max": 0.5,
    "y_min": -0.3,
    "y_max": 0.6,
    "z_table": 0.81,   # Table surface height
    "z_max": 1.4,      # Workspace ceiling
}

EXTENSION_DISTANCE = 0.1  # meters for "around" relationships
DEFAULT_RADIUS = 0.1   # meters - object sphere approximation

# Visualization parameters
ELLIPSOID_OPACITY = 0.08   # Lower = more transparent, better visibility of point cloud
ELLIPSOID_MODE = "surface" # Options: "surface" or "wireframe"
POINT_CLOUD_OPACITY = 0.95  # Point cloud visibility


# ============================================================================
# ELLIPSOID CONSTRUCTION (from minimal_cbf_demo_interactive.py)
# ============================================================================

def generate_sphere_pointcloud(center, radius=DEFAULT_RADIUS, n_points=100):
    """
    Generate a spherical point cloud around a center position.

    Source: minimal_cbf_demo_interactive.py line 150-163

    Args:
        center: (3,) array - object center position
        radius: float - sphere radius (default 0.05m)
        n_points: int - number of points to generate

    Returns:
        points: (n_points, 3) array - 3D point cloud
    """
    center = np.array(center)

    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = radius * np.cbrt(np.random.uniform(0, 1, n_points))

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    points = np.stack([x, y, z], axis=1) + center
    return points


def extend_pointcloud(points, relationship):
    """
    Extend point cloud based on spatial relationship semantics.

    Source: minimal_cbf_demo_interactive.py line 166-189

    Args:
        points: (N, 3) array - base point cloud
        relationship: str - spatial relationship type
            "above" → extend upward to workspace ceiling
            "below" → extend downward to table surface
            "around in front of" → extend in -y direction
            "around behind" → extend in +y direction
            "around" → no extension, just use the object points (boundary pointsof the object)

    Returns:
        extended: (M, 3) array - extended point cloud
    """
    z_obj = points[:, 2].mean()

    if relationship == "above":
        z_range = np.linspace(z_obj, WORKSPACE_BOUNDS["z_max"], 10)
        layers = [points + [0, 0, dz] for dz in (z_range - z_obj)]
        extended = np.vstack(layers)

    elif relationship == "below":
        z_range = np.linspace(WORKSPACE_BOUNDS["z_table"], z_obj, 10)
        layers = [points + [0, 0, dz] for dz in (z_range - z_obj)]
        extended = np.vstack(layers)

    elif relationship == "around in front of":
        extended = np.vstack([points, points + [0, -EXTENSION_DISTANCE, 0]])

    elif relationship == "around behind":
        extended = np.vstack([points, points + [0, +EXTENSION_DISTANCE, 0]])

    elif relationship == "around":
        extended = points

    else:
        extended = points

    return extended


def fit_ellipsoid(points):
    """
    Fit an ellipsoid to a point cloud using simple statistics.

    Source: minimal_cbf_demo_interactive.py line 192-197

    Args:
        points: (N, 3) array - point cloud

    Returns:
        center: (3,) array - ellipsoid center
        semi_axes: (3,) array - ellipsoid semi-axes [a, b, c]
    """
    center = np.mean(points, axis=0)
    semi_axes = np.std(points, axis=0) * 2.5
    semi_axes = np.maximum(semi_axes, 0.03)  # Minimum size
    return center, semi_axes


def evaluate_cbf(x_ee, center, semi_axes):
    """
    Evaluate ellipsoid-based Control Barrier Function.

    Source: minimal_cbf_demo_interactive.py line 200-205

    Args:
        x_ee: (3,) array - end-effector position
        center: (3,) array - ellipsoid center
        semi_axes: (3,) array - ellipsoid semi-axes

    Returns:
        h: float - barrier function value
            h > 0: safe (outside unsafe region)
            h = 0: boundary
            h < 0: unsafe (inside unsafe region)
    """
    diff = x_ee - center
    normalized_sq = (diff / semi_axes) ** 2
    h = np.sum(normalized_sq) - 1.0
    return h


# ============================================================================
# POINT CLOUD UTILITIES (from minimal_cbf_demo_interactive.py)
# ============================================================================

def align_modalities(rgb, depth, seg, metadata):
    """Ensure RGB / depth / seg all use OpenCV pixel layout (row 0 = top)."""
    align = metadata.get("image_alignment")
    if align is None:
        # Legacy: RGB already flipped to OpenCV, depth/seg still OpenGL.
        depth = np.flip(depth, axis=0).copy()
        seg = np.flip(seg, axis=0).copy()
    else:
        rgb_flipped = bool(align.get("rgb", False))
        depth_flipped = bool(align.get("depth", False))
        if rgb_flipped and not depth_flipped:
            depth = np.flip(depth, axis=0).copy()
            seg = np.flip(seg, axis=0).copy()
        elif depth_flipped and not rgb_flipped:
            rgb = np.flip(rgb, axis=0).copy()

    return rgb, depth, seg


def depth_to_pointcloud(rgb, depth, intrinsic, extrinsic, downsample=4, max_depth=2.0):
    """Back-project depth image to 3D world-frame point cloud with colors."""
    h, w = depth.shape
    ys = np.arange(0, h, downsample)
    xs = np.arange(0, w, downsample)
    xs, ys = np.meshgrid(xs, ys)
    xs = xs.flatten()
    ys = ys.flatten()

    d = depth[ys, xs]
    valid = (d > 0) & (d < max_depth)
    xs, ys, d = xs[valid], ys[valid], d[valid]

    # Unproject to OpenCV camera frame (X right, Y down, Z forward)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x = (xs - cx) * d / fx
    y = (ys - cy) * d / fy
    pts_cam = np.stack([x, y, d], axis=-1)

    # Transform to world coordinates
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    pts_world = pts_cam @ R.T + t

    # Workspace crop
    in_ws = (
        (pts_world[:, 0] > -1.0) & (pts_world[:, 0] < 1.0) &
        (pts_world[:, 1] > -1.0) & (pts_world[:, 1] < 1.0) &
        (pts_world[:, 2] > 0.5) & (pts_world[:, 2] < 2.0)
    )
    pts_world = pts_world[in_ws]
    xs, ys = xs[in_ws], ys[in_ws]

    colors = rgb[ys, xs, :3]
    return pts_world, colors


def load_episode_data(episode_folder, camera_key="agentview"):
    """Load RGB, depth, camera params, and metadata from episode folder."""
    with open(os.path.join(episode_folder, "metadata.json")) as f:
        metadata = json.load(f)
    with open(os.path.join(episode_folder, "camera_params.json")) as f:
        cam_params = json.load(f)

    # Load camera data
    rgb_path = os.path.join(episode_folder, f"{camera_key}_rgb.png")
    depth_path = os.path.join(episode_folder, f"{camera_key}_depth.npy")
    seg_path = os.path.join(episode_folder, f"{camera_key}_seg.npy")

    rgb = np.array(Image.open(rgb_path))
    depth = np.load(depth_path).squeeze()
    seg = np.load(seg_path).squeeze()

    rgb, depth, seg = align_modalities(rgb, depth, seg, metadata)

    intrinsic = np.array(cam_params[camera_key]["intrinsic"])
    extrinsic = np.array(cam_params[camera_key]["extrinsic"])

    return rgb, depth, seg, intrinsic, extrinsic, metadata


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def create_ellipsoid_mesh(center, semi_axes, n_grid=30):
    """Generate ellipsoid surface mesh for plotly visualization."""
    u = np.linspace(0, 2 * np.pi, n_grid)
    v = np.linspace(0, np.pi, n_grid)
    U, V = np.meshgrid(u, v)

    X = center[0] + semi_axes[0] * np.cos(U) * np.sin(V)
    Y = center[1] + semi_axes[1] * np.sin(U) * np.sin(V)
    Z = center[2] + semi_axes[2] * np.cos(V)

    return X, Y, Z


def get_cbf_color(h_value):
    """Map CBF value to RGB color (opacity controlled separately)."""
    if h_value > 0.1:
        return "rgb(50, 200, 50)"   # Green - safe
    elif h_value > -0.1:
        return "rgb(255, 200, 50)"  # Yellow - marginal
    else:
        return "rgb(255, 50, 50)"   # Red - violated


def get_object_color(name):
    """Assign a recognizable color to each object type."""
    color_map = {
        "bowl": "#1f1f1f",
        "cookies": "#c8a25c",
        "ramekin": "#e8dcc8",
        "plate": "#d4d4d4",
        "moka_pot": "#708090",
    }
    lower = name.lower()
    for key, color in color_map.items():
        if key in lower:
            return color
    return "#888888"


def visualize_combined_3d(pts_world, colors, constraints, objects, eef_pos,
                         task_desc, output_path):
    """
    Create interactive 3D visualization combining:
    - Point cloud from depth reconstruction
    - Superquadric CBF ellipsoids
    - Ground-truth object markers
    - Robot end-effector
    """
    fig = go.Figure()

    # 1. Point cloud from depth (background layer)
    color_strs = [f"rgb({r},{g},{b})" for r, g, b in colors]
    fig.add_trace(go.Scatter3d(
        x=pts_world[:, 0],
        y=pts_world[:, 1],
        z=pts_world[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=color_strs, opacity=POINT_CLOUD_OPACITY),
        name="Point Cloud (scene)",
        hoverinfo="skip"
    ))

    # 2. Ground-truth object markers
    for obj_name, obj_data in objects.items():
        pos = obj_data["position"]
        color = get_object_color(obj_name)
        label = obj_name.replace("_", " ").title()

        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode="markers+text",
            marker=dict(size=8, color=color, symbol="diamond",
                       line=dict(width=1, color="white")),
            text=[label],
            textposition="top center",
            textfont=dict(size=8, color="white"),
            name=label,
            hovertext=f"{label}<br>pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})",
            hoverinfo="text"
        ))

    # 3. Superquadric CBF ellipsoids
    for idx, c in enumerate(constraints):
        center = np.array(c["center"])
        semi_axes = np.array(c["semi_axes"])
        h_value = c["h_value"]

        X, Y, Z = create_ellipsoid_mesh(center, semi_axes, n_grid=20)
        color = get_cbf_color(h_value)

        label = f"{c['object']} - {c['relationship']}"

        if ELLIPSOID_MODE == "wireframe":
            # Wireframe mode - draw edges only for better visibility
            fig.add_trace(go.Mesh3d(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                alphahull=0,
                opacity=ELLIPSOID_OPACITY,
                color=color,
                name=f"{label} (h={h_value:.2f})",
                hovertemplate=f"{label}<br>h={h_value:.3f}<extra></extra>",
                showlegend=True,
                flatshading=True
            ))
        else:
            # Surface mode - semi-transparent surfaces
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                opacity=ELLIPSOID_OPACITY,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                name=f"{label} (h={h_value:.2f})",
                hovertemplate=f"{label}<br>h={h_value:.3f}<extra></extra>",
                showlegend=True,
                lighting=dict(ambient=0.8, diffuse=0.5, specular=0.2),
                lightposition=dict(x=1000, y=1000, z=1000)
            ))

    # 4. Robot end-effector
    fig.add_trace(go.Scatter3d(
        x=[eef_pos[0]], y=[eef_pos[1]], z=[eef_pos[2]],
        mode="markers+text",
        marker=dict(size=12, color="cyan", symbol="cross",
                   line=dict(width=2, color="white")),
        text=["Robot EEF"],
        textposition="top center",
        textfont=dict(size=10, color="cyan"),
        name="Robot End-Effector"
    ))

    # Calculate overall safety status
    h_min = min(c["h_value"] for c in constraints) if constraints else 0
    safety_status = "SAFE ✓" if h_min > 0 else "UNSAFE ✗"

    # Layout with dark theme (matching build_3d_map.py)
    fig.update_layout(
        title=dict(
            text=f"Semantic CBF Safety Filter + Point Cloud<br>"
                 f"<sub>Task: {task_desc}</sub><br>"
                 f"<sub>Status: {safety_status} (h_min={h_min:.3f})</sub>",
            font=dict(size=14, color="white")
        ),
        scene=dict(
            xaxis=dict(title="X (m)", backgroundcolor="rgb(20,20,20)",
                      gridcolor="gray", showbackground=True),
            yaxis=dict(title="Y (m)", backgroundcolor="rgb(20,20,20)",
                      gridcolor="gray", showbackground=True),
            zaxis=dict(title="Z (m)", backgroundcolor="rgb(20,20,20)",
                      gridcolor="gray", showbackground=True),
            aspectmode="data",
            bgcolor="rgb(15,15,15)"
        ),
        paper_bgcolor="rgb(10,10,10)",
        legend=dict(
            font=dict(color="white", size=8),
            bgcolor="rgba(20,20,20,0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        margin=dict(l=0, r=0, b=0, t=80)
    )

    fig.write_html(output_path)
    print(f"  ✓ Interactive 3D visualization saved to: {output_path}")


# ============================================================================
# VLM CONSTRAINT LOADING
# ============================================================================

def load_vlm_constraints(vlm_json_path):
    """
    Load VLM semantic constraints from JSON.

    Args:
        vlm_json_path: path to VLM output JSON

    Returns:
        vlm_objects: list of [object_name, [constraint1, constraint2, ...]]
        task_description: str
        end_object: str - high-level description of manipulated object
    """
    with open(vlm_json_path, 'r') as f:
        vlm_data = json.load(f)

    # Handle both formats: {"single": {...}} or {"level_I/task_0/episode_00": {...}}
    if "single" in vlm_data:
        vlm_entry = vlm_data["single"]
    else:
        # Take first episode key
        first_key = list(vlm_data.keys())[0]
        vlm_entry = vlm_data[first_key]

    task_description = vlm_entry.get("description", "")
    end_object = vlm_entry.get("end_object", "")
    vlm_objects = vlm_entry.get("objects", [])

    return vlm_objects, task_description, end_object


def load_object_positions(metadata_path):
    """
    Load ground-truth object positions from metadata JSON.

    Args:
        metadata_path: path to metadata.json (from save_vlm_inputs.py)

    Returns:
        objects: dict {object_name: position (3,)}
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    objects = {}
    for obj_name, obj_data in metadata.get("objects", {}).items():
        position = np.array(obj_data["position"])
        # Filter out objects outside workspace
        if abs(position[0]) < 5 and abs(position[1]) < 5:
            objects[obj_name] = position

    return objects


# ============================================================================
# OBJECT MATCHING
# ============================================================================

def match_target_object(end_object_desc, object_names):
    """
    Match high-level end_object description to actual object ID(s).

    Uses token-based substring matching to map descriptions like "black bowl"
    to structured object IDs like "akita_black_bowl_1".

    Args:
        end_object_desc: str - VLM description (e.g., "black bowl between the plate")
        object_names: list[str] - available object IDs from metadata

    Returns:
        set[str] - matched object IDs to exclude from CBF constraints

    Examples:
        >>> match_target_object("black bowl", ["akita_black_bowl_1", "plate_1"])
        {'akita_black_bowl_1'}

        >>> match_target_object("bowl", ["akita_black_bowl_1", "akita_black_bowl_2"])
        {'akita_black_bowl_1', 'akita_black_bowl_2'}  # Conservative: exclude both
    """
    if not end_object_desc or not object_names:
        return set()

    # Common filler words to ignore
    STOP_WORDS = {"the", "a", "an", "between", "and", "on", "in", "at", "to", "from", "near"}

    # Extract descriptive tokens from VLM description
    desc_tokens = set(
        token.lower()
        for token in end_object_desc.replace(",", " ").split()
        if token.lower() not in STOP_WORDS and not token.isdigit()
    )

    if not desc_tokens:
        return set()

    # Score each object by token overlap
    scores = {}
    for obj_name in object_names:
        # Convert "akita_black_bowl_1" → ["akita", "black", "bowl", "1"]
        obj_tokens = set(obj_name.lower().replace("_", " ").split())
        # Count matching descriptive tokens (ignore numbers)
        obj_tokens_no_digits = {t for t in obj_tokens if not t.isdigit()}
        match_count = len(desc_tokens & obj_tokens_no_digits)
        if match_count > 0:
            scores[obj_name] = match_count

    if not scores:
        return set()

    # Conservative approach: return ALL objects with highest match score
    # This handles cases like "bowl" matching both "black_bowl_1" and "black_bowl_2"
    max_score = max(scores.values())
    return {obj for obj, score in scores.items() if score == max_score}


def extract_placement_target(task_description):
    """
    Extract the target placement object from task description.

    Looks for patterns: "place it on the X", "put it in the X", etc.

    Args:
        task_description: str - Full task description

    Returns:
        str - Placement target description (e.g., "plate") or empty string

    Example:
        >>> extract_placement_target("pick up the black bowl and place it on the plate")
        "plate"
    """
    import re

    # Pattern: "place/put it on/in the <target>"
    patterns = [
        r'place it (?:on|in) (?:the )?(.+?)(?:\s|$)',
        r'put it (?:on|in) (?:the )?(.+?)(?:\s|$)',
    ]

    desc_lower = task_description.lower()
    for pattern in patterns:
        match = re.search(pattern, desc_lower)
        if match:
            return match.group(1).strip()

    return ""


# ============================================================================
# MAIN ELLIPSOID BUILDER
# ============================================================================

def build_ellipsoids(vlm_json_path, metadata_path, task_id=0):
    """
    Build all CBF ellipsoids for a task.

    Args:
        vlm_json_path: path to VLM constraints JSON
        metadata_path: path to episode metadata JSON
        task_id: int - task identifier

    Returns:
        ellipsoids_data: dict with task_id and ellipsoid list
        eef_pos: (3,) array - end-effector position
        gt_objects: dict - ground-truth object data with positions
    """
    print(f"\n{'='*70}")
    print(f"Building CBF Ellipsoids for Task {task_id}")
    print(f"{'='*70}\n")

    # Load VLM constraints
    print(f"[1/4] Loading VLM constraints from {vlm_json_path}...")
    vlm_objects, task_description, end_object = load_vlm_constraints(vlm_json_path)
    print(f"  Task: {task_description}")
    print(f"  Found {len(vlm_objects)} objects with constraints")

    # Load object positions and robot state
    print(f"\n[2/4] Loading ground-truth object positions from {metadata_path}...")
    gt_objects = load_object_positions(metadata_path)

    # Load metadata for robot state
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    eef_pos = np.array(metadata["robot_state"]["eef_pos"])

    print(f"  Found {len(gt_objects)} objects in workspace")
    print(f"  Robot EEF at: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}]")

    # Identify objects to exclude from CBF constraints
    objects_to_exclude = set()

    # 1. Manipulated object (being picked up)
    manipulated_objects = match_target_object(end_object, list(gt_objects.keys()))
    if manipulated_objects:
        objects_to_exclude.update(manipulated_objects)
        print(f"  🤏 Manipulated object(s): {manipulated_objects}")

    # 2. Target placement object (where object will be placed)
    placement_target_desc = extract_placement_target(task_description)
    if placement_target_desc:
        placement_objects = match_target_object(placement_target_desc, list(gt_objects.keys()))
        if placement_objects:
            objects_to_exclude.update(placement_objects)
            print(f"  🎯 Placement target(s): {placement_objects}")

    print(f"  ⏭️  Total objects to EXCLUDE: {objects_to_exclude}")

    # Build ellipsoids
    print(f"\n[3/4] Building ellipsoids...")
    ellipsoids = []

    for obj_name, constraint_list in vlm_objects:
        if obj_name == "end_effector":
            # Skip behavioral/pose constraints (handled differently at runtime)
            continue

        # Skip excluded objects (manipulated object and placement target)
        if obj_name in objects_to_exclude:
            print(f"  ⏭️  Skipping excluded object: {obj_name}")
            continue

        if obj_name not in gt_objects:
            print(f"  ⚠️  Object '{obj_name}' not found in ground truth, skipping")
            continue

        obj_position = gt_objects[obj_name]

        # Filter to spatial constraints only
        spatial_constraints = [c for c in constraint_list
                              if c in ["above", "below", "around", "around in front of", "around behind"]]

        if not spatial_constraints:
            continue

        print(f"\n  Object: {obj_name}")
        print(f"    Position: [{obj_position[0]:.3f}, {obj_position[1]:.3f}, {obj_position[2]:.3f}]")
        print(f"    Constraints: {spatial_constraints}")

        # Build ellipsoid for each constraint
        for relationship in spatial_constraints:
            # Generate sphere point cloud
            pc = generate_sphere_pointcloud(obj_position, radius=DEFAULT_RADIUS, n_points=100)

            # Extend based on relationship
            pc_extended = extend_pointcloud(pc, relationship)

            # Fit ellipsoid
            center, semi_axes = fit_ellipsoid(pc_extended)

            # Evaluate CBF at current end-effector position
            h_value = evaluate_cbf(eef_pos, center, semi_axes)

            # Store ellipsoid
            ellipsoid = {
                "object": obj_name,
                "relationship": relationship,
                "center": center.tolist(),
                "semi_axes": semi_axes.tolist(),
                "h_value": float(h_value),
            }
            ellipsoids.append(ellipsoid)

            print(f"      → {relationship}: center={center}, axes={semi_axes}, h={h_value:.3f}")

    print(f"\n  ✓ Built {len(ellipsoids)} ellipsoid constraints")

    # Package output
    ellipsoids_data = {
        "task_id": task_id,
        "task_description": task_description,
        "ellipsoids": ellipsoids,
        "workspace_bounds": WORKSPACE_BOUNDS,
        "eef_pos": eef_pos.tolist(),
    }

    # Prepare objects dict for visualization (with position arrays)
    objects_for_vis = {
        name: {"position": pos}
        for name, pos in gt_objects.items()
    }

    return ellipsoids_data, eef_pos, objects_for_vis


def save_ellipsoids(ellipsoids_data, output_path):
    """
    Save ellipsoids to JSON file.

    Args:
        ellipsoids_data: dict from build_ellipsoids()
        output_path: path to output JSON file
    """
    print(f"\n[4/5] Saving ellipsoids to {output_path}...")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(ellipsoids_data, f, indent=2)

    print(f"  ✓ Saved {len(ellipsoids_data['ellipsoids'])} ellipsoids")


def generate_visualization(episode_folder, ellipsoids_data, eef_pos, objects, output_html):
    """
    Generate interactive 3D HTML visualization of CBF constraints.

    Args:
        episode_folder: path to episode folder with RGB/depth/camera data
        ellipsoids_data: dict from build_ellipsoids()
        eef_pos: (3,) array - end-effector position
        objects: dict - ground-truth object data
        output_html: path to output HTML file
    """
    print(f"\n[5/5] Generating interactive 3D visualization...")

    try:
        # Load episode data
        rgb, depth, seg, intrinsic, extrinsic, metadata = load_episode_data(
            episode_folder, camera_key="agentview"
        )

        # Build point cloud
        pts_world, colors = depth_to_pointcloud(
            rgb, depth, intrinsic, extrinsic, downsample=3, max_depth=2.0
        )
        print(f"  Generated {len(pts_world)} 3D points from depth map")

        # Prepare constraints for visualization
        constraints = ellipsoids_data["ellipsoids"]
        task_desc = ellipsoids_data["task_description"]

        # Generate visualization
        visualize_combined_3d(
            pts_world, colors, constraints, objects, eef_pos, task_desc, output_html
        )

        print(f"\n{'='*70}")
        print(f"SUCCESS! Ellipsoids and visualization ready.")
        print(f"{'='*70}\n")
        print(f"📊 Outputs:")
        print(f"  - Ellipsoid parameters: {output_html.replace('.html', '.json')}")
        print(f"  - 3D Interactive visualization: {output_html}")
        print(f"\nOpen {output_html} in a browser to explore the constraints.")

    except Exception as e:
        print(f"  ⚠️  Visualization generation failed: {e}")
        print(f"  Ellipsoid JSON was still saved successfully.")
        print(f"\n{'='*70}")
        print(f"SUCCESS! Ellipsoids ready for runtime loading.")
        print(f"{'='*70}\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build CBF ellipsoids from VLM constraints + ground truth positions"
    )
    parser.add_argument(
        "--vlm_json",
        type=str,
        required=True,
        help="Path to VLM constraints JSON (e.g., results/m1_task0_ep00.json)"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to episode metadata JSON (e.g., vlm_inputs/.../metadata.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for ellipsoids JSON (e.g., cbf_outputs/task_0_ellipsoids.json)"
    )
    parser.add_argument(
        "--episode_folder",
        type=str,
        required=False,
        default=None,
        help="Path to episode folder for visualization (e.g., vlm_inputs/.../episode_00). If not provided, inferred from metadata path."
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="Task ID (default: 0)"
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="Skip 3D visualization generation"
    )

    args = parser.parse_args()

    # Build ellipsoids
    ellipsoids_data, eef_pos, objects = build_ellipsoids(
        vlm_json_path=args.vlm_json,
        metadata_path=args.metadata,
        task_id=args.task_id
    )

    # Save to JSON
    save_ellipsoids(ellipsoids_data, args.output)

    # Generate visualization (unless disabled)
    if not args.no_vis:
        # Infer episode folder from metadata path if not provided
        episode_folder = args.episode_folder
        if episode_folder is None:
            episode_folder = os.path.dirname(args.metadata)

        # Determine HTML output path (same directory as JSON, replace extension)
        output_html = args.output.replace('.json', '.html')

        generate_visualization(
            episode_folder=episode_folder,
            ellipsoids_data=ellipsoids_data,
            eef_pos=eef_pos,
            objects=objects,
            output_html=output_html
        )
    else:
        print(f"\n{'='*70}")
        print(f"SUCCESS! Ellipsoids ready for runtime loading.")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
