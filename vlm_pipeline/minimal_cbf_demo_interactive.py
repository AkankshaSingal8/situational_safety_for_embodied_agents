#!/usr/bin/env python3
"""
Minimal Semantic CBF Demo - Interactive HTML Version

Combines point cloud visualization from build_3d_map.py with
superquadric CBF overlays in an interactive plotly HTML file.

Usage:
    python minimal_cbf_demo_interactive.py
"""

import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go


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

EXTENSION_DISTANCE = 0.15  # meters for "around" relationships (reduced for tighter fit)
DEFAULT_RADIUS = 0.05      # meters - object sphere approximation

# Visualization parameters
ELLIPSOID_OPACITY = 0.08   # Lower = more transparent, better visibility of point cloud
ELLIPSOID_MODE = "surface" # Options: "surface" or "wireframe"
POINT_CLOUD_OPACITY = 0.95  # Point cloud visibility

# 2D Overlay parameters
OVERLAY_ALPHA = 0.5        # Blending factor for 2D overlay (0=invisible, 1=opaque)
OVERLAY_LINE_WIDTH = 2     # Thickness of ellipsoid boundaries in 2D


# ============================================================================
# POINT CLOUD UTILITIES (from build_3d_map.py)
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
# DATA LOADING (CBF)
# ============================================================================

def load_vlm_data(vlm_json_path):
    """Load VLM predictions."""
    with open(vlm_json_path, 'r') as f:
        vlm_data = json.load(f)

    vlm_single = vlm_data.get("single", {})
    task_desc = vlm_single.get("description", "")
    vlm_objects = vlm_single.get("objects", [])

    return vlm_objects, task_desc


# ============================================================================
# POINT CLOUD GENERATION (CBF)
# ============================================================================

def generate_sphere_pointcloud(center, radius=DEFAULT_RADIUS, n_points=100):
    """Generate a spherical point cloud around a center position."""
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
    """Extend point cloud based on spatial relationship semantics."""
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

    else:
        extended = points

    return extended


def fit_ellipsoid(points):
    """Fit an ellipsoid to a point cloud using simple statistics."""
    center = np.mean(points, axis=0)
    semi_axes = np.std(points, axis=0) * 2.5
    semi_axes = np.maximum(semi_axes, 0.03)
    return center, semi_axes


def evaluate_cbf(x_ee, center, semi_axes):
    """Evaluate ellipsoid-based Control Barrier Function."""
    diff = x_ee - center
    normalized_sq = (diff / semi_axes) ** 2
    h = np.sum(normalized_sq) - 1.0
    return h


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


# ============================================================================
# INTERACTIVE 3D VISUALIZATION
# ============================================================================

def visualize_2d_overlay(constraints, rgb, intrinsic, extrinsic, objects,
                        eef_pos, task_desc, output_path):
    """
    Create 2D visualization by projecting ellipsoid boundaries onto RGB image.
    Uses PIL for image composition (no matplotlib dependency).
    """
    h_img, w_img = rgb.shape[:2]

    # Create two panels: original and overlay
    panel_width = w_img
    panel_height = h_img
    combined_width = panel_width * 2 + 60  # +60 for spacing
    combined_height = panel_height + 100   # +100 for title and legend

    # Create combined image
    combined = Image.new('RGB', (combined_width, combined_height), color=(240, 240, 240))
    draw = ImageDraw.Draw(combined, 'RGBA')

    # Left panel: Original RGB
    img_original = Image.fromarray(rgb)
    combined.paste(img_original, (30, 60))

    # Right panel: Create overlay
    overlay = rgb.copy().astype(np.float64)

    K = np.array(intrinsic)
    T = np.array(extrinsic)

    # Draw ellipsoid boundaries
    for idx, c in enumerate(constraints):
        center = np.array(c["center"])
        semi_axes = np.array(c["semi_axes"])
        h_value = c["h_value"]

        # Generate ellipsoid surface points
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        U, V = np.meshgrid(u, v)

        pts_3d = np.stack([
            center[0] + semi_axes[0] * np.cos(U) * np.sin(V),
            center[1] + semi_axes[1] * np.sin(U) * np.sin(V),
            center[2] + semi_axes[2] * np.cos(V),
        ], axis=-1).reshape(-1, 3)

        # Transform to camera coordinates
        ones = np.ones((pts_3d.shape[0], 1))
        pts_h = np.hstack([pts_3d, ones])
        pts_cam = (T @ pts_h.T).T[:, :3]

        # Keep only points in front of camera
        mask = pts_cam[:, 2] > 0.01
        pts_cam = pts_cam[mask]
        if len(pts_cam) == 0:
            continue

        # Project to image plane
        pixels = (K @ pts_cam.T).T
        px = (pixels[:, 0] / pixels[:, 2]).astype(int)
        py = (pixels[:, 1] / pixels[:, 2]).astype(int)

        # Choose color based on safety status
        if h_value > 0.1:
            color = np.array([50, 200, 50])  # Green - safe
        elif h_value > -0.1:
            color = np.array([255, 200, 50])  # Yellow - marginal
        else:
            color = np.array([255, 50, 50])  # Red - violated

        # Draw on overlay with transparency
        for x, y in zip(px, py):
            if 0 <= x < w_img and 0 <= y < h_img:
                overlay[y, x] = overlay[y, x] * (1 - OVERLAY_ALPHA) + color * OVERLAY_ALPHA

    # Mark object centers with white crosses
    for obj_name, obj_data in objects.items():
        pos_3d = np.array(obj_data["position"])
        pos_h = np.array([pos_3d[0], pos_3d[1], pos_3d[2], 1.0])
        pos_cam = T @ pos_h

        if pos_cam[2] > 0.01:
            pixel = K @ pos_cam[:3]
            obj_x = int(pixel[0] / pixel[2])
            obj_y = int(pixel[1] / pixel[2])

            if 0 <= obj_x < w_img and 0 <= obj_y < h_img:
                marker_size = 10
                for dx in range(-marker_size, marker_size+1):
                    if 0 <= obj_x + dx < w_img:
                        overlay[obj_y, obj_x + dx] = [255, 255, 255]
                for dy in range(-marker_size, marker_size+1):
                    if 0 <= obj_y + dy < h_img:
                        overlay[obj_y + dy, obj_x] = [255, 255, 255]

    # Mark end-effector with cyan circle
    eef_h = np.array([eef_pos[0], eef_pos[1], eef_pos[2], 1.0])
    eef_cam = T @ eef_h
    if eef_cam[2] > 0.01:
        eef_pixel = K @ eef_cam[:3]
        eef_x = int(eef_pixel[0] / eef_pixel[2])
        eef_y = int(eef_pixel[1] / eef_pixel[2])

        if 0 <= eef_x < w_img and 0 <= eef_y < h_img:
            marker_size = 15
            for dy in range(-marker_size, marker_size+1):
                for dx in range(-marker_size, marker_size+1):
                    if dx*dx + dy*dy <= marker_size*marker_size:
                        y, x = eef_y + dy, eef_x + dx
                        if 0 <= x < w_img and 0 <= y < h_img:
                            overlay[y, x] = [0, 255, 255]

    # Paste overlay on right panel
    img_overlay = Image.fromarray(overlay.astype(np.uint8))
    combined.paste(img_overlay, (panel_width + 60, 60))

    # Add titles and legend using draw
    try:
        # Try to use a nice font
        title_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 20)
        label_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)
    except:
        # Fallback to default
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Main title
    h_min = min(c["h_value"] for c in constraints) if constraints else 0
    safety_status = "SAFE ✓" if h_min > 0 else "UNSAFE ✗"
    title_text = f"2D Ellipsoid Projection | Status: {safety_status} (h_min={h_min:.3f})"
    draw.text((combined_width // 2 - 200, 10), title_text, fill=(0, 0, 0), font=title_font)

    # Panel titles
    draw.text((panel_width // 2 - 50, 35), "Original Agent View", fill=(0, 0, 0), font=label_font)
    draw.text((panel_width + 90 + panel_width // 2 - 100, 35),
             "Agent View + CBF Overlays", fill=(0, 0, 0), font=label_font)

    # Legend
    legend_x = panel_width + 90
    legend_y = panel_height + 70
    legend_items = [
        ("Safe (h > 0.1)", (50, 200, 50)),
        ("Marginal (-0.1 < h < 0.1)", (255, 200, 50)),
        ("Violated (h < -0.1)", (255, 50, 50)),
        ("Robot EEF", (0, 255, 255)),
        ("Object Centers", (255, 255, 255)),
    ]

    for idx, (label, color) in enumerate(legend_items):
        y = legend_y + idx * 18
        # Color box
        draw.rectangle([legend_x, y, legend_x + 15, y + 12], fill=color)
        # Label
        draw.text((legend_x + 20, y - 2), label, fill=(0, 0, 0), font=small_font)

    combined.save(output_path, dpi=(150, 150))
    print(f"2D overlay visualization saved to: {output_path}")


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
        center = c["center"]
        semi_axes = c["semi_axes"]
        h_value = c["h_value"]

        X, Y, Z = create_ellipsoid_mesh(center, semi_axes, n_grid=20)
        color = get_cbf_color(h_value)

        label = f"{c['object']} - {c['relationship']}"

        if ELLIPSOID_MODE == "wireframe":
            # Wireframe mode - draw edges only for better visibility
            # Create wireframe by plotting lines
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)

            # Add as mesh3d with visible edges
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
    print(f"\nInteractive visualization saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Orchestrate the combined pipeline."""

    print("=" * 70)
    print("MINIMAL SEMANTIC CBF DEMO - INTERACTIVE HTML")
    print("=" * 70)

    # File paths
    vlm_json = "results/m1_task0_ep00.json"
    episode_folder = "vlm_inputs/safelibero_spatial/level_I/task_0/episode_00"

    print(f"\nVisualization Settings:")
    print(f"  Extension distance (around): {EXTENSION_DISTANCE} m")
    print(f"  Ellipsoid opacity: {ELLIPSOID_OPACITY}")
    print(f"  Ellipsoid mode: {ELLIPSOID_MODE}")
    print(f"  Point cloud opacity: {POINT_CLOUD_OPACITY}")

    # 1. Load episode data (point cloud)
    print("\n[1/5] Loading episode data (RGB, depth, camera params)...")
    rgb, depth, seg, intrinsic, extrinsic, metadata = load_episode_data(
        episode_folder, camera_key="agentview"
    )
    print(f"  Loaded {rgb.shape[0]}x{rgb.shape[1]} RGB-D data")

    # 2. Build point cloud from depth
    print("\n[2/5] Building 3D point cloud from depth map...")
    pts_world, colors = depth_to_pointcloud(
        rgb, depth, intrinsic, extrinsic, downsample=3, max_depth=2.0
    )
    print(f"  Generated {len(pts_world)} 3D points")

    # 3. Load VLM predictions
    print("\n[3/5] Loading VLM predictions and building CBF constraints...")
    vlm_objects, task_desc = load_vlm_data(vlm_json)

    eef_pos = np.array(metadata["robot_state"]["eef_pos"])
    gt_objects = metadata.get("objects", {})

    # Filter in-workspace objects
    objects = {}
    for obj_name, relationships in vlm_objects:
        if obj_name == "end_effector":
            continue
        if obj_name not in gt_objects:
            continue

        position = np.array(gt_objects[obj_name]["position"])
        if abs(position[0]) > 5 or abs(position[1]) > 5:
            continue

        spatial_rels = [r for r in relationships
                       if r not in ["caution", "rotation lock"]]

        if spatial_rels:
            objects[obj_name] = {
                "position": position,
                "relationships": spatial_rels
            }

    print(f"  Loaded {len(objects)} in-workspace objects")
    print(f"  Robot EEF at: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}]")

    # 4. Build CBF constraints
    print("\n[4/5] Generating superquadric constraints...")
    constraints = []
    total_relationships = sum(len(o["relationships"]) for o in objects.values())
    print(f"  Processing {total_relationships} spatial constraints...")

    for obj_name, obj_data in objects.items():
        position = obj_data["position"]

        for relationship in obj_data["relationships"]:
            pc = generate_sphere_pointcloud(position, radius=DEFAULT_RADIUS, n_points=100)
            pc_ext = extend_pointcloud(pc, relationship)
            center, semi_axes = fit_ellipsoid(pc_ext)
            h_value = evaluate_cbf(eef_pos, center, semi_axes)

            constraints.append({
                "object": obj_name,
                "relationship": relationship,
                "center": center,
                "semi_axes": semi_axes,
                "h_value": h_value
            })

    print(f"  Built {len(constraints)} CBF constraints")

    # 5. Create combined 3D visualization
    print("\n[5/6] Generating interactive 3D visualization...")
    output_html = "cbf_demo_interactive.html"
    visualize_combined_3d(
        pts_world, colors, constraints, objects, eef_pos, task_desc, output_html
    )

    # 6. Create 2D overlay visualization
    print("\n[6/6] Generating 2D overlay on agent view image...")
    output_2d = "cbf_demo_2d_overlay.png"
    visualize_2d_overlay(
        constraints, rgb, intrinsic, extrinsic, objects, eef_pos, task_desc, output_2d
    )

    # Print safety report
    print("\n" + "=" * 70)
    print("CBF SAFETY REPORT")
    print("=" * 70)
    h_min = min(c["h_value"] for c in constraints)
    h_max = max(c["h_value"] for c in constraints)
    overall = "SAFE ✓" if h_min > 0 else "UNSAFE ✗"

    print(f"Overall Safety: {overall}")
    print(f"  h_min = {h_min:.3f}  |  h_max = {h_max:.3f}")
    print(f"  Total constraints: {len(constraints)}")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\n📊 Visualizations saved:")
    print(f"  - 3D Interactive: {output_html} (open in browser)")
    print(f"  - 2D Overlay: {output_2d}")
    print(f"\nYou can rotate, zoom, and toggle layers in the 3D visualization.")


if __name__ == "__main__":
    main()
