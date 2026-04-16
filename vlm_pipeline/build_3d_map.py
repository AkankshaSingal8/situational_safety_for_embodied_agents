#!/usr/bin/env python3
"""
Build an interactive 3D map from SafeLibero episode data.

Uses RGB images, depth maps, segmentation, camera parameters, and metadata
to construct a colored 3D point cloud with object annotations, saved as HTML.

Usage:
    python build_3d_map.py <episode_folder>
    python build_3d_map.py vlm_inputs/safelibero_spatial/level_I/task_0/episode_00
"""

import sys
import os
import json
import numpy as np
from PIL import Image
import plotly.graph_objects as go


def align_modalities(rgb, depth, seg, metadata):
    """Ensure RGB / depth / seg all use OpenCV pixel layout (row 0 = top).

    Robosuite renders in OpenGL convention (row 0 = bottom).
    The intrinsic / extrinsic matrices assume OpenCV convention,
    so all pixel data must be flipped to match.

    Legacy captures flipped only RGB; newer captures flip all modalities.
    """
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


def load_episode(folder):
    """Load all data from an episode folder."""
    with open(os.path.join(folder, "metadata.json")) as f:
        metadata = json.load(f)
    with open(os.path.join(folder, "camera_params.json")) as f:
        cam_params = json.load(f)

    data = {}
    for cam_key, params in cam_params.items():
        cam_prefix = "eye_in_hand" if cam_key == "robot0_eye_in_hand" else cam_key
        rgb_path = os.path.join(folder, f"{cam_prefix}_rgb.png")
        depth_path = os.path.join(folder, f"{cam_prefix}_depth.npy")
        seg_path = os.path.join(folder, f"{cam_prefix}_seg.npy")
        if not (os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(seg_path)):
            continue
        rgb = np.array(Image.open(rgb_path))
        depth = np.load(depth_path).squeeze()
        seg = np.load(seg_path)
        if seg.ndim == 3 and seg.shape[-1] == 1:
            seg = seg.squeeze(-1)
        rgb, depth, seg = align_modalities(rgb, depth, seg, metadata)
        intrinsic = np.array(params["intrinsic"])
        extrinsic = np.array(params["extrinsic"])
        data[cam_prefix] = dict(
            rgb=rgb,
            depth=depth,
            seg=seg,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            camera_key=cam_key,
        )
    return metadata, data


def depth_to_pointcloud(
    rgb,
    depth,
    intrinsic,
    extrinsic,
    downsample=4,
    max_depth=2.0,
):
    """Back-project depth image to 3D world-frame point cloud with colors.

    All pixel data (rgb, depth) must be in OpenCV convention (row 0 = top)
    to match the intrinsic / extrinsic matrices from robosuite.
    """
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

    # Robosuite's get_camera_extrinsic_matrix returns camera→world pose
    # (make_pose @ axis_correction).  p_world = R @ p_cam + t
    # In row-vector form: pts_world = pts_cam @ R^T + t
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    pts_world = pts_cam @ R.T + t

    # Workspace crop: keep points near the tabletop workspace
    in_ws = (
        (pts_world[:, 0] > -1.0) & (pts_world[:, 0] < 1.0) &
        (pts_world[:, 1] > -1.0) & (pts_world[:, 1] < 1.0) &
        (pts_world[:, 2] > 0.5) & (pts_world[:, 2] < 2.0)
    )
    pts_world = pts_world[in_ws]
    xs, ys = xs[in_ws], ys[in_ws]

    colors = rgb[ys, xs, :3]
    return pts_world, colors


def get_object_color(name):
    """Assign a recognizable color to each object type."""
    color_map = {
        "bowl": "#1f1f1f",
        "cookies": "#c8a25c",
        "ramekin": "#e8dcc8",
        "plate": "#d4d4d4",
        "moka_pot": "#708090",
        "mug": "#cc2222",
        "wine_bottle": "#2d5a27",
        "milk": "#f5f5f5",
        "book": "#cccc22",
        "box_base": "#8b6914",
        "storage_box": "#eeeeee",
    }
    lower = name.lower()
    for key, color in color_map.items():
        if key in lower:
            return color
    return "#888888"


def build_3d_map(folder, output_html=None):
    if output_html is None:
        output_html = os.path.join(folder, "3d_map.html")

    metadata, cameras = load_episode(folder)

    fig = go.Figure()

    # --- Point clouds from both cameras ---
    for cam_name, cam_data in cameras.items():
        pts, colors = depth_to_pointcloud(
            cam_data["rgb"], cam_data["depth"],
            cam_data["intrinsic"], cam_data["extrinsic"],
            downsample=3, max_depth=2.0,
        )
        # Convert colors to plotly format
        color_strs = [f"rgb({r},{g},{b})" for r, g, b in colors]

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=1.5, color=color_strs, opacity=0.7),
            name=f"Point Cloud ({cam_name})",
            hoverinfo="skip",
        ))

    # --- Object markers from metadata ---
    objects = metadata.get("objects", {})
    # Filter out objects far from the scene (position component > 5m = off-screen placeholders)
    scene_objects = {}
    for obj_name, obj_data in objects.items():
        pos = obj_data["position"]
        if all(abs(p) < 5.0 for p in pos):
            scene_objects[obj_name] = obj_data

    obj_names = list(scene_objects.keys())
    obj_positions = np.array([scene_objects[n]["position"] for n in obj_names])
    obj_colors = [get_object_color(n) for n in obj_names]
    # Friendly labels
    obj_labels = [n.replace("_", " ").title() for n in obj_names]

    fig.add_trace(go.Scatter3d(
        x=obj_positions[:, 0], y=obj_positions[:, 1], z=obj_positions[:, 2],
        mode="markers+text",
        marker=dict(size=8, color=obj_colors, symbol="diamond",
                    line=dict(width=1, color="white")),
        text=obj_labels,
        textposition="top center",
        textfont=dict(size=10, color="white"),
        name="Objects (Ground Truth)",
        hovertext=[f"{n}<br>pos: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
                   for n, p in zip(obj_labels, obj_positions)],
        hoverinfo="text",
    ))

    # --- Robot end-effector marker ---
    eef = metadata["robot_state"]["eef_pos"]
    fig.add_trace(go.Scatter3d(
        x=[eef[0]], y=[eef[1]], z=[eef[2]],
        mode="markers+text",
        marker=dict(size=10, color="cyan", symbol="cross"),
        text=["Robot EEF"],
        textposition="top center",
        textfont=dict(size=10, color="cyan"),
        name="Robot End-Effector",
    ))

    # --- Obstacle highlight ---
    obstacle = metadata.get("obstacle", {})
    if obstacle.get("position"):
        opos = obstacle["position"]
        fig.add_trace(go.Scatter3d(
            x=[opos[0]], y=[opos[1]], z=[opos[2]],
            mode="markers+text",
            marker=dict(size=12, color="red", symbol="x", line=dict(width=2, color="yellow")),
            text=[f"OBSTACLE: {obstacle['name'].replace('_', ' ').title()}"],
            textposition="top center",
            textfont=dict(size=11, color="red"),
            name="Obstacle",
        ))

    # --- Camera position markers ---
    for cam_name, cam_data in cameras.items():
        ext = cam_data["extrinsic"]
        cam_pos = ext[:3, 3]  # camera-to-world: translation IS the camera position
        fig.add_trace(go.Scatter3d(
            x=[cam_pos[0]], y=[cam_pos[1]], z=[cam_pos[2]],
            mode="markers+text",
            marker=dict(size=7, color="lime", symbol="square"),
            text=[cam_name],
            textposition="top center",
            textfont=dict(size=9, color="lime"),
            name=f"Camera: {cam_name}",
        ))

    # --- Layout ---
    task_desc = metadata.get("task_description", "")
    fig.update_layout(
        title=dict(
            text=f"3D Scene Map<br><sub>Task: {task_desc}</sub>",
            font=dict(size=16, color="white"),
        ),
        scene=dict(
            xaxis=dict(title="X", backgroundcolor="rgb(20,20,20)", gridcolor="gray"),
            yaxis=dict(title="Y", backgroundcolor="rgb(20,20,20)", gridcolor="gray"),
            zaxis=dict(title="Z", backgroundcolor="rgb(20,20,20)", gridcolor="gray"),
            aspectmode="data",
            bgcolor="rgb(15,15,15)",
        ),
        paper_bgcolor="rgb(10,10,10)",
        legend=dict(font=dict(color="white")),
        margin=dict(l=0, r=0, b=0, t=60),
    )

    fig.write_html(output_html)
    print(f"3D map saved to: {output_html}")
    return output_html


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_3d_map.py <episode_folder> [output.html]")
        sys.exit(1)
    folder = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    build_3d_map(folder, output)