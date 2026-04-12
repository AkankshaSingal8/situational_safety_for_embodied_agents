"""
CBF Construction: From VLM predicates to Control Barrier Functions.

Takes VLM predicate JSON + episode observation data and produces:
  - Ellipsoid-based CBF constraints per (object, relationship) pair
  - 3D visualization, 2D slice heatmaps, RGB overlay, and dashboard
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from PIL import Image
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Workspace constants
# ---------------------------------------------------------------------------
WORKSPACE_BOUNDS = {
    "x_min": -0.5, "x_max": 0.5,
    "y_min": -0.3, "y_max": 0.6,
    "z_table": 0.81,
    "z_max": 1.4,
}
EXTENSION_DISTANCE = 0.12  # meters for front/behind/around
DEFAULT_OBJECT_RADIUS = 0.05  # fallback sphere radius
SAFETY_MARGIN = 1.3


def align_modalities(rgb, depth, seg, metadata):
    """Ensure RGB / depth / seg all use OpenCV pixel layout (row 0 = top).

    Robosuite renders in OpenGL convention (row 0 = bottom).
    The intrinsic / extrinsic matrices assume OpenCV convention,
    so all pixel data must be flipped to match.

    Legacy captures flipped only RGB; newer captures flip all modalities.
    """
    align = metadata.get("image_alignment")
    if align is None:
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


# ---------------------------------------------------------------------------
# 1. Build per-object point clouds from depth + segmentation
# ---------------------------------------------------------------------------

def _geom_ids_for_object(object_name, geom_id_to_name):
    """Return set of integer geom IDs belonging to *object_name*."""
    ids = set()
    for gid, gname in geom_id_to_name.items():
        # geom names look like "plate_1_g0", "plate_1_g1", ...
        # strip the trailing _gN to recover the object name
        base = "_".join(gname.split("_")[:-1])
        # also try exact match
        if base == object_name or gname.startswith(object_name + "_"):
            ids.add(int(gid))
    return ids


def build_object_point_cloud(seg, depth, geom_ids, intrinsic, extrinsic):
    """Back-project masked depth pixels to 3D world coordinates."""
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    cx, cy = intrinsic[0][2], intrinsic[1][2]

    mask = np.isin(seg, list(geom_ids))
    vs, us = np.where(mask)
    if len(vs) == 0:
        return None

    zs = depth[vs, us]
    valid = zs > 0
    vs, us, zs = vs[valid], us[valid], zs[valid]
    if len(zs) == 0:
        return None

    xs_cam = (us - cx) * zs / fx
    ys_cam = (vs - cy) * zs / fy
    pts_cam = np.stack([xs_cam, ys_cam, zs], axis=1)  # Nx3

    # Robosuite extrinsic is camera→world: p_world = R @ p_cam + t
    T = np.array(extrinsic)
    R = T[:3, :3]
    t = T[:3, 3]
    pts_world = pts_cam @ R.T + t
    return pts_world


def build_gt_point_cloud(position, radius=DEFAULT_OBJECT_RADIUS, n=200):
    """Fallback: approximate object as sphere of points around GT position."""
    center = np.array(position)
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.random.uniform(0, np.pi, n)
    r = radius * np.cbrt(np.random.uniform(0, 1, n))
    pts = np.stack([
        center[0] + r * np.sin(phi) * np.cos(theta),
        center[1] + r * np.sin(phi) * np.sin(theta),
        center[2] + r * np.cos(phi),
    ], axis=1)
    return pts


# ---------------------------------------------------------------------------
# 2. Extend point cloud by spatial relationship
# ---------------------------------------------------------------------------

def extend_point_cloud(points, relationship, ws=WORKSPACE_BOUNDS):
    """Extend point cloud in the direction implied by *relationship*."""
    ext = points.copy()

    if relationship == "above":
        upper = points.copy()
        upper[:, 2] = ws["z_max"]
        ext = np.vstack([points, upper])

    elif relationship == "below":
        lower = points.copy()
        lower[:, 2] = ws["z_table"]
        ext = np.vstack([points, lower])

    elif relationship == "around in front of":
        front = points.copy()
        front[:, 1] -= EXTENSION_DISTANCE
        ext = np.vstack([points, front])

    elif relationship == "around behind":
        behind = points.copy()
        behind[:, 1] += EXTENSION_DISTANCE
        ext = np.vstack([points, behind])

    elif relationship == "around":
        offsets = [
            [EXTENSION_DISTANCE, 0, 0],
            [-EXTENSION_DISTANCE, 0, 0],
            [0, EXTENSION_DISTANCE, 0],
            [0, -EXTENSION_DISTANCE, 0],
        ]
        parts = [points]
        for off in offsets:
            shifted = points.copy()
            shifted += np.array(off)
            parts.append(shifted)
        ext = np.vstack(parts)

    return ext


# ---------------------------------------------------------------------------
# 3. Fit ellipsoid (superquadric with eps1=eps2=1)
# ---------------------------------------------------------------------------

def fit_ellipsoid(points, safety_margin=SAFETY_MARGIN):
    center = points.mean(axis=0)
    scales = points.std(axis=0) * safety_margin
    scales = np.maximum(scales, 0.03)  # min 3 cm per axis
    return center, scales


# ---------------------------------------------------------------------------
# 4. CBF evaluation and gradient
# ---------------------------------------------------------------------------

def evaluate_cbf(x_ee, center, scales):
    """h(x) = sum_i ((x_i - c_i)/s_i)^2 - 1.  h>0 is safe."""
    diff = x_ee - center
    g = np.sum((diff / scales) ** 2)
    return g - 1.0


def cbf_gradient(x_ee, center, scales):
    diff = x_ee - center
    return 2.0 * diff / (scales ** 2)


# ---------------------------------------------------------------------------
# 5. Full pipeline: parse VLM JSON → build constraints
# ---------------------------------------------------------------------------

def build_constraints(vlm_json, obs_folder, camera_key="agentview"):
    """Return list of constraint dicts and behavioral flags."""
    with open(os.path.join(obs_folder, "metadata.json")) as f:
        meta = json.load(f)
    with open(os.path.join(obs_folder, "camera_params.json")) as f:
        cam = json.load(f)

    cam_prefix = "eye_in_hand" if camera_key == "robot0_eye_in_hand" else camera_key
    seg_element_path = os.path.join(obs_folder, f"{cam_prefix}_seg_element.npy")
    seg_default_path = os.path.join(obs_folder, f"{cam_prefix}_seg.npy")
    seg_path = seg_element_path if os.path.exists(seg_element_path) else seg_default_path
    print(f"  [build_constraints] using segmentation: {os.path.basename(seg_path)}")
    seg = np.load(seg_path)
    depth = np.load(os.path.join(obs_folder, f"{cam_prefix}_depth.npy"))
    depth = np.squeeze(depth)
    if seg.ndim == 3 and seg.shape[-1] == 1:
        seg = seg.squeeze(-1)

    rgb_path = os.path.join(obs_folder, f"{cam_prefix}_rgb.png")
    rgb = np.array(Image.open(rgb_path))
    rgb, depth, seg = align_modalities(rgb, depth, seg, meta)

    intrinsic = cam[camera_key]["intrinsic"]
    extrinsic = cam[camera_key]["extrinsic"]
    geom_id_to_name = meta["geom_id_to_name"]
    eef_pos = np.array(meta["robot_state"]["eef_pos"])

    objects_data = vlm_json["single"]["objects"]

    constraints = []
    behavioral = {"caution": False, "alpha_scale": 1.0}
    pose = {"rotation_lock": False, "w_rot": 0.0}

    for obj_name, relationships in objects_data:
        if obj_name == "end_effector":
            if "caution" in relationships:
                behavioral["caution"] = True
                behavioral["alpha_scale"] = 0.25
            if "rotation lock" in relationships:
                pose["rotation_lock"] = True
                pose["w_rot"] = 1.0
            continue

        # Skip objects far outside the workspace (off-scene)
        if obj_name in meta["objects"]:
            pos = meta["objects"][obj_name]["position"]
            if abs(pos[0]) > 5 or abs(pos[1]) > 5:
                continue

        # Build point cloud from segmentation
        geom_ids = _geom_ids_for_object(obj_name, geom_id_to_name)
        if geom_ids:
            pc = build_object_point_cloud(seg, depth, geom_ids, intrinsic, extrinsic)
        else:
            pc = None

        # Fallback to GT position
        if pc is None or len(pc) < 10:
            if obj_name in meta["objects"]:
                pc = build_gt_point_cloud(meta["objects"][obj_name]["position"])
            else:
                continue

        for rel in relationships:
            ext_pc = extend_point_cloud(pc, rel)
            center, scales = fit_ellipsoid(ext_pc)
            h_val = evaluate_cbf(eef_pos, center, scales)
            grad = cbf_gradient(eef_pos, center, scales)

            constraints.append({
                "object": obj_name,
                "relationship": rel,
                "center": center.tolist(),
                "scales": scales.tolist(),
                "extended_points": ext_pc,
                "object_points": pc,
                "h_at_eef": float(h_val),
                "gradient_at_eef": grad.tolist(),
            })

    return constraints, behavioral, pose, eef_pos


# ---------------------------------------------------------------------------
# 6. Visualizations
# ---------------------------------------------------------------------------

def _ellipsoid_surface(center, scales, n=30):
    """Return X, Y, Z meshes for an ellipsoid surface."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    U, V = np.meshgrid(u, v)
    X = center[0] + scales[0] * np.cos(U) * np.sin(V)
    Y = center[1] + scales[1] * np.sin(U) * np.sin(V)
    Z = center[2] + scales[2] * np.cos(V)
    return X, Y, Z


def vis_3d(constraints, eef_pos, save_path):
    """Visualization 1: 3D ellipsoids + EEF."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(constraints), 1)))
    for idx, c in enumerate(constraints):
        center = np.array(c["center"])
        scales = np.array(c["scales"])
        X, Y, Z = _ellipsoid_surface(center, scales)
        ax.plot_surface(X, Y, Z, alpha=0.12, color=colors[idx % len(colors)])
        ax.scatter(*center, s=40, color=colors[idx % len(colors)],
                   label=f'{c["object"]} / {c["relationship"]}')

    h_min = min(c["h_at_eef"] for c in constraints) if constraints else 1.0
    eef_color = "green" if h_min > 0 else "red"
    ax.scatter(*eef_pos, c=eef_color, s=250, marker="*",
               zorder=10, label=f"EEF (h_min={h_min:.2f})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("CBF Constraints — 3D View")
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def vis_slice(constraints, eef_pos, z_slice, save_path):
    """Visualization 2: 2D heatmap slice at fixed z."""
    ws = WORKSPACE_BOUNDS
    nx, ny = 200, 200
    xs = np.linspace(ws["x_min"], ws["x_max"], nx)
    ys = np.linspace(ws["y_min"], ws["y_max"], ny)
    X, Y = np.meshgrid(xs, ys)

    # Composite h: take the minimum across all constraints (most restrictive)
    H = np.full_like(X, np.inf)
    for c in constraints:
        center = np.array(c["center"])
        scales = np.array(c["scales"])
        for i in range(ny):
            for j in range(nx):
                pt = np.array([X[i, j], Y[i, j], z_slice])
                h = evaluate_cbf(pt, center, scales)
                H[i, j] = min(H[i, j], h)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(X, Y, H, levels=50, cmap="RdYlGn", vmin=-3, vmax=5)
    plt.colorbar(cf, ax=ax, label="h(x,y,z_fixed)")
    ax.contour(X, Y, H, levels=[0.0], colors="black", linewidths=2)
    ax.scatter(eef_pos[0], eef_pos[1], c="blue", s=120, marker="*",
               zorder=10, label="EEF")
    # Mark object centers
    seen = set()
    for c in constraints:
        key = c["object"]
        if key not in seen:
            seen.add(key)
            ax.scatter(c["center"][0], c["center"][1], c="white",
                       edgecolors="black", s=50, zorder=9, label=key)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"CBF slice at z = {z_slice:.2f} m")
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def vis_rgb_overlay(constraints, rgb_path, intrinsic, extrinsic, save_path):
    """Visualization 3: project h=0 ellipsoid boundaries onto RGB image."""
    img = np.array(Image.open(rgb_path))
    overlay = img.copy().astype(np.float64)

    K = np.array(intrinsic)
    T = np.array(extrinsic)
    # Extrinsic is camera→world; invert for world→camera projection
    R = T[:3, :3]
    t_vec = T[:3, 3]

    colors_list = [
        [255, 50, 50],
        [50, 50, 255],
        [50, 200, 50],
        [255, 165, 0],
        [200, 50, 200],
        [0, 200, 200],
    ]

    for idx, c in enumerate(constraints):
        center = np.array(c["center"])
        scales = np.array(c["scales"])
        u = np.linspace(0, 2 * np.pi, 80)
        v = np.linspace(0, np.pi, 40)
        U, V = np.meshgrid(u, v)
        pts_3d = np.stack([
            center[0] + scales[0] * np.cos(U) * np.sin(V),
            center[1] + scales[1] * np.sin(U) * np.sin(V),
            center[2] + scales[2] * np.cos(V),
        ], axis=-1).reshape(-1, 3)

        # World to camera: p_cam = R^T @ (p_world - t)
        pts_cam = (pts_3d - t_vec) @ R

        # Keep only in front of camera
        mask = pts_cam[:, 2] > 0.01
        pts_cam = pts_cam[mask]
        if len(pts_cam) == 0:
            continue

        # Project
        pixels = (K @ pts_cam.T).T
        px = (pixels[:, 0] / pixels[:, 2]).astype(int)
        py = (pixels[:, 1] / pixels[:, 2]).astype(int)

        color = colors_list[idx % len(colors_list)]
        h, w = img.shape[:2]
        for x, y in zip(px, py):
            if 0 <= x < w and 0 <= y < h:
                overlay[y, x] = overlay[y, x] * 0.4 + np.array(color) * 0.6

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(overlay.astype(np.uint8))
    ax.set_title("CBF h=0 boundaries projected on image")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def vis_dashboard(constraints, behavioral, pose, eef_pos, rgb_path,
                  intrinsic, extrinsic, z_slice, save_path, task_desc):
    """Visualization 4: combined dashboard."""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # --- Header text ---
    header = (f"Task: {task_desc}\n"
              f"Constraints: {len(constraints)} spatial | "
              f"caution={behavioral['caution']} | "
              f"rotation_lock={pose['rotation_lock']}")
    fig.suptitle(header, fontsize=11, y=0.98)

    # --- Panel 1: RGB with overlay ---
    ax_rgb = fig.add_subplot(gs[0, 0])
    img = np.array(Image.open(rgb_path))
    K = np.array(intrinsic)
    T = np.array(extrinsic)
    # Extrinsic is camera→world; invert for world→camera projection
    R_dash = T[:3, :3]
    t_dash = T[:3, 3]
    overlay = img.copy().astype(np.float64)
    colors_list = [[255, 50, 50], [50, 50, 255], [50, 200, 50],
                   [255, 165, 0], [200, 50, 200], [0, 200, 200]]
    for idx, c in enumerate(constraints):
        center = np.array(c["center"])
        scales = np.array(c["scales"])
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        U, V = np.meshgrid(u, v)
        pts_3d = np.stack([
            center[0] + scales[0] * np.cos(U) * np.sin(V),
            center[1] + scales[1] * np.sin(U) * np.sin(V),
            center[2] + scales[2] * np.cos(V),
        ], axis=-1).reshape(-1, 3)
        # World to camera: p_cam = R^T @ (p_world - t)
        pts_cam = (pts_3d - t_dash) @ R_dash
        mask = pts_cam[:, 2] > 0.01
        pts_cam = pts_cam[mask]
        if len(pts_cam) == 0:
            continue
        pixels = (K @ pts_cam.T).T
        px = (pixels[:, 0] / pixels[:, 2]).astype(int)
        py = (pixels[:, 1] / pixels[:, 2]).astype(int)
        color = colors_list[idx % len(colors_list)]
        h_img, w_img = img.shape[:2]
        for x, y in zip(px, py):
            if 0 <= x < w_img and 0 <= y < h_img:
                overlay[y, x] = overlay[y, x] * 0.4 + np.array(color) * 0.6
    ax_rgb.imshow(overlay.astype(np.uint8))
    ax_rgb.set_title("RGB + CBF overlay")
    ax_rgb.axis("off")

    # --- Panel 2: 3D view ---
    ax_3d = fig.add_subplot(gs[0, 1:], projection="3d")
    cm_colors = plt.cm.tab10(np.linspace(0, 1, max(len(constraints), 1)))
    for idx, c in enumerate(constraints):
        ctr = np.array(c["center"])
        sc = np.array(c["scales"])
        Xs, Ys, Zs = _ellipsoid_surface(ctr, sc, n=20)
        ax_3d.plot_surface(Xs, Ys, Zs, alpha=0.1, color=cm_colors[idx % len(cm_colors)])
        ax_3d.scatter(*ctr, s=30, color=cm_colors[idx % len(cm_colors)])
    h_min = min(c["h_at_eef"] for c in constraints) if constraints else 1.0
    eef_c = "green" if h_min > 0 else "red"
    ax_3d.scatter(*eef_pos, c=eef_c, s=200, marker="*")
    ax_3d.set_xlabel("X"); ax_3d.set_ylabel("Y"); ax_3d.set_zlabel("Z")
    ax_3d.set_title("3D ellipsoids")

    # --- Panel 3: 2D slice ---
    ax_slice = fig.add_subplot(gs[1, 0:2])
    ws = WORKSPACE_BOUNDS
    nx, ny = 150, 150
    xs = np.linspace(ws["x_min"], ws["x_max"], nx)
    ys = np.linspace(ws["y_min"], ws["y_max"], ny)
    XG, YG = np.meshgrid(xs, ys)
    HG = np.full_like(XG, np.inf)
    for c in constraints:
        ctr = np.array(c["center"])
        sc = np.array(c["scales"])
        for i in range(ny):
            for j in range(nx):
                pt = np.array([XG[i, j], YG[i, j], z_slice])
                hv = evaluate_cbf(pt, ctr, sc)
                HG[i, j] = min(HG[i, j], hv)
    cf = ax_slice.contourf(XG, YG, HG, levels=40, cmap="RdYlGn", vmin=-3, vmax=5)
    plt.colorbar(cf, ax=ax_slice, label="h")
    ax_slice.contour(XG, YG, HG, levels=[0.0], colors="black", linewidths=2)
    ax_slice.scatter(eef_pos[0], eef_pos[1], c="blue", s=100, marker="*")
    ax_slice.set_xlabel("X (m)"); ax_slice.set_ylabel("Y (m)")
    ax_slice.set_title(f"2D slice z={z_slice:.2f} m")

    # --- Panel 4: h-value bar chart ---
    ax_bar = fig.add_subplot(gs[1, 2])
    labels = [f'{c["object"]}\n{c["relationship"]}' for c in constraints]
    h_vals = [c["h_at_eef"] for c in constraints]
    bar_colors = ["green" if v > 0 else "red" for v in h_vals]
    if labels:
        ax_bar.barh(range(len(labels)), h_vals, color=bar_colors)
        ax_bar.set_yticks(range(len(labels)))
        ax_bar.set_yticklabels(labels, fontsize=6)
        ax_bar.axvline(0, color="black", linewidth=1)
    ax_bar.set_xlabel("h(x_ee)")
    ax_bar.set_title("CBF values at EEF")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CBF construction from VLM predicates")
    parser.add_argument("--vlm_json", required=True, help="Path to VLM predicate JSON")
    parser.add_argument("--obs_folder", required=True, help="Path to episode observation folder")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: cbf_outputs/)")
    args = parser.parse_args()

    # Load VLM JSON
    with open(args.vlm_json) as f:
        vlm = json.load(f)

    obs = args.obs_folder
    out = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cbf_outputs")
    os.makedirs(out, exist_ok=True)

    print("Building CBF constraints ...")
    constraints, behavioral, pose, eef_pos = build_constraints(vlm, obs, camera_key="agentview")
    print(f"  {len(constraints)} spatial constraints, caution={behavioral['caution']}, "
          f"rotation_lock={pose['rotation_lock']}")

    # Load camera params for overlays
    with open(os.path.join(obs, "camera_params.json")) as f:
        cam = json.load(f)

    # Numerical sanity checks
    print("\nNumerical checks:")
    for c in constraints:
        center = np.array(c["center"])
        scales = np.array(c["scales"])
        h_eef = c["h_at_eef"]
        grad_norm = np.linalg.norm(c["gradient_at_eef"])
        status = "SAFE" if h_eef > 0 else "VIOLATED"
        print(f"  {c['object']:30s} | {c['relationship']:20s} | h={h_eef:+8.3f} [{status}] | |grad|={grad_norm:.4f}")

    # Save CBF params JSON
    cbf_json = {
        "constraints": [
            {
                "object": c["object"],
                "relationship": c["relationship"],
                "type": "ellipsoid",
                "params": {
                    "center": c["center"],
                    "scales": c["scales"],
                    "epsilon1": 1.0,
                    "epsilon2": 1.0,
                },
                "h_at_eef": c["h_at_eef"],
                "gradient_at_eef": c["gradient_at_eef"],
            }
            for c in constraints
        ],
        "behavioral": behavioral,
        "pose": pose,
    }
    params_path = os.path.join(out, "cbf_params.json")
    with open(params_path, "w") as f:
        json.dump(cbf_json, f, indent=2)
    print(f"\n  Saved {params_path}")

    # Task description for titles
    task_desc = vlm.get("single", {}).get("description", "")

    # --- Visualizations ---
    print("\nGenerating visualizations ...")

    # Strip large point-cloud arrays before visualization (keep only what's needed)
    vis_constraints = []
    for c in constraints:
        vis_constraints.append({k: v for k, v in c.items()
                                if k not in ("extended_points", "object_points")})

    # Vis 1: 3D
    vis_3d(vis_constraints, eef_pos, os.path.join(out, "vis_3d.png"))

    # Vis 2: 2D slice at z ~ table + 0.1 m
    z_slice = WORKSPACE_BOUNDS["z_table"] + 0.10
    vis_slice(vis_constraints, eef_pos, z_slice, os.path.join(out, f"vis_slice_z{z_slice:.2f}.png"))

    # Vis 3: RGB overlay — agentview
    agentview_rgb = os.path.join(obs, "agentview_rgb.png")
    vis_rgb_overlay(vis_constraints, agentview_rgb,
                    cam["agentview"]["intrinsic"],
                    cam["agentview"]["extrinsic"],
                    os.path.join(out, "vis_agentview_overlay.png"))

    # Vis 3b: RGB overlay — eye in hand
    eih_rgb = os.path.join(obs, "eye_in_hand_rgb.png")
    vis_rgb_overlay(vis_constraints, eih_rgb,
                    cam["robot0_eye_in_hand"]["intrinsic"],
                    cam["robot0_eye_in_hand"]["extrinsic"],
                    os.path.join(out, "vis_eye_in_hand_overlay.png"))

    # Vis 4: Dashboard
    vis_dashboard(vis_constraints, behavioral, pose, eef_pos,
                  agentview_rgb,
                  cam["agentview"]["intrinsic"],
                  cam["agentview"]["extrinsic"],
                  z_slice,
                  os.path.join(out, "vis_dashboard.png"),
                  task_desc)

    print("\nDone. All outputs saved to:", out)


if __name__ == "__main__":
    main()
