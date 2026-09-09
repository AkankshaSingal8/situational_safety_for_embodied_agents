"""
CBF Construction with Superquadric Fitting.

Uses full superquadric shapes (with ε1, ε2 shape parameters) instead of
simple ellipsoids for tighter CBF constraint envelopes.

Outputs:
  - cbf_params.json          : superquadric parameters per constraint
  - vis_3d.html              : interactive 3D plotly visualization
  - vis_agentview_overlay.png: h=0 boundary projected on agentview image
  - vis_eye_in_hand_overlay.png: h=0 boundary projected on eye-in-hand image
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
EXTENSION_DISTANCE = 0.12
DEFAULT_OBJECT_RADIUS = 0.05
SAFETY_MARGIN = 1.3


# ---------------------------------------------------------------------------
# 1. Per-object point clouds from depth + segmentation
# ---------------------------------------------------------------------------

def _geom_ids_for_object(object_name, geom_id_to_name):
    """Return set of integer geom IDs belonging to *object_name*."""
    ids = set()
    for gid, gname in geom_id_to_name.items():
        base = "_".join(gname.split("_")[:-1])
        if base == object_name or gname.startswith(object_name + "_"):
            ids.add(int(gid))
    return ids


def build_object_point_cloud(seg, depth, geom_ids, intrinsic, extrinsic):
    """Back-project masked depth pixels to 3D world coordinates."""
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    cx, cy = intrinsic[0][2], intrinsic[1][2]

    # Squeeze extra dimensions (e.g. (H,W,1) → (H,W))
    depth = np.squeeze(depth)
    seg = np.squeeze(seg)

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
    pts_cam = np.stack([xs_cam, ys_cam, zs], axis=1)

    T = np.array(extrinsic)
    R = T[:3, :3]
    t = T[:3, 3]
    pts_world = (R.T @ (pts_cam - t).T).T
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
# 3. Superquadric fitting
# ---------------------------------------------------------------------------

def _superquadric_inside_outside(points, params):
    """Evaluate g(p; θ) for all points. Surface is g=1."""
    cx, cy, cz, ax, ay, az, e1, e2 = params
    local = points - np.array([cx, cy, cz])

    # Clamp exponents to avoid numerical issues
    e1 = np.clip(e1, 0.05, 2.0)
    e2 = np.clip(e2, 0.05, 2.0)

    term_xy = (np.abs(local[:, 0] / ax) ** (2.0 / e2) +
               np.abs(local[:, 1] / ay) ** (2.0 / e2)) ** (e2 / e1)
    term_z = np.abs(local[:, 2] / az) ** (2.0 / e1)
    return term_xy + term_z


def fit_superquadric(points, safety_margin=SAFETY_MARGIN):
    """
    Fit a superquadric to a point cloud.

    Optimises center (cx,cy,cz), scales (ax,ay,az), and shape (ε1,ε2).
    The objective minimises Σ(g(p_i) - 1)² so that the surface passes
    through the point cloud, then inflates scales by safety_margin.

    Returns: (cx, cy, cz, ax, ay, az, e1, e2)
    """
    center_init = points.mean(axis=0)
    scales_init = points.std(axis=0) * 2.0
    scales_init = np.maximum(scales_init, 0.03)
    x0 = np.array([*center_init, *scales_init, 1.0, 1.0])

    def objective(params):
        params = np.array(params)
        g = _superquadric_inside_outside(points, params)
        return np.sum((g - 1.0) ** 2)

    bounds = (
        [(None, None)] * 3 +       # center: unconstrained
        [(0.01, None)] * 3 +        # scales: positive
        [(0.05, 2.0), (0.05, 2.0)]  # shape exponents
    )

    result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-10})

    params = result.x.copy()
    # Inflate scales by safety margin
    params[3:6] *= safety_margin
    return params


# ---------------------------------------------------------------------------
# 4. CBF evaluation and gradient (superquadric)
# ---------------------------------------------------------------------------

def evaluate_cbf(x_ee, sq_params):
    """
    h(x) = g(x; θ) - 1.
    h > 0 → safe (outside), h < 0 → violated (inside).
    """
    cx, cy, cz, ax, ay, az, e1, e2 = sq_params
    dx = x_ee[0] - cx
    dy = x_ee[1] - cy
    dz = x_ee[2] - cz

    e1 = np.clip(e1, 0.05, 2.0)
    e2 = np.clip(e2, 0.05, 2.0)

    term_xy = (np.abs(dx / ax) ** (2.0 / e2) +
               np.abs(dy / ay) ** (2.0 / e2)) ** (e2 / e1)
    term_z = np.abs(dz / az) ** (2.0 / e1)
    g = term_xy + term_z
    return g - 1.0


def cbf_gradient(x_ee, sq_params):
    """
    Compute ∂h/∂x_ee analytically for the superquadric.
    Uses chain rule through g(x; θ).
    """
    cx, cy, cz, ax, ay, az, e1, e2 = sq_params
    dx = x_ee[0] - cx
    dy = x_ee[1] - cy
    dz = x_ee[2] - cz

    e1 = np.clip(e1, 0.05, 2.0)
    e2 = np.clip(e2, 0.05, 2.0)

    eps = 1e-12  # avoid division by zero

    # Intermediate terms
    abs_x = np.abs(dx / ax) + eps
    abs_y = np.abs(dy / ay) + eps
    abs_z = np.abs(dz / az) + eps

    pow_x = abs_x ** (2.0 / e2)
    pow_y = abs_y ** (2.0 / e2)
    sum_xy = pow_x + pow_y + eps

    pow_xy = sum_xy ** (e2 / e1)

    # ∂g/∂x
    dg_dx = (pow_xy / sum_xy) * (2.0 / e1) * pow_x / (abs_x * ax) * np.sign(dx)
    # ∂g/∂y
    dg_dy = (pow_xy / sum_xy) * (2.0 / e1) * pow_y / (abs_y * ay) * np.sign(dy)
    # ∂g/∂z
    dg_dz = (2.0 / e1) * abs_z ** (2.0 / e1 - 1.0) / az * np.sign(dz)

    return np.array([dg_dx, dg_dy, dg_dz])


# ---------------------------------------------------------------------------
# 5. Full pipeline: parse VLM JSON → build constraints
# ---------------------------------------------------------------------------

def build_constraints(vlm_json, obs_folder, camera_key="agentview"):
    """Return list of constraint dicts and behavioral flags."""
    with open(os.path.join(obs_folder, "metadata.json")) as f:
        meta = json.load(f)
    with open(os.path.join(obs_folder, "camera_params.json")) as f:
        cam = json.load(f)

    seg = np.load(os.path.join(obs_folder, f"{camera_key}_seg.npy"))
    depth = np.load(os.path.join(obs_folder, f"{camera_key}_depth.npy"))
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

        if obj_name in meta["objects"]:
            pos = meta["objects"][obj_name]["position"]
            if abs(pos[0]) > 5 or abs(pos[1]) > 5:
                continue

        geom_ids = _geom_ids_for_object(obj_name, geom_id_to_name)
        if geom_ids:
            pc = build_object_point_cloud(seg, depth, geom_ids, intrinsic, extrinsic)
        else:
            pc = None

        if pc is None or len(pc) < 10:
            if obj_name in meta["objects"]:
                pc = build_gt_point_cloud(meta["objects"][obj_name]["position"])
            else:
                continue

        for rel in relationships:
            ext_pc = extend_point_cloud(pc, rel)
            sq_params = fit_superquadric(ext_pc)
            h_val = evaluate_cbf(eef_pos, sq_params)
            grad = cbf_gradient(eef_pos, sq_params)

            constraints.append({
                "object": obj_name,
                "relationship": rel,
                "sq_params": sq_params.tolist(),
                "extended_points": ext_pc,
                "object_points": pc,
                "h_at_eef": float(h_val),
                "gradient_at_eef": grad.tolist(),
            })

    return constraints, behavioral, pose, eef_pos


# ---------------------------------------------------------------------------
# 6. Superquadric surface generation (parametric)
# ---------------------------------------------------------------------------

def superquadric_surface(sq_params, n_u=60, n_v=30):
    """
    Generate parametric surface points for a superquadric.
    Returns (X, Y, Z) meshes of shape (n_v, n_u).
    """
    cx, cy, cz, ax, ay, az, e1, e2 = sq_params
    e1 = np.clip(e1, 0.05, 2.0)
    e2 = np.clip(e2, 0.05, 2.0)

    eta = np.linspace(-np.pi / 2, np.pi / 2, n_v)  # latitude
    omega = np.linspace(-np.pi, np.pi, n_u)          # longitude
    ETA, OMEGA = np.meshgrid(eta, omega, indexing="ij")

    def signed_pow(base, exp):
        return np.sign(base) * np.abs(base) ** exp

    cos_eta = np.cos(ETA)
    sin_eta = np.sin(ETA)
    cos_omega = np.cos(OMEGA)
    sin_omega = np.sin(OMEGA)

    X = cx + ax * signed_pow(cos_eta, e1) * signed_pow(cos_omega, e2)
    Y = cy + ay * signed_pow(cos_eta, e1) * signed_pow(sin_omega, e2)
    Z = cz + az * signed_pow(sin_eta, e1)

    return X, Y, Z


# ---------------------------------------------------------------------------
# 7. Visualizations
# ---------------------------------------------------------------------------

def _build_scene_point_cloud(obs_folder, camera_keys, cam_params,
                             max_points_per_cam=50000):
    """Back-project full depth+RGB images into a colored world-space point cloud.

    Returns (pts_world, colors_rgb) where colors_rgb is in [0,255] uint8.
    """
    all_pts = []
    all_colors = []

    for cam_key in camera_keys:
        depth_path = os.path.join(obs_folder, f"{cam_key}_depth.npy")
        rgb_path = os.path.join(obs_folder, f"{cam_key}_rgb.png")
        if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
            continue

        depth = np.squeeze(np.load(depth_path))
        rgb = np.array(Image.open(rgb_path))
        intrinsic = np.array(cam_params[cam_key]["intrinsic"])
        extrinsic = np.array(cam_params[cam_key]["extrinsic"])

        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        h, w = depth.shape[:2]
        vs, us = np.where(depth > 0)
        zs = depth[vs, us]

        # Subsample if too many points
        if len(zs) > max_points_per_cam:
            idx = np.random.choice(len(zs), max_points_per_cam, replace=False)
            vs, us, zs = vs[idx], us[idx], zs[idx]

        xs_cam = (us - cx) * zs / fx
        ys_cam = (vs - cy) * zs / fy
        pts_cam = np.stack([xs_cam, ys_cam, zs], axis=1)

        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        pts_world = (R.T @ (pts_cam - t).T).T

        colors = rgb[vs, us, :3]  # grab RGB per point

        all_pts.append(pts_world)
        all_colors.append(colors)

    if all_pts:
        return np.vstack(all_pts), np.vstack(all_colors).astype(np.uint8)
    return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)


def _make_scene_fig(obs_folder, cam_params, eef_pos):
    """Build a plotly Figure with just the scene point cloud and EEF marker."""
    import plotly.graph_objects as go

    fig = go.Figure()

    if obs_folder is not None and cam_params is not None:
        camera_keys = list(cam_params.keys())
        pts, colors = _build_scene_point_cloud(obs_folder, camera_keys,
                                               cam_params)
        if len(pts) > 0:
            color_strs = [f"rgb({r},{g},{b})" for r, g, b in colors]
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=color_strs, opacity=0.6),
                name="Scene point cloud",
                hoverinfo="skip",
            ))

    # End-effector marker
    fig.add_trace(go.Scatter3d(
        x=[eef_pos[0]], y=[eef_pos[1]], z=[eef_pos[2]],
        mode="markers",
        marker=dict(size=10, color="blue", symbol="diamond"),
        name="End-effector",
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
        width=1100, height=900,
        legend=dict(font=dict(size=10)),
    )
    return fig


def vis_scene_html(eef_pos, save_path, obs_folder=None, cam_params=None):
    """Save a 3D scene point cloud (no superquadrics) as HTML."""
    fig = _make_scene_fig(obs_folder, cam_params, eef_pos)
    fig.update_layout(title="3D Scene Map")
    fig.write_html(save_path)
    print(f"  Saved {save_path}")


def vis_3d_html(constraints, eef_pos, save_path, obs_folder=None,
                cam_params=None):
    """Scene point cloud + superquadric overlays, saved as HTML."""
    import plotly.graph_objects as go

    fig = _make_scene_fig(obs_folder, cam_params, eef_pos)

    # Update EEF color based on constraint status
    h_min = min(c["h_at_eef"] for c in constraints) if constraints else 1.0
    eef_color = "green" if h_min > 0 else "red"
    # Update the EEF trace (last trace added by _make_scene_fig)
    fig.data[-1].marker.color = eef_color
    fig.data[-1].name = f"EEF (h_min={h_min:.2f})"

    # --- Superquadric constraint surfaces ---
    line_colors = [
        "rgb(255,50,50)", "rgb(50,50,255)",
        "rgb(50,200,50)", "rgb(255,165,0)",
        "rgb(200,50,200)", "rgb(0,200,200)",
    ]

    for idx, c in enumerate(constraints):
        sq = np.array(c["sq_params"])
        X, Y, Z = superquadric_surface(sq, n_u=50, n_v=25)
        lc = line_colors[idx % len(line_colors)]

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            opacity=0.25,
            colorscale=[[0, lc], [1, lc]],
            showscale=False,
            name=f'{c["object"]} / {c["relationship"]}',
            hoverinfo="name",
        ))

        fig.add_trace(go.Scatter3d(
            x=[sq[0]], y=[sq[1]], z=[sq[2]],
            mode="markers",
            marker=dict(size=5, color=lc),
            name=f'{c["object"]} center',
            showlegend=False,
        ))

    fig.update_layout(title="CBF Superquadric Constraints — 3D Scene")
    fig.write_html(save_path)
    print(f"  Saved {save_path}")


def vis_rgb_overlay(constraints, rgb_path, intrinsic, extrinsic, save_path):
    """Project h=0 superquadric surface onto RGB image."""
    img = np.array(Image.open(rgb_path))
    overlay = img.copy().astype(np.float64)

    K = np.array(intrinsic)
    T = np.array(extrinsic)

    colors_list = [
        [255, 50, 50], [50, 50, 255], [50, 200, 50],
        [255, 165, 0], [200, 50, 200], [0, 200, 200],
    ]

    for idx, c in enumerate(constraints):
        sq = np.array(c["sq_params"])
        X, Y, Z = superquadric_surface(sq, n_u=80, n_v=40)
        pts_3d = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        # World to camera
        ones = np.ones((pts_3d.shape[0], 1))
        pts_cam = (T @ np.hstack([pts_3d, ones]).T).T[:, :3]

        mask = pts_cam[:, 2] > 0.01
        pts_cam = pts_cam[mask]
        if len(pts_cam) == 0:
            continue

        pixels = (K @ pts_cam.T).T
        px = (pixels[:, 0] / pixels[:, 2]).astype(int)
        py = (pixels[:, 1] / pixels[:, 2]).astype(int)

        color = np.array(colors_list[idx % len(colors_list)])
        h_img, w_img = img.shape[:2]
        valid = (px >= 0) & (px < w_img) & (py >= 0) & (py < h_img)
        px, py = px[valid], py[valid]
        overlay[py, px] = overlay[py, px] * 0.4 + color * 0.6

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(overlay.astype(np.uint8))
    ax.set_title("CBF superquadric h=0 boundaries")
    ax.axis("off")

    # Add legend
    for idx, c in enumerate(constraints):
        color = np.array(colors_list[idx % len(colors_list)]) / 255.0
        ax.plot([], [], color=color, linewidth=4,
                label=f'{c["object"]} /
                 {c["relationship"]}')
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CBF construction with superquadric fitting")
    parser.add_argument("--vlm_json", required=True,
                        help="Path to VLM predicate JSON")
    parser.add_argument("--obs_folder", required=True,
                        help="Path to episode observation folder")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: cbf_sq_outputs/)")
    args = parser.parse_args()

    with open(args.vlm_json) as f:
        vlm = json.load(f)

    obs = args.obs_folder
    out = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cbf_sq_outputs")
    os.makedirs(out, exist_ok=True)

    print("Building superquadric CBF constraints ...")
    constraints, behavioral, pose, eef_pos = build_constraints(
        vlm, obs, camera_key="agentview")
    print(f"  {len(constraints)} spatial constraints, "
          f"caution={behavioral['caution']}, "
          f"rotation_lock={pose['rotation_lock']}")

    # Camera params for overlays
    with open(os.path.join(obs, "camera_params.json")) as f:
        cam = json.load(f)

    # Numerical sanity checks
    print("\nNumerical checks:")
    for c in constraints:
        sq = np.array(c["sq_params"])
        h_eef = c["h_at_eef"]
        grad_norm = np.linalg.norm(c["gradient_at_eef"])
        status = "SAFE" if h_eef > 0 else "VIOLATED"
        print(f"  {c['object']:30s} | {c['relationship']:20s} | "
              f"h={h_eef:+8.3f} [{status}] | |grad|={grad_norm:.4f} | "
              f"e1={sq[6]:.2f} e2={sq[7]:.2f}")

    # Save CBF params JSON
    cbf_json = {
        "constraints": [
            {
                "object": c["object"],
                "relationship": c["relationship"],
                "type": "superquadric",
                "params": {
                    "center": c["sq_params"][:3],
                    "scales": c["sq_params"][3:6],
                    "epsilon1": c["sq_params"][6],
                    "epsilon2": c["sq_params"][7],
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

    # Strip point clouds for visualization dicts
    vis_constraints = []
    for c in constraints:
        vis_constraints.append({k: v for k, v in c.items()
                                if k not in ("extended_points", "object_points")})

    # --- Visualizations ---
    print("\nGenerating visualizations ...")

    # 3D scene map (point cloud only, no superquadrics)
    vis_scene_html(eef_pos, os.path.join(out, "vis_scene_3d.html"),
                   obs_folder=obs, cam_params=cam)

    # 3D scene + superquadric overlays
    vis_3d_html(vis_constraints, eef_pos, os.path.join(out, "vis_3d.html"),
                obs_folder=obs, cam_params=cam)

    # RGB overlay — agentview
    agentview_rgb = os.path.join(obs, "agentview_rgb.png")
    vis_rgb_overlay(vis_constraints, agentview_rgb,
                    cam["agentview"]["intrinsic"],
                    cam["agentview"]["extrinsic"],
                    os.path.join(out, "vis_agentview_overlay.png"))

    # RGB overlay — eye in hand
    eih_rgb = os.path.join(obs, "eye_in_hand_rgb.png")
    vis_rgb_overlay(vis_constraints, eih_rgb,
                    cam["robot0_eye_in_hand"]["intrinsic"],
                    cam["robot0_eye_in_hand"]["extrinsic"],
                    os.path.join(out, "vis_eye_in_hand_overlay.png"))

    print("\nDone. All outputs saved to:", out)


if __name__ == "__main__":
    main()
