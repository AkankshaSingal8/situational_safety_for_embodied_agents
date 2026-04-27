"""
CBF Construction with Superquadric Fitting.

Uses full superquadric shapes (with ε1, ε2 shape parameters) instead of
simple ellipsoids for tighter CBF constraint envelopes.

Outputs:
  - cbf_params.json          : superquadric parameters per constraint
  - vis_3d.html              : interactive 3D plotly visualization

The gripper is approximated as a sphere by inflating each obstacle
superquadric during CBF evaluation, while visualization keeps the raw
obstacle surface and renders the gripper sphere explicitly.
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import minimize
from scipy.special import betaln

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
AROUND_DUPLICATE_DISTANCE = 0.0
DEFAULT_OBJECT_RADIUS = 0.05
SAFETY_MARGIN = 1.0
GRIPPER_SPHERE_RADIUS = 0.12
GRIPPER_SPHERE_CENTER_OFFSET = np.array([0.0, 0.0, -0.08], dtype=np.float64)
MASK_ERODE_PIXELS = 1
MIN_ERODED_PIXELS = 30
ENCLOSURE_PENALTY = 1e6
MAX_FIT_POINTS = 8000


def _quat_to_rotation_matrix(quat):
    """Return a 3x3 rotation matrix for an (x, y, z, w) quaternion."""
    x, y, z, w = np.asarray(quat, dtype=np.float64)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


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


def offset_gripper_sphere_center(eef_pos, eef_quat=None, offset=GRIPPER_SPHERE_CENTER_OFFSET):
    """Place the gripper sphere using an offset in the end-effector frame."""
    center = np.asarray(eef_pos, dtype=np.float64).copy()
    offset = np.asarray(offset, dtype=np.float64)
    if eef_quat is not None:
        quat = np.asarray(eef_quat, dtype=np.float64)
        if quat.shape == (4,) and np.all(np.isfinite(quat)):
            offset = _quat_to_rotation_matrix(quat).dot(offset)
    center += offset
    return center


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


def _binary_erosion_square(mask, radius=1):
    """Binary erosion with a square kernel implemented in NumPy."""
    if radius <= 0:
        return mask
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")

    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    h, w = mask.shape
    out = np.ones((h, w), dtype=bool)

    # Keep a pixel only if all neighbors in the kernel are True.
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out &= padded[dy:dy + h, dx:dx + w]
    return out


def build_object_point_cloud(seg, depth, geom_ids, intrinsic, extrinsic):
    """Back-project masked depth pixels to 3D world coordinates."""
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    cx, cy = intrinsic[0][2], intrinsic[1][2]

    # Squeeze extra dimensions (e.g. (H,W,1) → (H,W))
    depth = np.squeeze(depth)
    seg = np.squeeze(seg)

    raw_mask = np.isin(seg, list(geom_ids))
    if MASK_ERODE_PIXELS > 0:
        eroded_mask = _binary_erosion_square(raw_mask, radius=MASK_ERODE_PIXELS)
        # Avoid wiping out tiny masks; keep raw mask when erosion is too aggressive.
        mask = eroded_mask if np.count_nonzero(eroded_mask) >= MIN_ERODED_PIXELS else raw_mask
    else:
        mask = raw_mask

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


def _point_cloud_debug(points, reference=None):
    """Compact JSON-safe point-cloud summary for debugging fits."""
    if points is None or len(points) == 0:
        return {"num_points": 0}

    pts = np.asarray(points, dtype=np.float64)
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if len(pts) == 0:
        return {"num_points": 0, "num_nonfinite": int(len(points))}

    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    summary = {
        "num_points": int(len(pts)),
        "num_nonfinite": int(len(points) - len(pts)),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_extent": (bbox_max - bbox_min).tolist(),
        "centroid": pts.mean(axis=0).tolist(),
    }

    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64)
        d = np.linalg.norm(pts - ref, axis=1)
        summary["distance_to_reference"] = {
            "p50": float(np.percentile(d, 50)),
            "p95": float(np.percentile(d, 95)),
            "p99": float(np.percentile(d, 99)),
            "max": float(np.max(d)),
            "count_gt_0p25": int(np.count_nonzero(d > 0.25)),
        }

    return summary


def _mask_debug(seg, depth, geom_ids):
    """Mirror build_object_point_cloud mask logic without changing behavior."""
    seg = np.squeeze(seg)
    depth = np.squeeze(depth)
    raw_mask = np.isin(seg, list(geom_ids))
    if MASK_ERODE_PIXELS > 0:
        eroded_mask = _binary_erosion_square(raw_mask, radius=MASK_ERODE_PIXELS)
        use_eroded = np.count_nonzero(eroded_mask) >= MIN_ERODED_PIXELS
        mask = eroded_mask if use_eroded else raw_mask
    else:
        eroded_mask = raw_mask
        use_eroded = False
        mask = raw_mask

    valid_depth = mask & (depth > 0)
    return {
        "raw_pixels": int(np.count_nonzero(raw_mask)),
        "eroded_pixels": int(np.count_nonzero(eroded_mask)),
        "used_eroded_mask": bool(use_eroded),
        "mask_pixels": int(np.count_nonzero(mask)),
        "valid_depth_pixels": int(np.count_nonzero(valid_depth)),
    }


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
            [AROUND_DUPLICATE_DISTANCE, 0, 0],
            [-AROUND_DUPLICATE_DISTANCE, 0, 0],
            [0, AROUND_DUPLICATE_DISTANCE, 0],
            [0, -AROUND_DUPLICATE_DISTANCE, 0],
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

    e1 = np.clip(e1, 0.2, 2.0)
    e2 = np.clip(e2, 0.2, 2.0)

    term_xy = (np.abs(local[:, 0] / ax) ** (2.0 / e2) +
               np.abs(local[:, 1] / ay) ** (2.0 / e2)) ** (e2 / e1)
    term_z = np.abs(local[:, 2] / az) ** (2.0 / e1)
    return term_xy + term_z


def _superquadric_log_volume(params):
    """Log volume of a superquadric / superellipsoid."""
    _, _, _, ax, ay, az, e1, e2 = params
    ax = max(ax, 1e-9)
    ay = max(ay, 1e-9)
    az = max(az, 1e-9)
    e1 = np.clip(e1, 0.2, 2.0)
    e2 = np.clip(e2, 0.2, 2.0)

    # V = 2 a b c e1 e2 B(e1/2 + 1, e1) B(e2/2, e2/2)
    return (
        np.log(2.0)
        + np.log(ax) + np.log(ay) + np.log(az)
        + np.log(e1) + np.log(e2)
        + betaln(e1 / 2.0 + 1.0, e1)
        + betaln(e2 / 2.0, e2 / 2.0)
    )


def fit_superquadric(points, safety_margin=SAFETY_MARGIN, return_debug=False):
    """
    Fit a minimum-volume enclosing superquadric to a point cloud.

    Optimizes center (cx,cy,cz), scales (ax,ay,az), and shape (ε1,ε2)
    by minimizing superquadric volume while enforcing enclosure:

      g(p_i; theta) <= 1 for all points p_i

    We use an L-BFGS-B objective with a strong hinge penalty on any
    outside points. Large clouds are subsampled during optimization for
    speed, then the full cloud is enclosed via the smallest uniform
    dilation needed to guarantee enclosure numerically.

    Returns: (cx, cy, cz, ax, ay, az, e1, e2)
    """
    points = np.asarray(points, dtype=np.float64)
    fit_points = points
    if len(points) > MAX_FIT_POINTS:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(points), MAX_FIT_POINTS, replace=False)
        fit_points = points[idx]

    center_init = fit_points.mean(axis=0)
    scales_init = fit_points.std(axis=0) * 2.0
    scales_init = np.maximum(scales_init, 0.03)
    x0 = np.array([*center_init, *scales_init, 1.0, 1.0])

    def objective(params):
        params = np.array(params)
        g = _superquadric_inside_outside(fit_points, params)
        violations = np.maximum(g - 1.0, 0.0)
        enclosure_penalty = np.mean(violations ** 2)
        return _superquadric_log_volume(params) + ENCLOSURE_PENALTY * enclosure_penalty

    data_min = fit_points.min(axis=0)
    data_max = fit_points.max(axis=0)
    data_extent = data_max - data_min
    margin = np.maximum(data_extent * 0.5, 0.05)

    bounds = (
        [(lo, hi) for lo, hi in zip(data_min - margin, data_max + margin)] +
        [(0.01, np.max(data_extent) * 5.0 + 0.1)] * 3 +
        [(0.2, 2.0), (0.2, 2.0)]
    )

    result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-10})

    params = result.x.copy()
    g_final = _superquadric_inside_outside(points, params)
    g_max = float(np.max(g_final))
    dilation_factor = 1.0
    if g_max > 1.0:
        e1 = np.clip(params[6], 0.2, 2.0)
        dilation_factor = float(g_max ** (e1 / 2.0))
        params[3:6] *= dilation_factor

    params[3:6] *= safety_margin
    if not return_debug:
        return params

    final_g = _superquadric_inside_outside(points, params)
    full_min = points.min(axis=0)
    full_max = points.max(axis=0)
    full_extent = full_max - full_min
    max_extent = float(np.max(full_extent))
    scale_to_extent = float(np.max(params[3:6]) / max(max_extent, 1e-9))

    debug = {
        "num_input_points": int(len(points)),
        "num_fit_points": int(len(fit_points)),
        "subsampled_for_fit": bool(len(fit_points) != len(points)),
        "fit_data_bbox_min": data_min.tolist(),
        "fit_data_bbox_max": data_max.tolist(),
        "fit_data_bbox_extent": data_extent.tolist(),
        "full_data_bbox_min": full_min.tolist(),
        "full_data_bbox_max": full_max.tolist(),
        "full_data_bbox_extent": full_extent.tolist(),
        "initial_center": center_init.tolist(),
        "initial_scales": scales_init.tolist(),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_nit": int(getattr(result, "nit", -1)),
        "optimizer_fun": float(result.fun),
        "pre_dilation_g_max": g_max,
        "dilation_factor": dilation_factor,
        "final_g_max": float(np.max(final_g)),
        "log_volume": float(_superquadric_log_volume(params)),
        "scale_to_max_data_extent": scale_to_extent,
    }
    return params, debug


# ---------------------------------------------------------------------------
# 4. CBF evaluation and gradient (superquadric)
# ---------------------------------------------------------------------------

def _superquadric_radial_clearance(x_ee, sq_params):
    """Signed radial clearance from x_ee to the SQ surface along the center ray."""
    cx, cy, cz, ax, ay, az, e1, e2 = np.array(sq_params, dtype=np.float64)
    local = np.array([x_ee[0] - cx, x_ee[1] - cy, x_ee[2] - cz], dtype=np.float64)
    dist = np.linalg.norm(local)
    if dist < 1e-12:
        return -min(ax, ay, az)

    e1 = np.clip(e1, 0.05, 2.0)
    e2 = np.clip(e2, 0.05, 2.0)
    eps = 1e-12
    tx = np.abs(local[0] / ax + eps) ** (2.0 / e2)
    ty = np.abs(local[1] / ay + eps) ** (2.0 / e2)
    g = (tx + ty + eps) ** (e2 / e1) + np.abs(local[2] / az + eps) ** (2.0 / e1)
    g = max(float(g), eps)
    surface_scale = g ** (-0.5 * e1)
    return dist * (1.0 - surface_scale)


def evaluate_cbf(x_ee, sq_params, gripper_radius=GRIPPER_SPHERE_RADIUS):
    """Approximate sphere-vs-SQ safety as radial clearance minus sphere radius."""
    return _superquadric_radial_clearance(x_ee, sq_params) - max(float(gripper_radius), 0.0)


def cbf_gradient(x_ee, sq_params, gripper_radius=GRIPPER_SPHERE_RADIUS):
    """Finite-difference gradient of the radial-clearance CBF."""
    eps = 1e-6
    grad = np.zeros(3, dtype=np.float64)
    for i in range(3):
        pp = np.array(x_ee, dtype=np.float64); pp[i] += eps
        pm = np.array(x_ee, dtype=np.float64); pm[i] -= eps
        grad[i] = (
            evaluate_cbf(pp, sq_params, gripper_radius)
            - evaluate_cbf(pm, sq_params, gripper_radius)
        ) / (2.0 * eps)
    if np.linalg.norm(grad) < 1e-10:
        delta = np.array(x_ee, dtype=np.float64) - np.array(sq_params[:3], dtype=np.float64)
        norm = np.linalg.norm(delta)
        grad = delta / norm if norm > 1e-10 else np.array([0.0, 0.0, 1.0])
    return grad


# ---------------------------------------------------------------------------
# 5. Full pipeline: parse VLM JSON → build constraints
# ---------------------------------------------------------------------------

def build_constraints(vlm_json, obs_folder, camera_keys=None):
    """Return list of constraint dicts and behavioral flags.

    Object point clouds are fused across all requested cameras.
    """
    with open(os.path.join(obs_folder, "metadata.json")) as f:
        meta = json.load(f)
    with open(os.path.join(obs_folder, "camera_params.json")) as f:
        cam = json.load(f)

    preferred_order = ["agentview", "robot0_eye_in_hand", "backview"]
    if camera_keys is None:
        camera_keys = [k for k in preferred_order if k in cam]
        camera_keys += [k for k in cam.keys() if k not in camera_keys]
    else:
        camera_keys = [k for k in camera_keys if k in cam]

    valid_cameras = []
    camera_data = {}
    for camera_key in camera_keys:
        cam_prefix = "eye_in_hand" if camera_key == "robot0_eye_in_hand" else camera_key
        depth_path = os.path.join(obs_folder, f"{cam_prefix}_depth.npy")
        rgb_path = os.path.join(obs_folder, f"{cam_prefix}_rgb.png")
        seg_element_path = os.path.join(obs_folder, f"{cam_prefix}_seg_element.npy")
        seg_default_path = os.path.join(obs_folder, f"{cam_prefix}_seg.npy")
        seg_path = seg_element_path if os.path.exists(seg_element_path) else seg_default_path
        if not (os.path.exists(depth_path) and os.path.exists(rgb_path) and os.path.exists(seg_path)):
            continue

        seg = np.load(seg_path)
        depth = np.squeeze(np.load(depth_path))
        if seg.ndim == 3 and seg.shape[-1] == 1:
            seg = seg.squeeze(-1)
        rgb = np.array(Image.open(rgb_path))
        rgb, depth, seg = align_modalities(rgb, depth, seg, meta)

        camera_data[camera_key] = {
            "seg": seg,
            "depth": depth,
            "intrinsic": cam[camera_key]["intrinsic"],
            "extrinsic": cam[camera_key]["extrinsic"],
            "seg_name": os.path.basename(seg_path),
        }
        valid_cameras.append(camera_key)
        print(f"  [build_constraints] {camera_key}: using {os.path.basename(seg_path)}")

    if not valid_cameras:
        raise RuntimeError("No valid camera data found for constraint construction.")

    geom_id_to_name = meta["geom_id_to_name"]
    eef_state = meta["robot_state"]
    eef_pos = np.array(eef_state["eef_pos"], dtype=np.float64)
    eef_quat = np.array(eef_state.get("eef_quat", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
    gripper_center = offset_gripper_sphere_center(eef_pos, eef_quat)

    objects_data = vlm_json["single"]["objects"]

    constraints = []
    behavioral = {"caution": False, "alpha_scale": 1.0}
    pose = {"rotation_lock": False, "w_rot": 0.0}
    sq_fit_total_seconds = 0.0
    sq_fit_count = 0

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
        object_reference = None
        if obj_name in meta["objects"]:
            object_reference = meta["objects"][obj_name]["position"]

        camera_debug = []
        pc_parts = []
        if geom_ids:
            for camera_key in valid_cameras:
                d = camera_data[camera_key]
                cam_debug = {
                    "camera": camera_key,
                    "seg_name": d["seg_name"],
                    "mask": _mask_debug(d["seg"], d["depth"], geom_ids),
                }
                pc_cam = build_object_point_cloud(
                    d["seg"], d["depth"], geom_ids, d["intrinsic"], d["extrinsic"]
                )
                if pc_cam is not None and len(pc_cam) > 0:
                    cam_debug["points_before_downsample"] = _point_cloud_debug(
                        pc_cam, reference=object_reference
                    )
                    # Keep fitting stable and balanced across views.
                    if len(pc_cam) > 12000:
                        idx = np.random.choice(len(pc_cam), 12000, replace=False)
                        pc_cam = pc_cam[idx]
                        cam_debug["downsampled"] = True
                        cam_debug["downsampled_to"] = int(len(pc_cam))
                        cam_debug["downsample_rng"] = "np.random.choice_global_state"
                    else:
                        cam_debug["downsampled"] = False
                    cam_debug["points_after_downsample"] = _point_cloud_debug(
                        pc_cam, reference=object_reference
                    )
                    pc_parts.append(pc_cam)
                else:
                    cam_debug["points_before_downsample"] = {"num_points": 0}
                    cam_debug["points_after_downsample"] = {"num_points": 0}
                    cam_debug["downsampled"] = False
                camera_debug.append(cam_debug)
        pc = np.vstack(pc_parts) if pc_parts else None

        used_gt_fallback = False
        if pc is None or len(pc) < 10:
            if obj_name in meta["objects"]:
                pc = build_gt_point_cloud(meta["objects"][obj_name]["position"])
                used_gt_fallback = True
            else:
                continue

        object_pc_debug = _point_cloud_debug(pc, reference=object_reference)

        for rel in relationships:
            ext_pc = extend_point_cloud(pc, rel)
            sq_fit_start = time.perf_counter()
            sq_params, fit_debug = fit_superquadric(ext_pc, return_debug=True)
            fit_elapsed = time.perf_counter() - sq_fit_start
            sq_fit_total_seconds += fit_elapsed
            sq_fit_count += 1
            h_val = evaluate_cbf(gripper_center, sq_params)
            grad = cbf_gradient(gripper_center, sq_params)
            extended_pc_debug = _point_cloud_debug(ext_pc, reference=object_reference)

            object_extent = np.array(object_pc_debug.get("bbox_extent", [0.0, 0.0, 0.0]))
            extended_extent = np.array(extended_pc_debug.get("bbox_extent", [0.0, 0.0, 0.0]))
            max_object_extent = float(np.max(object_extent)) if object_extent.size else 0.0
            max_extended_extent = float(np.max(extended_extent)) if extended_extent.size else 0.0
            max_scale = float(np.max(sq_params[3:6]))
            fit_debug["elapsed_seconds"] = float(fit_elapsed)
            fit_debug["scale_to_max_object_extent"] = float(max_scale / max(max_object_extent, 1e-9))
            fit_debug["scale_to_max_extended_extent"] = float(max_scale / max(max_extended_extent, 1e-9))

            debug = {
                "obs_folder": obs_folder,
                "valid_cameras": valid_cameras,
                "geom_ids": sorted(int(gid) for gid in geom_ids),
                "used_gt_fallback": bool(used_gt_fallback),
                "object_reference_position": object_reference,
                "per_camera": camera_debug,
                "object_point_cloud": object_pc_debug,
                "extended_point_cloud": extended_pc_debug,
                "fit": fit_debug,
            }

            print(
                "  [fit_debug] "
                f"{obj_name} | {rel} | cameras={valid_cameras} | "
                f"n_obj={object_pc_debug.get('num_points', 0)} | "
                f"extent={np.round(object_extent, 4).tolist()} | "
                f"scales={np.round(sq_params[3:6], 4).tolist()} | "
                f"scale/object_extent={fit_debug['scale_to_max_object_extent']:.3f} | "
                f"success={fit_debug['optimizer_success']} status={fit_debug['optimizer_status']} | "
                f"g_max={fit_debug['final_g_max']:.3f}"
            )

            constraints.append({
                "object": obj_name,
                "relationship": rel,
                "sq_params": sq_params.tolist(),
                "extended_points": ext_pc,
                "object_points": pc,
                "h_at_eef": float(h_val),
                "gradient_at_eef": grad.tolist(),
                "debug": debug,
            })

    print(f"  [timing] total SQ fitting time: {sq_fit_total_seconds:.3f}s for {sq_fit_count} superquadrics")
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


def sphere_surface(center, radius, n_u=50, n_v=25):
    """Generate a parametric sphere surface for gripper visualization."""
    center = np.asarray(center, dtype=np.float64)
    radius = max(float(radius), 1e-6)

    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    U, V = np.meshgrid(u, v, indexing="xy")

    X = center[0] + radius * np.cos(U) * np.sin(V)
    Y = center[1] + radius * np.sin(U) * np.sin(V)
    Z = center[2] + radius * np.cos(V)
    return X, Y, Z


# ---------------------------------------------------------------------------
# 7. Visualizations
# ---------------------------------------------------------------------------

def _build_scene_point_cloud(obs_folder, camera_keys, cam_params,
                             max_points_per_cam=50000):
    """Back-project full depth+RGB images into a colored world-space point cloud.

    Returns (pts_world, colors_rgb) where colors_rgb is in [0,255] uint8.
    """
    with open(os.path.join(obs_folder, "metadata.json")) as f:
        meta = json.load(f)

    all_pts = []
    all_colors = []

    for cam_key in camera_keys:
        cam_prefix = "eye_in_hand" if cam_key == "robot0_eye_in_hand" else cam_key
        depth_path = os.path.join(obs_folder, f"{cam_prefix}_depth.npy")
        rgb_path = os.path.join(obs_folder, f"{cam_prefix}_rgb.png")
        seg_path = os.path.join(obs_folder, f"{cam_prefix}_seg.npy")
        if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
            continue

        depth = np.squeeze(np.load(depth_path))
        rgb = np.array(Image.open(rgb_path))
        seg = np.load(seg_path) if os.path.exists(seg_path) else np.zeros_like(depth, dtype=np.int32)
        if seg.ndim == 3 and seg.shape[-1] == 1:
            seg = seg.squeeze(-1)
        rgb, depth, seg = align_modalities(rgb, depth, seg, meta)

        intrinsic = np.array(cam_params[cam_key]["intrinsic"])
        extrinsic = np.array(cam_params[cam_key]["extrinsic"])

        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        h, w = depth.shape[:2]
        valid_mask = (depth > 0) & (depth < 2.0)
        vs, us = np.where(valid_mask)
        zs = depth[vs, us]

        if len(zs) > max_points_per_cam:
            idx = np.random.choice(len(zs), max_points_per_cam, replace=False)
            vs, us, zs = vs[idx], us[idx], zs[idx]

        xs_cam = (us - cx) * zs / fx
        ys_cam = (vs - cy) * zs / fy
        pts_cam = np.stack([xs_cam, ys_cam, zs], axis=1)

        # Robosuite extrinsic is camera→world: p_world = R @ p_cam + t
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        pts_world = pts_cam @ R.T + t

        # Workspace crop: discard points far from the tabletop area
        in_ws = (
            (pts_world[:, 0] > -1.0) & (pts_world[:, 0] < 1.0) &
            (pts_world[:, 1] > -1.0) & (pts_world[:, 1] < 1.0) &
            (pts_world[:, 2] > 0.5) & (pts_world[:, 2] < 2.0)
        )
        pts_world = pts_world[in_ws]
        vs, us = vs[in_ws], us[in_ws]

        colors = rgb[vs, us, :3]

        all_pts.append(pts_world)
        all_colors.append(colors)

    if all_pts:
        return np.vstack(all_pts), np.vstack(all_colors).astype(np.uint8)
    return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)


def _make_scene_fig(obs_folder, cam_params, eef_pos, eef_quat=None,
                    gripper_radius=GRIPPER_SPHERE_RADIUS,
                    gripper_color="cyan", gripper_name="Gripper sphere"):
    """Build a plotly Figure with scene point cloud, GT object markers, and gripper."""
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

    # GT object markers from metadata
    if obs_folder is not None:
        meta_path = os.path.join(obs_folder, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if eef_quat is None:
                eef_quat = meta.get("robot_state", {}).get("eef_quat")
            objects = meta.get("objects", {})
            obj_names, obj_pos = [], []
            for name, data in objects.items():
                pos = data["position"]
                if all(abs(p) < 5.0 for p in pos):
                    obj_names.append(name.replace("_", " ").title())
                    obj_pos.append(pos)
            if obj_pos:
                obj_pos = np.array(obj_pos)
                fig.add_trace(go.Scatter3d(
                    x=obj_pos[:, 0], y=obj_pos[:, 1], z=obj_pos[:, 2],
                    mode="markers+text",
                    marker=dict(size=6, color="orange", symbol="diamond",
                                line=dict(width=1, color="white")),
                    text=obj_names,
                    textposition="top center",
                    textfont=dict(size=9, color="white"),
                    name="Objects (GT)",
                    hovertext=[f"{n}<br>({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
                               for n, p in zip(obj_names, obj_pos)],
                    hoverinfo="text",
                ))

    # Gripper sphere plus center marker
    gripper_center = offset_gripper_sphere_center(eef_pos, eef_quat)
    Xg, Yg, Zg = sphere_surface(gripper_center, gripper_radius, n_u=40, n_v=20)
    fig.add_trace(go.Surface(
        x=Xg, y=Yg, z=Zg,
        opacity=0.45,
        colorscale=[[0, gripper_color], [1, gripper_color]],
        showscale=False,
        name=gripper_name,
        hoverinfo="name",
    ))
    fig.add_trace(go.Scatter3d(
        x=[gripper_center[0]], y=[gripper_center[1]], z=[gripper_center[2]],
        mode="markers",
        marker=dict(size=5, color=gripper_color, symbol="cross"),
        name="Gripper center",
        showlegend=False,
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
            bgcolor="rgb(15,15,15)",
        ),
        paper_bgcolor="rgb(10,10,10)",
        width=1100, height=900,
        legend=dict(font=dict(size=10, color="white")),
    )
    return fig


def vis_scene_html(eef_pos, save_path, obs_folder=None, cam_params=None):
    """Save a 3D scene point cloud (no superquadrics) as HTML."""
    fig = _make_scene_fig(obs_folder, cam_params, eef_pos)
    fig.update_layout(title="3D Scene Map")
    fig.write_html(save_path)
    print(f"  Saved {save_path}")


def vis_pointcloud_only_html(constraints, eef_pos, save_path, obs_folder=None,
                             cam_params=None):
    """Save point-cloud-only visualization (no superquadric surfaces)."""
    import plotly.graph_objects as go

    fig = _make_scene_fig(obs_folder, cam_params, eef_pos)

    colors = [
        "rgb(255,50,50)", "rgb(50,50,255)",
        "rgb(50,200,50)", "rgb(255,165,0)",
        "rgb(200,50,200)", "rgb(0,200,200)",
    ]

    # Keep one "object seg cloud" trace per object to avoid duplicate legend items.
    added_object_names = set()

    for idx, c in enumerate(constraints):
        lc = colors[idx % len(colors)]
        obj_name = c["object"]
        rel_name = c["relationship"]

        obj_pts = c.get("object_points")
        if obj_pts is not None and len(obj_pts) > 0 and obj_name not in added_object_names:
            obj_pts = np.asarray(obj_pts)
            fig.add_trace(go.Scatter3d(
                x=obj_pts[:, 0], y=obj_pts[:, 1], z=obj_pts[:, 2],
                mode="markers",
                marker=dict(size=3, color=lc, opacity=0.9),
                name=f"Object seg cloud: {obj_name}",
                hoverinfo="name",
            ))
            added_object_names.add(obj_name)

        ext_pts = c.get("extended_points")
        if ext_pts is not None and len(ext_pts) > 0:
            ext_pts = np.asarray(ext_pts)
            fig.add_trace(go.Scatter3d(
                x=ext_pts[:, 0], y=ext_pts[:, 1], z=ext_pts[:, 2],
                mode="markers",
                marker=dict(size=2, color=lc, opacity=0.25),
                name=f"Constraint cloud: {obj_name} ({rel_name})",
                visible="legendonly",
                hoverinfo="name",
            ))

    fig.update_layout(title="Point Cloud Only (No Superquadrics)")
    fig.write_html(save_path)
    print(f"  Saved {save_path}")


def vis_3d_html(constraints, eef_pos, save_path, obs_folder=None,
                cam_params=None):
    """Scene point cloud + object point clouds + superquadric overlays as HTML."""
    import plotly.graph_objects as go

    h_min = min(c["h_at_eef"] for c in constraints) if constraints else 1.0
    gripper_color = "green" if h_min > 0 else "red"
    fig = _make_scene_fig(
        obs_folder, cam_params, eef_pos,
        gripper_color=gripper_color,
        gripper_name=f"Gripper sphere (h_min={h_min:.2f})",
    )

    line_colors = [
        "rgb(255,50,50)", "rgb(50,50,255)",
        "rgb(50,200,50)", "rgb(255,165,0)",
        "rgb(200,50,200)", "rgb(0,200,200)",
    ]

    for idx, c in enumerate(constraints):
        sq = np.array(c["sq_params"])
        X, Y, Z = superquadric_surface(sq, n_u=50, n_v=25)
        lc = line_colors[idx % len(line_colors)]
        obj_name = c["object"]
        rel_name = c["relationship"]
        group_name = f"{obj_name} ({rel_name})"

        # Object point cloud (if available)
        obj_pts = c.get("object_points")
        if obj_pts is not None and len(obj_pts) > 0:
            obj_pts = np.asarray(obj_pts)
            fig.add_trace(go.Scatter3d(
                x=obj_pts[:, 0], y=obj_pts[:, 1], z=obj_pts[:, 2],
                mode="markers",
                marker=dict(size=3, color=lc, opacity=0.8),
                name=f"Object points: {obj_name}",
                legendgroup=group_name,
                showlegend=False,
            ))

        # Extended point cloud (constraint region)
        ext_pts = c.get("extended_points")
        if ext_pts is not None and len(ext_pts) > 0:
            ext_pts = np.asarray(ext_pts)
            fig.add_trace(go.Scatter3d(
                x=ext_pts[:, 0], y=ext_pts[:, 1], z=ext_pts[:, 2],
                mode="markers",
                marker=dict(size=2, color=lc, opacity=0.3),
                name=f"Extended points: {obj_name} ({rel_name})",
                visible="legendonly",
                legendgroup=group_name,
                showlegend=False,
            ))

        # Superquadric surface
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            opacity=0.25,
            colorscale=[[0, lc], [1, lc]],
            showscale=False,
            name=f"SQ surface: {obj_name} ({rel_name})",
            hoverinfo="name",
            legendgroup=group_name,
            showlegend=True,
        ))

        # Superquadric center marker
        fig.add_trace(go.Scatter3d(
            x=[sq[0]], y=[sq[1]], z=[sq[2]],
            mode="markers",
            marker=dict(size=5, color=lc),
            name=f'{c["object"]} center',
            legendgroup=group_name,
            showlegend=False,
        ))

    fig.update_layout(
        title="CBF Superquadric Constraints — 3D Scene",
        legend=dict(groupclick="togglegroup"),
    )
    fig.write_html(save_path)
    print(f"  Saved {save_path}")


def vis_rgb_overlay(constraints, rgb_path, intrinsic, extrinsic, save_path):
    """Project h=0 superquadric surface onto RGB image."""
    img = np.array(Image.open(rgb_path))
    overlay = img.copy().astype(np.float64)

    K = np.array(intrinsic)
    T = np.array(extrinsic)
    # Extrinsic is camera→world; invert for world→camera projection
    R = T[:3, :3]
    t_vec = T[:3, 3]

    colors_list = [
        [255, 50, 50], [50, 50, 255], [50, 200, 50],
        [255, 165, 0], [200, 50, 200], [0, 200, 200],
    ]

    for idx, c in enumerate(constraints):
        sq = np.array(c["sq_params"])
        X, Y, Z = superquadric_surface(sq, n_u=80, n_v=40)
        pts_3d = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        # World to camera: p_cam = R^T @ (p_world - t)
        pts_cam = (pts_3d - t_vec) @ R

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
                label=f'{c["object"]} / {c["relationship"]}')
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
    constraint_build_start = time.perf_counter()
    constraints, behavioral, pose, eef_pos = build_constraints(
        vlm, obs, camera_keys=None)
    constraint_build_seconds = time.perf_counter() - constraint_build_start
    print(f"  [timing] constraint construction total: {constraint_build_seconds:.3f}s")
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
                "debug": c.get("debug", {}),
            }
            for c in constraints
        ],
        "behavioral": behavioral,
        "pose": pose,
        "gripper": {
            "type": "sphere",
            "radius": GRIPPER_SPHERE_RADIUS,
            "center_offset": GRIPPER_SPHERE_CENTER_OFFSET.tolist(),
            "approximation": "radial_clearance_minus_sphere_radius",
        },
    }
    params_path = os.path.join(out, "cbf_params.json")
    with open(params_path, "w") as f:
        json.dump(cbf_json, f, indent=2)
    print(f"\n  Saved {params_path}")

    # --- Visualization ---
    print("\nGenerating vis_3d.html ...")
    vis_3d_html(constraints, eef_pos, os.path.join(out, "vis_3d.html"),
                obs_folder=obs, cam_params=cam)

    print("\nDone. Saved cbf_params.json and vis_3d.html to:", out)


if __name__ == "__main__":
    main()
