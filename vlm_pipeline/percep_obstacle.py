"""Runtime RGB-D obstacle localization for the no-GT tier (Phase 5, geometry leg).

Estimates the active obstacle's 3D position from camera observations instead
of reading `obs[f"{name}_pos"]`. At runtime the camera matrices come straight
from the live sim (no saved-array orientation ambiguity — that bug lives only
in the offline captures).

Tier semantics:
- `region_source="gt_seg"`: pixel region from GT instance segmentation →
  isolates GEOMETRY error (Tier-SemID ablation arm for E3).
- `region_source="detector"`: pixel region from an external detector bbox
  (Tier-Percep; detector callable injected by the caller).

Position estimator: per-camera depth-continuous mask -> (centroid px, median
depth) -> back-projection; multi-camera mean fusion. Offline floor measured at
med 0.067 m / p90 0.152 m with 2-view fusion (see ledger, 2026-07-16);
runtime should do no worse.
"""

import os

import numpy as np

# E5 fail-direction stress test: deterministic position corruption, env-gated
# (default off). Direction rotates per call so errors cover the xy plane.
_E5_CORRUPT = os.environ.get("E5_CORRUPT", "")
_E5_POS_NOISE = float(os.environ.get("E5_POS_NOISE", "0.08"))
_e5_call_count = 0


WORKSPACE = {"x": (-0.6, 0.6), "y": (-0.6, 0.6), "z": (0.60, 1.45)}


def _in_workspace(p):
    return (WORKSPACE["x"][0] < p[0] < WORKSPACE["x"][1]
            and WORKSPACE["y"][0] < p[1] < WORKSPACE["y"][1]
            and WORKSPACE["z"][0] < p[2] < WORKSPACE["z"][1])


def make_gdino_detector(
    python_bin="/ocean/projects/cis250185p/asingal/envs/gdino_py310/bin/python",
    service=None, box_threshold=0.30, workdir=None,
):
    """Persistent GroundingDINO subprocess detector (heavyweight torch env
    stays out of the eval process). Returns detector(rgb, name) ->
    [((x0,y0,x1,y1), logit), ...] sorted handling left to the caller.
    Returns None on service failure (caller falls back / skips the view).
    """
    import json as _json
    import pathlib
    import subprocess
    import tempfile

    if service is None:
        service = str(pathlib.Path(__file__).parent / "gdino_service.py")
    workdir = pathlib.Path(workdir or tempfile.mkdtemp(prefix="gdino_"))
    proc = subprocess.Popen([python_bin, service], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, bufsize=1)
    for line in proc.stdout:
        if line.strip() == "READY":
            break

    _n = [0]

    def detector(rgb, name):
        from PIL import Image
        _n[0] += 1
        img_path = workdir / f"det_{_n[0] % 4}.png"
        Image.fromarray(rgb).save(img_path)
        proc.stdin.write(_json.dumps({
            "image": str(img_path),
            "phrase": name.replace("_obstacle", "").replace("_", " "),
            "box_threshold": box_threshold}) + "\n")
        proc.stdin.flush()
        resp = _json.loads(proc.stdout.readline())
        if "error" in resp:
            return None
        return list(zip([tuple(b) for b in resp["boxes"]], resp["logits"]))

    detector.proc = proc  # for lifecycle management by the caller
    return detector


def _camera_transform(sim, camera_name, camera_height, camera_width):
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )
    K = get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width)
    T = get_camera_extrinsic_matrix(sim, camera_name)
    return K, T


def _backproject(u, v, z, K, T):
    p_cam = z * (np.linalg.inv(K) @ np.array([u, v, 1.0]))
    p_w = T @ np.array([*p_cam, 1.0])
    return p_w[:3]


def _region_estimate(depth, region_mask, K, T, depth_window=0.20):
    """Centroid-pixel + median-depth back-projection over a pixel region."""
    from scipy import ndimage

    ys, xs = np.nonzero(region_mask)
    if len(ys) < 20:
        return None
    # anchor on the region's median-depth pixel, then keep the
    # depth-continuous connected component (drops background bbox pixels)
    zs_all = depth[region_mask]
    ok = (zs_all > 0.1) & (zs_all < 5.0)
    if ok.sum() < 20:
        return None
    z_anchor = float(np.median(zs_all[ok]))
    cont = region_mask & (np.abs(depth - z_anchor) < depth_window)
    lab, n = ndimage.label(cont)
    if n == 0:
        return None
    sizes = ndimage.sum(cont, lab, range(1, n + 1))
    mask = lab == (1 + int(np.argmax(sizes)))
    ys, xs = np.nonzero(mask)
    zs = depth[mask]
    zs = zs[(zs > 0.1) & (zs < 5.0)]
    if len(zs) < 10:
        return None
    return _backproject(float(xs.mean()), float(ys.mean()), float(np.median(zs)), K, T)


def estimate_obstacle_pos(sim, obstacle_name, cameras=("agentview", "birdview"),
                          camera_height=256, camera_width=256,
                          region_source="gt_seg", detector=None):
    """Estimate the obstacle's world position from RGB-D renders.

    Renders depth (+ instance segmentation for gt_seg) per camera via
    sim.render, builds the obstacle's pixel region, back-projects, and fuses
    across cameras by mean. Returns (pos, n_views) or (None, 0).
    """
    from robosuite.utils.camera_utils import get_real_depth_map

    ests = []
    for cam in cameras:
        try:
            K, T = _camera_transform(sim, cam, camera_height, camera_width)
        except Exception:
            continue
        depth = sim.render(camera_name=cam, height=camera_height,
                           width=camera_width, depth=True)[1][::-1]
        depth = get_real_depth_map(sim, depth[..., None] if depth.ndim == 2 else depth).squeeze()

        if region_source == "gt_seg":
            seg = sim.render(camera_name=cam, height=camera_height,
                             width=camera_width,
                             segmentation=True)[::-1, :, 1]
            geom_ids = [g for g in range(sim.model.ngeom)
                        if obstacle_name in (sim.model.geom_id2name(g) or "")]
            region = np.isin(seg, geom_ids)
        elif region_source == "detector":
            rgb = sim.render(camera_name=cam, height=camera_height,
                             width=camera_width)[::-1]
            cands = detector(rgb, obstacle_name) if detector else None
            if not cands:
                continue
            if isinstance(cands, (tuple, list)) and len(cands) == 4 \
                    and np.isscalar(cands[0]):
                cands = [(cands, 1.0)]  # legacy single-bbox detectors
            # Try boxes in logit order; keep the first whose back-projection
            # lands inside the workspace (kills wall/background false
            # positives — the dominant open-vocab failure mode).
            est_c = None
            for bbox, _logit in sorted(cands, key=lambda c: -c[1]):
                x0, y0, x1, y1 = [int(round(b)) for b in bbox]
                region = np.zeros(depth.shape, dtype=bool)
                region[max(0, y0):y1, max(0, x0):x1] = True
                e = _region_estimate(depth, region, K, T)
                if e is not None and _in_workspace(e):
                    est_c = e
                    break
            if est_c is not None:
                ests.append(est_c)
            continue
        else:
            raise ValueError(region_source)

        est = _region_estimate(depth, region, K, T)
        if est is not None:
            ests.append(est)
    if not ests:
        return None, 0
    fused = np.mean(ests, axis=0)
    if _E5_CORRUPT in ("position", "both"):
        global _e5_call_count
        ang = _e5_call_count * 2.399963  # golden-angle rotation, deterministic
        _e5_call_count += 1
        fused = fused + _E5_POS_NOISE * np.array([np.cos(ang), np.sin(ang), 0.0])
    return fused, len(ests)


def estimate_obstacle_extent(sim, obstacle_name, cameras=("agentview", "birdview"),
                             camera_height=256, camera_width=256,
                             region_source="gt_seg", detector=None):
    """Percep-tier half-extents: per-axis p5-p95 spread of the obstacle's
    masked-depth point cloud, fused across cameras by max (each view sees a
    different projection of the object). Visible-surface bias underestimates
    the hidden axis; the guidance pad (eef_radius + d_safe) absorbs that.
    Returns (3,) half-extents in meters, or None (fail-safe: caller falls
    back to the scalar radius sphere)."""
    from robosuite.utils.camera_utils import get_real_depth_map

    best = None
    for cam in cameras:
        try:
            K, T = _camera_transform(sim, cam, camera_height, camera_width)
        except Exception:
            continue
        depth = sim.render(camera_name=cam, height=camera_height,
                           width=camera_width, depth=True)[1][::-1]
        depth = get_real_depth_map(sim, depth[..., None] if depth.ndim == 2 else depth).squeeze()
        if region_source == "gt_seg":
            seg = sim.render(camera_name=cam, height=camera_height,
                             width=camera_width, segmentation=True)[::-1, :, 1]
            geom_ids = [g for g in range(sim.model.ngeom)
                        if obstacle_name in (sim.model.geom_id2name(g) or "")]
            region = np.isin(seg, geom_ids)
        else:
            cands = detector(sim.render(camera_name=cam, height=camera_height,
                                        width=camera_width)[::-1], obstacle_name) \
                if detector else None
            if not cands:
                continue
            bbox = max(cands, key=lambda c: c[1])[0]
            x0, y0, x1, y1 = [int(round(b)) for b in bbox]
            region = np.zeros(depth.shape, dtype=bool)
            region[max(0, y0):y1, max(0, x0):x1] = True
        ys, xs = np.nonzero(region)
        if len(ys) < 30:
            continue
        zs = depth[ys, xs]
        ok = (zs > 0.1) & (zs < 5.0)
        if ok.sum() < 30:
            continue
        ys, xs, zs = ys[ok], xs[ok], zs[ok]
        z_med = np.median(zs)
        keep = np.abs(zs - z_med) < 0.25
        pix = np.stack([xs[keep], ys[keep], np.ones(keep.sum())]).astype(np.float64)
        cam_pts = np.linalg.inv(K) @ pix * zs[keep][None, :]
        world = (T @ np.vstack([cam_pts, np.ones((1, cam_pts.shape[1]))]))[:3].T
        half = (np.percentile(world, 95, axis=0) - np.percentile(world, 5, axis=0)) / 2.0
        best = half if best is None else np.maximum(best, half)
    if best is None:
        return None
    return np.clip(best, 0.02, 0.35).astype(np.float32)


def estimate_object_positions(sim, names, cameras=("agentview",),
                              camera_height=256, camera_width=256,
                              region_source="gt_seg", detector=None,
                              z_correction=-0.03):
    """Batch per-object localization for the full Tier-Percep entity stack.

    Returns {name: pos} for every object whose estimate succeeded; failures
    are simply absent (fail-safe: an unlocalized object cannot become an
    identity candidate or grant a corridor). One render pass per object per
    camera — called once per episode, so cost is negligible.
    """
    out = {}
    for name in names:
        pos, n_views = estimate_obstacle_pos(
            sim, name, cameras=cameras,
            camera_height=camera_height, camera_width=camera_width,
            region_source=region_source, detector=detector)
        if pos is not None:
            out[name] = pos + np.array([0.0, 0.0, z_correction])
    return out


if __name__ == "__main__":
    print(__doc__)
