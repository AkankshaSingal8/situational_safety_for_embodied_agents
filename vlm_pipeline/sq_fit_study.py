"""Offline study: full superquadric fit on the obstacle's masked-depth cloud
vs the current percentile-box extents (no-GT obstacle_scales source).

Arms per saved scene (19 obstacle scenes, same set as extent_id_smoke):
  B  percentile box  — center = cloud median, scales = (p95-p5)/2, yaw 0,
                       eps fixed (whatever the server flag says; 0.4 boxy).
  C  SQ fit          — center(3) + log-scales(3) + eps + yaw fitted to the
                       cloud by inside-outside residual, after table-plane
                       completion (mirror about the vertical plane through
                       the centroid along the camera ray + clamp to table).

Judged WITHOUT GT extents (not saved offline):
  coverage  = % of raw cloud points inside the shape (F<=1). Fail-safe axis:
              undercoverage -> barrier misses real object volume.
  volume    = 8*sx*sy*sz box-volume proxy. Fail-live axis: inflation ->
              grasp smothering (the t1 failure).
  center_e  = ||fitted center - GT center|| (metadata GT, xy and 3d).
  yaw_e     = |fitted yaw - GT yaw| mod 90deg (box-symmetric).
Gate for adopting C: coverage_C >= coverage_B - 2pp AND volume_C < volume_B
on the median scene, center_e no worse.
"""

import json
import math
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import extent_id_smoke as eis  # noqa: E402
import vlm_slot_bench as vb  # noqa: E402

Z_TABLE = 0.862  # G5 plane-fit value (hardcoded 0.81 is ~5cm low)


def obstacle_cloud(ep_dir, ob_name, gt_pos):
    ep = pathlib.Path(ep_dir)
    cp = json.load(open(ep / "camera_params.json"))["agentview"]
    K = np.array(cp["intrinsic"]); T = np.array(cp["extrinsic"])
    depth = np.load(ep / "agentview_depth.npy").squeeze()[::-1, ::-1]
    seg = np.load(ep / "agentview_seg.npy").squeeze()[::-1, ::-1]
    best = None
    for sid in np.unique(seg):
        if sid <= 0:
            continue
        cloud = eis.region_cloud(depth, seg == sid, K, T)
        if cloud is None or len(cloud) < 30:
            continue
        cen = np.median(cloud, axis=0)
        if not (0.6 < cen[2] < 1.3):
            continue
        d = np.linalg.norm(cen[:2] - gt_pos[:2])
        if d < 0.10 and (best is None or len(cloud) > len(best)):
            best = cloud
    return best


def complete_cloud(cloud, cam_pos):
    """Single-view completion: mirror about the vertical plane through the
    xy-centroid perpendicular to the camera ray, and clamp base to table."""
    cen = np.median(cloud, axis=0)
    ray = cen[:2] - cam_pos[:2]
    ray = ray / (np.linalg.norm(ray) + 1e-9)
    d = (cloud[:, :2] - cen[:2]) @ ray
    mirrored = cloud.copy()
    mirrored[:, :2] -= 2.0 * d[:, None] * ray[None, :]
    out = np.vstack([cloud, mirrored])
    # ground the base: add the xy-footprint projected to the table plane
    foot = out[out[:, 2] < np.percentile(out[:, 2], 30)].copy()
    foot[:, 2] = Z_TABLE
    return np.vstack([out, foot])


def sq_inside(pts, center, scales, eps, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    q = (pts - center) @ R.T
    a = np.maximum(scales, 1e-3)
    p = 2.0 / max(eps, 0.25)
    F = (np.abs(q[:, 0] / a[0]) ** p + np.abs(q[:, 1] / a[1]) ** p) ** (eps / 1.0)
    Fz = np.abs(q[:, 2] / a[2]) ** 2.0
    return F + Fz  # <=1 inside (eps blends xy only; z kept quadric — barrier analog)


def fit_sq(cloud, cam_pos):
    comp = complete_cloud(cloud, cam_pos)
    cen0 = np.median(comp, axis=0)
    ext0 = np.maximum((np.percentile(comp, 95, axis=0) - np.percentile(comp, 5, axis=0)) / 2.0, 0.015)

    def loss(x):
        center = x[:3]; scales = np.exp(x[3:6]); eps = x[6]; yaw = x[7]
        F = sq_inside(comp, center, scales, eps, yaw)
        # points should sit ON or IN the surface: hinge outside residual +
        # small volume regularizer to keep the shape tight
        outside = np.maximum(F - 1.0, 0.0)
        return float(np.mean(outside ** 2) + 0.02 * np.prod(scales) / np.prod(ext0))

    x0 = np.concatenate([cen0, np.log(ext0), [0.6], [0.0]])
    bounds = ([(cen0[0] - 0.08, cen0[0] + 0.08), (cen0[1] - 0.08, cen0[1] + 0.08),
               (cen0[2] - 0.08, cen0[2] + 0.08)]
              + [(math.log(0.015), math.log(0.40))] * 3
              + [(0.3, 1.2), (-math.pi / 4, math.pi / 4)])
    res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
    x = res.x
    return dict(center=x[:3], scales=np.exp(x[3:6]), eps=float(x[6]), yaw=float(x[7]))


def yaw_from_quat(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main():
    eps_list = vb.load_episodes(["safelibero_spatial"], 5)
    rows = []
    for e in eps_list:
        ep = pathlib.Path(e["ep"])
        ob = json.load(open(ep / "obstacle.json"))["active_obstacle"]
        if not ob:
            continue
        gt_pos = np.array(ob["position"])
        cp = json.load(open(ep / "camera_params.json"))["agentview"]
        cam_pos = np.array(cp["extrinsic"])[:3, 3]
        cloud = obstacle_cloud(e["ep"], ob["name"], gt_pos)
        if cloud is None:
            rows.append({"ep": e["ep"], "name": ob["name"], "fail": "no cloud"})
            continue
        # Arm B
        cenB = np.median(cloud, axis=0)
        sB = np.maximum((np.percentile(cloud, 95, axis=0) - np.percentile(cloud, 5, axis=0)) / 2.0, 0.015)
        covB = float(np.mean(sq_inside(cloud, cenB, sB, 0.4, 0.0) <= 1.0))
        volB = float(8.0 * np.prod(sB))
        # Arm C
        fit = fit_sq(cloud, cam_pos)
        covC = float(np.mean(sq_inside(cloud, fit["center"], fit["scales"], fit["eps"], fit["yaw"]) <= 1.0))
        volC = float(8.0 * np.prod(fit["scales"]))
        gt_yaw = yaw_from_quat(ob["quaternion"])
        yaw_e = abs((fit["yaw"] - gt_yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4)
        rows.append({
            "ep": str(e["ep"]).split("safelibero_spatial/")[-1], "name": ob["name"],
            "covB": round(covB, 3), "covC": round(covC, 3),
            "volB_cm3": round(volB * 1e6), "volC_cm3": round(volC * 1e6),
            "cenB_err": round(float(np.linalg.norm(cenB - gt_pos)), 3),
            "cenC_err": round(float(np.linalg.norm(fit["center"] - gt_pos)), 3),
            "epsC": round(fit["eps"], 2), "yawC_err_deg": round(math.degrees(yaw_e), 1),
        })
    ok = [r for r in rows if "fail" not in r]
    for r in rows:
        print(json.dumps(r))
    if ok:
        med = lambda k: float(np.median([r[k] for r in ok]))
        print(json.dumps({
            "n": len(ok),
            "median": {k: med(k) for k in
                       ["covB", "covC", "volB_cm3", "volC_cm3", "cenB_err", "cenC_err", "yawC_err_deg"]},
        }, indent=1))


if __name__ == "__main__":
    main()


def fit_yaw_only(cloud):
    """Arm D: percentile scales FIXED (robust), fit yaw alone by minimizing the
    rotated-frame axis-aligned box volume of the cloud (min-volume orientation
    — no inside/outside residual, no completion, no inflation possible)."""
    cen = np.median(cloud, axis=0)
    q = cloud[:, :2] - cen[:2]

    def vol2d(yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        r = q @ np.array([[c, -s], [s, c]])
        e = np.percentile(r, 95, axis=0) - np.percentile(r, 5, axis=0)
        return float(e[0] * e[1])

    grid = np.linspace(-math.pi / 4, math.pi / 4, 91)
    yaw = float(grid[int(np.argmin([vol2d(y) for y in grid]))])
    c, s = math.cos(yaw), math.sin(yaw)
    r = q @ np.array([[c, -s], [s, c]])
    exy = (np.percentile(r, 95, axis=0) - np.percentile(r, 5, axis=0)) / 2.0
    ez = (np.percentile(cloud[:, 2], 95) - np.percentile(cloud[:, 2], 5)) / 2.0
    return yaw, np.array([exy[0], exy[1], ez])


def main_yaw():
    eps_list = vb.load_episodes(["safelibero_spatial"], 5)
    rows = []
    for e in eps_list:
        ep = pathlib.Path(e["ep"])
        ob = json.load(open(ep / "obstacle.json"))["active_obstacle"]
        if not ob:
            continue
        gt_pos = np.array(ob["position"])
        cloud = obstacle_cloud(e["ep"], ob["name"], gt_pos)
        if cloud is None:
            continue
        yaw, sc = fit_yaw_only(cloud)
        gt_yaw = yaw_from_quat(ob["quaternion"])
        err = abs((yaw - gt_yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4)
        sB = np.maximum((np.percentile(cloud, 95, axis=0) - np.percentile(cloud, 5, axis=0)) / 2.0, 0.015)
        rows.append({"name": ob["name"], "yaw_deg": round(math.degrees(yaw), 1),
                     "gt_yaw_deg": round(math.degrees(gt_yaw), 1),
                     "err_deg": round(math.degrees(err), 1),
                     "vol_axisaligned_cm3": round(8e6 * np.prod(sB)),
                     "vol_rotated_cm3": round(8e6 * np.prod(np.maximum(sc, 0.015)))})
    for r in rows:
        print(json.dumps(r))
    if rows:
        errs = [r["err_deg"] for r in rows]
        shrink = [r["vol_rotated_cm3"] / max(r["vol_axisaligned_cm3"], 1) for r in rows]
        print(json.dumps({"n": len(rows), "yaw_err_median_deg": float(np.median(errs)),
                          "yaw_err_p90_deg": float(np.percentile(errs, 90)),
                          "volume_ratio_median": round(float(np.median(shrink)), 3)}))
