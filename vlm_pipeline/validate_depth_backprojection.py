"""No-GT tier perception, step 1: validate RGB-D -> 3D obstacle localization.

Question: if a detector hands us a 2D region for the obstacle, how accurately
does bbox/mask-median depth back-projection recover its 3D world position,
compared against the GT position the sim recorded (`obstacle.json`)?

Validation-only shortcut: the obstacle's pixel region comes from the saved GT
instance segmentation (the seg id is found by projecting the GT position into
the image). This isolates GEOMETRY error (depth + calibration + median
aggregation) from detection error, which the VLM/detector layer owns. A bbox
variant (fill the mask's bounding box) approximates detector-quality regions.

Camera convention is auto-calibrated: both candidate pixel->camera-frame
conventions are tried on the first episode and the one whose projection of the
GT position lands nearest the object's mask wins.

Usage: python validate_depth_backprojection.py [vlm_inputs_root]
"""

import json
import pathlib
import sys

import numpy as np

# robosuite/mujoco camera frame: x right, y UP, z BACK (OpenGL). Image pixels:
# x right, y DOWN. Convention A applies the y/z flip; B assumes z forward.
CONV = {
    "A": np.diag([1.0, -1.0, -1.0]),
    "B": np.eye(3),
}


def backproject(u, v, z, K, T_wc, conv):
    p_img = np.array([u, v, 1.0])
    p_cam = z * (np.linalg.inv(K) @ p_img)
    p_cam = conv @ p_cam
    p_w = T_wc @ np.array([*p_cam, 1.0])
    return p_w[:3]


def project(p_world, K, T_wc, conv):
    p_cam = np.linalg.inv(T_wc) @ np.array([*p_world, 1.0])
    p_cam = np.linalg.inv(conv) @ p_cam[:3]
    if p_cam[2] <= 0:
        return None
    uv = K @ (p_cam / p_cam[2])
    return uv[:2]


def episode_error(ep_dir, cam="agentview", conv_key="A"):
    obst = json.load(open(ep_dir / "obstacle.json"))["active_obstacle"]
    if obst is None:
        return None
    gt = np.array(obst["position"])
    meta = json.load(open(ep_dir / "metadata.json"))
    eef = np.array(meta["robot_state"]["eef_pos"])
    cams = json.load(open(ep_dir / "camera_params.json"))
    K = np.array(cams[cam]["intrinsic"])
    T = np.array(cams[cam]["extrinsic"])
    depth = np.load(ep_dir / f"{cam}_depth.npy").squeeze()
    seg = np.load(ep_dir / f"{cam}_seg.npy").squeeze()
    conv = CONV[conv_key]

    # Saved arrays have INCONSISTENT vertical orientation across tasks
    # (task_1/3 renders are v-flipped vs task_0/2). Auto-orient per episode:
    # the EEF is a known 3D landmark — pick the orientation whose depth at the
    # EEF's projected pixel best matches its camera-frame z.
    uv_e = project(eef, K, T, conv)
    orientations = [lambda a: a, lambda a: a[::-1, :]]
    flip_score = []
    for orient in orientations:
        d_o = orient(depth)
        if uv_e is None:
            flip_score.append(np.inf)
            continue
        ue, ve = int(round(uv_e[0])), int(round(uv_e[1]))
        z_exp = (np.linalg.inv(T) @ np.array([*eef, 1.0]))[2]
        if 0 <= ve < d_o.shape[0] and 0 <= ue < d_o.shape[1]:
            flip_score.append(abs(float(d_o[ve, ue]) - float(z_exp)))
        else:
            flip_score.append(np.inf)
    best_orient = orientations[int(np.argmin(flip_score))]

    uv = project(gt, K, T, conv)
    if uv is None:
        return None
    u, v = int(round(uv[0])), int(round(uv[1]))
    h, w = seg.shape
    for flip in (False,):
        s = best_orient(seg)
        d = best_orient(depth)
        if 0 <= v < h and 0 <= u < w and s[v, u] > 0:
            seg_id = s[v, u]
            mask = s == seg_id
            if mask.sum() < 20:
                continue
            # Instance ids can merge the obstacle with table/other bodies
            # (observed 32k-px "masks"). Keep only the connected component
            # around the hit pixel that is depth-continuous (±20 cm).
            from scipy import ndimage
            cont = mask & (np.abs(d - d[v, u]) < 0.20)
            lab, _ = ndimage.label(cont)
            if lab[v, u] == 0:
                continue
            mask = lab == lab[v, u]
            if mask.sum() < 20:
                continue
            out = {}
            ys, xs = np.nonzero(mask)
            for name, region in (
                ("mask", mask),
                ("bbox", np.zeros_like(mask)),
            ):
                if name == "bbox":
                    region = np.zeros_like(mask)
                    region[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
                zs = d[region]
                zs = zs[(zs > 0.1) & (zs < 5.0)]
                z_med = np.median(zs)
                u_c, v_c = xs.mean() if name == "mask" else (xs.min() + xs.max()) / 2, \
                           ys.mean() if name == "mask" else (ys.min() + ys.max()) / 2
                est = backproject(u_c, v_c, z_med, K, T, conv)
                # Surface-to-center correction: median depth is the visible
                # SURFACE; push along the viewing ray by the apparent radius
                # (half the mean pixel extent scaled to metric at that depth).
                r_est = 0.25 * ((xs.max() - xs.min()) + (ys.max() - ys.min())) \
                    * z_med / K[0, 0]
                cam_center = T[:3, 3]
                ray = est - cam_center
                ray = ray / (np.linalg.norm(ray) + 1e-9)
                est_c = est + ray * r_est
                out[name] = float(np.linalg.norm(est_c - gt))
                out[f"{name}_xy"] = float(np.linalg.norm((est_c - gt)[:2]))
            out["flip"] = flip
            return out
    return None


def main(root):
    eps = sorted(pathlib.Path(root).glob("level_*/task_*/episode_*"))
    # run BOTH conventions; the one with lower median mask error is correct —
    # a single-episode hit test can false-positive on a neighboring mask.
    for conv_key in ("A", "B"):
        errs = {"mask": [], "bbox": [], "mask_xy": [], "bbox_xy": []}
        misses = 0
        for ep in eps:
            r = episode_error(ep, conv_key=conv_key)
            if r is None:
                misses += 1
                continue
            for k in errs:
                errs[k].append(r[k])
        n = len(errs["mask"])
        print(f"\nconvention {conv_key}: episodes resolved {n}/{len(eps)} (misses {misses})")
        for k, v in errs.items():
            if not v:
                continue
            v = np.array(v)
            print(f"  {k:8s} err m: median={np.median(v):.4f} mean={v.mean():.4f} "
                  f"p90={np.percentile(v, 90):.4f} max={v.max():.4f}")
    print("\nContext: obstacle radii are 0.065-0.08 m and corridor radius 0.07 m — "
          "localization error must sit well below these for the no-GT tier to inherit "
          "Tier-GT guidance unchanged.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/vlm_inputs/safelibero_spatial")
