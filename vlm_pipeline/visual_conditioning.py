"""pi0.7-style visual conditioning for safety steering (image channel, not text).

pi0.7 (PI blog, 2026) conditions the policy through its INPUT CHANNELS —
metadata, control-modality labels, and test-time visual subgoals — rather
than only language. The offline prompt-arm study (ledger 2026-07-25) showed
the text channel moves TASK behavior but not avoidance, so this module
implements the perception-channel analogue: project the guarded keep-out
region (translucent red disk) and the sanctioned target (green ring) into
the agentview image the policy consumes. The policy's own visual generality
does the steering — no barrier math, no retraining (Bitter-Lesson arm:
leans on the pretrained perception stack instead of hand-coded geometry).

Pure-numpy; conventions validated offline against saved captures
(vlm_inputs/*/camera_params.json) before any GPU use.
"""

import numpy as np


def _project(K, extrinsic_c2w, points_w):
    """World points (N,3) -> pixel (row, col) float array (N,2) + depth (N,).

    `extrinsic_c2w` is the camera-to-world transform as returned by
    robosuite's get_camera_extrinsic_matrix (and saved in
    camera_params.json). Its camera frame is already CV-convention —
    +z looks INTO the scene (verified: obstacle depth +0.93 m on the saved
    captures) — so K applies directly with no axis flip.
    """
    w2c = np.linalg.inv(np.asarray(extrinsic_c2w))
    pts = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    cam = (w2c @ homo.T).T[:, :3]
    z = np.maximum(cam[:, 2], 1e-6)
    # Column-mirror fix (2026-07-28 forensics, spatial t1 multi-object
    # validation): robosuite's saved extrinsic yields +z depth and correct
    # rows, but the camera x-axis is LEFT-pointing relative to the CV
    # convention — without the negation every projected point lands at
    # (W-1-col). Verified: 5 named objects land on their pixels only with
    # the flip. NOTE: this bug shipped in the visual-overlay pilot (disks
    # drawn mirrored) — that NO-GO is invalidated pending rerun.
    u = K[0, 0] * (-cam[:, 0]) / z + K[0, 2]  # col
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]  # row
    return np.stack([v, u], axis=1), z


def _pixel_radius(K, extrinsic_c2w, center_w, radius_m):
    px, _ = _project(K, extrinsic_c2w,
                     [center_w, np.asarray(center_w) + [radius_m, 0, 0],
                      np.asarray(center_w) + [0, 0, radius_m]])
    return float(max(np.linalg.norm(px[1] - px[0]), np.linalg.norm(px[2] - px[0])))


def _disk_mask(h, w, center_rc, radius_px):
    rr, cc = np.mgrid[0:h, 0:w]
    d2 = (rr - center_rc[0]) ** 2 + (cc - center_rc[1]) ** 2
    return d2 <= radius_px ** 2


def overlay_safety(img, K, extrinsic_c2w, obstacle_pos, obstacle_radius,
                   target_pos=None, rotated=True, alpha=0.45):
    """Blend the keep-out disk (red) and target ring (green) into `img`.

    img: (H, W, 3) uint8, the image ABOUT TO BE FED to the policy. If it has
    already been rotated 180 deg from the raw render (the client convention
    `[::-1, ::-1]`), pass rotated=True and projections are flipped to match.
    Returns a new array; the original is untouched.
    """
    h, w = img.shape[:2]
    out = img.astype(np.float32).copy()

    def to_img(rc):
        return (h - 1 - rc[0], w - 1 - rc[1]) if rotated else (rc[0], rc[1])

    px, z = _project(K, extrinsic_c2w, [obstacle_pos])
    if z[0] > 0.05:
        r_px = _pixel_radius(K, extrinsic_c2w, obstacle_pos, obstacle_radius)
        mask = _disk_mask(h, w, to_img(px[0]), max(r_px, 4.0))
        out[mask] = (1 - alpha) * out[mask] + alpha * np.array([220.0, 30.0, 30.0])

    if target_pos is not None and np.linalg.norm(np.asarray(target_pos)) < 5.0:
        tpx, tz = _project(K, extrinsic_c2w, [target_pos])
        if tz[0] > 0.05:
            ring_out = _disk_mask(h, w, to_img(tpx[0]), 14.0)
            ring_in = _disk_mask(h, w, to_img(tpx[0]), 9.0)
            ring = ring_out & ~ring_in
            out[ring] = (1 - alpha) * out[ring] + alpha * np.array([30.0, 220.0, 60.0])

    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_from_sim(img, sim, camera, obstacle_pos, obstacle_radius,
                     target_pos=None, rotated=True):
    """Runtime entry point: camera matrices straight from the live sim."""
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )
    h, w = img.shape[:2]
    K = get_camera_intrinsic_matrix(sim, camera, h, w)
    ext = get_camera_extrinsic_matrix(sim, camera)
    return overlay_safety(img, K, ext, obstacle_pos, obstacle_radius,
                          target_pos=target_pos, rotated=rotated)


if __name__ == "__main__":
    # Offline convention validation on a saved capture: draw the overlay for
    # the KNOWN active obstacle and eyeball the png.
    import json
    import pathlib
    import sys

    ep = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                      "/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/"
                      "vlm_inputs/safelibero_spatial/level_I/task_0/episode_00")
    import imageio.v2 as imageio

    cp = json.load(open(ep / "camera_params.json"))["agentview"]
    ob = json.load(open(ep / "obstacle.json"))["active_obstacle"]
    md = json.load(open(ep / "metadata.json"))
    raw = imageio.imread(ep / "agentview_rgb.png")
    img = np.ascontiguousarray(raw[::-1, ::-1])  # the client's policy-input rotation
    tgt = None
    for name, o in md.get("objects", {}).items():
        if isinstance(o, dict) and "position" in o and "bowl" in name:
            tgt = o["position"]
            break
    out = overlay_safety(img, np.asarray(cp["intrinsic"]), np.asarray(cp["extrinsic"]),
                         ob["position"], 0.09, target_pos=tgt, rotated=True)
    dst = ep.name + "_overlay_check.png"
    imageio.imwrite(dst, out)
    print("wrote", dst, "obstacle", ob["name"], "at", np.round(ob["position"], 3).tolist())
