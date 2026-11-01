"""Scene sampling, the fixed base pick-and-place trajectory, and the oracle
safe trajectories used to build residual targets.

Action representation: K waypoints of the grip site, each
    [x, y, z, rx, ry, rz]
where r is a world-frame tilt rotation vector applied on top of the nominal
top-down grasp orientation R0 (orientation = exp(r) @ R0). The mug is welded
upright under the hand, so ||r|| is the mug's tilt from vertical.

Phase layout (fixed indices):
  0-5   descend onto the mug        6-7  grasp (weld on at GRASP_K=7)
  8-11  lift to carry height        12-29 carry (start -> target)
  30-33 lower to place height       34 settle, 35-36 release (weld off at RELEASE_K)
  37-39 retreat
"""
from __future__ import annotations

import numpy as np

K = 40
GRASP_K = 7
RELEASE_K = 35
CARRY = (11, 29)          # inclusive waypoint indices spanning the carry
MUG_R = 0.04
MUG_H = 0.10
LAPTOP_TOP = 0.16
A_MARGIN = 0.03           # extra clearance the oracle detour keeps
MAX_DETOUR = 0.35         # beyond this the detour leaves the comfortable workspace

R0 = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # hand z pointing down


def rotvec_to_mat(r):
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def laptop_clearance(xy, scene):
    """Signed clearance of mug-center xy to the mug-inflated laptop footprint."""
    lx, ly = scene["laptop_center"]
    lw, ld = scene["laptop_size"]
    c, s = np.cos(scene["laptop_yaw"]), np.sin(scene["laptop_yaw"])
    dx, dy = xy[..., 0] - lx, xy[..., 1] - ly
    u, v = c * dx + s * dy, -s * dx + c * dy
    return np.maximum(np.abs(u) - lw / 2 - MUG_R, np.abs(v) - ld / 2 - MUG_R)


def sample_scene(rng: np.random.Generator) -> dict:
    while True:
        mug = np.array([rng.uniform(-0.26, -0.20), rng.uniform(-0.10, 0.10)])
        tgt = np.array([rng.uniform(0.18, 0.24), rng.uniform(-0.10, 0.10)])
        d = tgt - mug
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        frac = rng.uniform(0.45, 0.55)
        lat = rng.uniform(-0.06, 0.06)
        center = mug + frac * d + lat * n
        scene = dict(
            mug_start=mug.tolist(),
            target=tgt.tolist(),
            laptop_center=center.tolist(),
            laptop_size=[rng.uniform(0.10, 0.14), rng.uniform(0.24, 0.32)],
            laptop_yaw=float(np.arctan2(d[1], d[0]) + rng.uniform(-0.25, 0.25)),
            carry_z=float(rng.uniform(0.25, 0.33)),
            # base-trajectory nuisance parameters (the base policy's habits)
            tilt_deg=float(rng.uniform(35, 60)),
            tilt_axis_angle=float(rng.uniform(0, 2 * np.pi)),
            tilt_start=int(rng.integers(12, 18)),
            tilt_len=int(rng.integers(8, 12)),
            progress_gamma=float(rng.uniform(0.8, 1.25)),
        )
        # start/target must be clear of the laptop so only the carry can violate A
        if laptop_clearance(mug, scene) > 0.06 and laptop_clearance(tgt, scene) > 0.06:
            if safe_A(scene, base_trajectory(scene)) is not None:   # a feasible correction must exist
                return scene


def sample_scene_v2(rng: np.random.Generator) -> dict:
    """Wider randomization than v1, plus stochastic base-policy habits.
    The base trajectory is still a deterministic function of the scene dict;
    the randomness lives in the per-scene parameters drawn here.
    Rejection keeps the spec's control: the commanded base carry crosses the
    laptop by >= 3 cm, start/target are clear of it, the tilt is >= 25 deg,
    and a feasible detour exists."""
    while True:
        mug = np.array([rng.uniform(-0.25, -0.12), rng.uniform(-0.15, 0.15)])   # mug x < -0.26 folded the arm in the 2026-10-20 smoke run
        tgt = np.array([rng.uniform(0.12, 0.27), rng.uniform(-0.15, 0.15)])
        d = tgt - mug
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        center = mug + rng.uniform(0.38, 0.62) * d + rng.uniform(-0.08, 0.08) * n
        ts = int(rng.integers(12, 19))
        scene = dict(
            profile="v2",
            mug_start=mug.tolist(),
            target=tgt.tolist(),
            laptop_center=center.tolist(),
            laptop_size=[rng.uniform(0.09, 0.15), rng.uniform(0.20, 0.34)],
            laptop_yaw=float(np.arctan2(d[1], d[0]) + rng.uniform(-0.5, 0.5)),
            carry_z=float(rng.uniform(0.26, 0.36)),
            tilt_deg=float(rng.uniform(25, 70)),
            tilt_axis_angle=float(rng.uniform(0, 2 * np.pi)),
            tilt_start=ts,
            tilt_len=int(rng.integers(6, min(13, CARRY[1] - ts) + 1)),
            progress_gamma=float(rng.uniform(0.7, 1.4)),
            bend=float(rng.uniform(-0.05, 0.05)),
            wob_lat=rng.uniform(-0.01, 0.01, size=2).tolist(),
            wob_z=rng.uniform(-0.01, 0.01, size=2).tolist(),
        )
        if laptop_clearance(mug, scene) <= 0.06 or laptop_clearance(tgt, scene) <= 0.06:
            continue
        b = base_trajectory(scene)
        k0, k1 = CARRY
        if laptop_clearance(_dense_xy(b[k0:k1 + 1, :2]), scene).min() > -0.03:
            continue                      # base must clearly pass over the laptop
        if np.degrees(np.linalg.norm(b[:, 3:], axis=-1).max()) < 25:
            continue                      # realized peak tilt (odd tilt_len undershoots tilt_deg)
        if safe_A(scene, b) is None:
            continue
        return scene


def scene_features(scene) -> np.ndarray:
    return np.array([
        *scene["mug_start"], *scene["target"], *scene["laptop_center"],
        *scene["laptop_size"], np.cos(scene["laptop_yaw"]), np.sin(scene["laptop_yaw"]),
    ], dtype=np.float32)


def _carry_progress(scene):
    k0, k1 = CARRY
    u = np.linspace(0, 1, k1 - k0 + 1)
    return _smoothstep(u ** scene["progress_gamma"])


def _path_normal(scene):
    d = np.array(scene["target"]) - np.array(scene["mug_start"])
    return np.array([-d[1], d[0]]) / np.linalg.norm(d)


def _carry_habits(scene, prog):
    """v2 base-policy habits on the carry: a lateral bend (via-point), smooth
    low-frequency lateral wobble, and a carry-height wobble. Every term vanishes
    at both carry endpoints. v1 scenes lack these keys and get zeros."""
    lat = scene.get("bend", 0.0) * np.sin(np.pi * prog)
    dz = np.zeros_like(prog)
    for j, c in enumerate(scene.get("wob_lat", []), start=2):
        lat = lat + c * np.sin(j * np.pi * prog)
    for j, c in enumerate(scene.get("wob_z", []), start=1):
        dz = dz + c * np.sin(j * np.pi * prog)
    return lat, dz


def base_trajectory(scene) -> np.ndarray:
    mug = np.array(scene["mug_start"]); tgt = np.array(scene["target"])
    zc, zg = scene["carry_z"], MUG_H / 2
    z_pre = zg + 0.18
    wp = np.zeros((K, 6))
    for k in range(0, 6):            # descend
        a = k / 5
        wp[k, :3] = [*mug, (1 - a) * z_pre + a * zg]
    for k in (6, 7):
        wp[k, :3] = [*mug, zg]
    for k in range(8, 12):           # lift
        a = _smoothstep((k - 7) / 4)
        wp[k, :3] = [*mug, (1 - a) * zg + a * zc]
    k0, k1 = CARRY
    prog = _carry_progress(scene)
    lat, dz = _carry_habits(scene, prog)
    n = _path_normal(scene)
    for i, k in enumerate(range(k0, k1 + 1)):
        wp[k, :2] = (1 - prog[i]) * mug + prog[i] * tgt + lat[i] * n
        wp[k, 2] = zc + dz[i]
    zp = zg + 0.01                   # place height: mug bottom 1 cm above the table
    for k in range(30, 34):          # lower
        a = _smoothstep((k - 29) / 4)
        wp[k, :3] = [*tgt, (1 - a) * zc + a * zp]
    for k in (34, 35, 36):           # settle, release at RELEASE_K, wait
        wp[k, :3] = [*tgt, zp]
    for k in range(37, 40):
        a = (k - 36) / 3
        wp[k, :3] = [*tgt, (1 - a) * zp + a * z_pre]
    # the base policy's bad habit: a mid-carry tilt of the held mug
    th = np.radians(scene["tilt_deg"])
    ax = np.array([np.cos(scene["tilt_axis_angle"]), np.sin(scene["tilt_axis_angle"]), 0.0])
    ks, L = scene["tilt_start"], scene["tilt_len"]
    for k in range(ks, min(ks + L, k1) + 1):
        b = np.sin(np.pi * (k - ks) / L) ** 2
        wp[k, 3:] = th * b * ax
    return wp


def _dense_xy(wp_xy):
    """Positions the controller actually passes through (linear between waypoints)."""
    a = np.linspace(0, 1, 9)[:, None, None]
    seg = (1 - a) * wp_xy[:-1][None] + a * wp_xy[1:][None]
    return seg.reshape(-1, 2)


def safe_A(scene, base) -> np.ndarray:
    """Detour around the laptop: lateral bump on the carry segment, smallest
    magnitude that clears the inflated footprint, on the cheaper side."""
    mug = np.array(scene["mug_start"]); tgt = np.array(scene["target"])
    d = tgt - mug
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    k0, k1 = CARRY
    prog = _carry_progress(scene)
    shape = _smoothstep(prog / 0.15) * _smoothstep((1 - prog) / 0.15)
    best = None
    for sign in (+1, -1):
        for D in np.arange(0.0, MAX_DETOUR, 0.005):
            wp = base.copy()
            wp[k0:k1 + 1, :2] = base[k0:k1 + 1, :2] + sign * D * shape[:, None] * n
            if np.min(laptop_clearance(_dense_xy(wp[k0:k1 + 1, :2]), scene)) > A_MARGIN:
                if best is None or D < best[0]:
                    best = (D, wp)
                break
    return None if best is None else best[1]


def safe_B(scene, base) -> np.ndarray:
    wp = base.copy()
    wp[:, 3:] = 0.0
    return wp


def safe_AB(scene, base) -> np.ndarray:
    wp = safe_A(scene, base).copy()
    wp[:, 3:] = 0.0
    return wp
