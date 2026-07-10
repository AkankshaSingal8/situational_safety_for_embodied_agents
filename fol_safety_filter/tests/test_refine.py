"""Tests for Task 3: multi-frame refinement + re-ground retry."""

import numpy as np
import pytest

from fol_safety_filter.visual_grounder import (
    VisualObstacleGrounder, _lsq_rays, _closest_point,
    RAY_GAP_MAX, UNCERT_TRIANGULATED,
)
from fol_safety_filter.filter import FOLSafetyFilter
from fol_safety_filter.primitives import ObjectState


def _rays_through(p, dirs, offsets):
    origins = [np.asarray(p) - t * np.asarray(d) / np.linalg.norm(d)
               for d, t in zip(dirs, offsets)]
    return origins, [np.asarray(d) / np.linalg.norm(d) for d in dirs]


def test_lsq_rays_recovers_known_point():
    p = np.array([0.1, 0.2, 0.95])
    dirs = [[1.0, 0.2, -0.5], [-0.3, 1.0, -0.4], [0.5, -0.6, -1.0]]
    origins, d = _rays_through(p, dirs, [1.0, 1.3, 0.8])
    sol, gap = _lsq_rays(origins, d)
    assert sol is not None
    assert np.linalg.norm(sol - p) < 1e-9
    assert gap < 1e-9


def test_lsq_rays_parallel_degenerate():
    d = np.array([0.0, 0.0, 1.0])
    sol, gap = _lsq_rays([np.zeros(3), np.array([0.1, 0.0, 0.0])], [d, d])
    assert sol is None
    assert gap == np.inf


def test_lsq_rays_matches_closest_point_two_rays():
    rng = np.random.default_rng(3)
    for _ in range(50):
        c1, c2 = rng.normal(size=3), rng.normal(size=3)
        d1, d2 = rng.normal(size=3), rng.normal(size=3)
        d1, d2 = d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2)
        ref, _ = _closest_point(c1, d1, c2, d2)
        if ref is None:
            continue
        sol, _ = _lsq_rays([c1, c2], [d1, d2])
        assert np.allclose(sol, ref, atol=1e-8)


class _NullClient:
    def infer(self, *a, **k):
        return "[]"


def _grounder_with_state(pos, extra_ray_pairs=1):
    g = VisualObstacleGrounder(_NullClient())
    pos = np.asarray(pos, dtype=np.float64)
    dirs = [[0.3, 0.4, -1.0], [-0.5, 0.2, -1.0]][:1 + extra_ray_pairs]
    origins, d = _rays_through(pos, dirs, [1.2] * len(dirs))
    g._refine_state = {"name": "moka pot", "pos": pos.copy(),
                       "rays": list(zip(origins, d))}
    return g


def _patch_crop_detect(g, target_uv):
    # one detection whose center maps exactly back to target_uv (scale=1, x1=y1=0)
    locs = [{"bbox": [target_uv[0] - 5, target_uv[1] - 5,
                      target_uv[0] + 5, target_uv[1] + 5], "name": "obj"}]
    g._crop_detect = lambda img, pts: (locs, 0, 0, 1, (1000, 1000))


def _wrist_pose_looking_at(pos):
    # eef straight above the point, wrist camera looking down
    eef_pos = np.asarray(pos) + np.array([0.0, 0.0, 0.35])
    eef_quat = np.array([1.0, 0.0, 0.0, 0.0])  # x-180: z-down-ish frame
    return eef_pos, eef_quat


def test_refine_no_state_returns_none():
    g = VisualObstacleGrounder(_NullClient())
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    assert g.refine(img, np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])) is None


def test_refine_accepts_consistent_ray_and_updates():
    from fol_safety_filter.visual_grounder import (
        EIH_K_512, T_HANDEYE, _project,
    )
    true_pos = np.array([0.05, 0.15, 0.95])
    g = _grounder_with_state(true_pos)
    eef_pos, eef_quat = _wrist_pose_looking_at(true_pos)
    # compute where the true point lands in the (rotated) wrist image
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(eef_quat).as_matrix()
    T[:3, 3] = eef_pos
    Ee = T @ T_HANDEYE
    wh = 128
    Ke = EIH_K_512 * (wh / 512.0)
    Ke[2, 2] = 1.0
    uv = _project(true_pos, Ke, np.linalg.inv(Ee))
    if uv is None or not (0 <= uv[0] < wh and 0 <= uv[1] < wh):
        pytest.skip("synthetic pose does not see the point; test geometry bad")
    _patch_crop_detect(g, uv)
    img = np.zeros((wh, wh, 3), dtype=np.uint8)
    out = g.refine(img[::-1, ::-1], eef_pos, eef_quat)  # raw = rotated back
    assert out is not None
    assert np.linalg.norm(out["pos"] - true_pos) < 0.02
    assert len(g._refine_state["rays"]) == 3
    assert np.isclose(out["uncertainty_m"], max(0.015, UNCERT_TRIANGULATED * 2 / 3))


def test_refine_rejects_large_jump():
    true_pos = np.array([0.05, 0.15, 0.95])
    g = _grounder_with_state(true_pos)
    g._refine_state["pos"] = true_pos + np.array([0.3, 0.0, 0.0])  # stale, far
    g._refine_state["rays"] = [
        (o, d) for o, d in zip(*_rays_through(true_pos,
                                              [[0.3, 0.4, -1.0], [-0.5, 0.2, -1.0]],
                                              [1.2, 1.2]))
    ]
    from fol_safety_filter.visual_grounder import EIH_K_512, T_HANDEYE, _project
    from scipy.spatial.transform import Rotation
    eef_pos, eef_quat = _wrist_pose_looking_at(true_pos + np.array([0.3, 0.0, 0.0]))
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(eef_quat).as_matrix()
    T[:3, 3] = eef_pos
    wh = 128
    Ke = EIH_K_512 * (wh / 512.0)
    Ke[2, 2] = 1.0
    uv = _project(true_pos + np.array([0.3, 0.0, 0.0]), Ke, np.linalg.inv(T @ T_HANDEYE))
    if uv is None:
        pytest.skip("bad geometry")
    _patch_crop_detect(g, uv)
    img = np.zeros((wh, wh, 3), dtype=np.uint8)
    # new lsq solution lands near true_pos (rays dominate) — >0.15 from stale pos
    out = g.refine(img[::-1, ::-1], eef_pos, eef_quat)
    assert out is None


def test_uncertainty_schedule():
    assert max(0.015, UNCERT_TRIANGULATED * 2 / 2) == pytest.approx(0.03)
    assert max(0.015, UNCERT_TRIANGULATED * 2 / 3) == pytest.approx(0.02)
    assert max(0.015, UNCERT_TRIANGULATED * 2 / 4) == pytest.approx(0.015)
    assert max(0.015, UNCERT_TRIANGULATED * 2 / 5) == pytest.approx(0.015)


class _FakeRefineGrounder:
    def __init__(self, res):
        self.res = res
        self.calls = 0

    def refine(self, *a, **k):
        self.calls += 1
        return self.res


def _bare_filter_with_visual_obstacle():
    f = FOLSafetyFilter.__new__(FOLSafetyFilter)
    synth = "visual_moka_pot"
    f._obstacle_names = [synth]
    f._obstacle_radii = {synth: 0.23}
    f._obstacle_ellipsoids = {synth: {
        "frame_R": np.eye(3),
        "warn_radii": np.array([0.3, 0.2, 0.2]),
        "hard_radii": np.array([0.15, 0.1, 0.1]),
    }}
    f._object_states = {synth: ObjectState(
        name=synth, pos=np.array([0.1, 0.1, 0.95]),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        bbox_half=np.array([0.05, 0.05, 0.05]))}
    f._visual_res = {"name": "moka pot", "frame_R": np.eye(3),
                     "half_extents": np.array([0.05, 0.05, 0.08])}
    f._refine_steps = {6, 12, 18}
    f._reground_attempted = False
    f._task_description = "put the bowl on the plate"
    return f


def _maintenance_obs():
    return {
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.0, 0.0, 1.2]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    }


def test_maintenance_updates_at_refine_step():
    f = _bare_filter_with_visual_obstacle()
    new_pos = np.array([0.12, 0.08, 0.96])
    f._visual_grounder = _FakeRefineGrounder(
        {"pos": new_pos, "uncertainty_m": 0.02})
    f._t = 6
    f._vision_maintenance(_maintenance_obs())
    synth = "visual_moka_pot"
    assert np.allclose(f._object_states[synth].pos, new_pos)
    assert np.isclose(f._obstacle_radii[synth], 0.20 + 0.02 + 0.0, atol=0.15)
    u = 0.02
    base = np.array([0.05, 0.05, 0.08]) + np.array([2.5 * u, u, u])
    assert np.allclose(f._obstacle_ellipsoids[synth]["warn_radii"],
                       np.maximum(base + 0.08, [0.20, 0.14, 0.14]))
    assert f._visual_grounder.calls == 1


def test_maintenance_skips_off_step():
    f = _bare_filter_with_visual_obstacle()
    f._visual_grounder = _FakeRefineGrounder(
        {"pos": np.zeros(3), "uncertainty_m": 0.02})
    f._t = 7
    old_pos = f._object_states["visual_moka_pot"].pos.copy()
    f._vision_maintenance(_maintenance_obs())
    assert f._visual_grounder.calls == 0
    assert np.allclose(f._object_states["visual_moka_pot"].pos, old_pos)
