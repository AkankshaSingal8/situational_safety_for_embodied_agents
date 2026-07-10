"""Tests for Task 2: anisotropic uncertainty frame + image-derived extents."""

import numpy as np
import pytest

from fol_safety_filter.visual_grounder import _obstacle_frame, _bbox_extents
from fol_safety_filter.filter import FOLSafetyFilter


def test_frame_orthonormal_random_pairs():
    rng = np.random.default_rng(0)
    for _ in range(200):
        dda = rng.normal(size=3)
        dda /= np.linalg.norm(dda)
        ddb = rng.normal(size=3)
        ddb /= np.linalg.norm(ddb)
        R = _obstacle_frame(dda, ddb)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_frame_near_antiparallel_rays():
    dda = np.array([1.0, 0.0, 0.0])
    ddb = -dda + np.array([0.0, 1e-9, 0.0])
    R = _obstacle_frame(dda, ddb)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    # falls back to dda_hat for axis0
    assert np.allclose(np.abs(R[:, 0]), np.abs(dda), atol=1e-6)


def test_frame_axis0_is_bisector():
    dda = np.array([1.0, 0.0, -1.0]) / np.sqrt(2)
    ddb = np.array([0.0, 1.0, -1.0]) / np.sqrt(2)
    R = _obstacle_frame(dda, ddb)
    bis = dda + ddb
    bis /= np.linalg.norm(bis)
    cos = abs(float(R[:, 0] @ bis))
    assert cos > 0.999  # within ~2.5 degrees


def test_bbox_extents_math():
    f = 618.04
    bb = [100.0, 50.0, 200.0, 250.0]  # 100 px wide, 200 px tall
    ext = _bbox_extents(bb, f, rng=1.0)
    w = 100.0 / f
    assert np.allclose(ext[0], w / 2, atol=1e-9)
    assert np.allclose(ext[1], w / 2, atol=1e-9)
    assert np.isclose(ext[2], 0.12)  # h/2 = 0.1618 capped at 0.12


class _FakeGrounder:
    def __init__(self, res):
        self._res = res

    def ground(self, *a, **k):
        return self._res


def _bare_filter():
    f = FOLSafetyFilter.__new__(FOLSafetyFilter)
    f._visual_grounder = None
    f._object_states = {}
    f._obstacle_ellipsoids = {}
    return f


def _obs():
    return {
        "robot0_eef_pos": np.array([0.0, 0.0, 1.0]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    }


def test_vision_ground_builds_ellipsoid():
    filt = _bare_filter()
    R = np.eye(3)
    filt._visual_grounder = _FakeGrounder({
        "name": "moka pot", "pos": np.array([0.1, 0.1, 0.95]),
        "uncertainty_m": 0.03, "method": "visual-triangulated",
        "target_xy": None,
        "frame_R": R,
        "u_axes": np.array([0.075, 0.03, 0.03]),
        "half_extents": np.array([0.05, 0.05, 0.08]),
    })
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    out = filt._vision_ground(_obs(), "put the bowl on the plate", img, img)
    synth = out["obstacle_obs_key"]
    assert synth in filt._obstacle_ellipsoids
    e = filt._obstacle_ellipsoids[synth]
    base = np.array([0.05, 0.05, 0.08]) + np.array([0.075, 0.03, 0.03])
    assert np.allclose(e["warn_radii"],
                       np.maximum(base + 0.08, [0.20, 0.14, 0.14]))
    assert np.allclose(e["hard_radii"], np.maximum(base + 0.02, 0.08))
    assert np.allclose(e["frame_R"], R)


def test_vision_ground_zband_stays_spherical():
    filt = _bare_filter()
    filt._visual_grounder = _FakeGrounder({
        "name": "moka pot", "pos": np.array([0.1, 0.1, 0.95]),
        "uncertainty_m": 0.08, "method": "visual-zband",
        "target_xy": None,
    })
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    out = filt._vision_ground(_obs(), "put the bowl on the plate", img, img)
    assert out["obstacle_obs_keys"]
    assert filt._obstacle_ellipsoids == {}
