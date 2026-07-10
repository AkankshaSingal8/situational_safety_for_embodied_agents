"""Tests for the ellipsoid barrier generalization in FOLSafetyFilter._apply_point_cbf."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fol_safety_filter.filter import FOLSafetyFilter


def _make_filter() -> FOLSafetyFilter:
    return FOLSafetyFilter.__new__(FOLSafetyFilter)


def test_spherical_equivalence():
    f = _make_filter()
    rng = np.random.default_rng(0)
    for _ in range(200):
        point = rng.uniform(-0.5, 0.5, 3)
        obs_pos = rng.uniform(-0.5, 0.5, 3)
        u_cmd = np.zeros(7)
        u_cmd[:3] = rng.uniform(-0.05, 0.05, 3)
        w = rng.uniform(0.05, 0.5)
        h = rng.uniform(0.02, w)

        u_sphere = u_cmd.copy()
        f._apply_point_cbf(u_sphere, point, obs_pos,
                           warning_r=w, hard_r=h, push_hard=0.08)

        u_ellip = u_cmd.copy()
        f._apply_point_cbf(u_ellip, point, obs_pos,
                           warning_r=w, hard_r=h, push_hard=0.08,
                           frame_R=np.eye(3),
                           warn_radii=np.array([w, w, w]),
                           hard_radii=np.array([h, h, h]))

        assert np.allclose(u_sphere, u_ellip, atol=1e-12)


ANISO_KWARGS = dict(
    warning_r=0.4, hard_r=0.2, push_hard=0.08,
    warn_radii=np.array([0.4, 0.15, 0.15]),
    hard_radii=np.array([0.2, 0.08, 0.08]),
)


def test_anisotropy():
    f = _make_filter()
    obs_pos = np.zeros(3)

    u_long = np.zeros(7)
    u_long[:3] = [-0.02, 0.0, 0.0]
    f._apply_point_cbf(u_long, np.array([0.3, 0.0, 0.0]), obs_pos,
                       frame_R=np.eye(3), **ANISO_KWARGS)
    assert not np.allclose(u_long[:3], [-0.02, 0.0, 0.0])
    assert u_long[0] >= -1e-12

    u_lat = np.zeros(7)
    u_lat[:3] = [0.0, -0.02, 0.0]
    f._apply_point_cbf(u_lat, np.array([0.0, 0.3, 0.0]), obs_pos,
                       frame_R=np.eye(3), **ANISO_KWARGS)
    assert np.allclose(u_lat[:3], [0.0, -0.02, 0.0], atol=1e-12)


def test_rotation_swaps_lateral_roles():
    f = _make_filter()
    obs_pos = np.zeros(3)
    frame_R = Rotation.from_euler("z", 90, degrees=True).as_matrix()

    u_y = np.zeros(7)
    u_y[:3] = [0.0, -0.02, 0.0]
    f._apply_point_cbf(u_y, np.array([0.0, 0.3, 0.0]), obs_pos,
                       frame_R=frame_R, **ANISO_KWARGS)
    assert not np.allclose(u_y[:3], [0.0, -0.02, 0.0])
    assert u_y[1] >= -1e-12

    u_x = np.zeros(7)
    u_x[:3] = [-0.02, 0.0, 0.0]
    f._apply_point_cbf(u_x, np.array([0.3, 0.0, 0.0]), obs_pos,
                       frame_R=frame_R, **ANISO_KWARGS)
    assert np.allclose(u_x[:3], [-0.02, 0.0, 0.0], atol=1e-12)


def test_hard_push_increases_normalized_distance():
    f = _make_filter()
    obs_pos = np.zeros(3)
    point = np.array([0.1, 0.0, 0.0])
    hard_radii = ANISO_KWARGS["hard_radii"]

    s_h_before = np.linalg.norm(np.eye(3).T @ (point - obs_pos) / hard_radii)
    assert s_h_before < 1.0

    u = np.zeros(7)
    f._apply_point_cbf(u, point, obs_pos, frame_R=np.eye(3), **ANISO_KWARGS)
    s_h_after = np.linalg.norm(
        np.eye(3).T @ (point + u[:3] - obs_pos) / hard_radii)
    assert s_h_after > s_h_before


def test_tangential_preservation():
    f = _make_filter()
    obs_pos = np.zeros(3)

    u_sphere = np.zeros(7)
    u_sphere[:3] = [0.0, 0.01, 0.0]
    f._apply_point_cbf(u_sphere, np.array([0.4, 0.0, 0.0]), obs_pos,
                       warning_r=0.4, hard_r=0.2, push_hard=0.08)
    assert np.allclose(u_sphere[:3], [0.0, 0.01, 0.0], atol=1e-12)

    u_ellip = np.zeros(7)
    u_ellip[:3] = [0.0, 0.01, 0.0]
    f._apply_point_cbf(u_ellip, np.array([0.4, 0.0, 0.0]), obs_pos,
                       frame_R=np.eye(3), **ANISO_KWARGS)
    assert np.allclose(u_ellip[:3], [0.0, 0.01, 0.0], atol=1e-12)
