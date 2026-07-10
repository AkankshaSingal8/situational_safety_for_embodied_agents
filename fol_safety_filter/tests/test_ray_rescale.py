"""Tests for ray-rescaling barrier mode (FOL_RAY_RESCALE=1).

Instead of cancelling the approach component inside the warning ellipsoid,
the commanded step is scaled along its own direction to stop at the hard
boundary. Direction preserved; hard-zone penetration impossible.
"""

import numpy as np
import pytest

from fol_safety_filter.filter import FOLSafetyFilter


def _f():
    return FOLSafetyFilter.__new__(FOLSafetyFilter)


R = np.eye(3)
WARN = np.array([0.30, 0.20, 0.20])
HARD = np.array([0.15, 0.10, 0.10])


def _apply(point, u3, ray_rescale=True):
    f = _f()
    u = np.zeros(7)
    u[:3] = u3
    f._apply_point_cbf(
        u, np.asarray(point, dtype=np.float64), np.zeros(3),
        warning_r=0.30, hard_r=0.10, push_hard=0.08,
        frame_R=R, warn_radii=WARN, hard_radii=HARD,
        ray_rescale=ray_rescale,
    )
    return u[:3]


def test_step_ending_outside_hard_untouched():
    # from outside warning, small step toward obstacle that stays outside hard
    out = _apply([0.40, 0.0, 0.0], [-0.02, 0.0, 0.0])
    assert np.allclose(out, [-0.02, 0.0, 0.0])


def test_corridor_approach_inside_warning_preserved():
    # inside the warning zone (corridor case), moving obliquely past the
    # obstacle while remaining outside hard: direction and magnitude kept
    p = [0.25, 0.05, 0.0]  # ||s_warn|| < 1, outside hard
    u3 = [-0.01, 0.015, 0.0]
    out = _apply(p, u3)
    assert np.allclose(out, u3)


def test_step_into_hard_is_rescaled_to_boundary():
    p = [0.20, 0.0, 0.0]
    u3 = [-0.10, 0.0, 0.0]  # would land at x=0.10 < hard 0.15
    out = _apply(p, u3)
    # direction preserved
    assert out[0] < 0 and np.allclose(out[1:], 0)
    # lands on (or just outside) the hard boundary
    land = np.asarray(p) + out
    s_h = np.linalg.norm(land / HARD)
    assert s_h >= 1.0 - 1e-9
    assert s_h < 1.1
    # and it moved (not fully cancelled)
    assert abs(out[0]) > 0.03


def test_inside_hard_still_pushed_out():
    p = [0.10, 0.0, 0.0]  # inside hard (0.15)
    out = _apply(p, [0.0, 0.0, 0.0])
    land = np.asarray(p) + out
    assert np.linalg.norm(land / HARD) > np.linalg.norm(np.asarray(p) / HARD)


def test_knob_off_keeps_cancellation_semantics():
    # ray_rescale=False → v19 default: approach component cancelled in warn zone
    p = [0.25, 0.0, 0.0]
    u3 = [-0.02, 0.0, 0.0]
    out = _apply(p, u3, ray_rescale=False)
    assert np.allclose(out, [0.0, 0.0, 0.0], atol=1e-12)
