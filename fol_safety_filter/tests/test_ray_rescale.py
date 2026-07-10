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


def _apply_cbf_mode(mode, aiming, monkeypatch_env):
    import os
    from fol_safety_filter.cbf_mapper import MappedCBFSet
    from fol_safety_filter.primitives import ObjectState
    os.environ["FOL_RAY_RESCALE"] = mode
    os.environ["FOL_ELLIPSOID"] = "1"
    try:
        f = _f()
        synth = "visual_pot"
        f._obstacle_names = [synth]
        f._obstacle_radii = {synth: 0.30}
        f._obstacle_ellipsoids = {synth: {
            "frame_R": R, "warn_radii": WARN, "hard_radii": HARD}}
        f.velocity_limit_default = None
        f._vision_fallback_active = False
        f._fallback_vlim = 0.010
        # eef inside warning zone at +x; target placed so u aims at it or not
        f._vision_target_xy = np.array([0.05, 0.05]) if aiming else np.array([0.6, 0.6])

        class _S:
            ee_pos = np.array([0.25, 0.0, 0.0])
            objects = {synth: ObjectState(
                name=synth, pos=np.zeros(3),
                quat=np.array([0.0, 0.0, 0.0, 1.0]),
                bbox_half=np.array([0.05, 0.05, 0.05]))}

        u = np.zeros(7)
        u[:3] = [-0.01, 0.005, 0.0]  # inward + tangential, ends outside hard
        out, _ = f._apply_cbf(u, _S(), MappedCBFSet(
            semantic_envelopes=[], collision_envelopes=[]), None)
        return out[:3]
    finally:
        del os.environ["FOL_RAY_RESCALE"]


def test_target_mode_carves_corridor_when_aiming(monkeypatch=None):
    out = _apply_cbf_mode("target", aiming=True, monkeypatch_env=None)
    # ray-rescale: full commanded step preserved (endpoint outside hard)
    assert np.allclose(out, [-0.01, 0.005, 0.0])


def test_target_mode_slides_when_not_aiming(monkeypatch=None):
    out = _apply_cbf_mode("target", aiming=False, monkeypatch_env=None)
    # cancellation: inward (-x) component removed, tangential kept
    assert out[0] > -1e-12
    assert out[1] == pytest.approx(0.005)
