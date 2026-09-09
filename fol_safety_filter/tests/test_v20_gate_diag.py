"""Tests for v20: multi-target aiming gate + diagnostic-only GT logging."""

import numpy as np
import pytest

from fol_safety_filter.filter import FOLSafetyFilter
from fol_safety_filter.primitives import ObjectState
from fol_safety_filter.cbf_mapper import MappedCBFSet


def _gate_filter(targets):
    f = FOLSafetyFilter.__new__(FOLSafetyFilter)
    synth = "visual_pot"
    f._obstacle_names = [synth]
    f._obstacle_radii = {synth: 0.20}
    f._obstacle_ellipsoids = {}
    f.velocity_limit_default = None
    f._vision_fallback_active = False
    f._fallback_vlim = 0.010
    f._vision_target_xy = targets[0] if targets else None
    f._vision_targets_xy = [np.asarray(t) for t in targets]
    f._objects = None

    class _S:
        ee_pos = np.array([0.0, 0.0, 1.10])
        objects = {synth: ObjectState(
            name=synth, pos=np.array([0.0, 0.0, 0.95]),
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            bbox_half=np.array([0.05, 0.05, 0.05]))}

    return f, _S()


def _descend(f, state):
    u = np.zeros(7)
    u[:3] = [0.0, 0.0, -0.02]
    out, _ = f._apply_cbf(u, state, MappedCBFSet(
        semantic_envelopes=[], collision_envelopes=[]), None)
    return out


def test_gate_fires_on_second_mentioned_target():
    # grasp target far away, PLACE goal directly below the EEF
    f, s = _gate_filter([np.array([0.5, 0.5]), np.array([0.0, 0.0])])
    out = _descend(f, s)
    assert out[2] < -0.005  # relaxation (0.6 cancel) leaves 40% of descent


def test_gate_silent_when_no_target_near():
    f, s = _gate_filter([np.array([0.5, 0.5])])
    out = _descend(f, s)
    assert out[2] == pytest.approx(0.0, abs=1e-12)  # full cancellation


def test_gate_never_fires_with_empty_targets():
    f, s = _gate_filter([])
    out = _descend(f, s)
    assert out[2] == pytest.approx(0.0, abs=1e-12)


def test_grounder_result_contains_mentioned_xy():
    import inspect
    from fol_safety_filter import visual_grounder as vg
    src = inspect.getsource(vg.VisualObstacleGrounder.ground)
    assert '"mentioned_xy"' in src


class _FakeGrounder:
    def __init__(self, res):
        self._res = res

    def ground(self, *a, **k):
        return self._res


def _vision_filter():
    f = FOLSafetyFilter.__new__(FOLSafetyFilter)
    f._visual_grounder = _FakeGrounder({
        "name": "moka pot", "pos": np.array([0.1, 0.1, 0.95]),
        "uncertainty_m": 0.03, "method": "visual-triangulated",
        "target_xy": np.array([0.0, 0.2]),
        "mentioned_xy": [np.array([0.0, 0.2]), np.array([-0.2, 0.0])],
        "frame_R": np.eye(3),
        "u_axes": np.array([0.075, 0.03, 0.03]),
        "half_extents": np.array([0.05, 0.05, 0.08]),
    })
    f._object_states = {}
    f._obstacle_ellipsoids = {}
    return f


def _obs(with_gt):
    o = {"robot0_eef_pos": np.array([0.0, 0.0, 1.0]),
         "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0])}
    if with_gt:
        o["moka_pot_obstacle_pos"] = np.array([0.12, 0.09, 0.96])
    return o


def test_vision_ground_stores_targets_list():
    f = _vision_filter()
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    f._vision_ground(_obs(False), "put the bowl on the plate", img, img)
    assert len(f._vision_targets_xy) == 2
    assert np.allclose(f._vision_targets_xy[1], [-0.2, 0.0])


def test_diag_gt_logging_is_control_inert():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    f1, f2 = _vision_filter(), _vision_filter()
    out1 = f1._vision_ground(_obs(True), "task", img, img)
    out2 = f2._vision_ground(_obs(False), "task", img, img)
    # identical control-relevant output with and without GT keys in obs
    assert out1["obstacle_obs_keys"] == out2["obstacle_obs_keys"]
    assert out1["obstacle_radii"] == out2["obstacle_radii"]
    synth = out1["obstacle_obs_key"]
    assert np.allclose(f1._object_states[synth].pos,
                       f2._object_states[synth].pos)
    assert np.allclose(
        f1._obstacle_ellipsoids[synth]["warn_radii"],
        f2._obstacle_ellipsoids[synth]["warn_radii"])


def test_diag_gt_line_emitted(caplog):
    import logging
    f = _vision_filter()
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    with caplog.at_level(logging.INFO, logger="fol_safety_filter.filter"):
        f._vision_ground(_obs(True), "task", img, img)
    assert any("[DIAG-GT]" in r.message for r in caplog.records)
