"""Tests for Task 3 review fixes + C5 verified-safe fallback."""

import numpy as np
import pytest

from fol_safety_filter.visual_grounder import VisualObstacleGrounder
from fol_safety_filter.filter import FOLSafetyFilter
from fol_safety_filter.primitives import ObjectState


def test_grounder_has_state():
    g = VisualObstacleGrounder(object())
    assert g.has_state is False
    g._refine_state = {"pos": np.zeros(3), "rays": []}
    assert g.has_state is True


class _RecordingGrounder:
    def __init__(self):
        self.ground_calls = 0
        self.props_calls = []

    def ground(self, obs, object_states, rgb_image=None):
        self.ground_calls += 1

    def apply_vlm_properties(self, object_states, obstacle_properties):
        self.props_calls.append(dict(obstacle_properties))


class _KB:
    def __init__(self):
        self.rules = []

    def add_rule(self, rule):
        self.rules.append(rule)


def _reground_filter():
    f = FOLSafetyFilter.__new__(FOLSafetyFilter)
    synth = "visual_moka_pot"
    f._obstacle_names = []
    f._obstacle_radii = {}
    f._obstacle_properties = {}
    f._obstacle_ellipsoids = {}
    f._obstacle_name = None
    f._object_states = {synth: ObjectState(
        name=synth, pos=np.array([0.1, 0.1, 0.95]),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        bbox_half=np.array([0.05, 0.05, 0.05]))}
    f._refine_steps = {6, 12, 18}
    f._reground_attempted = False
    f._task_description = "put the bowl on the plate"
    f._vision_fallback_active = True
    f.grounder = _RecordingGrounder()
    f.kb = _KB()
    f._vision_ground = lambda obs, task, av, eih: {
        "obstacle_obs_keys": [synth],
        "obstacle_radii": {synth: 0.23},
        "obstacle_properties": {synth: ["IS_HOT"]},
    }
    return f


def _obs():
    z = np.zeros((8, 8, 3), dtype=np.uint8)
    return {"agentview_image": z, "robot0_eye_in_hand_image": z,
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0])}


def test_reground_propagates_properties_and_rules():
    f = _reground_filter()
    f._t = 10
    f._vision_maintenance(_obs())
    assert f._obstacle_names == ["visual_moka_pot"]
    assert f.grounder.ground_calls == 1
    assert f.grounder.props_calls == [{"visual_moka_pot": ["IS_HOT"]}]
    assert len(f.kb.rules) > 0
    assert f._vision_fallback_active is False


def test_reground_failure_keeps_fallback():
    f = _reground_filter()
    f._vision_ground = lambda obs, task, av, eih: {"obstacle_obs_keys": []}
    f._t = 10
    f._vision_maintenance(_obs())
    assert f._obstacle_names == []
    assert f._vision_fallback_active is True
    assert f._reground_attempted is True


def _cbf_filter(fallback):
    from fol_safety_filter.cbf_mapper import MappedCBFSet
    f = FOLSafetyFilter.__new__(FOLSafetyFilter)
    f._obstacle_names = []
    f._obstacle_radii = {}
    f._obstacle_ellipsoids = {}
    f.velocity_limit_default = None
    f._vision_fallback_active = fallback
    f._fallback_vlim = 0.010
    cbf_set = MappedCBFSet(semantic_envelopes=[], collision_envelopes=[])
    return f, cbf_set


class _State:
    ee_pos = np.zeros(3)
    objects = {}


def test_fallback_speed_cap_active():
    f, cbf_set = _cbf_filter(fallback=True)
    u = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    out, _ = f._apply_cbf(u, _State(), cbf_set, None)
    assert np.linalg.norm(out[:3]) == pytest.approx(0.010)


def test_no_fallback_no_cap():
    f, cbf_set = _cbf_filter(fallback=False)
    u = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    out, _ = f._apply_cbf(u, _State(), cbf_set, None)
    assert np.linalg.norm(out[:3]) == pytest.approx(0.02)


class _FakeSim:
    class model:
        @staticmethod
        def body_name2id(name):
            ids = {"robot0_link4": 4, "robot0_link6": 6, "robot0_link7": 7}
            if name not in ids:
                raise ValueError(name)
            return ids[name]
    class data:
        body_xpos = {4: np.array([0.1, 0.0, 1.2]),
                     6: np.array([0.2, 0.0, 1.1]),
                     7: np.array([0.25, 0.0, 1.0])}


class _FakeEnv:
    sim = _FakeSim()


def test_arm_checkpoints_include_hand_with_tight_radii():
    f = FOLSafetyFilter.__new__(FOLSafetyFilter)
    cps = f._get_arm_checkpoints(_FakeEnv())
    names = [c["name"] for c in cps]
    assert names == ["robot0_link4", "robot0_link6", "robot0_link7"]
    hand = cps[-1]
    assert hand["warning_r"] == pytest.approx(0.10)
    assert hand["hard_r"] == pytest.approx(0.06)
    assert cps[0]["warning_r"] == pytest.approx(0.15)


def test_hand_checkpoint_brakes_plow_through():
    from fol_safety_filter.cbf_mapper import MappedCBFSet
    from fol_safety_filter.primitives import ObjectState
    f = FOLSafetyFilter.__new__(FOLSafetyFilter)
    synth = "visual_pot"
    f._obstacle_names = [synth]
    f._obstacle_radii = {synth: 0.20}
    f._obstacle_ellipsoids = {}
    f.velocity_limit_default = None
    f._vision_fallback_active = False
    f._fallback_vlim = 0.010
    f._vision_target_xy = None
    pot = np.array([0.0, 0.0, 0.95])

    class _S:
        ee_pos = np.array([-0.30, 0.0, 0.95])  # EEF well past/away from pot
        objects = {synth: ObjectState(
            name=synth, pos=pot, quat=np.array([0.0, 0.0, 0.0, 1.0]),
            bbox_half=np.array([0.05, 0.05, 0.05]))}

    hand_cp = [{"pos": np.array([0.05, 0.0, 0.98]), "warning_r": 0.10,
                "hard_r": 0.06, "push": 0.06, "name": "robot0_link7"}]
    u = np.zeros(7)
    u[:3] = [-0.02, 0.0, 0.0]  # dragging hand through the pot
    out, _ = f._apply_cbf(u, _S(), MappedCBFSet(
        semantic_envelopes=[], collision_envelopes=[]), hand_cp)
    # hand checkpoint inside hard zone: the outward-direction component of
    # the filtered step must be non-negative (approach cancelled + push)
    d = hand_cp[0]["pos"] - pot
    out_dir = d / np.linalg.norm(d)
    assert float(np.array([-0.02, 0.0, 0.0]) @ out_dir) < 0  # raw was inward
    assert float(out[:3] @ out_dir) >= -1e-9


def test_corridor_relaxation_fires_on_vertical_descent():
    from fol_safety_filter.cbf_mapper import MappedCBFSet
    f = FOLSafetyFilter.__new__(FOLSafetyFilter)
    synth = "visual_pot"
    f._obstacle_names = [synth]
    f._obstacle_radii = {synth: 0.20}
    f._obstacle_ellipsoids = {}
    f.velocity_limit_default = None
    f._vision_fallback_active = False
    f._fallback_vlim = 0.010
    f._vision_target_xy = np.array([0.0, 0.0])  # target directly below EEF

    class _S:
        ee_pos = np.array([0.0, 0.0, 1.10])  # above obstacle, inside warning
        objects = {synth: ObjectState(
            name=synth, pos=np.array([0.0, 0.0, 0.95]),
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            bbox_half=np.array([0.05, 0.05, 0.05]))}

    u = np.zeros(7)
    u[:3] = [0.0, 0.0, -0.02]  # pure vertical grasp descent, zero xy speed
    out, _ = f._apply_cbf(u, _S(), MappedCBFSet(
        semantic_envelopes=[], collision_envelopes=[]), None)
    # xy-only gate cancels 100% (out_z=0); 3D-aware gate relaxes to 0.6
    # leaving 40% of the descent
    assert out[2] < -0.005
