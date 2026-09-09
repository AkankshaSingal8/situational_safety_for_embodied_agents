"""v20 Task A: multi-target (grasp + place) aiming gate; Task B: diag-GT inertness."""

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
    f._object_states = {synth: ObjectState(
        name=synth, pos=np.array([0.0, 0.0, 0.95]),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        bbox_half=np.array([0.05, 0.05, 0.05]))}
    return f, synth


class _S:
    ee_pos = np.array([0.15, 0.0, 0.95])  # inside warning (0.20) of pot
    objects = None


def _run(f, u3):
    s = _S()
    s.objects = f._object_states
    u = np.zeros(7)
    u[:3] = u3
    out, _ = f._apply_cbf(u, s, MappedCBFSet(
        semantic_envelopes=[], collision_envelopes=[]), None)
    return out[:3]


def test_gate_fires_toward_second_mentioned_object():
    # grasp target far away, PLACE goal right beyond the eef heading
    f, _ = _gate_filter([np.array([0.6, 0.6]), np.array([0.05, 0.05])])
    out = _run(f, [-0.015, 0.008, 0.0])  # aiming at second target
    # relaxation: approach not fully cancelled (cancel_scale 0.6 leaves 40%)
    assert out[0] < -1e-4


def test_gate_silent_toward_unmentioned_direction():
    f, _ = _gate_filter([np.array([0.6, 0.6]), np.array([0.0, -0.6])])
    out = _run(f, [-0.015, 0.0, 0.0])  # aiming at neither
    assert out[0] >= -1e-12  # full cancellation


def test_gate_empty_list_never_fires():
    f, _ = _gate_filter([])
    out = _run(f, [-0.015, 0.0, 0.0])
    assert out[0] >= -1e-12
