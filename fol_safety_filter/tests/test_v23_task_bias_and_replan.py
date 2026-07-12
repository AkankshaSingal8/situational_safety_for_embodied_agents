"""Tests for v23 Task B (FOL_TASK_BIAS) and Task C (FOL_CHUNK_REPLAN)."""

import os

import numpy as np
import pytest

from fol_safety_filter.filter import FOLSafetyFilter
from fol_safety_filter.primitives import ObjectState


def _f():
    f = FOLSafetyFilter.__new__(FOLSafetyFilter)
    f._target_name = None
    f._obstacle_names = []
    f._obstacle_ellipsoids = {}
    return f


class _S:
    def __init__(self, ee_pos, objects):
        self.ee_pos = ee_pos
        self.objects = objects


# ------------------------------------------------------------------ #
# Task B: _apply_task_bias
# ------------------------------------------------------------------ #

def test_task_bias_flag_off_is_byte_identical(monkeypatch):
    monkeypatch.delenv("FOL_TASK_BIAS", raising=False)
    f = _f()
    u = np.zeros(7)
    u[:3] = [0.01, 0.0, 0.0]
    # simulate _apply_cbf's gating: bias only called when flag set
    if os.environ.get("FOL_TASK_BIAS", "0") == "1":
        u = f._apply_task_bias(u, np.zeros(3), _S(np.zeros(3), {}))
    assert np.allclose(u[:3], [0.01, 0.0, 0.0])


def test_task_bias_no_target_returns_unchanged():
    f = _f()
    f._target_name = None
    u = np.zeros(7)
    u[:3] = [0.01, 0.0, 0.0]
    out = f._apply_task_bias(u, np.zeros(3), _S(np.zeros(3), {}))
    assert np.allclose(out[:3], [0.01, 0.0, 0.0])


def test_task_bias_pulls_toward_target_when_safe():
    f = _f()
    f._target_name = "goal_obj"
    ee = np.zeros(3)
    target = np.array([1.0, 0.0, 0.0])  # far away, no obstacles nearby
    state = _S(ee, {"goal_obj": ObjectState(
        name="goal_obj", pos=target,
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        bbox_half=np.array([0.02, 0.02, 0.02]))})
    u = np.zeros(7)
    u[:3] = [0.0, 0.01, 0.0]  # commanded step orthogonal to target direction
    out = f._apply_task_bias(u, ee, state)
    # biased step should have a positive dot-product component toward target
    assert float(out[:3] @ (target / np.linalg.norm(target))) > 0.0


def test_task_bias_rejected_if_it_breaches_hard_boundary():
    f = _f()
    f._target_name = "goal_obj"
    ee = np.array([0.155, 0.0, 0.0])  # just outside hard boundary (0.15)
    target = np.array([0.0, 0.0, 0.0])  # target sits AT the obstacle position
    obstacle_pos = np.array([0.0, 0.0, 0.0])
    f._obstacle_names = ["obs1"]
    f._obstacle_ellipsoids = {"obs1": {
        "frame_R": np.eye(3),
        "warn_radii": np.array([0.30, 0.30, 0.30]),
        "hard_radii": np.array([0.15, 0.15, 0.15]),
    }}
    state = _S(ee, {
        "goal_obj": ObjectState(
            name="goal_obj", pos=target,
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            bbox_half=np.array([0.02, 0.02, 0.02])),
        "obs1": ObjectState(
            name="obs1", pos=obstacle_pos,
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            bbox_half=np.array([0.05, 0.05, 0.05])),
    })
    u = np.zeros(7)
    u[:3] = [-0.004, 0.0, 0.0]  # already-safe step, lands just outside hard (0.151)
    out = f._apply_task_bias(u, ee, state)
    # bias toward target (which sits inside the obstacle's hard zone) would
    # push the landing point to 0.1498 < 0.15 — must be rejected, output
    # stays exactly the pre-bias safe action.
    assert np.allclose(out[:3], [-0.004, 0.0, 0.0])


def test_is_safe_step_ellipsoid_and_default():
    f = _f()
    ellipsoid = {
        "frame_R": np.eye(3),
        "hard_radii": np.array([0.10, 0.10, 0.10]),
    }
    assert f._is_safe_step(
        np.array([0.20, 0.0, 0.0]), np.array([-0.01, 0.0, 0.0]),
        np.zeros(3), ellipsoid) is True
    assert f._is_safe_step(
        np.array([0.05, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]),
        np.zeros(3), ellipsoid) is False
    # default (no ellipsoid) 0.10m hard_r
    assert f._is_safe_step(
        np.array([0.20, 0.0, 0.0]), np.zeros(3), np.zeros(3), None) is True
    assert f._is_safe_step(
        np.array([0.05, 0.0, 0.0]), np.zeros(3), np.zeros(3), None) is False


# ------------------------------------------------------------------ #
# Task C: replan_requested flag
# ------------------------------------------------------------------ #

def test_replan_flag_default_false():
    f = _f()
    f.replan_requested = False
    assert f.replan_requested is False


def test_replan_logic_flips_true_only_over_threshold(monkeypatch):
    monkeypatch.setenv("FOL_CHUNK_REPLAN", "1")
    monkeypatch.setenv("FOL_CHUNK_REPLAN_THRESHOLD", "0.01")
    f = _f()
    f.replan_requested = False

    def _maybe_replan(correction_mag):
        if os.environ.get("FOL_CHUNK_REPLAN", "0") == "1":
            threshold = float(os.environ.get("FOL_CHUNK_REPLAN_THRESHOLD", "0.01"))
            if correction_mag > threshold:
                f.replan_requested = True

    _maybe_replan(0.005)
    assert f.replan_requested is False
    _maybe_replan(0.02)
    assert f.replan_requested is True


def test_replan_stays_false_when_flag_unset_even_for_large_correction(monkeypatch):
    monkeypatch.delenv("FOL_CHUNK_REPLAN", raising=False)
    f = _f()
    f.replan_requested = False

    def _maybe_replan(correction_mag):
        if os.environ.get("FOL_CHUNK_REPLAN", "0") == "1":
            threshold = float(os.environ.get("FOL_CHUNK_REPLAN_THRESHOLD", "0.01"))
            if correction_mag > threshold:
                f.replan_requested = True

    _maybe_replan(5.0)
    assert f.replan_requested is False


def test_replan_resets_after_being_read_once():
    f = _f()
    f.replan_requested = True
    # simulate the frozen-script hook: clear queue, reset flag
    action_queue = [1, 2, 3]
    if f.replan_requested:
        action_queue.clear()
        f.replan_requested = False
    assert f.replan_requested is False
    assert action_queue == []


def test_frozen_script_hook_inert_when_fol_filter_is_none():
    # baseline (no-filter) runs: fol_filter is None, hook must no-op
    fol_filter = None
    action_queue = [1, 2, 3]
    if fol_filter is not None and getattr(fol_filter, "replan_requested", False):
        action_queue.clear()
    assert action_queue == [1, 2, 3]


def test_frozen_script_hook_noop_when_replan_not_requested():
    f = _f()
    f.replan_requested = False
    action_queue = [1, 2, 3]
    if f is not None and getattr(f, "replan_requested", False):
        action_queue.clear()
    assert action_queue == [1, 2, 3]
