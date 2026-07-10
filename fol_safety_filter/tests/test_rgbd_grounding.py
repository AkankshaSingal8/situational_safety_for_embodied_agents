"""Tests for v21-D: RGB-D one-view grounding (FOL_RGBD=1)."""

import json

import numpy as np
import pytest

from fol_safety_filter.visual_grounder import (
    VisualObstacleGrounder, AGENTVIEW_K_512, AGENTVIEW_E,
    _deproject_depth, UNCERT_DEPTH,
)
from fol_safety_filter.filter import FOLSafetyFilter, _zbuffer_to_metric


def test_deproject_roundtrip_camera_model():
    # project a known world point with the validated model, then deproject
    # its pixel + camera-z back to world
    p = np.array([0.05, 0.15, 0.95])
    Einv = np.linalg.inv(AGENTVIEW_E)
    pc = Einv @ np.append(p, 1.0)
    K = AGENTVIEW_K_512
    u = K[0, 2] + K[0, 0] * pc[0] / pc[2]
    v = K[1, 2] + K[1, 1] * pc[1] / pc[2]
    out = _deproject_depth(u, v, float(pc[2]), K, AGENTVIEW_E)
    assert np.linalg.norm(out - p) < 1e-9


def test_zbuffer_to_metric_formula():
    near, far = 0.01, 50.0
    # z-buffer value for a point at 1.0 m: invert d = near/(1 - z(1-near/far))
    d_m = 1.0
    z = (1 - near / d_m) / (1 - near / far)
    assert _zbuffer_to_metric(np.array([z]), near, far)[0] == pytest.approx(1.0, rel=1e-9)


class _ScriptedClient:
    """Returns a fixed detection list for dense-detect; names crops on query."""

    def __init__(self, dets, crop_answer="moka pot"):
        self.dets = dets
        self.crop_answer = crop_answer

    def infer(self, prompt, images, max_tokens=512):
        if "1-3 words" in prompt:
            return self.crop_answer
        return json.dumps(self.dets)


def _depth_image_for(points_px, K, E, wh=512, bg=2.0):
    """Metric depth image with each (u, v, world_p) painted in a small patch."""
    d = np.full((wh, wh), bg, dtype=np.float64)
    Einv = np.linalg.inv(E)
    for u, v, p in points_px:
        z = float((Einv @ np.append(p, 1.0))[2])
        d[max(0, int(v) - 12):int(v) + 12, max(0, int(u) - 12):int(u) + 12] = z
    return d


def _px_of(p, K, E):
    Einv = np.linalg.inv(E)
    pc = Einv @ np.append(p, 1.0)
    return (K[0, 2] + K[0, 0] * pc[0] / pc[2],
            K[1, 2] + K[1, 1] * pc[1] / pc[2])


def test_rgbd_ground_selects_hovering_unmentioned():
    # two candidates: hovering pot (true obstacle) and a bowl on the table
    K, E = AGENTVIEW_K_512, AGENTVIEW_E
    pot = np.array([0.10, -0.10, 1.00])    # hovering (bottom well above table)
    bowl = np.array([-0.05, 0.15, 0.86])   # sitting on the table
    u1, v1 = _px_of(pot, K, E)
    u2, v2 = _px_of(bowl, K, E)
    dets = [
        {"bbox_2d": [int(u1) - 20, int(v1) - 20, int(u1) + 20, int(v1) + 20],
         "label": "kettle"},
        {"bbox_2d": [int(u2) - 20, int(v2) - 20, int(u2) + 20, int(v2) + 20],
         "label": "bowl"},
    ]
    g = VisualObstacleGrounder(_ScriptedClient(dets, crop_answer="kettle"))
    depth = _depth_image_for([(u1, v1, pot), (u2, v2, bowl)], K, E)
    rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    # grounder rotates inputs 180°; pass pre-rotated-back arrays
    res = g.ground(rgb[::-1, ::-1], rgb[::-1, ::-1],
                   "pick up the black bowl and place it on the plate",
                   np.array([-0.2, 0.0, 1.1]),
                   np.array([0.0, 0.0, 0.0, 1.0]),
                   agent_depth_m=depth[::-1, ::-1])
    assert res is not None
    assert res["method"] == "visual-rgbd"
    assert np.linalg.norm(res["pos"] - pot) < 0.05
    assert res["uncertainty_m"] == pytest.approx(UNCERT_DEPTH)
    assert res.get("frame_R") is not None
    # depth path needs no multi-frame refinement
    assert g._refine_state is None


def test_rgbd_prefers_hovering_over_nearer_table_object():
    # unmentioned table-level box is NEARER the path than the hovering pot;
    # the hover gate must still pick the pot
    K, E = AGENTVIEW_K_512, AGENTVIEW_E
    pot = np.array([0.10, -0.10, 1.00])
    box = np.array([-0.10, 0.05, 0.86])
    u1, v1 = _px_of(pot, K, E)
    u2, v2 = _px_of(box, K, E)
    dets = [
        {"bbox_2d": [int(u1) - 20, int(v1) - 20, int(u1) + 20, int(v1) + 20],
         "label": "kettle"},
        {"bbox_2d": [int(u2) - 20, int(v2) - 20, int(u2) + 20, int(v2) + 20],
         "label": "box"},
    ]
    g = VisualObstacleGrounder(_ScriptedClient(dets, crop_answer="snack box"))
    depth = _depth_image_for([(u1, v1, pot), (u2, v2, box)], K, E)
    rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    res = g.ground(rgb[::-1, ::-1], rgb[::-1, ::-1],
                   "pick up the black bowl and place it on the plate",
                   np.array([-0.2, 0.0, 1.1]),
                   np.array([0.0, 0.0, 0.0, 1.0]),
                   agent_depth_m=depth[::-1, ::-1])
    assert res is not None
    assert np.linalg.norm(res["pos"] - pot) < 0.05
