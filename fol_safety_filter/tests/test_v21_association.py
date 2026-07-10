"""v21: size-consistent cross-view association (_match_crop_detection)."""

import numpy as np
import pytest

from fol_safety_filter.visual_grounder import (
    _match_crop_detection, _make_ray, _project,
    AGENTVIEW_K_512, AGENTVIEW_E, EIH_K_512, T_HANDEYE,
)
from scipy.spatial.transform import Rotation


def _setup(true_p):
    Ka = AGENTVIEW_K_512.copy()
    Ke = EIH_K_512.copy()
    eef_pos = np.asarray(true_p) + np.array([0.02, 0.0, 0.35])
    eef_quat = np.array([1.0, 0.0, 0.0, 0.0])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(eef_quat).as_matrix()
    T[:3, 3] = eef_pos
    Ee = T @ T_HANDEYE
    # agentview ray through the true point
    uv_a = _project(true_p, Ka, np.linalg.inv(AGENTVIEW_E))
    ca, dda = _make_ray(uv_a[0], uv_a[1], Ka, AGENTVIEW_E)
    # true point in the wrist view
    uv_e = _project(true_p, Ke, np.linalg.inv(Ee))
    assert uv_e is not None
    return Ka, Ke, Ee, ca, dda, uv_e


def _crop_loc(uv, half_px, x1=0, y1=0, scale=1):
    u, v = (uv[0] - x1) * scale, (uv[1] - y1) * scale
    return {"bbox": [u - half_px, v - half_px, u + half_px, v + half_px],
            "name": "obj"}


def test_size_consistent_candidate_beats_nearer_inconsistent():
    true_p = np.array([0.05, 0.10, 0.95])
    Ka, Ke, Ee, ca, dda, uv_e = _setup(true_p)
    # agentview bbox: pot-sized (0.15m at ~1.0m range with f=618 → ~93px)
    bb_a = [200.0, 200.0, 293.0, 293.0]
    # candidate A: right ON the epipolar segment but tiny (0.02m-equivalent)
    locA = _crop_loc(uv_e, half_px=3)
    # candidate B: 25px off the segment, correctly sized for the pot
    # (~0.15m at ~0.35m wrist range with f=334 → ~143px box → half 71)
    locB = _crop_loc((uv_e[0] + 25, uv_e[1]), half_px=71)
    seg_c = [uv_e]
    m = _match_crop_detection([locA, locB], seg_c, 0, 0, 1,
                              Ke, Ee, ca, dda, bb_a, Ka)
    assert m is not None
    # B's center is 25px off; the resulting triangulation stays near truth
    assert np.linalg.norm(m["p"] - true_p) < 0.06
    # and it must NOT be candidate A's (A is size-inconsistent);
    # A would triangulate essentially exactly at true_p (gap ~0) — the chosen
    # match has the small lateral offset from B
    assert np.linalg.norm(m["p"][:2] - true_p[:2]) > 0.005


def test_no_consistent_candidate_returns_none():
    true_p = np.array([0.05, 0.10, 0.95])
    Ka, Ke, Ee, ca, dda, uv_e = _setup(true_p)
    bb_a = [200.0, 200.0, 293.0, 293.0]
    locA = _crop_loc(uv_e, half_px=2)   # far too small
    m = _match_crop_detection([locA], [uv_e], 0, 0, 1,
                              Ke, Ee, ca, dda, bb_a, Ka)
    assert m is None


def test_epipolar_gap_gate_still_applies():
    true_p = np.array([0.05, 0.10, 0.95])
    Ka, Ke, Ee, ca, dda, uv_e = _setup(true_p)
    bb_a = [200.0, 200.0, 293.0, 293.0]
    # correctly-sized but 200px away → ray misses the epipolar line → big gap
    locB = _crop_loc((uv_e[0] + 200, uv_e[1] + 200), half_px=71)
    m = _match_crop_detection([locB], [uv_e], 0, 0, 1,
                              Ke, Ee, ca, dda, bb_a, Ka)
    assert m is None


def test_clipped_small_bbox_not_rejected():
    true_p = np.array([0.05, 0.10, 0.95])
    Ka, Ke, Ee, ca, dda, uv_e = _setup(true_p)
    bb_a = [200.0, 200.0, 293.0, 293.0]
    # undersized bbox centered on the true pixel but touching the crop
    # border (clipped view of a big object)
    loc = {"bbox": [uv_e[0] - 5, uv_e[1] - 5, uv_e[0] + 5, uv_e[1] + 5],
           "name": "obj"}
    m = _match_crop_detection([loc], [uv_e], 0, 0, 1,
                              Ke, Ee, ca, dda, bb_a, Ka,
                              crop_wh=(int(uv_e[0] + 5), int(uv_e[1] + 200)))
    assert m is not None  # clipped → under-size violation forgiven
    # same tiny bbox NOT touching the border stays rejected
    m2 = _match_crop_detection([loc], [uv_e], 0, 0, 1,
                               Ke, Ee, ca, dda, bb_a, Ka,
                               crop_wh=(int(uv_e[0] + 200), int(uv_e[1] + 200)))
    assert m2 is None
