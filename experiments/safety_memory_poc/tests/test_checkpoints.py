import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.filter import apply_keep_away_multi
from src.rules import KeepAway

T = {"ram": np.array([0.0, 0.0, 0.0])}
C = [KeepAway("eef", "ram", 0.13, "x")]


def test_far_checkpoints_do_not_intervene():
    pts = {"eef": np.array([0.5, 0, 0]), "robot0_link4": np.array([0.8, 0, 0])}
    out, rep = apply_keep_away_multi(np.zeros(7), pts, {}, C, T)
    assert not rep.intervened


def test_a_close_ARM_link_triggers_even_when_the_eef_is_far():
    """The defect this module exists for: the end effector is clear but the
    forearm is inside the margin."""
    pts = {"eef": np.array([0.5, 0, 0]), "robot0_link4": np.array([0.05, 0, 0])}
    scales = {"robot0_link4": 0.65, "eef": 1.0}
    u = np.array([-1.0, 0, 0, 0, 0, 0, 0])
    out, rep = apply_keep_away_multi(u, pts, scales, C, T)
    assert rep.intervened
    assert any(d["point"] == "robot0_link4" for d in rep.details)


def test_point_scale_shrinks_the_margin():
    pts = {"robot0_link4": np.array([0.10, 0, 0])}
    u = np.array([-1.0, 0, 0, 0, 0, 0, 0])
    _, big = apply_keep_away_multi(u, pts, {"robot0_link4": 1.0}, C, T)
    _, small = apply_keep_away_multi(u, pts, {"robot0_link4": 0.5}, C, T)
    assert big.intervened and not small.intervened


def test_no_constraints_is_identity_multi():
    pts = {"eef": np.array([0.01, 0, 0])}
    u = np.array([1.0, 0, 0, 0, 0, 0, 0])
    out, rep = apply_keep_away_multi(u, pts, {}, [], T)
    assert np.allclose(out, u) and rep.active == 0
