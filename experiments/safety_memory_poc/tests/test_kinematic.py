import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.kinematic_sim import KinematicEnv


def test_reset_is_deterministic_per_seed():
    e = KinematicEnv(); e.seed(3); a = e.reset()
    e2 = KinematicEnv(); e2.seed(3); b = e2.reset()
    for k in a:
        assert np.allclose(a[k], b[k])


def test_different_seeds_move_objects():
    e = KinematicEnv(); e.seed(1); a = e.reset()
    e.seed(2); b = e.reset()
    assert not np.allclose(a["plate_pos"], b["plate_pos"])


def test_action_moves_eef_in_commanded_direction():
    e = KinematicEnv(noise=0.0); e.seed(0); o = e.reset()
    p0 = o["robot0_eef_pos"].copy()
    for _ in range(10):
        o, _, _, _ = e.step(np.array([1.0, 0, 0, 0, 0, 0, -1]))
    assert o["robot0_eef_pos"][0] - p0[0] > 0.02


def test_all_roles_present():
    e = KinematicEnv(); e.seed(0); o = e.reset()
    for k in ("plate_pos", "raw_source_pos", "lookalike_source_pos", "sink_pos"):
        assert k in o
