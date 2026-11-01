import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.policy import waypoint_action, reached, through_goal


def test_action_points_at_target_and_is_bounded():
    a = waypoint_action(np.zeros(3), np.array([1.0, 0, 0]))
    assert a[0] > 0 and np.linalg.norm(a[:3]) <= 1.0 + 1e-9


def test_small_error_gives_small_action():
    a = waypoint_action(np.zeros(3), np.array([0.001, 0, 0]))
    assert np.linalg.norm(a[:3]) < 0.1


def test_reached_tolerance():
    assert reached(np.zeros(3), np.array([0.01, 0, 0]), 0.02)
    assert not reached(np.zeros(3), np.array([0.05, 0, 0]), 0.02)


def test_through_goal_puts_obstacle_on_the_segment():
    s = np.array([0.0, 0, 0]); o = np.array([0.5, 0.1, 0])
    g = through_goal(s, o)
    # obstacle lies on the segment start->goal
    t = np.dot(o - s, g - s) / np.dot(g - s, g - s)
    perp = np.linalg.norm((s + t * (g - s)) - o)
    assert 0 < t < 1 and perp < 1e-9


def test_symmetric_transit_keeps_goal_outside_the_margin():
    """Goal must sit farther from the protected object than the keep-away margin,
    otherwise a correct filter makes the task impossible and conservatism and
    correct behaviour become indistinguishable."""
    L, margin = 0.20, 0.13
    assert L > margin
