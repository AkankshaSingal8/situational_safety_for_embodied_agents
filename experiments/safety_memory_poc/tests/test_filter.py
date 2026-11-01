import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.filter import apply_keep_away
from src.rules import KeepAway

T = {"plate": np.array([0.0, 0.0, 0.0])}


def test_no_constraints_is_identity():
    u = np.array([1.0, 0, 0, 0, 0, 0, 0])
    out, rep = apply_keep_away(u, np.array([0.5, 0, 0]), [], T)
    assert np.allclose(out, u) and not rep.intervened


def test_far_away_action_untouched():
    c = [KeepAway("eef", "plate", 0.08, "x")]
    u = np.array([-1.0, 0, 0, 0, 0, 0, 0])   # moving away
    out, rep = apply_keep_away(u, np.array([0.5, 0, 0]), c, T)
    assert np.allclose(out[:3], u[:3]) and not rep.intervened


def test_approach_into_margin_is_repaired():
    c = [KeepAway("eef", "plate", 0.08, "x")]
    p = np.array([0.09, 0.0, 0.0])            # just outside margin
    u = np.array([-1.0, 0, 0, 0, 0, 0, 0])    # driving straight at the target
    out, rep = apply_keep_away(u, p, c, T)
    assert rep.intervened
    # repaired action must not decrease the barrier faster than allowed
    n = p / np.linalg.norm(p)
    assert float(np.dot(n, 0.05 * out[:3])) >= -0.5 * (np.linalg.norm(p) - 0.08) - 1e-9


def test_repair_is_minimum_norm_along_the_normal():
    c = [KeepAway("eef", "plate", 0.08, "x")]
    p = np.array([0.09, 0.0, 0.0])
    u = np.array([-1.0, 0.3, 0.0, 0, 0, 0, 0])
    out, _ = apply_keep_away(u, p, c, T)
    # tangential component is preserved; only the normal direction changes
    assert abs(out[1] - 0.3) < 1e-9


def test_inside_margin_pushes_outward():
    c = [KeepAway("eef", "plate", 0.08, "x")]
    p = np.array([0.04, 0.0, 0.0])            # inside the margin
    u = np.zeros(7)
    out, rep = apply_keep_away(u, p, c, T)
    assert rep.intervened and out[0] > 0       # pushed away from the target


def test_degenerate_coincident_position_does_not_crash():
    c = [KeepAway("eef", "plate", 0.08, "x")]
    out, rep = apply_keep_away(np.zeros(7), np.zeros(3), c, T)
    assert np.all(np.isfinite(out)) and rep.intervened


def test_slide_adds_tangential_motion_when_blocked():
    """Head-on approach: without slide the repaired action nearly cancels;
    with slide there is real tangential travel toward the goal."""
    import numpy as np
    from src.filter import apply_keep_away
    from src.rules import KeepAway
    c = [KeepAway("eef", "plate", 0.13, "x")]
    p = np.array([-0.13, 0.0, 0.0])          # directly in front of the target
    goal = np.array([0.20, 0.0, 0.0])        # goal is straight through it
    u = np.array([1.0, 0, 0, 0, 0, 0, 0])
    no_slide, _ = apply_keep_away(u, p, c, T, goal=goal, slide=0.0)
    with_slide, _ = apply_keep_away(u, p, c, T, goal=goal, slide=0.6)
    assert np.linalg.norm(with_slide[:3]) > np.linalg.norm(no_slide[:3])


def test_slide_is_inactive_when_filter_does_not_intervene():
    import numpy as np
    from src.filter import apply_keep_away
    from src.rules import KeepAway
    c = [KeepAway("eef", "plate", 0.08, "x")]
    u = np.array([-1.0, 0, 0, 0, 0, 0, 0])
    out, rep = apply_keep_away(u, np.array([0.5, 0, 0]), c, T,
                               goal=np.array([1.0, 0, 0]), slide=0.6)
    assert not rep.intervened and np.allclose(out[:3], u[:3])
