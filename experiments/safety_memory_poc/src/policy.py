"""Scripted nominal policy: proportional waypoint following in EEF space.

A scripted policy is used deliberately. The question under test is whether
memory changes safety outcomes, and a deterministic nominal controller removes
VLA sampling variance from that measurement. It also makes every arm share an
identical nominal trajectory, so differences are attributable to the filter
acting on beliefs. Porting to a learned policy is step 3.
"""
from __future__ import annotations

import numpy as np


def waypoint_action(eef_pos: np.ndarray, target: np.ndarray,
                    kp: float = 12.0, max_norm: float = 1.0,
                    gripper: float = -1.0) -> np.ndarray:
    """One delta-EEF action driving the end effector toward `target`."""
    err = np.asarray(target, dtype=float) - np.asarray(eef_pos, dtype=float)
    u = kp * err
    n = float(np.linalg.norm(u))
    if n > max_norm:
        u = u * (max_norm / n)
    a = np.zeros(7)
    a[:3] = u
    a[6] = gripper
    return a


def reached(eef_pos: np.ndarray, target: np.ndarray, tol: float) -> bool:
    return float(np.linalg.norm(np.asarray(eef_pos) - np.asarray(target))) <= tol


def through_goal(start: np.ndarray, obstacle: np.ndarray,
                 overshoot: float = 1.0) -> np.ndarray:
    """A goal placed so the straight path from `start` passes through `obstacle`.

    This guarantees the unfiltered nominal policy drives into the protected
    object, which is what makes the hidden-hazard cells informative: the
    memoryless arm has no reason to deviate.
    """
    start = np.asarray(start, dtype=float)
    obstacle = np.asarray(obstacle, dtype=float)
    d = obstacle - start
    return obstacle + overshoot * d
