"""Pure control-frame math for the AEGIS ground-truth safety filter.

Kept dependency-free (numpy only, no cvxpy) so the frame and scale conventions
can be unit-tested without a QP solver installed.

Convention
----------
The QP reasons in a scaled body frame. For a nominal 6-DOF `action`:

    u_v_ref     = 5 * (R1.T @ action[:3])     # body frame, x5
    u_omega_ref = 5 * action[3:6]             # world frame, x5

and the executed command is reconstructed as

    action_input[:3]  = 0.2 * (R1 @ u_v)
    action_input[3:6] = 0.2 * u_omega

The 5 and the 0.2 cancel (the 0.2 is the benchmark's dynamics constant,
p_dot = 0.2u -- arXiv:2512.11891 Sec. V.A), so feeding the reference vectors
straight through reproduces the nominal action exactly.
"""
from typing import Tuple

import numpy as np

# p_dot = 0.2 u (SafeLIBERO / AEGIS dynamics); the QP works in 1/0.2 = 5x units.
DYNAMICS_GAIN = 0.2
CONTROL_SCALE = 1.0 / DYNAMICS_GAIN  # 5.0


def qp_reference_controls(action: np.ndarray, R1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reference (unfiltered) controls in the QP's scaled body frame."""
    action = np.asarray(action, dtype=np.float64)
    u_v_ref = CONTROL_SCALE * (R1.T @ action[:3])
    u_omega_ref = CONTROL_SCALE * action[3:6]
    return u_v_ref, u_omega_ref


def qp_fallback_controls(action: np.ndarray, R1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Controls to use when the QP is infeasible: pass the nominal action through.

    This must return the same vectors as :func:`qp_reference_controls`. Assigning
    the raw base-frame ``action[:3]`` here -- as this driver previously did --
    makes the executed command ``0.2 * R1 @ action[:3]``: five times too small
    and rotated by R1 instead of frame-cancelled.
    """
    return qp_reference_controls(action, R1)


def executed_command(u_v: np.ndarray, u_omega: np.ndarray, R1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct the linear/angular command actually sent to the environment."""
    return DYNAMICS_GAIN * (R1 @ np.asarray(u_v)), DYNAMICS_GAIN * np.asarray(u_omega)
