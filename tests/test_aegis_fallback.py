"""L4: the AEGIS QP-infeasible fallback used the wrong frame and scale.

The QP works in a scaled body frame: `u_v_ref = 5 * (R1.T @ action[:3])`, and the
executed command is reconstructed as `action_input[:3] = 0.2 * R1 @ u_v`. Those
factors cancel (0.2 * 5 == 1) so a pass-through must feed the REFERENCE vectors
back in.

The fallback instead assigned the raw base-frame `action[:3]`, so the executed
command became `0.2 * R1 @ action[:3]` -- 5x attenuated AND rotated by R1
instead of frame-cancelled. The angular channel had the same defect:
`0.2 * action[3:6]` instead of `action[3:6]`.

The 5 is 1/0.2 from the benchmark's dynamics, p_dot = 0.2u (arXiv:2512.11891
Sec. V.A). Upstream main_aegis.py:376 crashed here on a bare name; the port
turned a loud stop into a silent wrong command.
"""
import numpy as np

from aegis_controls import qp_fallback_controls, executed_command as _executed


IDENTITY = np.eye(3)
# 90 deg about z: makes a frame error visible rather than a no-op.
ROT_Z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_fallback_is_a_true_passthrough_under_rotation():
    action = np.array([0.10, -0.20, 0.05, 0.01, -0.02, 0.03])
    u_v, u_omega = qp_fallback_controls(action, ROT_Z90)
    lin, ang = _executed(u_v, u_omega, ROT_Z90)
    np.testing.assert_allclose(lin, action[:3], atol=1e-12)
    np.testing.assert_allclose(ang, action[3:6], atol=1e-12)


def test_fallback_is_a_true_passthrough_with_identity_rotation():
    action = np.array([0.03, 0.04, -0.05, 0.0, 0.1, 0.0])
    u_v, u_omega = qp_fallback_controls(action, IDENTITY)
    lin, ang = _executed(u_v, u_omega, IDENTITY)
    np.testing.assert_allclose(lin, action[:3], atol=1e-12)
    np.testing.assert_allclose(ang, action[3:6], atol=1e-12)


def test_fallback_matches_the_qp_reference_vectors():
    """The fallback must equal what the QP would be handed as its reference."""
    action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    u_v, u_omega = qp_fallback_controls(action, ROT_Z90)
    np.testing.assert_allclose(u_v, 5 * (ROT_Z90.T @ action[:3]), atol=1e-12)
    np.testing.assert_allclose(u_omega, 5 * action[3:6], atol=1e-12)


def test_regression_old_fallback_was_attenuated_and_rotated():
    """Pin the magnitude of the defect so it cannot silently return."""
    action = np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0])
    old_lin, _ = _executed(action[:3], action[3:6], ROT_Z90)
    new_lin, _ = _executed(*qp_fallback_controls(action, ROT_Z90), ROT_Z90)
    # Old: 0.2 * R1 @ [0.1,0,0] = [0, 0.02, 0] -- rotated onto +y and 5x small.
    np.testing.assert_allclose(old_lin, np.array([0.0, 0.02, 0.0]), atol=1e-12)
    np.testing.assert_allclose(new_lin, np.array([0.10, 0.0, 0.0]), atol=1e-12)
    ratio = np.linalg.norm(new_lin) / np.linalg.norm(old_lin)
    assert abs(ratio - 5.0) < 1e-9, f"expected a 5x attenuation, got {ratio}"
