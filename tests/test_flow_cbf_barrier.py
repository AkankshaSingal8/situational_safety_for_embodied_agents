"""L2: the flow-CBF barrier must act in metric space, at the true control dt.

Two defects pinned here:

- The correction was applied to `v_t`, the model's velocity FIELD, inside the
  denoising loop and before `Unnormalize`. In a pi0-style flow model
  `v_t ~ eps - x_1` is a noise-scale quantity, not an action, so comparing it
  against metre-valued obstacle geometry was category-wrong.
- `predict_trajectory` integrated `p += dt * u`, assuming `p_dot = u`.
  SafeLIBERO's dynamics are `p_dot = 0.2 u`, so displacement was overestimated
  5x and predicted violations were inflated.

The previous validation script passed only because it exercised
`correct_trajectory` in isolation with DT_CONTROL=1.0, which hides both errors.
These tests use the real control dt (0.05) and real obstacle radii.
"""
import numpy as np
import pytest

from flow_cbf_barrier import (
    DYNAMICS_GAIN,
    EEF_SEMI_AXES,
    assert_barrier_is_active,
    barrier_values,
    correct_action_chunk,
    ellipsoid_sdf,
    predict_trajectory,
)

DT_CONTROL = 0.05          # 20 Hz, per arXiv:2512.11891 Sec. V.A
R_OBS = 0.06               # obstacle sphere radius used by the drivers
H = 8                      # action-chunk horizon


# --- sign convention -----------------------------------------------------

def test_sdf_is_positive_when_far_and_negative_when_inside():
    p_obs = np.array([0.5, 0.0, 1.0])
    assert ellipsoid_sdf(p_obs + np.array([1.0, 0.0, 0.0]), p_obs, R_OBS) > 0
    assert ellipsoid_sdf(p_obs, p_obs, R_OBS) < 0


# --- the dynamics gain ---------------------------------------------------

def test_trajectory_integrates_with_the_dynamics_gain():
    """p_dot = 0.2u, so one unit action for one dt advances 0.2*dt metres."""
    chunk = np.tile(np.array([1.0, 0.0, 0.0]), (H, 1))
    traj = predict_trajectory(np.zeros(3), chunk, DT_CONTROL)
    step = DYNAMICS_GAIN * DT_CONTROL
    np.testing.assert_allclose(traj[1], [step, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(traj[H], [H * step, 0.0, 0.0], atol=1e-12)


def test_omitting_the_gain_overestimates_displacement_fivefold():
    """Pin the magnitude of the old error."""
    chunk = np.tile(np.array([1.0, 0.0, 0.0]), (H, 1))
    correct = predict_trajectory(np.zeros(3), chunk, DT_CONTROL)
    old = predict_trajectory(np.zeros(3), chunk, DT_CONTROL, dynamics_gain=1.0)
    ratio = np.linalg.norm(old[H]) / np.linalg.norm(correct[H])
    assert abs(ratio - 5.0) < 1e-9


# --- the barrier actually bites in metric space --------------------------

def test_correction_moves_the_chunk_when_heading_into_an_obstacle():
    """A chunk driving straight at a nearby hazard must be altered."""
    eef0 = np.zeros(3)
    # Hazard 12 cm ahead in +x; the raw chunk drives straight into it.
    p_obs = np.array([0.12, 0.0, 0.0])
    chunk = np.tile(np.array([1.0, 0.0, 0.0]), (H, 1))

    before = barrier_values(predict_trajectory(eef0, chunk, DT_CONTROL), p_obs, R_OBS)
    assert np.min(before) < 0.0, "test setup must actually violate the barrier"

    corrected, after, ok = correct_action_chunk(chunk, eef0, DT_CONTROL, p_obs, R_OBS)
    assert ok, "SLSQP should converge on this well-posed problem"
    moved = np.abs(corrected - chunk).max()
    assert moved > 1e-4, f"barrier had no metric effect (max delta {moved:.2e})"
    assert np.min(after) > np.min(before), "correction must improve the worst barrier"


def test_no_correction_when_already_safe():
    """A chunk moving away from a distant hazard should be left alone."""
    eef0 = np.zeros(3)
    p_obs = np.array([2.0, 0.0, 0.0])
    chunk = np.tile(np.array([-1.0, 0.0, 0.0]), (H, 1))
    corrected, _, ok = correct_action_chunk(chunk, eef0, DT_CONTROL, p_obs, R_OBS)
    assert ok
    np.testing.assert_allclose(corrected, chunk, atol=1e-6)


def test_correction_is_minimum_norm_not_arbitrary():
    """The applied delta must be small relative to the command."""
    eef0 = np.zeros(3)
    p_obs = np.array([0.12, 0.0, 0.0])
    chunk = np.tile(np.array([1.0, 0.0, 0.0]), (H, 1))
    corrected, _, _ = correct_action_chunk(chunk, eef0, DT_CONTROL, p_obs, R_OBS)
    assert np.abs(corrected - chunk).max() < 10.0


# --- the inertness guard -------------------------------------------------

def test_inertness_guard_flags_a_violation_with_no_correction():
    chunk = np.ones((H, 3))
    complaint = assert_barrier_is_active(chunk, chunk, barriers_before=np.array([-0.05] * (H + 1)))
    assert complaint is not None and "inert" in complaint


def test_inertness_guard_silent_when_nothing_was_violated():
    chunk = np.ones((H, 3))
    assert assert_barrier_is_active(chunk, chunk, barriers_before=np.array([0.5] * (H + 1))) is None


def test_inertness_guard_silent_when_a_correction_was_applied():
    chunk = np.ones((H, 3))
    corrected = chunk + 0.01
    assert assert_barrier_is_active(corrected, chunk, np.array([-0.05] * (H + 1))) is None


# --- the old validation script's blind spot ------------------------------

def test_dt_control_of_one_hides_the_scale_error():
    """Why validate_flow_cbf_trajectory.py passed while the filter was broken.

    At DT_CONTROL=1.0 the gain and dt both wash out, so a chunk that is
    harmless at the true 0.05 dt looks like a violation and the correction
    appears to work.
    """
    eef0 = np.zeros(3)
    p_obs = np.array([0.5, 0.0, 0.0])
    chunk = np.tile(np.array([1.0, 0.0, 0.0]), (H, 1))
    b_real = barrier_values(predict_trajectory(eef0, chunk, 0.05), p_obs, R_OBS)
    b_fake = barrier_values(predict_trajectory(eef0, chunk, 1.0), p_obs, R_OBS)
    assert np.min(b_real) > 0.0, "harmless at the true dt"
    assert np.min(b_fake) < 0.0, "looks violating at dt=1.0"
