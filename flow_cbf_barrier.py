"""Metric-space barrier math for the flow-CBF filter (Algorithm 1, Eq 7-10).

Split out of flow_cbf_sample.py, which imports jax at module scope, so these
pure numpy/scipy functions can be unit-tested without a GPU stack.

WHAT WAS WRONG
--------------
Two independent errors, both fixed here:

1. **Wrong space.** The correction was applied to ``v_t``, the model's velocity
   *field* output inside the denoising loop. In a pi0-style flow model
   ``v_t ~ eps - x_1`` is a noise-scale quantity, not an action, and the
   ``Unnormalize`` output transform (policy_config.py:86) has not yet run. The
   barrier's ``p_obs`` / ``r_obs`` / ``EEF_SEMI_AXES`` are in metres, so the
   comparison was category-wrong regardless of magnitude. The barrier must act
   on the DECODED, UNNORMALISED action chunk instead -- see
   ``correct_action_chunk``.

2. **Missing dynamics gain.** ``predict_trajectory`` integrated
   ``p += dt_control * u``, i.e. it assumed ``p_dot = u``. SafeLIBERO's
   dynamics are ``p_dot = 0.2 u`` (arXiv:2512.11891 Sec. V.A), so a control step
   advances the end-effector by ``0.2 * u * dt``. Integrating without the gain
   overestimated displacement 5x, which inflates every predicted barrier
   violation.

NOT YET VALIDATED ON HARDWARE: the arm this serves needs a GPU re-run before
any number from it is reported. ``assert_barrier_is_active`` exists to make a
silently inert filter fail loudly when that re-run happens.
"""
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

# Discount in the discrete barrier recursion B_j >= (1-GAMMA) B_{j-1}.
GAMMA = 0.9
# Extra clearance in metres beyond the geometric boundary.
DSAFE = 0.01
# EEF ellipsoid semi-axes in metres: SafeLIBERO's MVEE
# Q_ef = diag(0.06, 0.12, 0.11) (arXiv:2512.11891 Sec. V.A).
EEF_SEMI_AXES = np.array([0.06, 0.12, 0.11])
# p_dot = 0.2 u (arXiv:2512.11891 Sec. V.A).
DYNAMICS_GAIN = 0.2


def ellipsoid_sdf(p_ee, p_obs, r_obs, semi_axes=EEF_SEMI_AXES):
    """Approximate signed distance, EEF ellipsoid vs spherical obstacle.

    Positive = safe, matching the h>0 convention used throughout this project
    and in vlsa-aegis/main/utils.py's compute_h_ij.
    """
    d = np.asarray(p_ee, dtype=np.float64) - np.asarray(p_obs, dtype=np.float64)
    scaled = d / semi_axes
    n_scaled = np.linalg.norm(scaled)
    ellipsoid_dist = n_scaled - 1.0
    world_scale = np.linalg.norm(d) / (n_scaled + 1e-9) if n_scaled > 1e-9 else 1.0
    return ellipsoid_dist * world_scale - r_obs - DSAFE


def predict_trajectory(eef_pos0, chunk_xyz, dt_control, dynamics_gain=DYNAMICS_GAIN):
    """Eq 7. Roll the action chunk out into predicted EEF positions, in metres.

    Returns shape (H+1, 3): the current position followed by one point per
    action in the chunk. ``dynamics_gain`` converts a commanded action into a
    velocity (p_dot = gain * u); pass 1.0 only if the chunk is already a metric
    velocity.
    """
    chunk_xyz = np.asarray(chunk_xyz, dtype=np.float64)
    H = chunk_xyz.shape[0]
    step = dynamics_gain * dt_control
    traj = np.zeros((H + 1, 3))
    traj[0] = eef_pos0
    traj[1:] = eef_pos0 + step * np.cumsum(chunk_xyz, axis=0)
    return traj


def barrier_values(traj, p_obs, r_obs):
    """B_j for j=0..H (Eq 8), one per predicted trajectory point."""
    return np.array([ellipsoid_sdf(p, p_obs, r_obs) for p in traj])


def correct_action_chunk(
    chunk_xyz,
    eef_pos0,
    dt_control,
    p_obs,
    r_obs,
    dynamics_gain=DYNAMICS_GAIN,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Algorithm 1 / Eq 9-10, applied to a DECODED, UNNORMALISED chunk.

    One SLSQP solve for a joint minimum-norm correction delta over the whole
    predicted trajectory, with all H consecutive-point barrier constraints
    enforced simultaneously -- not H independent single-step corrections.

    Returns ``(corrected_chunk, final_barriers, solver_ok)``. ``solver_ok`` is
    False when the solve failed and the chunk was returned UNCORRECTED, which
    means the filter was effectively off for this step; callers must count it.
    """
    chunk_xyz = np.asarray(chunk_xyz, dtype=np.float64)
    H = chunk_xyz.shape[0]

    def barriers_of(delta_flat):
        delta = delta_flat.reshape(H, 3)
        traj = predict_trajectory(eef_pos0, chunk_xyz + delta, dt_control, dynamics_gain)
        return barrier_values(traj, p_obs, r_obs)

    def objective(delta_flat):
        return float(np.dot(delta_flat, delta_flat))

    def make_constraint(j):
        def c(delta_flat):
            b = barriers_of(delta_flat)
            return b[j] - (1.0 - GAMMA) * b[j - 1]
        return c

    constraints = [{"type": "ineq", "fun": make_constraint(j)} for j in range(1, H + 1)]
    res = minimize(
        objective,
        x0=np.zeros(H * 3),
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 50, "ftol": 1e-8},
    )
    if res.success:
        delta = res.x.reshape(H, 3)
        solver_ok = True
    else:
        delta = np.zeros((H, 3))
        solver_ok = False
    corrected = chunk_xyz + delta
    return corrected, barriers_of(delta.flatten()), solver_ok


def assert_barrier_is_active(
    corrected_chunk,
    original_chunk,
    barriers_before,
    min_displacement_m: float = 1e-4,
) -> Optional[str]:
    """Return a complaint string if a violated barrier produced no correction.

    A filter that reports a violation and then changes nothing is inert. Wire
    this in at the call site so that condition surfaces instead of passing for
    a working filter.
    """
    if np.min(barriers_before) >= 0.0:
        return None  # nothing was violated; no correction expected
    delta = np.abs(np.asarray(corrected_chunk) - np.asarray(original_chunk)).max()
    if delta < min_displacement_m:
        return (
            f"barrier reported a violation (min B = {np.min(barriers_before):.4f}) "
            f"but the chunk changed by only {delta:.2e} -- the filter is inert"
        )
    return None
