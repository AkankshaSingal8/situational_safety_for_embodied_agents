"""
validate_flow_cbf_trajectory.py

New check (does NOT exist for the earlier single-step version, which had no
way to fail this and still look correct): with an obstacle placed directly
in the predicted H-step path, correct_trajectory() must return a delta such
that ALL H+1 predicted trajectory points satisfy the barrier constraint
B_j >= (1-gamma)*B_{j-1} -- not just the first point. This is the actual
differentiator between "predictive, trajectory-level" correction (the
paper's claim) and a single-action filter repeated H times.

Pure numpy/scipy -- no model/checkpoint needed, since correct_trajectory()
only takes a velocity chunk + obstacle position.
"""

import numpy as np

from flow_cbf_barrier import GAMMA, correct_action_chunk, predict_trajectory, barrier_values

# The TRUE control period (20 Hz, arXiv:2512.11891 Sec. V.A). This was 1.0,
# which made the script pass while the filter was broken in two ways: at
# dt=1.0 both the dynamics gain and the timestep wash out, so a chunk that is
# harmless at 0.05 looks violating and the correction appears to work. See
# tests/test_flow_cbf_barrier.py::test_dt_control_of_one_hides_the_scale_error.
DT_CONTROL = 0.05
R_OBS = 0.06


def run_case(name, eef_pos0, v_chunk_xyz, p_obs, expect_violation=None):
    print(f"--- case: {name} ---")
    traj_before = predict_trajectory(eef_pos0, v_chunk_xyz, DT_CONTROL)
    b_before = barrier_values(traj_before, p_obs, R_OBS)
    print(f"barriers before correction: {np.round(b_before, 4)}")

    # Guard against a vacuous pass: a case meant to exercise the correction
    # must actually violate the barrier, or "no violation after" proves nothing.
    violated = bool(np.min(b_before) < 0.0)
    if expect_violation is not None and violated != expect_violation:
        print(f"  SETUP ERROR: expected violation={expect_violation} but min(B)="
              f"{np.min(b_before):.4f} -- this case tests nothing")
        return False

    corrected_xyz, b_after, solver_ok = correct_action_chunk(
        v_chunk_xyz, eef_pos0, DT_CONTROL, p_obs, R_OBS
    )
    if not solver_ok:
        print('  WARNING: SLSQP failed; chunk returned UNCORRECTED')
    print(f"barriers after correction:  {np.round(b_after, 4)}")

    ok = True
    for j in range(1, len(b_after)):
        if b_after[j] < (1.0 - GAMMA) * b_after[j - 1] - 1e-6:
            print(f"  VIOLATION at j={j}: B_j={b_after[j]:.4f} < (1-gamma)*B_(j-1)={(1.0 - GAMMA) * b_after[j - 1]:.4f}")
            ok = False
    delta_norm = float(np.linalg.norm(corrected_xyz - v_chunk_xyz))
    print(f"correction norm ||delta||: {delta_norm:.4f}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def main():
    H = 10
    all_pass = True

    # Case 1: obstacle directly in the middle of a straight-line path -- the
    # single-step version could only react to the FIRST step's proximity,
    # which at t=0 may still be far from the obstacle (violation invisible
    # to it); the trajectory-level version must see the future collision
    # and correct now.
    eef_pos0 = np.zeros(3)
    # Realistic magnitudes: pi0 LIBERO actions are ~unit-scale, and with
    # p_dot = 0.2u at dt = 0.05 each step advances 0.01 m, so an H=10 chunk
    # reaches ~0.10 m. The previous fixture used 0.05-magnitude actions and
    # obstacles at 0.25-0.48 m, which only intruded because the old integration
    # omitted the dynamics gain and used dt = 1.0 (a 100x longer reach). At the
    # true scale those cases became no-ops and the script passed vacuously.
    v_chunk_xyz = np.tile(np.array([1.0, 0.0, 0.0]), (H, 1))  # ~0.10 m of travel
    # 0.15 m: B[0] = +0.02 (starts SAFE) but B[H] = -0.08 (collides late in
    # the chunk) -- exactly the case a single-step filter cannot see.
    p_obs = np.array([0.15, 0.0, 0.0])
    all_pass &= run_case("obstacle mid-path (not visible at j=0)", eef_pos0, v_chunk_xyz,
                         p_obs, expect_violation=True)

    # Case 2: obstacle near the END of the path only.
    # 0.20 m: violation confined to the last few steps.
    p_obs2 = np.array([0.20, 0.0, 0.0])
    all_pass &= run_case("obstacle near end of path", eef_pos0, v_chunk_xyz, p_obs2,
                         expect_violation=True)

    # Case 3: no violation anywhere -- correction should be a no-op.
    p_obs3 = np.array([10.0, 10.0, 10.0])
    all_pass &= run_case("no obstacle nearby (expect zero correction)", eef_pos0, v_chunk_xyz,
                         p_obs3, expect_violation=False)

    print()
    print("OVERALL:", "PASS" if all_pass else "FAIL")


if __name__ == "__main__":
    main()
