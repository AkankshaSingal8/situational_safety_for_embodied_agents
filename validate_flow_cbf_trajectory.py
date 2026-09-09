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

from flow_cbf_sample import GAMMA, correct_trajectory, predict_trajectory, barrier_values

DT_CONTROL = 1.0  # treat v_chunk_xyz entries as per-step displacements directly (test-scale only)
R_OBS = 0.06


def run_case(name, eef_pos0, v_chunk_xyz, p_obs):
    print(f"--- case: {name} ---")
    traj_before = predict_trajectory(eef_pos0, v_chunk_xyz, DT_CONTROL)
    b_before = barrier_values(traj_before, p_obs, R_OBS)
    print(f"barriers before correction: {np.round(b_before, 4)}")

    corrected_xyz, b_after = correct_trajectory(v_chunk_xyz, eef_pos0, DT_CONTROL, p_obs, R_OBS)
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
    v_chunk_xyz = np.tile(np.array([0.05, 0.0, 0.0]), (H, 1))  # moves +0.5m in x over the chunk
    p_obs = np.array([0.25, 0.0, 0.0])  # sits mid-path -- far from p_ee(0), close to p_ee(H/2)
    all_pass &= run_case("obstacle mid-path (not visible at j=0)", eef_pos0, v_chunk_xyz, p_obs)

    # Case 2: obstacle near the END of the path only.
    p_obs2 = np.array([0.48, 0.0, 0.0])
    all_pass &= run_case("obstacle near end of path", eef_pos0, v_chunk_xyz, p_obs2)

    # Case 3: no violation anywhere -- correction should be a no-op.
    p_obs3 = np.array([10.0, 10.0, 10.0])
    all_pass &= run_case("no obstacle nearby (expect zero correction)", eef_pos0, v_chunk_xyz, p_obs3)

    print()
    print("OVERALL:", "PASS" if all_pass else "FAIL")


if __name__ == "__main__":
    main()
