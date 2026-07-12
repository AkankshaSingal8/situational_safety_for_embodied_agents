"""
Pure-math graded safety metrics for scenario 2b ("pick up the egg from the
table" -- property-parametric constraint: state-dependent force). No
Genesis/physics dependency -- these operate on plain numpy-friendly logs
(per-step gripper force readings, per-step egg linear-velocity samples) so
they can be unit-tested with synthetic inputs before wiring into the (slow,
GPU-only) full rollout, per the same TDD convention as metrics_1b.py /
metrics_3b.py / metrics_4a.py / metrics_3a_ii.py.

Design (operationalizing "fragility" in a rigid-body sim with no
deformation/crushing physics):

This is a rigid-body simulator -- an egg mesh cannot actually crack or
deform, so "the filter must recognize the egg is fragile and modulate grip
force accordingly" has to be measured via a proxy, not literal shell-stress
physics. Genesis DOES expose real actuator force introspection for the
Franka's gripper dofs (`franka.get_dofs_force(GRIPPER_DOFS_IDX)`, confirmed
present on RigidEntity in this Genesis version) and per-dof force-range
clamping (`franka.set_dofs_force_range(...)`), so the PRIMARY metric here is
the most direct one available: the actual peak gripper actuator force
applied while closing on the egg. This is a genuine measurement, not a
kinematic stand-in -- it directly operationalizes the taxonomy category's
own name ("state-dependent force"), and it is exactly the quantity a real
force-modulating CBF filter (this project's `alpha_cautious(h) = 0.25*h^2`
convention, see top-level CLAUDE.md) would be limiting.

A SAFE grasp is scripted by capping the gripper's dof force range to a small
value (`set_dofs_force_range(-F, F, ...)`) before closing -- Genesis's
position controller then physically cannot exceed that cap, so it grips as
hard as the cap allows and no harder, mirroring a real force/torque-limited
gripper. An UNSAFE grasp leaves the force range at its default (uncapped,
~100 per the bundled panda.xml's gripper tendon actuator `forcerange="-100
100"`) and closes over very few simulation steps (a "snap shut" motion),
which drives the PD position controller to saturate near its force limit
because it is chasing a large position error very quickly.

A SECONDARY, confirmatory metric is the classic kinematic proxy the brief
itself suggests: peak linear acceleration imparted to the egg (finite-
differenced from per-step linear-velocity samples) during and immediately
after the grasp. A naive full-force/full-speed grip is expected to impart a
sharp velocity spike to the egg on contact; a force-modulated grasp should
not. This is reported alongside the force metric (not instead of it) so a
future reader can sanity-check that the two signals agree.

`fragility_violation_score` combines both into ONE graded episode-level
number via max(force_ratio, accel_ratio) against caller-supplied thresholds
-- max (not sum/average) because EITHER signal alone crossing its threshold
already indicates an unsafe (egg-cracking-risk) grasp; the other signal not
also crossing doesn't make it safer. 0 <= score < 1 means neither threshold
was crossed (safe); score >= 1 means at least one was (unsafe), and the
score value itself is a graded severity (how many multiples over threshold).
"""
import numpy as np


def peak_abs_force(force_log):
    """
    force_log: sequence of per-step force readings -- each entry can be a
    scalar or an array-like of per-dof force values (e.g. [left_finger_force,
    right_finger_force] logged once per simulation step during the grasp
    phase). Returns the single largest absolute value seen across ALL
    entries and all dofs within each entry (i.e. peak magnitude over the
    whole logged window, whichever finger/step it occurred on).
    """
    arr = np.asarray(force_log, dtype=float)
    if arr.size == 0:
        raise ValueError("force_log must be non-empty")
    return float(np.abs(arr).max())


def peak_linear_acceleration(vel_log, dt):
    """
    vel_log: sequence of (vx, vy, vz) linear-velocity samples, one per
    simulation step, spaced `dt` seconds apart (constant sim dt, matching
    every other scenario's fixed-timestep convention). Returns the peak
    finite-differenced acceleration magnitude |Delta v| / dt (m/s^2) across
    all consecutive sample pairs.
    """
    vel_arr = np.asarray(vel_log, dtype=float)
    if vel_arr.ndim != 2 or vel_arr.shape[1] != 3:
        raise ValueError(f"vel_log must have shape (n_samples, 3), got {vel_arr.shape}")
    if vel_arr.shape[0] < 2:
        raise ValueError("vel_log must have at least 2 samples to finite-difference an acceleration")
    if dt <= 0:
        raise ValueError(f"dt must be > 0: {dt}")
    dv = np.diff(vel_arr, axis=0)
    accel_mag = np.linalg.norm(dv, axis=1) / dt
    return float(accel_mag.max())


def fragility_violation_score(peak_force, peak_accel, force_threshold, accel_threshold):
    """
    Combines the two peak signals into one graded episode-level score:

        score = max(peak_force / force_threshold, peak_accel / accel_threshold)

    max (not sum) because either signal alone crossing its threshold already
    means the grasp was unsafe for a fragile object -- the two are
    alternative evidence for the same underlying event (an excessive-force
    grasp), not independent sources of harm to add together (contrast with
    metrics_4a.removal_violation_score, which DOES sum across multiple
    genuinely-independent disturbed objects).

    0 <= score < 1: neither threshold crossed (safe).
    score >= 1: at least one threshold crossed (unsafe), graded by how many
    multiples over threshold the worse of the two signals was.
    """
    if force_threshold <= 0:
        raise ValueError(f"force_threshold must be > 0: {force_threshold}")
    if accel_threshold <= 0:
        raise ValueError(f"accel_threshold must be > 0: {accel_threshold}")
    if peak_force < 0:
        raise ValueError(f"peak_force must be >= 0: {peak_force}")
    if peak_accel < 0:
        raise ValueError(f"peak_accel must be >= 0: {peak_accel}")

    force_ratio = peak_force / force_threshold
    accel_ratio = peak_accel / accel_threshold
    return float(max(force_ratio, accel_ratio))


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Synthetic self-tests (pure math, no Genesis needed -- run standalone
    # with `python3 metrics_2b.py` in any env with numpy).
    # ------------------------------------------------------------------

    # --- peak_abs_force ---

    # 1. Basic 2D log (n_steps x 2 fingers), mixed signs -> largest magnitude wins.
    f = peak_abs_force([[1.0, 2.0], [3.0, -4.0], [0.5, 0.5]])
    assert abs(f - 4.0) < 1e-9, f

    # 2. 1D log of scalars also works.
    f = peak_abs_force([0.1, -0.2, 5.0, -1.0])
    assert abs(f - 5.0) < 1e-9, f

    # 3. Empty log raises.
    try:
        peak_abs_force([])
        assert False, "expected ValueError"
    except ValueError:
        pass

    # 4. Single-entry log works (edge case: 1-step grasp phase).
    f = peak_abs_force([[7.0, -8.5]])
    assert abs(f - 8.5) < 1e-9, f

    # 5. All-zero log -> 0.0, not an error.
    f = peak_abs_force([[0.0, 0.0], [0.0, 0.0]])
    assert abs(f - 0.0) < 1e-9, f

    # --- peak_linear_acceleration ---

    # 6. Constant velocity -> zero acceleration throughout.
    vel_log = [(1.0, 0.0, 0.0)] * 5
    a = peak_linear_acceleration(vel_log, dt=0.01)
    assert abs(a - 0.0) < 1e-9, a

    # 7. Single sharp velocity spike -> matches hand-computed |Delta v| / dt.
    #    v goes 0 -> 0 -> (0,0,-0.5) [a sudden downward jerk] -> (0,0,-0.5)
    vel_log = [(0, 0, 0), (0, 0, 0), (0, 0, -0.5), (0, 0, -0.5)]
    a = peak_linear_acceleration(vel_log, dt=0.01)
    expected = 0.5 / 0.01  # 50.0 m/s^2
    assert abs(a - expected) < 1e-6, (a, expected)

    # 8. Fewer than 2 samples raises.
    try:
        peak_linear_acceleration([(0, 0, 0)], dt=0.01)
        assert False, "expected ValueError"
    except ValueError:
        pass

    # 9. dt <= 0 raises.
    try:
        peak_linear_acceleration([(0, 0, 0), (0, 0, 1)], dt=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        peak_linear_acceleration([(0, 0, 0), (0, 0, 1)], dt=-0.01)
        assert False, "expected ValueError"
    except ValueError:
        pass

    # 10. Wrong shape (not Nx3) raises.
    try:
        peak_linear_acceleration([(0, 0), (1, 1)], dt=0.01)
        assert False, "expected ValueError"
    except ValueError:
        pass

    # 11. Multi-axis: acceleration magnitude combines all 3 components correctly
    #     (3-4-5 triangle scaled), verified against hand calc.
    vel_log = [(0, 0, 0), (0.03, 0.04, 0.0)]  # |Delta v| = 0.05
    a = peak_linear_acceleration(vel_log, dt=0.005)
    expected = 0.05 / 0.005  # 10.0 m/s^2
    assert abs(a - expected) < 1e-6, (a, expected)

    # 12. Peak is taken over ALL consecutive pairs, not just the last one.
    vel_log = [(0, 0, 0), (0, 0, 1.0), (0, 0, 1.0), (0, 0, 1.0)]  # spike only in the first interval
    a = peak_linear_acceleration(vel_log, dt=0.1)
    expected = 1.0 / 0.1  # 10.0, from the first interval; later intervals are 0
    assert abs(a - expected) < 1e-6, (a, expected)

    # --- fragility_violation_score ---

    # 13. Both signals well under threshold -> score < 1 (safe).
    s = fragility_violation_score(peak_force=1.0, peak_accel=2.0, force_threshold=5.0, accel_threshold=20.0)
    assert s < 1.0, s
    assert abs(s - max(1.0 / 5.0, 2.0 / 20.0)) < 1e-9, s

    # 14. Force exceeds its threshold, accel does not -> score driven by force, >= 1.
    s = fragility_violation_score(peak_force=10.0, peak_accel=1.0, force_threshold=5.0, accel_threshold=20.0)
    assert s >= 1.0, s
    assert abs(s - 2.0) < 1e-9, s  # 10/5 = 2.0, dominates over 1/20=0.05

    # 15. Accel exceeds its threshold, force does not -> score driven by accel, >= 1.
    s = fragility_violation_score(peak_force=1.0, peak_accel=100.0, force_threshold=5.0, accel_threshold=20.0)
    assert s >= 1.0, s
    assert abs(s - 5.0) < 1e-9, s  # 100/20 = 5.0, dominates over 1/5=0.2

    # 16. Both exceed -> takes the MAX ratio, not the sum.
    s = fragility_violation_score(peak_force=10.0, peak_accel=100.0, force_threshold=5.0, accel_threshold=20.0)
    assert abs(s - 5.0) < 1e-9, s  # max(2.0, 5.0) = 5.0, NOT 2.0+5.0=7.0

    # 17. Zero peaks (perfectly still egg, no contact force at all) -> score exactly 0.
    s = fragility_violation_score(peak_force=0.0, peak_accel=0.0, force_threshold=5.0, accel_threshold=20.0)
    assert abs(s - 0.0) < 1e-9, s

    # 18. Negative peak_force / peak_accel raise (physically meaningless -- these are magnitudes).
    try:
        fragility_violation_score(peak_force=-1.0, peak_accel=0.0, force_threshold=5.0, accel_threshold=20.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        fragility_violation_score(peak_force=0.0, peak_accel=-1.0, force_threshold=5.0, accel_threshold=20.0)
        assert False, "expected ValueError"
    except ValueError:
        pass

    # 19. Non-positive thresholds raise.
    try:
        fragility_violation_score(peak_force=1.0, peak_accel=1.0, force_threshold=0.0, accel_threshold=20.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        fragility_violation_score(peak_force=1.0, peak_accel=1.0, force_threshold=5.0, accel_threshold=-1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass

    # 20. Doubling a threshold exactly halves that ratio's contribution (scaling sanity).
    s1 = fragility_violation_score(peak_force=10.0, peak_accel=0.0, force_threshold=5.0, accel_threshold=20.0)
    s2 = fragility_violation_score(peak_force=10.0, peak_accel=0.0, force_threshold=10.0, accel_threshold=20.0)
    assert abs(s1 - 2.0) < 1e-9 and abs(s2 - 1.0) < 1e-9 and abs(s1 - 2 * s2) < 1e-9, (s1, s2)

    # 21. End-to-end discriminative sanity: a canonical SAFE grasp (low
    #     force-capped peak force, low resulting accel) scores well under 1,
    #     a canonical UNSAFE grasp (uncapped high peak force AND a sharp
    #     velocity spike) scores well over 1 -- this is the core
    #     discriminative-power claim the scenario script must reproduce
    #     empirically with real physics.
    safe_score = fragility_violation_score(
        peak_force=2.5, peak_accel=3.0, force_threshold=8.0, accel_threshold=15.0,
    )
    unsafe_score = fragility_violation_score(
        peak_force=45.0, peak_accel=60.0, force_threshold=8.0, accel_threshold=15.0,
    )
    assert safe_score < 1.0, safe_score
    assert unsafe_score > safe_score, (safe_score, unsafe_score)
    assert unsafe_score >= 1.0, unsafe_score

    print("ALL METRICS_2B SELF-TESTS PASSED")
