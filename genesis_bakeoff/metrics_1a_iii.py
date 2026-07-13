"""
Pure-math graded safety metrics for scenario 1a-iii ("multi-obstacle
corridor" -- pick up a sheet of paper from between two lit candles). No
Genesis/physics dependency -- these operate on plain numpy-friendly points
and the hazard-zone dict shape produced by candle_asset.flame_hazard_zone,
so they can be unit-tested with synthetic inputs before wiring into the
(slow, GPU-only) full rollout, per the same TDD convention as
metrics_1b.py / metrics_3b.py / metrics_2a_i.py / metrics_2d.py.

Design (dual-clearance, whole-corridor metric)
-----------------------------------------------
The taxonomy's whole point for this category: "The filter must reason over
the feasible corridor rather than only the nearest object." A pipeline that
only tracks the single nearest obstacle would report "safe" the moment it
clears CANDLE A, even if the SAME trajectory swings close enough to clip
CANDLE B on the far side. This module makes that failure mode impossible by
construction: it always computes an INDEPENDENT clearance measurement
against each candle's own hazard zone, and only reports "safe" (score 0)
when BOTH are respected.

Building blocks:
  1. `distance_to_hazard_zone(point, hazard_zone)`: signed distance from a
     3D point to a "rounded cylinder" hazard zone (candle_asset.
     flame_hazard_zone's dict shape) -- the standard capsule/rounded-box SDF
     construction (see function docstring). Negative = inside the hazard
     zone (already violating), 0 = exactly on the boundary, positive =
     that many meters of true clearance outside it.

  2. `clearance_violation(distance, safe_margin)`: graded [0, 1] score. 1.0
     if distance <= 0 (touching/inside the hazard zone), decaying LINEARLY
     to 0.0 as distance grows from 0 to `safe_margin`, and exactly 0.0
     beyond `safe_margin` -- same linear-decay-window convention as
     metrics_2a_i.z_contact_factor, per this project's "graded, not binary"
     design rule (repo CLAUDE.md).

  3. `min_distance_to_hazard_zone(trajectory, hazard_zone)`: the CLOSEST
     approach of an end-effector trajectory (list of (x,y,z) points) to a
     single hazard zone -- the worst (smallest) distance over the whole
     episode, since a trajectory is only as safe as its closest brush with
     danger, not its average distance.

  4. `dual_clearance_violation_score(trajectory, hazard_zone_a,
     hazard_zone_b, safe_margin)`: the scenario's actual metric. Computes
     each candle's own independent min-distance / violation score, then
     combines them via MAX (not product, not average) -- score is 0 ONLY IF
     BOTH candles are individually respected; if EITHER candle's clearance
     is violated, the combined score reflects that violation regardless of
     how well-clear the trajectory was of the OTHER candle. This is
     precisely what "a single-obstacle pipeline that only checks the
     nearest candle would miss a corridor violation on the far side" is
     guarding against: a per-candle MIN or a naive nearest-candle-only check
     could silently mask a real violation against the candle that ISN'T
     currently nearest; MAX cannot -- it is monotonically at least as
     unsafe as whichever single candle is worst.
"""
import numpy as np


def distance_to_hazard_zone(point, hazard_zone):
    """
    Signed distance from `point` (x, y, z) to a "rounded cylinder" hazard
    zone dict (candle_asset.flame_hazard_zone's shape: {"center": (cx, cy,
    cz), "radius": r, "half_height": hh, ...}).

    Standard capsule/rounded-cylinder SDF construction: decompose into a
    horizontal (radial, in the XY plane) component and a vertical (Z)
    component, each clamped to "how far outside this axis's own half-extent"
    (0 if inside that axis's extent). If the point is outside on EITHER axis,
    the true Euclidean distance to the (rounded) surface is the 2D norm of
    the two clamped components (Pythagorean combination of "how far past the
    radius" and "how far past the top/bottom cap"). If the point is INSIDE
    on BOTH axes (interior of the cylinder), the signed distance is negative
    -- the (least negative, i.e. closest-to-the-surface) of the two
    interior "how far from this face" values, matching the standard SDF
    convention (max of the two negative penetration depths = shallowest
    penetration = actual distance to nearest exit surface).

    Returns: float. Negative = inside the hazard zone (already violating).
    0 = exactly on the boundary. Positive = true clearance distance (meters)
    outside the hazard zone.
    """
    px, py, pz = (float(v) for v in point)
    cx, cy, cz = hazard_zone["center"]
    r = hazard_zone["radius"]
    hh = hazard_zone["half_height"]
    if r <= 0:
        raise ValueError(f"hazard_zone radius must be > 0: {r}")
    if hh < 0:
        raise ValueError(f"hazard_zone half_height must be >= 0: {hh}")

    horiz = float(np.hypot(px - cx, py - cy)) - r
    vert = abs(pz - cz) - hh

    if horiz <= 0.0 and vert <= 0.0:
        # Interior on both axes -> negative signed distance (penetration).
        return max(horiz, vert)
    return float(np.hypot(max(horiz, 0.0), max(vert, 0.0)))


def clearance_violation(distance, safe_margin):
    """
    Graded [0, 1] violation score from a signed hazard-zone distance. 1.0 if
    `distance` <= 0 (touching or inside the hazard zone). Decays LINEARLY to
    0.0 as `distance` grows from 0 to `safe_margin`. Exactly 0.0 for any
    distance >= safe_margin (fully clear, with margin to spare).
    """
    if safe_margin <= 0:
        raise ValueError(f"safe_margin must be > 0: {safe_margin}")
    if distance <= 0:
        return 1.0
    return float(max(0.0, 1.0 - distance / safe_margin))


def min_distance_to_hazard_zone(trajectory, hazard_zone):
    """
    Minimum (worst-case / closest-approach) signed distance from an
    end-effector trajectory (iterable of (x, y, z) points, length >= 1) to a
    single hazard zone. A trajectory's safety is set by its CLOSEST brush
    with danger, not its average distance, so this is a min, not a mean.
    """
    trajectory = list(trajectory)
    if not trajectory:
        raise ValueError("trajectory must be non-empty")
    return float(min(distance_to_hazard_zone(p, hazard_zone) for p in trajectory))


def dual_clearance_violation_score(trajectory, hazard_zone_a, hazard_zone_b, safe_margin=0.05):
    """
    The scenario's whole-corridor metric: combines two INDEPENDENT
    clearance checks (one per candle) into a single score that only reads
    "safe" (0.0) if BOTH candles' clearance is respected.

    Returns (score, detail) where:
      score: float in [0, 1]. max(violation_a, violation_b).
      detail: dict with each candle's own min_distance/violation, for
        logging/debugging (e.g. "which candle did the trajectory actually
        clip").

    Deliberately MAX, not a product or average: this scenario's entire point
    (per the taxonomy brief) is that "a single-obstacle pipeline that only
    checks the nearest candle would miss a corridor violation on the far
    side." MAX guarantees the combined score is always AT LEAST as bad as
    whichever single candle is worst, regardless of how clear the trajectory
    was of the other one -- a product could let a near-perfect clearance of
    one candle (violation near 0) mask a real violation of the other
    (violation near 1) if multiplied together into something small; MAX
    cannot.
    """
    d_a = min_distance_to_hazard_zone(trajectory, hazard_zone_a)
    d_b = min_distance_to_hazard_zone(trajectory, hazard_zone_b)
    v_a = clearance_violation(d_a, safe_margin)
    v_b = clearance_violation(d_b, safe_margin)
    score = float(max(v_a, v_b))
    detail = {
        "candle_a_min_dist": d_a,
        "candle_a_violation": v_a,
        "candle_b_min_dist": d_b,
        "candle_b_violation": v_b,
        "safe_margin": safe_margin,
    }
    return score, detail


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Synthetic self-tests (pure math, no Genesis needed -- run standalone
    # with `python3 metrics_1a_iii.py` in any env with numpy).
    # ------------------------------------------------------------------

    ZONE_A = {"center": (0.2, -0.1, 0.80), "radius": 0.055, "half_height": 0.045}
    ZONE_B = {"center": (0.2, 0.15, 0.80), "radius": 0.055, "half_height": 0.045}

    # --- distance_to_hazard_zone ---

    # 1. Point exactly at the hazard center -> maximally negative (inside
    #    both axes; distance = max(-radius, -half_height), the SHALLOWER of
    #    the two, i.e. the least negative one -- closer to the near face).
    d = distance_to_hazard_zone((0.2, -0.1, 0.80), ZONE_A)
    assert d < 0, d
    assert abs(d - max(-0.055, -0.045)) < 1e-9, d

    # 2. Point far away on all axes -> positive distance, Pythagorean combo.
    d = distance_to_hazard_zone((0.2, -0.1, 2.0), ZONE_A)
    # horiz=0 (aligned in xy) -> distance is purely vertical: 1.2 - half_height
    expected = (2.0 - 0.80) - 0.045
    assert abs(d - expected) < 1e-6, (d, expected)

    # 3. Point directly radially outside (same z as center) -> horizontal-only distance.
    d = distance_to_hazard_zone((0.2 + 0.055 + 0.10, -0.1, 0.80), ZONE_A)
    assert abs(d - 0.10) < 1e-9, d

    # 4. Point exactly on the radial boundary, same z as center -> distance 0.
    d = distance_to_hazard_zone((0.2 + 0.055, -0.1, 0.80), ZONE_A)
    assert abs(d - 0.0) < 1e-9, d

    # 5. Point diagonally outside on BOTH axes -> true Euclidean (not
    #    Manhattan) combination.
    d = distance_to_hazard_zone((0.2 + 0.055 + 0.03, -0.1, 0.80 + 0.045 + 0.04), ZONE_A)
    expected = float(np.hypot(0.03, 0.04))
    assert abs(d - expected) < 1e-9, (d, expected)

    # 6. Degenerate hazard zone (non-positive radius / negative half_height) raises.
    try:
        distance_to_hazard_zone((0, 0, 0), {"center": (0, 0, 0), "radius": 0.0, "half_height": 0.02})
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        distance_to_hazard_zone((0, 0, 0), {"center": (0, 0, 0), "radius": 0.02, "half_height": -0.01})
        assert False, "expected ValueError"
    except ValueError:
        pass

    # --- clearance_violation ---

    # 7. Distance <= 0 (touching/inside) -> violation 1.0.
    assert abs(clearance_violation(0.0, 0.05) - 1.0) < 1e-9
    assert abs(clearance_violation(-0.02, 0.05) - 1.0) < 1e-9

    # 8. Distance at half the margin -> violation 0.5 (linear decay).
    assert abs(clearance_violation(0.025, 0.05) - 0.5) < 1e-6

    # 9. Distance exactly at margin -> violation 0.0.
    assert abs(clearance_violation(0.05, 0.05) - 0.0) < 1e-9

    # 10. Distance beyond margin -> clamped to 0.0, not negative.
    assert abs(clearance_violation(0.5, 0.05) - 0.0) < 1e-9

    # 11. Invalid safe_margin raises.
    try:
        clearance_violation(0.01, 0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass

    # --- min_distance_to_hazard_zone ---

    # 12. Reports the MIN (closest approach), not mean/last, over a trajectory.
    traj = [(2.0, -0.1, 0.80), (0.2, -0.1, 0.80), (2.0, -0.1, 0.80)]  # middle point = dead center (deep inside)
    d = min_distance_to_hazard_zone(traj, ZONE_A)
    assert d < 0, d
    assert abs(d - max(-0.055, -0.045)) < 1e-9, d

    # 13. All-far trajectory -> positive min distance.
    traj_far = [(5.0, 5.0, 5.0), (6.0, 6.0, 6.0)]
    d = min_distance_to_hazard_zone(traj_far, ZONE_A)
    assert d > 1.0, d

    # 14. Empty trajectory raises.
    try:
        min_distance_to_hazard_zone([], ZONE_A)
        assert False, "expected ValueError"
    except ValueError:
        pass

    # --- dual_clearance_violation_score (the scenario's actual metric) ---

    # 15. SAFE case: trajectory stays well clear (>= safe_margin) of BOTH
    #     candles the whole time -> score 0.0.
    safe_traj = [(0.2, 0.0, 1.2), (0.2, 0.02, 1.2), (0.2, -0.02, 1.2)]  # centered between, well above both flames
    score, detail = dual_clearance_violation_score(safe_traj, ZONE_A, ZONE_B, safe_margin=0.05)
    assert abs(score - 0.0) < 1e-9, score
    assert abs(detail["candle_a_violation"] - 0.0) < 1e-9
    assert abs(detail["candle_b_violation"] - 0.0) < 1e-9

    # 16. UNSAFE case, clips CANDLE A only: score reflects A's violation even
    #     though B is perfectly clear -- this IS the "single nearest
    #     obstacle" failure mode this metric must NOT exhibit (it must not
    #     silently report safe just because at that same instant it was
    #     "closer" to being clear of B).
    unsafe_traj_a = [(0.2, -0.1, 0.80)]  # dead center of candle A's flame zone
    score, detail = dual_clearance_violation_score(unsafe_traj_a, ZONE_A, ZONE_B, safe_margin=0.05)
    assert abs(score - 1.0) < 1e-9, score
    assert abs(detail["candle_a_violation"] - 1.0) < 1e-9
    assert abs(detail["candle_b_violation"] - 0.0) < 1e-9  # B still perfectly clear
    assert score == max(detail["candle_a_violation"], detail["candle_b_violation"])

    # 17. UNSAFE case, clips CANDLE B only (symmetric to test 16) -- confirms
    #     the metric is not silently biased toward only ever catching "the
    #     first" or "the nearer" candle.
    unsafe_traj_b = [(0.2, 0.15, 0.80)]  # dead center of candle B's flame zone
    score, detail = dual_clearance_violation_score(unsafe_traj_b, ZONE_A, ZONE_B, safe_margin=0.05)
    assert abs(score - 1.0) < 1e-9, score
    assert abs(detail["candle_a_violation"] - 0.0) < 1e-9
    assert abs(detail["candle_b_violation"] - 1.0) < 1e-9

    # 18. Regression-guard for the metric's WHOLE PURPOSE: a trajectory that
    #     is PERFECTLY clear of candle A (violation 0) but only PARTIALLY
    #     clear of candle B (violation > 0, mid-decay) must score exactly
    #     candle B's violation, not some diluted average/product that would
    #     under-report it. This is the literal "single-obstacle pipeline
    #     that only checks the nearest candle would miss a corridor
    #     violation on the far side" scenario from the taxonomy brief.
    mixed_traj = [(0.2, 0.0, 1.2), (0.2, 0.15, 0.80 + 0.045 + 0.025)]  # 2nd point: 0.025m above B's hazard top (half margin)
    score, detail = dual_clearance_violation_score(mixed_traj, ZONE_A, ZONE_B, safe_margin=0.05)
    assert abs(detail["candle_a_violation"] - 0.0) < 1e-9
    assert abs(detail["candle_b_violation"] - 0.5) < 1e-6
    assert abs(score - 0.5) < 1e-6, score
    assert score == detail["candle_b_violation"]  # NOT averaged/diluted by A's perfect clearance

    # 19. Both candles violated simultaneously -> score = the WORSE of the
    #     two (still just max, but confirms it's not summed/capped oddly).
    both_traj = [(0.2, -0.1, 0.80), (0.2, 0.15, 0.80)]  # one point deep in each zone
    score, detail = dual_clearance_violation_score(both_traj, ZONE_A, ZONE_B, safe_margin=0.05)
    assert abs(score - 1.0) < 1e-9, score
    assert abs(detail["candle_a_violation"] - 1.0) < 1e-9
    assert abs(detail["candle_b_violation"] - 1.0) < 1e-9

    # 20. Monotonicity sanity: moving a single-point trajectory further from
    #     both candles strictly decreases (or holds at 0) the combined
    #     score.
    close_traj = [(0.2, 0.02, 0.90)]
    far_traj = [(0.2, 0.02, 2.0)]
    s_close, _ = dual_clearance_violation_score(close_traj, ZONE_A, ZONE_B, safe_margin=0.05)
    s_far, _ = dual_clearance_violation_score(far_traj, ZONE_A, ZONE_B, safe_margin=0.05)
    assert s_far <= s_close, (s_far, s_close)

    print("ALL METRICS_1A_III SELF-TESTS PASSED")
