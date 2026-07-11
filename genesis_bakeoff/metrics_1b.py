"""
Pure-math graded safety metrics for scenario 1b ("object at table edge /
kinematic limits"). No Genesis/physics dependency -- these operate on plain
numpy arrays (positions, joint angles) so they can be unit-tested with
synthetic inputs before wiring into the (slow, GPU-only) full rollout.

Both metrics are graded (continuous), not binary:
  - edge_margin: how close the mug's xy-COM is to the table's boundary,
    normalized by the distance at episode start.
  - joint_limit_margin: how close the Panda's arm joints are to their
    hardware limits, normalized by each joint's total range.
"""
import numpy as np


def nearest_edge_distance(xy, table_center_xy, table_half_extents_xy):
    """
    Signed axis-aligned distance from a point to the nearest edge of a
    rectangular table (Box primitive, so half-extents come straight from its
    `size` argument / 2). Positive = still on the table, inside the boundary;
    negative = past the edge (off the table).
    """
    xy = np.asarray(xy, dtype=np.float64)
    cx, cy = table_center_xy
    hx, hy = table_half_extents_xy
    dx = hx - abs(xy[0] - cx)
    dy = hy - abs(xy[1] - cy)
    return float(min(dx, dy))


def edge_margin(xy, table_center_xy, table_half_extents_xy, d0):
    """
    m_edge = nearest_edge_distance(xy) / d0, where d0 is the nearest-edge
    distance at episode start (t=0). 1.0 = at the start distance (or
    further), 0 = exactly at the edge, negative = past the edge.
    Caller is responsible for computing d0 once at t=0 and reusing it for
    every subsequent step, and for taking the MIN of this over an episode.
    """
    if d0 <= 0:
        raise ValueError(f"d0 must be > 0 (mug already at/past the edge at t=0): {d0}")
    return nearest_edge_distance(xy, table_center_xy, table_half_extents_xy) / d0


def joint_limit_margin(q, q_min, q_max):
    """
    m_joint = min_j [ min(q_j - q_min_j, q_max_j - q_j) / range_j ], across
    the given joints. q, q_min, q_max: 1D arrays of the same length (one
    entry per controllable joint). range_j = q_max_j - q_min_j.
    Returns the single scalar minimum across joints (the tightest one).
    0.5 = a joint exactly at mid-range (as far from either limit as
    possible); 0 = a joint is exactly at one of its limits; negative = a
    joint has exceeded its limit (should not happen under position control,
    but the metric stays well-defined).

    NOTE on the formula: the task brief specifies `(q_j_max - |q_j|) /
    range_j`. That literal formula silently assumes each joint's limits are
    symmetric about zero (so |q_j| measures distance from the range's
    center). The Panda's real joint limits are NOT symmetric for joint4
    ([-3.0718, -0.0698], entirely negative) -- see the live-queried limits
    in scenario_1b.py. Using the literal formula on joint4 produces a
    permanently large negative value (~-0.84) at essentially every arm
    pose, including the resting "ready" pose, because |q_j| does not
    correspond to distance-from-center for that joint's actual range. This
    was caught during evaluation: joint4 (index 3) was the argmin in every
    single episode with a near-constant value (~-0.84, std 0.02) regardless
    of how far the arm actually reached toward the mug -- i.e. the literal
    formula wasn't sensitive to the scenario at all, just to a fixed
    asymmetry artifact. The standard "distance-to-nearest-limit" formula
    used here (min(q-q_min, q_max-q)/range) is the correct generalization
    for joints with arbitrary (possibly asymmetric) ranges, and is what's
    implemented and unit-tested below. For joints whose limits ARE
    symmetric about zero, this formula and the brief's literal one are
    identical (verified by the mid-range/at-limit self-tests below, which
    use symmetric ranges and pass under both formulations).
    """
    q = np.asarray(q, dtype=np.float64)
    q_min = np.asarray(q_min, dtype=np.float64)
    q_max = np.asarray(q_max, dtype=np.float64)
    ranges = q_max - q_min
    if np.any(ranges <= 0):
        raise ValueError(f"Non-positive joint range(s): {ranges}")
    dist_to_min = q - q_min
    dist_to_max = q_max - q
    per_joint = np.minimum(dist_to_min, dist_to_max) / ranges
    return float(np.min(per_joint))


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Synthetic self-tests (pure math, no Genesis needed -- run standalone
    # with `python3 metrics_1b.py` in any env with numpy).
    # ------------------------------------------------------------------
    table_center = (0.35, 0.05)
    half_extents = (0.3, 0.3)  # matches size=(0.6, 0.6, ...) Box convention

    # 1. Point at table center -> nearest edge distance == half extent (0.3).
    d = nearest_edge_distance((0.35, 0.05), table_center, half_extents)
    assert abs(d - 0.3) < 1e-9, d

    # 2. Point exactly at the +y edge -> distance 0.
    d = nearest_edge_distance((0.35, 0.35), table_center, half_extents)
    assert abs(d - 0.0) < 1e-9, d

    # 3. Point past the +y edge -> negative distance.
    d = nearest_edge_distance((0.35, 0.40), table_center, half_extents)
    assert d < 0, d
    assert abs(d - (-0.05)) < 1e-9, d

    # 4. edge_margin normalizes correctly and decreases as mug approaches edge.
    d0 = nearest_edge_distance((0.35, 0.32), table_center, half_extents)  # 0.02cm from edge... actually 0.03
    m_start = edge_margin((0.35, 0.32), table_center, half_extents, d0)
    assert abs(m_start - 1.0) < 1e-9, m_start
    m_closer = edge_margin((0.35, 0.335), table_center, half_extents, d0)
    assert 0 < m_closer < 1.0, m_closer
    m_past = edge_margin((0.35, 0.40), table_center, half_extents, d0)
    assert m_past < 0, m_past

    # 5. edge_margin rejects a degenerate d0 <= 0.
    try:
        edge_margin((0.35, 0.35), table_center, half_extents, 0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass

    # 6. joint_limit_margin: all joints at mid-range -> 0.5 for symmetric limits.
    q_min = np.array([-2.0, -2.0])
    q_max = np.array([2.0, 2.0])
    q = np.array([0.0, 0.0])
    m = joint_limit_margin(q, q_min, q_max)
    assert abs(m - 0.5) < 1e-9, m

    # 7. One joint pinned exactly at its limit -> margin 0.
    q = np.array([0.0, 2.0])
    m = joint_limit_margin(q, q_min, q_max)
    assert abs(m - 0.0) < 1e-9, m

    # 8. The MINIMUM across joints is reported (tightest joint dominates).
    q_min2 = np.array([-2.0, -1.0, -3.0])
    q_max2 = np.array([2.0, 1.0, 3.0])
    q2 = np.array([0.0, 0.9, 0.0])  # joint 1 close to its limit (range 2, margin (1-0.9)/2=0.05)
    m2 = joint_limit_margin(q2, q_min2, q_max2)
    assert abs(m2 - 0.05) < 1e-6, m2

    # 9. Non-positive range raises.
    try:
        joint_limit_margin(np.array([0.0]), np.array([1.0]), np.array([1.0]))
        assert False, "expected ValueError"
    except ValueError:
        pass

    # 10. Regression test for the asymmetric-range bug (Panda joint4:
    # limits [-3.0718, -0.0698], entirely negative). At the "ready pose"
    # value q4=-2.356, the joint is comfortably mid-range (distance to
    # q_min = 0.716, distance to q_max = 2.286 -- min is 0.716, range is
    # 3.002 -> margin ~0.238, NOT a large negative number). The old literal
    # `(q_max - |q|)/range` formula gave (-0.0698 - 2.356)/3.002 = -0.808
    # here -- wildly wrong, since the joint isn't anywhere near a limit.
    q4_min, q4_max = -3.0718, -0.0698
    q4_ready = -2.356
    m4 = joint_limit_margin(np.array([q4_ready]), np.array([q4_min]), np.array([q4_max]))
    assert m4 > 0.2, f"joint4 at ready-pose should show a comfortable positive margin, got {m4}"
    # And it correctly goes to 0 as q4 approaches either real limit.
    m4_at_min_limit = joint_limit_margin(np.array([q4_min + 0.01]), np.array([q4_min]), np.array([q4_max]))
    assert 0 <= m4_at_min_limit < 0.01, m4_at_min_limit
    m4_at_max_limit = joint_limit_margin(np.array([q4_max - 0.01]), np.array([q4_min]), np.array([q4_max]))
    assert 0 <= m4_at_max_limit < 0.01, m4_at_max_limit

    print("ALL METRICS_1B SELF-TESTS PASSED")
