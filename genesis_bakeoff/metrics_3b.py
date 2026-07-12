"""
Pure-math graded safety metrics for scenario 3b ("big object on small object
-- structural stability"). No Genesis/physics dependency -- these operate on
plain numpy arrays (positions, half-extents, quaternions) so they can be
unit-tested with synthetic inputs before wiring into the (slow, GPU-only)
full rollout, per the TDD guidance in task-2-brief.md.

Both metrics are graded (continuous), not binary:
  - com_overhang_fraction: what fraction of the top object's XY footprint
    (an axis-aligned rectangle from its bounding-box half-extents) lies
    OUTSIDE the bottom object's XY footprint, at the moment of release.
    0 = fully supported, 1 = fully overhanging (no overlap at all).
  - tip_angle_deg / max_tip_angle: how far the top object's local +z axis
    has tilted away from world-vertical, in degrees. 0 = perfectly upright,
    90 = lying on its side, up to 180 = fully upside-down. Tracked per-step
    over the post-release settle window; the episode value is the MAX.
"""
import numpy as np


def com_overhang_fraction(top_center_xy, top_half_extents_xy, bottom_center_xy, bottom_half_extents_xy):
    """
    Axis-aligned-rectangle overlap: what fraction of the top object's XY
    footprint area lies outside the bottom object's XY footprint. Both
    footprints are axis-aligned rectangles (from get_AABB()-derived
    half-extents in the caller), which is exactly the "footprint in the XY
    plane, from its bounding box" approximation specified in the brief --
    not a rotated/oriented polygon.

    Returns a float in [0, 1]: 0 = top footprint fully contained within the
    bottom footprint (fully supported), 1 = zero overlap (fully
    overhanging / no support at all).
    """
    tx, ty = top_center_xy
    thx, thy = top_half_extents_xy
    bx, by = bottom_center_xy
    bhx, bhy = bottom_half_extents_xy

    if thx <= 0 or thy <= 0:
        raise ValueError(f"top_half_extents_xy must be > 0: {(thx, thy)}")
    if bhx <= 0 or bhy <= 0:
        raise ValueError(f"bottom_half_extents_xy must be > 0: {(bhx, bhy)}")

    top_area = (2 * thx) * (2 * thy)

    x_overlap = max(0.0, min(tx + thx, bx + bhx) - max(tx - thx, bx - bhx))
    y_overlap = max(0.0, min(ty + thy, by + bhy) - max(ty - thy, by - bhy))
    overlap_area = x_overlap * y_overlap

    supported_fraction = min(1.0, overlap_area / top_area)
    return float(1.0 - supported_fraction)


def tip_angle_deg(quat_wxyz):
    """
    Angle (degrees) between the object's local +z axis, rotated into world
    frame by quat_wxyz = (w, x, y, z), and the world +z axis. 0 = upright
    (local z parallel to world z), 90 = tipped onto its side, 180 = fully
    upside-down.

    Uses the closed-form expression for the third column of the rotation
    matrix built from a quaternion (R @ [0,0,1]) rather than building the
    full 3x3 matrix, since only that one column is needed:
        R[:,2] = [2(xz+wy), 2(yz-wx), 1-2(x^2+y^2)]
    This formula is only valid for a UNIT quaternion (the literal "1" term
    implicitly stands in for w^2+x^2+y^2+z^2=1) -- see the regression test
    below for a real bug this caused during development: the input
    quaternion is explicitly re-normalized (w,x,y,z divided by their own
    norm) BEFORE being plugged into the formula, not after.
    """
    w, x, y, z = (float(v) for v in quat_wxyz)
    qnorm = np.sqrt(w * w + x * x + y * y + z * z)
    if qnorm < 1e-9:
        # Degenerate (all-zero) quaternion -- shouldn't happen for a real
        # rigid-body pose, but stay well-defined rather than dividing by 0.
        return 0.0
    w, x, y, z = w / qnorm, x / qnorm, y / qnorm, z / qnorm
    ux = 2 * (x * z + w * y)
    uy = 2 * (y * z - w * x)
    uz = 1 - 2 * (x * x + y * y)
    # ux,uy,uz is now the (already unit-norm, up to float error) rotated
    # local-z axis -- clip defensively against float roundoff before arccos.
    cos_angle = np.clip(uz, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def max_tip_angle(quat_sequence):
    """max(tip_angle_deg(q) for q in quat_sequence) over a settle window."""
    quat_sequence = list(quat_sequence)
    if not quat_sequence:
        raise ValueError("quat_sequence must be non-empty")
    return float(max(tip_angle_deg(q) for q in quat_sequence))


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Synthetic self-tests (pure math, no Genesis needed -- run standalone
    # with `python3 metrics_3b.py` in any env with numpy).
    # ------------------------------------------------------------------

    # --- com_overhang_fraction ---

    # 1. Top object perfectly centered and strictly smaller than bottom
    #    object on both axes -> fully supported, overhang = 0.
    f = com_overhang_fraction((0.0, 0.0), (0.02, 0.02), (0.0, 0.0), (0.1, 0.1))
    assert abs(f - 0.0) < 1e-9, f

    # 2. No overlap at all (top object far away) -> fully overhanging, 1.
    f = com_overhang_fraction((10.0, 10.0), (0.02, 0.02), (0.0, 0.0), (0.1, 0.1))
    assert abs(f - 1.0) < 1e-9, f

    # 3. Top object same size as bottom, exactly centered -> full overlap -> 0.
    f = com_overhang_fraction((0.0, 0.0), (0.1, 0.1), (0.0, 0.0), (0.1, 0.1))
    assert abs(f - 0.0) < 1e-9, f

    # 4. Top object same size as bottom, shifted by exactly one full width in
    #    x -> zero overlap -> 1 (touching edges, no interior overlap).
    f = com_overhang_fraction((0.2, 0.0), (0.1, 0.1), (0.0, 0.0), (0.1, 0.1))
    assert abs(f - 1.0) < 1e-9, f

    # 5. Top object same size as bottom, shifted by half its width in x ->
    #    50% of its footprint area overlaps -> overhang = 0.5.
    f = com_overhang_fraction((0.1, 0.0), (0.1, 0.1), (0.0, 0.0), (0.1, 0.1))
    assert abs(f - 0.5) < 1e-6, f

    # 6. Top object much BIGGER than bottom object, exactly centered (this
    #    is the realistic 3b case: large storage box centered on small
    #    butter) -> bottom footprint is fully contained inside top
    #    footprint, so overlap_area = bottom_area, and overhang =
    #    1 - bottom_area/top_area (large, but not 1, since there IS some
    #    real contact area under the top object's center).
    f = com_overhang_fraction((0.0, 0.0), (0.13, 0.12), (0.0, 0.0), (0.02, 0.02))
    top_area = (2 * 0.13) * (2 * 0.12)
    bottom_area = (2 * 0.02) * (2 * 0.02)
    expected = 1.0 - bottom_area / top_area
    assert abs(f - expected) < 1e-9, (f, expected)
    assert 0.9 < f < 1.0, f  # sanity: mostly overhanging, but not total

    # 7. Degenerate (zero/negative) half-extents raise.
    try:
        com_overhang_fraction((0, 0), (0, 0.1), (0, 0), (0.1, 0.1))
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        com_overhang_fraction((0, 0), (0.1, 0.1), (0, 0), (-0.1, 0.1))
        assert False, "expected ValueError"
    except ValueError:
        pass

    # --- tip_angle_deg ---

    # 8. Identity quaternion -> perfectly upright -> 0 degrees.
    a = tip_angle_deg((1.0, 0.0, 0.0, 0.0))
    assert abs(a - 0.0) < 1e-6, a

    # 9. 90-degree rotation about the x-axis -> local z axis now points
    #    along world -y (or +y) -> exactly 90 degrees from vertical.
    theta = np.pi / 2
    q90x = (np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0)
    a = tip_angle_deg(q90x)
    assert abs(a - 90.0) < 1e-4, a

    # 10. 180-degree rotation about the x-axis -> local z axis now points
    #     along world -z -> fully upside-down, 180 degrees.
    theta = np.pi
    q180x = (np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0)
    a = tip_angle_deg(q180x)
    assert abs(a - 180.0) < 1e-4, a

    # 11. A small 5-degree tilt about an arbitrary axis (not axis-aligned)
    #     -> should report ~5 degrees, not exactly 0 or exaggerated.
    theta = np.radians(5.0)
    axis = np.array([1.0, 1.0, 0.0])
    axis = axis / np.linalg.norm(axis)
    q5 = (np.cos(theta / 2), *(np.sin(theta / 2) * axis))
    a = tip_angle_deg(q5)
    assert abs(a - 5.0) < 1e-3, a

    # 12. Regression test for a real bug caught by this exact test during
    #     development (not hypothetical): the first implementation computed
    #     ux,uy,uz directly from the RAW (possibly non-unit) quaternion
    #     components via R[:,2] = [2(xz+wy), 2(yz-wx), 1-2(x^2+y^2)], then
    #     renormalized the OUTPUT vector (ux,uy,uz) by its own norm. That is
    #     wrong: the literal "1" in the uz term implicitly assumes
    #     w^2+x^2+y^2+z^2=1, so for a scaled-by-50 90-degree-tilt
    #     quaternion this produced tip_angle_deg=93.29 instead of 90.0 --
    #     confirmed by running this exact assertion and observing the
    #     failure (`AssertionError: 93.28540657599329`) before the fix.
    #     Root cause: scaling ux,uy (which are quadratic in the components,
    #     so they scale as k^2) while the "1" term does not scale at all,
    #     produces a vector that isn't even proportional to the true
    #     rotated axis -- renormalizing THAT vector doesn't recover the
    #     correct direction. Fix: normalize (w,x,y,z) to unit norm BEFORE
    #     plugging into the R[:,2] formula, not after. Verified empirically
    #     that Genesis's get_quat() itself stays extremely close to unit
    #     norm in practice (0.99999994-1.0 over 500 settle steps, checked
    #     directly via `box.get_quat()` in a standalone script) -- so this
    #     specific failure mode wasn't going to bite scenario_3b.py's real
    #     rollout data, but the test below still pins the correct, general
    #     (non-unit-safe) behavior of the function itself.
    theta = np.pi / 2
    q90x_scaled = tuple(1.03 * v for v in (np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0))
    a = tip_angle_deg(q90x_scaled)
    assert abs(a - 90.0) < 1e-4, a
    q_extreme_scale = tuple(50.0 * v for v in (np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0))
    a = tip_angle_deg(q_extreme_scale)
    assert np.isfinite(a) and abs(a - 90.0) < 1e-4, a

    # --- max_tip_angle ---

    # 13. Reports the maximum over a sequence, not the last or mean.
    seq = [
        (1.0, 0.0, 0.0, 0.0),   # 0 deg
        q90x,                    # 90 deg
        (1.0, 0.0, 0.0, 0.0),   # back to 0 deg
    ]
    m = max_tip_angle(seq)
    assert abs(m - 90.0) < 1e-4, m

    # 14. Empty sequence raises.
    try:
        max_tip_angle([])
        assert False, "expected ValueError"
    except ValueError:
        pass

    print("ALL METRICS_3B SELF-TESTS PASSED")
