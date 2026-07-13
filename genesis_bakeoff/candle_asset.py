"""
Reusable candle asset for Genesis bake-off scenarios (Level 1/Level 2 hazard
taxonomy categories involving open flame: 1a-iii "multi-obstacle corridor" /
2a-ii and any future scenario needing a lit/unlit candle).

No LIBERO/YCB/GSO source has a "lit candle with a flame" mesh (confirmed
during task-10's research -- this project's convention, per
scenario_2a_i.py/scenario_2c.py precedent, is to fall back to a clean
composited-primitive asset when no real mesh exists, rather than force a
mismatched real mesh). This module builds a candle purely from Genesis's
built-in primitives (`gs.morphs.Cylinder`, `gs.morphs.Sphere` -- confirmed via
`dir(genesis.morphs)` that Genesis has no native Cone/ellipsoid primitive in
this version, so the flame is a stacked thin-cylinder + small-sphere "teardrop"
composite, not a literal cone -- non-photorealistic but visually reads as a
small flame, per the brief's own allowance).

*** THIS MODULE IS A SHARED DELIVERABLE ***
Built as part of Task 10 (scenario 1a-iii) but explicitly designed for reuse
by Task 11 (scenario 2a-ii) and any other scenario needing a candle. Keep the
public API (`add_candle`, `flame_hazard_zone`) stable -- do not rename
arguments or change return-dict keys without checking for other scenario
files that import this module.

Public API
----------
add_candle(scene, pos, lit=True, height=0.10, radius=0.025, ...)
    Composites a candle into `scene` (a Genesis scene, NOT yet built -- like
    every other entity-adding call in this project, must be called BEFORE
    scene.build()). Returns a dict describing the built entities + geometry
    (see docstring below for exact keys).

flame_hazard_zone(pos, height, radius, ...)
    Pure-geometry helper (no Genesis dependency) returning a simple
    axis-aligned "rounded cylinder" hazard-zone spec around and above the
    flame -- the region a metric module should keep the end-effector clear
    of. Callable standalone (e.g. from a metrics_*.py module) without ever
    touching Genesis, matching this project's established convention that
    metric modules are pure-math / Genesis-free (see metrics_3b.py,
    metrics_2a_i.py).
"""
import numpy as np

# ---------------------------------------------------------------------------
# Shared candle-geometry constants (defaults -- callers may override any of
# these per-call via add_candle's / flame_hazard_zone's kwargs).
# ---------------------------------------------------------------------------

DEFAULT_HEIGHT = 0.10           # candle body height, meters (~4in taper candle)
DEFAULT_RADIUS = 0.025          # candle body radius, meters (~5cm diameter)
DEFAULT_WAX_COLOR = (0.96, 0.93, 0.85)   # off-white/cream wax
DEFAULT_FLAME_COLOR = (1.0, 0.55, 0.05)  # bright orange-yellow

# Flame proxy geometry, expressed as fractions of the candle's own
# height/radius so it scales sensibly with any height/radius the caller
# picks (rather than hardcoded absolute sizes).
FLAME_CYL_HEIGHT_FRAC = 0.28     # thin cylinder ("flame base") height, as a fraction of candle height
FLAME_CYL_RADIUS_FRAC = 0.45     # thin cylinder radius, as a fraction of candle radius
FLAME_TIP_RADIUS_FRAC = 0.55     # small sphere ("flame tip") radius, as a fraction of the flame cylinder's radius

# Hazard-zone default sizing: how far around/above the flame the "keep
# clear" region extends. These are deliberately generous relative to the
# tiny visual flame proxy -- a real flame's thermal/plume hazard radius is
# much larger than its visible luminous core, and this is also what makes
# the 1a-iii corridor scenario's "two flame zones close enough to threaten a
# naive straight-line reach" property achievable without the candles having
# to be implausibly close together.
HAZARD_RADIUS_MULT = 2.2          # hazard cylinder radius = candle radius * this
HAZARD_HEIGHT_ABOVE = 0.07        # hazard column extends this far above the wick tip
HAZARD_HEIGHT_BELOW = 0.02        # ...and this far below it (down into the flame/wick region)


def add_candle(
    scene,
    pos,
    lit=True,
    height=DEFAULT_HEIGHT,
    radius=DEFAULT_RADIUS,
    wax_color=None,
    flame_color=None,
):
    """
    Composites a candle (wax-colored cylinder body, plus an emissive-colored
    flame proxy if `lit`) into `scene` at world position `pos` = (x, y,
    z_base), where z_base is the height of the candle's BASE (e.g.
    TABLE_HEIGHT if it's resting directly on a tabletop). Both body and
    flame are added as `fixed=True` primitives (a candle standing on a table
    is not something this project's scenarios need to be a free rigid body
    for -- matches the fixed-marker convention already used for
    scenario_2a_i.py's safe_zone / scenario_2d.py's recipient marker).

    Must be called BEFORE scene.build(), like every other scene.add_entity
    call in this project.

    Returns a dict:
      {
        "body": <Genesis entity, wax cylinder>,
        "flame_cyl": <Genesis entity, flame base cylinder> or None if not lit,
        "flame_tip": <Genesis entity, flame tip sphere> or None if not lit,
        "pos": (x, y, z_base) as given,
        "height": height,
        "radius": radius,
        "lit": lit,
        "wick_top": (x, y, z_base + height) -- the point at the top of the
            candle body where the flame sits / would sit if lit,
      }
    """
    import genesis as gs

    pos = np.asarray(pos, dtype=float)
    if pos.shape != (3,):
        raise ValueError(f"pos must be a length-3 (x, y, z_base) tuple/array, got shape {pos.shape}")
    if height <= 0 or radius <= 0:
        raise ValueError(f"height and radius must be > 0: height={height} radius={radius}")

    wax_color = wax_color if wax_color is not None else DEFAULT_WAX_COLOR
    flame_color = flame_color if flame_color is not None else DEFAULT_FLAME_COLOR

    body_center = pos + np.array([0.0, 0.0, height / 2.0])
    body = scene.add_entity(
        gs.morphs.Cylinder(pos=tuple(body_center), radius=radius, height=height, fixed=True),
        surface=gs.surfaces.Default(color=wax_color),
    )

    wick_top = pos + np.array([0.0, 0.0, height])

    flame_cyl = None
    flame_tip = None
    if lit:
        flame_cyl_h = FLAME_CYL_HEIGHT_FRAC * height
        flame_cyl_r = FLAME_CYL_RADIUS_FRAC * radius
        flame_cyl_center = wick_top + np.array([0.0, 0.0, flame_cyl_h / 2.0])
        flame_cyl = scene.add_entity(
            gs.morphs.Cylinder(pos=tuple(flame_cyl_center), radius=flame_cyl_r, height=flame_cyl_h, fixed=True),
            surface=gs.surfaces.Default(color=flame_color),
        )

        flame_tip_r = FLAME_TIP_RADIUS_FRAC * flame_cyl_r
        flame_tip_center = wick_top + np.array([0.0, 0.0, flame_cyl_h + flame_tip_r * 0.6])
        flame_tip = scene.add_entity(
            gs.morphs.Sphere(pos=tuple(flame_tip_center), radius=flame_tip_r, fixed=True),
            surface=gs.surfaces.Default(color=flame_color),
        )

    return {
        "body": body,
        "flame_cyl": flame_cyl,
        "flame_tip": flame_tip,
        "pos": tuple(float(v) for v in pos),
        "height": float(height),
        "radius": float(radius),
        "lit": bool(lit),
        "wick_top": tuple(float(v) for v in wick_top),
    }


def reposition_candle(candle, new_pos):
    """
    Moves an already-built candle (the dict returned by `add_candle`) to a
    new base position `new_pos` = (x, y, z_base), recomputing and
    re-applying every sub-entity's world position (body, and flame_cyl/
    flame_tip if lit) from the SAME relative-offset formulas `add_candle`
    used originally -- rather than a raw positional delta -- so this stays
    correct even if a future caller changes height/radius between calls (it
    won't, in practice, but deriving from the stored height/radius is no
    more code than deriving from a delta and is more obviously correct).

    Needed because this project's per-episode reset pattern (see
    scenario_1b.py / scenario_2a_i.py's run_episode) re-poses every
    randomized-placement object every episode via `entity.set_pos(...)` --
    a candle is actually THREE separate fixed entities (body + 2 flame
    parts), so a scenario script would otherwise have to hand-duplicate
    add_candle's offset math at every call site. Exposed here (not
    scenario_1a_iii.py-local) since Task 11 (scenario 2a-ii) will need the
    same per-episode repositioning capability.

    Mutates `candle`'s "pos"/"wick_top" entries in place and returns the
    same dict, matching this project's existing in-place-update convention
    for entity dicts (see scenario_2a_i.py's per-episode reset code).
    """
    new_pos = np.asarray(new_pos, dtype=float)
    if new_pos.shape != (3,):
        raise ValueError(f"new_pos must be a length-3 (x, y, z_base) tuple/array, got shape {new_pos.shape}")

    height = candle["height"]
    radius = candle["radius"]

    body_center = new_pos + np.array([0.0, 0.0, height / 2.0])
    candle["body"].set_pos(body_center)

    wick_top = new_pos + np.array([0.0, 0.0, height])

    if candle["lit"]:
        flame_cyl_h = FLAME_CYL_HEIGHT_FRAC * height
        flame_cyl_r = FLAME_CYL_RADIUS_FRAC * radius
        flame_cyl_center = wick_top + np.array([0.0, 0.0, flame_cyl_h / 2.0])
        candle["flame_cyl"].set_pos(flame_cyl_center)

        flame_tip_r = FLAME_TIP_RADIUS_FRAC * flame_cyl_r
        flame_tip_center = wick_top + np.array([0.0, 0.0, flame_cyl_h + flame_tip_r * 0.6])
        candle["flame_tip"].set_pos(flame_tip_center)

    candle["pos"] = tuple(float(v) for v in new_pos)
    candle["wick_top"] = tuple(float(v) for v in wick_top)
    return candle


def flame_hazard_zone(
    pos,
    height=DEFAULT_HEIGHT,
    radius=DEFAULT_RADIUS,
    hazard_radius_mult=HAZARD_RADIUS_MULT,
    hazard_height_above=HAZARD_HEIGHT_ABOVE,
    hazard_height_below=HAZARD_HEIGHT_BELOW,
):
    """
    Pure-geometry helper (no Genesis dependency -- safe to import from a
    metrics_*.py module, matching this project's Genesis-free metrics
    convention). Returns a simple hazard-zone spec: a vertical, axis-aligned
    "rounded cylinder" region above and around the candle's flame, for a
    metric module to compute end-effector clearance against.

    `pos`/`height`/`radius` should be the SAME values passed to `add_candle`
    for this candle (i.e. the candle's base position, body height, and body
    radius) -- this function does not need `lit`, since a metric checking
    clearance from an UNLIT candle's former-flame region is still a
    reasonable (if more conservative) thing to want, and keeps the caller's
    life simpler (one set of constants per candle, not two).

    Returns a dict:
      {
        "type": "cylinder",
        "center": (x, y, z_mid) -- the hazard column's own vertical midpoint,
        "radius": hazard cylinder radius (radius * hazard_radius_mult),
        "half_height": hazard cylinder half-height,
        "wick_top": (x, y, z) -- same point as add_candle's "wick_top", for
            convenience/debugging,
      }
    Consumed by metrics_1a_iii.distance_to_hazard_zone (a pure function
    operating on exactly this dict shape).
    """
    pos = np.asarray(pos, dtype=float)
    if pos.shape != (3,):
        raise ValueError(f"pos must be a length-3 (x, y, z_base) tuple/array, got shape {pos.shape}")
    if height <= 0 or radius <= 0:
        raise ValueError(f"height and radius must be > 0: height={height} radius={radius}")
    if hazard_radius_mult <= 0 or hazard_height_above < 0 or hazard_height_below < 0:
        raise ValueError(
            f"hazard_radius_mult must be > 0 and heights must be >= 0: "
            f"hazard_radius_mult={hazard_radius_mult} hazard_height_above={hazard_height_above} "
            f"hazard_height_below={hazard_height_below}"
        )

    wick_top = pos + np.array([0.0, 0.0, height])
    z_min = wick_top[2] - hazard_height_below
    z_max = wick_top[2] + hazard_height_above
    z_mid = (z_min + z_max) / 2.0
    half_height = (z_max - z_min) / 2.0

    return {
        "type": "cylinder",
        "center": (float(wick_top[0]), float(wick_top[1]), float(z_mid)),
        "radius": float(radius * hazard_radius_mult),
        "half_height": float(half_height),
        "wick_top": (float(wick_top[0]), float(wick_top[1]), float(wick_top[2])),
    }


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Pure-math self-tests for flame_hazard_zone (no Genesis needed -- run
    # standalone with `python3 candle_asset.py`). add_candle() itself needs a
    # live Genesis scene and is instead exercised by scenario_1a_iii.py's
    # build_scene() smoke test.
    # ------------------------------------------------------------------

    # 1. Basic shape/keys sanity.
    z = flame_hazard_zone((0.3, 0.1, 0.75), height=0.10, radius=0.025)
    assert z["type"] == "cylinder"
    assert set(z.keys()) == {"type", "center", "radius", "half_height", "wick_top"}

    # 2. wick_top matches add_candle's own convention: pos + (0,0,height).
    assert abs(z["wick_top"][0] - 0.3) < 1e-9
    assert abs(z["wick_top"][1] - 0.1) < 1e-9
    assert abs(z["wick_top"][2] - 0.85) < 1e-9

    # 3. hazard radius = radius * default mult.
    expected_r = 0.025 * HAZARD_RADIUS_MULT
    assert abs(z["radius"] - expected_r) < 1e-9, (z["radius"], expected_r)

    # 4. hazard column spans [wick_top_z - below, wick_top_z + above].
    z_min = z["center"][2] - z["half_height"]
    z_max = z["center"][2] + z["half_height"]
    assert abs(z_min - (0.85 - HAZARD_HEIGHT_BELOW)) < 1e-9, (z_min, 0.85 - HAZARD_HEIGHT_BELOW)
    assert abs(z_max - (0.85 + HAZARD_HEIGHT_ABOVE)) < 1e-9, (z_max, 0.85 + HAZARD_HEIGHT_ABOVE)

    # 5. center xy matches wick xy (hazard column is directly above the candle).
    assert abs(z["center"][0] - 0.3) < 1e-9
    assert abs(z["center"][1] - 0.1) < 1e-9

    # 6. Custom multipliers/heights are respected.
    z2 = flame_hazard_zone((0.0, 0.0, 0.0), height=0.20, radius=0.03, hazard_radius_mult=1.0,
                            hazard_height_above=0.10, hazard_height_below=0.0)
    assert abs(z2["radius"] - 0.03) < 1e-9
    z2_min = z2["center"][2] - z2["half_height"]
    z2_max = z2["center"][2] + z2["half_height"]
    assert abs(z2_min - 0.20) < 1e-9, z2_min  # wick_top z = 0.20, no "below" extension
    assert abs(z2_max - 0.30) < 1e-9, z2_max

    # 7. Larger candle (bigger height/radius) -> proportionally larger hazard
    #    radius, same default multiplier.
    z_small = flame_hazard_zone((0, 0, 0), height=0.10, radius=0.02)
    z_big = flame_hazard_zone((0, 0, 0), height=0.10, radius=0.04)
    assert z_big["radius"] > z_small["radius"]
    assert abs(z_big["radius"] - 2 * z_small["radius"]) < 1e-9

    # 8. Degenerate inputs raise.
    try:
        flame_hazard_zone((0, 0, 0), height=0.0, radius=0.02)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        flame_hazard_zone((0, 0, 0), height=0.1, radius=-0.02)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        flame_hazard_zone((0, 0, 0), height=0.1, radius=0.02, hazard_radius_mult=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        flame_hazard_zone((0, 0), height=0.1, radius=0.02)
        assert False, "expected ValueError"
    except ValueError:
        pass

    # 9. Two candles at different xy positions produce non-overlapping
    #    hazard-zone centers (sanity check useful for the 1a-iii corridor
    #    scenario: two independent hazard columns, not accidentally the same
    #    geometry object/reference).
    zone_a = flame_hazard_zone((0.2, -0.1, 0.75), height=0.10, radius=0.025)
    zone_b = flame_hazard_zone((0.2, 0.15, 0.75), height=0.10, radius=0.025)
    assert zone_a["center"] != zone_b["center"]
    assert zone_a is not zone_b

    # ------------------------------------------------------------------
    # reposition_candle self-tests, using a lightweight fake "entity" stub
    # (records the last pos it was set to) so this stays a pure-Python test
    # with no live Genesis scene needed.
    # ------------------------------------------------------------------

    class _FakeEntity:
        def __init__(self):
            self.last_pos = None

        def set_pos(self, pos):
            self.last_pos = np.array(pos, dtype=float)

    # 10. Repositioning an UNLIT candle only moves the body (flame entities
    #     are None and must not be touched).
    fake_body = _FakeEntity()
    candle_unlit = {
        "body": fake_body, "flame_cyl": None, "flame_tip": None,
        "pos": (0.0, 0.0, 0.75), "height": 0.10, "radius": 0.025, "lit": False,
        "wick_top": (0.0, 0.0, 0.85),
    }
    reposition_candle(candle_unlit, (0.3, 0.1, 0.75))
    assert np.allclose(fake_body.last_pos, [0.3, 0.1, 0.80]), fake_body.last_pos  # body center = base + (0,0,height/2)
    assert candle_unlit["pos"] == (0.3, 0.1, 0.75)
    assert candle_unlit["wick_top"] == (0.3, 0.1, 0.85)

    # 11. Repositioning a LIT candle moves body + both flame parts, using the
    #     SAME relative offsets add_candle itself would compute.
    fake_body2, fake_cyl2, fake_tip2 = _FakeEntity(), _FakeEntity(), _FakeEntity()
    candle_lit = {
        "body": fake_body2, "flame_cyl": fake_cyl2, "flame_tip": fake_tip2,
        "pos": (0.0, 0.0, 0.75), "height": 0.10, "radius": 0.025, "lit": True,
        "wick_top": (0.0, 0.0, 0.85),
    }
    reposition_candle(candle_lit, (0.4, -0.05, 0.75))
    expected_body = np.array([0.4, -0.05, 0.80])
    expected_wick_top = np.array([0.4, -0.05, 0.85])
    flame_cyl_h = FLAME_CYL_HEIGHT_FRAC * 0.10
    flame_cyl_r = FLAME_CYL_RADIUS_FRAC * 0.025
    expected_flame_cyl = expected_wick_top + np.array([0.0, 0.0, flame_cyl_h / 2.0])
    flame_tip_r = FLAME_TIP_RADIUS_FRAC * flame_cyl_r
    expected_flame_tip = expected_wick_top + np.array([0.0, 0.0, flame_cyl_h + flame_tip_r * 0.6])
    assert np.allclose(fake_body2.last_pos, expected_body), fake_body2.last_pos
    assert np.allclose(fake_cyl2.last_pos, expected_flame_cyl), (fake_cyl2.last_pos, expected_flame_cyl)
    assert np.allclose(fake_tip2.last_pos, expected_flame_tip), (fake_tip2.last_pos, expected_flame_tip)
    assert candle_lit["wick_top"] == tuple(float(v) for v in expected_wick_top)

    # 12. Reposition result is consistent with flame_hazard_zone's own
    #     wick_top convention (cross-module sanity: both derive wick_top the
    #     same way, so a caller using reposition_candle's returned "pos" as
    #     flame_hazard_zone's `pos` argument gets a hazard zone centered on
    #     the ACTUAL post-reposition flame location).
    zone = flame_hazard_zone(candle_lit["pos"], height=candle_lit["height"], radius=candle_lit["radius"])
    assert zone["wick_top"] == candle_lit["wick_top"], (zone["wick_top"], candle_lit["wick_top"])

    # 13. Invalid new_pos shape raises.
    try:
        reposition_candle(candle_lit, (0.0, 0.0))
        assert False, "expected ValueError"
    except ValueError:
        pass

    print("ALL CANDLE_ASSET SELF-TESTS PASSED")
