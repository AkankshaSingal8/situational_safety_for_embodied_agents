"""
Generic, scenario-agnostic declarative placement sampler for Genesis scenes.

Mirrors the level of approximation robosuite's own placement initializer uses:
region-range sampling + minimum-distance (footprint) rejection, NOT full
contact-mesh collision detection. This is pure geometry, no Genesis/physics
dependency in this module -- it produces (x, y, z) candidates that a caller
then applies to real entities and settles with real physics as an additional
sanity check (see apply_and_settle.py).
"""
import numpy as np


def _circles_overlap(xy_a, r_a, xy_b, r_b, margin=0.0):
    d = np.hypot(xy_a[0] - xy_b[0], xy_a[1] - xy_b[1])
    return d < (r_a + r_b + margin)


def sample_placement(region_specs, existing_placements=None, max_tries=200, seed=None, margin=0.0):
    """
    region_specs: dict of {name: {"xy_range": ((x_min, y_min), (x_max, y_max)),
                                    "z": fixed_height, "footprint_radius": r}}
    existing_placements: optional list of (name, (x, y), footprint_radius) to
        reject against, for objects placed by a different call (e.g. fixed
        furniture) -- not required for the common case of sampling all
        objects in one call, since that's handled internally.
    Returns: dict of {name: (x, y, z)} on success, or None if max_tries
        exhausted without a valid joint placement.
    """
    rng = np.random.default_rng(seed)
    existing_placements = list(existing_placements or [])

    names = list(region_specs.keys())
    for _attempt in range(max_tries):
        placed = list(existing_placements)  # (name, xy, radius)
        result = {}
        ok = True
        for name in names:
            spec = region_specs[name]
            (x_min, y_min), (x_max, y_max) = spec["xy_range"]
            r = spec["footprint_radius"]
            found = False
            for _inner_try in range(max_tries):
                x = rng.uniform(x_min, x_max)
                y = rng.uniform(y_min, y_max)
                if all(not _circles_overlap((x, y), r, xy_b, r_b, margin) for (_n, xy_b, r_b) in placed):
                    found = True
                    break
            if not found:
                ok = False
                break
            placed.append((name, (x, y), r))
            result[name] = (x, y, spec["z"])
        if ok:
            return result
    return None
