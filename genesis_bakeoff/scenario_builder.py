"""
Reusable Genesis scenario-build pipeline: the Genesis-side equivalent of
SafeLIBERO's generate_pruned_init.py.

Extracted from genesis_bakeoff/spike_placement_sampler.py once that spike
confirmed the pattern generalizes. A scenario script should only need to:
  1. write a build_scene() function returning (scene, entities_dict, cam)
  2. define REGION_SPECS for the objects that need randomized placement
  3. call generate_init_states(...) and save_init_states(...)

Nothing in this module is scenario-specific -- object names, ranges, and
z-heights are all caller-supplied data (REGION_SPECS), not code here.
"""
import numpy as np

from placement_sampler import sample_placement


def _to_numpy(x):
    return np.array(x.cpu() if hasattr(x, "cpu") else x)


def generate_init_states(
    scene,
    entities,
    region_specs,
    n_episodes=50,
    settle_steps=80,
    max_geometric_tries=200,
    min_final_dist=None,
    z_tolerance=0.08,
    seed=0,
):
    """
    scene: a built (scene.build() already called) Genesis scene.
    entities: dict of {name: Genesis entity} for every name that appears in
        region_specs -- must support .set_pos() and .get_pos().
    region_specs: same format as placement_sampler.sample_placement's
        region_specs argument.
    min_final_dist: optional dict {(name_a, name_b): min_dist} of pairwise
        post-settle distance floors used as a physics-based sanity check on
        top of the geometric sampler (catches interpenetration the 2D
        footprint check might miss). If None, derived as the sum of the two
        objects' footprint_radius values minus a small slack.

    Returns: (valid_states, stats)
      valid_states: list of dicts {name: (x, y, z)} -- the SETTLED positions
        (not the raw sampled ones), one entry per successfully-validated
        episode. This is the array to persist (see save_init_states).
      stats: dict of counters (geometric_fail, physics_bad, first_try_ok,
        retry_ok) for reporting -- mirrors what the placement-sampler spike
        logged, so scenario scripts get this diagnostic for free.
    """
    names = list(region_specs.keys())
    if min_final_dist is None:
        min_final_dist = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                min_final_dist[(a, b)] = (
                    region_specs[a]["footprint_radius"] + region_specs[b]["footprint_radius"]
                ) * 0.7  # slack: settled objects can legitimately end up closer than the initial rejection margin

    valid_states = []
    stats = {"geometric_fail": 0, "physics_bad": 0, "first_try_ok": 0, "retry_ok": 0}

    attempt = 0
    while len(valid_states) < n_episodes and attempt < n_episodes * 4:
        attempt += 1
        placement = sample_placement(region_specs, max_tries=max_geometric_tries, seed=seed + attempt)
        if placement is None:
            stats["geometric_fail"] += 1
            continue

        for name, (x, y, z) in placement.items():
            entities[name].set_pos(np.array([x, y, z]))
            if hasattr(entities[name], "set_dofs_velocity"):
                try:
                    entities[name].set_dofs_velocity(np.zeros(6))
                except Exception:
                    pass

        for _ in range(settle_steps):
            scene.step()

        finals = {name: _to_numpy(entities[name].get_pos()) for name in names}

        bad = False
        for name in names:
            z_expect = region_specs[name]["z"]
            if not np.all(np.isfinite(finals[name])):
                bad = True
            elif abs(finals[name][2] - z_expect) > z_tolerance:
                bad = True  # fell through the table or got launched
        for (a, b), floor in min_final_dist.items():
            d = np.hypot(finals[a][0] - finals[b][0], finals[a][1] - finals[b][1])
            if d < floor:
                bad = True

        if bad:
            stats["physics_bad"] += 1
            continue

        valid_states.append({name: tuple(float(v) for v in finals[name]) for name in names})

    return valid_states, stats


def save_init_states(valid_states, output_path):
    """
    Saves valid_states (list of {name: (x, y, z)}) to a .npz file: one array
    per object name, shape (n_episodes, 3). This is the Genesis-side
    equivalent of a .pruned_init file -- load with load_init_states().
    """
    if not valid_states:
        raise ValueError("No valid states to save -- generate_init_states produced an empty list")
    names = list(valid_states[0].keys())
    arrays = {name: np.array([s[name] for s in valid_states]) for name in names}
    np.savez(output_path, **arrays)


def load_init_states(path):
    """Returns list of {name: (x, y, z)} dicts, inverse of save_init_states."""
    data = np.load(path)
    names = list(data.keys())
    n = data[names[0]].shape[0]
    return [{name: tuple(data[name][i]) for name in names} for i in range(n)]
