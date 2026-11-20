"""Paired seen / novel test scenes from one scenes.py file (test layouts 40-49).

    python novel_scenes.py --suite safelibero_spatial --scenes data_v2/spatial/scenes_pair.npz \
        --out_seen data_v2/spatial/scenes_pair_seen.npz --out_novel data_v2/spatial/scenes_pair_novel.npz

For every scene the same init state is loaded twice: in the official env (seen obstacle) and in the novel env
(novel_objects.py swaps the obstacle for an unseen look-alike at the same pose). A scene is kept only if
  - its obstacle was actually swapped (some scenes have no free look-alike, e.g. every bottle is a task object),
  - after settling, the novel obstacle stands within 2 cm (xy, body origin) and 3 cm (z, bottom of its collision boxes) of where the seen one stands,
  - and it is upright (its z axis within 20 deg of the seen obstacle's).
Both outputs hold the same states in the same order; meta "obstacles" uses the scene's own names
(seen names in one, novel names in the other), so eval_sl --scenes names the right object.
"""
import argparse
import collections
import json

import numpy as np


def zaxis(q):  # robosuite obs quats are xyzw
    x, y, z, w = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def collision_box(sim, name):
    """World AABB of the collision box geoms of one object. object_aabb also counts the visual mesh, whose box is
    unreliable for some assets (orange juice: 42 cm tall), so the stability check uses collision boxes only."""
    import sl_env
    m, d = sim.model, sim.data
    ids = set(sl_env.body_tree_ids(sim, name))
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in ids and m.geom_type[g] == 6 and (m.geom_contype[g] or m.geom_conaffinity[g]):
            half = np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_size[g]
            lo, hi = np.minimum(lo, d.geom_xpos[g] - half), np.maximum(hi, d.geom_xpos[g] + half)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out_seen", required=True)
    ap.add_argument("--out_novel", required=True)
    ap.add_argument("--settle", type=int, default=20)
    a = ap.parse_args()
    import novel_objects
    import sl_env
    from eval_sl import load_scenes
    sl_env.set_suite(a.suite)
    sl_env.ENV_SEED, sl_env.SETTLE = 0, a.settle
    orig_make = sl_env.make_env
    novel_objects.enable()
    novel_make = sl_env.make_env
    states, meta = load_scenes(a.scenes)
    keep, why = [], collections.Counter()
    nmeta = []
    by_cell = collections.defaultdict(list)
    for i, m in enumerate(meta):
        by_cell[(m["level"], m["task"])].append(i)
    for (lv, t), idx in sorted(by_cell.items()):
        env_s, _, _ = orig_make(lv, t, res=128)
        env_n, _, _ = novel_make(lv, t, res=128)
        for i in idx:
            m = meta[i]
            env_s.reset(); env_s.set_init_state(states[i]); os_ = sl_env.settle(env_s)
            env_n.reset(); env_n.set_init_state(states[i]); on = sl_env.settle(env_n)
            ps, pn = sl_env.active_obstacles(os_), sl_env.active_obstacles(on)
            seen = [n for n in m["obstacles"] if n in ps]
            if len(seen) != 1 or len(pn) != 1:
                why["obstacle count"] += 1
                continue
            s, n = seen[0], pn[0]
            if sl_env.canon(n) == sl_env.canon(s):
                why[f"no free look-alike for {sl_env.canon(s)}"] += 1
                continue
            d = on[n + "_pos"] - os_[s + "_pos"]
            # body origins differ between meshes (orange juice sits 7 cm lower than milk), so compare resting heights
            # by the bottom of each object's collision boxes
            bn, bs = collision_box(env_n.sim, n), collision_box(env_s.sim, s)
            if np.isfinite(bn[0]).all() and np.isfinite(bs[0]).all():    # else (no collision boxes) keep origins
                d[2] = bn[0][2] - bs[0][2]
                dc = (bn[0][:2] + bn[1][:2]) / 2 - (bs[0][:2] + bs[1][:2]) / 2   # xy: origin or box centre may match
                if np.abs(dc).max() < np.abs(d[:2]).max():
                    d[:2] = dc
            tilt = np.degrees(np.arccos(np.clip(zaxis(on[n + "_quat"]) @ zaxis(os_[s + "_quat"]), -1, 1)))
            if np.abs(d[:2]).max() > 0.02 or abs(d[2]) > 0.03 or tilt > 20:
                why[f"unstable {sl_env.canon(n)}"] += 1
                continue
            keep.append(i)
            nmeta.append(dict(m, obstacles=[n], seen_obstacle=s, novel=True))
            why[f"ok {sl_env.canon(s)} -> {sl_env.canon(n)}"] += 1
        env_s.close(); env_n.close()
        print(lv, t, dict(why), flush=True)

    def save(path, mm):
        arr = np.empty(len(keep), dtype=object)
        for j, i in enumerate(keep):
            arr[j] = states[i]
        np.savez_compressed(path, states=arr, meta=json.dumps(mm))

    save(a.out_seen, [dict(meta[i], obstacles=[nm["seen_obstacle"]]) for i, nm in zip(keep, nmeta)])
    save(a.out_novel, nmeta)
    print("kept", len(keep), "of", len(meta), dict(why))


if __name__ == "__main__":
    main()
