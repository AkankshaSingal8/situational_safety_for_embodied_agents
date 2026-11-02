"""Generate obstacle scenes for residual training and for the two-obstacle text test.

    python scenes.py --out data/scenes_train.npz --layouts 0-24 --per_layout 8 --seed 0
    python scenes.py --out data/scenes_test2.npz --layouts 25-49 --per_layout 2 --n_obs 2 --seed 1

Each scene starts from an official SafeLIBERO-Spatial layout (bowls, plate, ramekin, ...),
clears all six obstacles off the table, then places 1 or 2 obstacle types at random either
near the target bowl (level-I-like) or along the bowl->plate path (level-II-like). Scenes
are kept only if every placed obstacle comes to rest upright, does not touch another object
and sits within reach. Official episodes 25-49 are never used for training layouts, so the
official benchmark on those episodes stays held out.
"""
import argparse
import json
import os

import numpy as np

import sl_env

from collections import Counter

REJECT = Counter()
CLEAR = lambda i: [30.0 + 3 * i, 30.0, 0.0]  # noqa: E731
TASK_OBJS = ["akita_black_bowl_1", "akita_black_bowl_2", "plate_1", "glazed_rim_porcelain_ramekin_1", "cookies_1"]


def table_top(env):
    return sl_env.object_aabb(env.sim, "plate_1")[0][2]


def footprint(env, name):
    lo, hi = sl_env.object_aabb(env.sim, name)
    return 0.5 * max(hi[0] - lo[0], hi[1] - lo[1]), lo, hi


def place(env, name, xy, ztop):
    sl_env.move_object(env, name, [xy[0], xy[1], 1.4])
    lo, hi = sl_env.object_aabb(env.sim, name)
    z_body = env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(name + "_joint0")[0] + 2]
    sl_env.move_object(env, name, [xy[0], xy[1], z_body + (ztop + 0.003 - lo[2])])


def sample_xy(rng, bowl, plate, mode, start=np.array([-0.21, 0.0])):
    if mode == "start":  # between the gripper's start and the bowl, where level-II obstacles stand
        u = rng.uniform(0.3, 0.7)
        d = bowl[:2] - start
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        return start + u * d + n * rng.normal(0, 0.04)
    if mode == "near":
        r, th = rng.uniform(0.10, 0.17), rng.uniform(0, 2 * np.pi)
        return bowl[:2] + r * np.array([np.cos(th), np.sin(th)])
    u = rng.uniform(0.3, 0.7)
    d = plate[:2] - bowl[:2]
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    return bowl[:2] + u * d + n * rng.normal(0, 0.04)


def in_env(env, name):
    try:
        env.sim.model.get_joint_qpos_addr(name + "_joint0")
        return True
    except (ValueError, KeyError):
        return False


def make_scene(env, init_state, rng, n_obs, types=None, p_elevated=0.0, pool=None, layout=None):
    """p_elevated: chance that the first obstacle stands on the box_base platform (as in level II).
    layout: an extended env (bddl_x) starts from sl_env.capture_layout(...) instead of an official state.
    pool: obstacle names to draw from (default: every obstacle the env has)."""
    if layout is not None:
        obs = sl_env.apply_layout(env, layout)
    else:
        env.reset()
        obs = env.set_init_state(init_state)
    present = [n for n in sl_env.OBSTACLES if in_env(env, n)]
    for i, n in enumerate(present):
        sl_env.move_object(env, n, CLEAR(i))
    sl_env.move_object(env, "box_base_1", [0.0, 20.0, 0.0])
    obs = sl_env.settle(env, 5)
    zt = table_top(env)
    bowl, plate = obs["akita_black_bowl_1_pos"], obs["plate_1_pos"]
    others = {n: obs[n + "_pos"][:2] for n in TASK_OBJS if n + "_pos" in obs}
    pool = [n for n in (pool or present) if n in present]
    chosen = types or list(rng.choice(pool, n_obs, replace=False))
    placed = {}
    elevated = "box_base_1_pos" in obs and rng.random() < p_elevated
    if elevated:
        ok = False
        for _ in range(40):
            xy = sample_xy(rng, bowl, plate, rng.choice(["near", "path", "start"]))
            if not (-0.30 < xy[0] < 0.22 and -0.30 < xy[1] < 0.35):
                continue
            place(env, "box_base_1", xy, zt)
            r_b = footprint(env, "box_base_1")[0]
            if all(np.linalg.norm(xy - p) > r_b + 0.07 for p in others.values()):
                ok = True
                break
        if not ok:
            REJECT["no_free_spot_platform"] += 1
            return None
        placed["box_base_1"] = xy
        box_top = sl_env.object_aabb(env.sim, "box_base_1")[1][2]
    for k, name in enumerate(chosen):
        ok = False
        for _ in range(40):
            on_box = elevated and k == 0
            xy = placed["box_base_1"] if on_box else sample_xy(rng, bowl, plate, rng.choice(["near", "path", "start"]))
            if not (-0.30 < xy[0] < 0.22 and -0.30 < xy[1] < 0.35):
                continue
            place(env, name, xy, box_top if on_box else zt)
            if on_box:
                ok = True
                break
            r_obs = footprint(env, name)[0]
            clear = all(np.linalg.norm(xy - p) > r_obs + 0.07 for p in others.values())
            clear &= all(np.linalg.norm(xy - p) > r_obs + 0.08 for p in placed.values())
            if "box_base_1" in placed:
                clear &= np.linalg.norm(xy - placed["box_base_1"]) > r_obs + footprint(env, "box_base_1")[0] + 0.02
            if clear:
                ok = True
                break
        if not ok:
            REJECT["no_free_spot"] += 1
            return None
        placed[name] = xy
    before = {n: env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(n + "_joint0")[0]:][:7].copy() for n in placed}
    obs = sl_env.settle(env, 25)
    for n in placed:
        q = env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(n + "_joint0")[0]:][:7]
        if np.linalg.norm(q[:3] - before[n][:3]) > 0.02 or abs(abs(np.dot(q[3:], before[n][3:])) - 1) > 0.01:
            REJECT["moved_or_toppled:" + n] += 1
            return None  # fell over, slid or got pushed
        if not sl_env.on_table(obs[n + "_pos"]):
            REJECT["off_table"] += 1
            return None
    # task objects must not have been disturbed by the placement
    for n, p in others.items():
        if np.linalg.norm(obs[n + "_pos"][:2] - p) > 0.01:
            REJECT["disturbed:" + n] += 1
            return None
    return env.sim.get_state().flatten().copy(), chosen, elevated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="I,II")
    ap.add_argument("--tasks", default="0,1,2,3")
    ap.add_argument("--layouts", default="0-24")
    ap.add_argument("--per_layout", type=int, default=8)
    ap.add_argument("--n_obs", default="1,2", help="choices for the number of obstacles per scene")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p_elev_I", type=float, default=0.2)
    ap.add_argument("--p_elev_II", type=float, default=0.6)
    ap.add_argument("--extended", action="store_true", help="bddl_x scenes with 19 obstacle types (extend_bddl.py)")
    ap.add_argument("--types", default="", help="comma-separated obstacle names to draw from (default: all in the env)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    lo, hi = map(int, a.layouts.split("-"))
    nchoice = [int(x) for x in a.n_obs.split(",")]
    states, meta, tries = [], [], 0
    for lv in a.levels.split(","):
        for t in [int(x) for x in a.tasks.split(",")]:
            env, lang, inits = sl_env.make_env(lv, t)
            layouts = {}
            if a.extended:  # capture the official layouts, then work in the extended env
                for e in range(lo, hi + 1):
                    env.reset(); env.set_init_state(inits[e])
                    layouts[e] = sl_env.capture_layout(env)
                env.close()
                env, _, _ = sl_env.make_env(lv, t, extended=True)
            pool = [x for x in a.types.split(",") if x] or None
            for e in range(lo, hi + 1):
                made = 0
                tries_here = 0
                while made < a.per_layout and tries_here < 60:
                    tries_here += 1
                    tries += 1
                    r = make_scene(env, inits[e], rng, int(rng.choice(nchoice)), p_elevated=a.p_elev_II if lv == "II" else a.p_elev_I,
                                   pool=pool, layout=layouts.get(e))
                    if r is None:
                        continue
                    states.append(r[0]); meta.append(dict(level=lv, task=t, layout=e, obstacles=r[1], elevated=bool(r[2]), lang=lang))
                    if a.extended:
                        meta[-1].update(extended=True, bodies=layouts[e]["body"])
                    made += 1
            env.close()
            print(lv, t, len(states), "scenes, tries", tries, dict(REJECT), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    arr = np.empty(len(states), dtype=object)
    for i, st in enumerate(states):  # state length differs between tasks (different fixtures)
        arr[i] = st
    np.savez_compressed(a.out, states=arr, meta=json.dumps(meta))
    print("wrote", a.out, len(states))


if __name__ == "__main__":
    main()
