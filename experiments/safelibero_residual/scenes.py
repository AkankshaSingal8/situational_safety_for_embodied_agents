"""Generate obstacle scenes for residual training and for the two-obstacle text test (all SafeLIBERO suites).

    python scenes.py --suite safelibero_object --out data/object/scenes_train.npz --layouts 0-19 --per_layout 8 --seed 0
    python scenes.py --suite safelibero_object --out data/object/scenes_test2.npz --layouts 25-49 --per_layout 1 --n_obs 2 --seed 1

Each scene starts from an official SafeLIBERO layout, clears every obstacle (and the level-II platform) off
the table, then places 1 or 2 obstacle types at random: near the manipulated object or a target
(level-I-like), along the object->target path, or between the gripper's start and the object (level-II-like).
Anchors come from the task's BDDL (:obj_of_interest), so nothing is specific to one suite. Scenes are kept
only if every placed obstacle comes to rest upright, does not touch another object and sits in the workspace.
Layouts of official episodes 20-49 are never used for training scenes (20-24 validation, 25-49 test).
"""
import argparse
import json
import os

import numpy as np

import sl_env

from collections import Counter

REJECT = Counter()
CLEAR = lambda i: [30.0 + 3 * i, 30.0, 0.0]  # noqa: E731
def task_objects(env, obs):
    """Every free-joint object that is neither an obstacle nor the level-II platform."""
    names = [n[:-len("_joint0")] for n in env.sim.model.joint_names if n.endswith("_joint0")]
    return [n for n in names if "obstacle" not in n and n != "box_base_1" and n + "_pos" in obs]


def anchor_pos(env, obs, name):
    """World position of an obj_of_interest entry: an object, or a fixture/region ('wooden_cabinet_1_top_region')."""
    if name + "_pos" in obs:
        return obs[name + "_pos"]
    m = env.sim.model
    for cand in (name, name.rsplit("_", 2)[0], name.rsplit("_", 1)[0]):
        ids = sl_env.body_tree_ids(env.sim, cand)
        if ids:
            return env.sim.data.body_xpos[ids[0]].copy()
    return None


def table_top(env, obs, objs):
    """Support height: the lowest bottom of the task objects resting in the workspace."""
    return min(sl_env.object_aabb(env.sim, n)[0][2] for n in objs)


def workspace(obs, objs, eef):
    """xy bounds for obstacle placement: task objects and the gripper start, padded by 12 cm."""
    pts = np.array([obs[n + "_pos"][:2] for n in objs] + [eef[:2]])
    return pts.min(0) - 0.12, pts.max(0) + 0.12


def footprint(env, name):
    lo, hi = sl_env.object_aabb(env.sim, name)
    return 0.5 * max(hi[0] - lo[0], hi[1] - lo[1]), lo, hi


def place(env, name, xy, ztop, quat=None):
    sl_env.move_object(env, name, [xy[0], xy[1], 1.4], quat)
    lo, hi = sl_env.object_aabb(env.sim, name)
    z_body = env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(name + "_joint0")[0] + 2]
    sl_env.move_object(env, name, [xy[0], xy[1], z_body + (ztop + 0.003 - lo[2])])


def sample_xy(rng, bowl, plate, mode, start):
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


def _mode(rng):
    """Placement mode. SCENE_MODES="path=0.7,near=0.2,start=0.1" reweights it (training collection only, e.g. to put
    Object-suite obstacles on the arm's path more often); unset = the original uniform choice."""
    w = os.environ.get("SCENE_MODES")
    if not w:
        return rng.choice(["near", "path", "start"])
    kv = dict(x.split("=") for x in w.replace(";", ",").split(","))   # ";" also works (sbatch --export splits on ",")
    ks = [k for k in ("near", "path", "start") if k in kv]
    p = np.array([float(kv[k]) for k in ks])
    return ks[int(rng.choice(len(ks), p=p / p.sum()))]


def make_scene(env, init_state, rng, n_obs, types=None, p_elevated=0.0, quats=None):
    """p_elevated: chance that the first obstacle stands on the box_base platform (as in level II).
    quats: {obstacle name: wxyz} orientation to place it with (GSO models are z-up; default keeps the init's)."""
    quats = quats or {}
    env.reset()
    obs = env.set_init_state(init_state)
    start = obs["robot0_eef_pos"][:2].copy()
    scene_obs = sl_env.scene_obstacles(obs)
    for i, n in enumerate(scene_obs):
        sl_env.move_object(env, n, CLEAR(i))
    if "box_base_1_pos" in obs:
        sl_env.move_object(env, "box_base_1", [0.0, 20.0, 0.0])
    obs = sl_env.settle(env, 5)
    objs = [n for n in task_objects(env, obs) if sl_env.on_table(obs[n + "_pos"])]
    # drop objects parked under the table (Goal parks cream_cheese at z ~ 0.2 inside the xy workspace; it made
    # table_top 0.2 below the real table and every Goal I0/I1/II0/II1/II2 scene was rejected)
    bot = {n: sl_env.object_aabb(env.sim, n)[0][2] for n in objs}
    med = float(np.median(list(bot.values()))) if bot else 0.0
    objs = [n for n in objs if bot[n] > med - 0.3]
    zt = table_top(env, obs, objs)
    lo_xy, hi_xy = workspace(obs, objs, obs["robot0_eef_pos"])
    ooi = [(n, anchor_pos(env, obs, n)) for n in getattr(env, "ooi", [])]
    ooi = [(n, p) for n, p in ooi if p is not None]
    movable = [(n, p) for n, p in ooi if n in objs] or ooi[:1]
    pick_name, bowl = movable[rng.integers(len(movable))]
    targets = [p for n, p in ooi if n != pick_name] or [bowl]
    plate = targets[rng.integers(len(targets))]
    others = {n: obs[n + "_pos"][:2] for n in objs}
    chosen = types or list(rng.choice(scene_obs, n_obs, replace=False))
    placed = {}
    elevated = "box_base_1_pos" in obs and rng.random() < p_elevated
    in_ws = lambda xy: bool(np.all(xy > lo_xy) and np.all(xy < hi_xy))  # noqa: E731
    if elevated:
        ok = False
        for _ in range(40):
            xy = sample_xy(rng, bowl, plate, _mode(rng), start)
            if not in_ws(xy):
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
            xy = placed["box_base_1"] if on_box else sample_xy(rng, bowl, plate, _mode(rng), start)
            if not in_ws(xy):
                continue
            place(env, name, xy, box_top if on_box else zt, quats.get(name))
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
    ap.add_argument("--suite", default="safelibero_spatial")
    ap.add_argument("--levels", default="I,II")
    ap.add_argument("--tasks", default="0,1,2,3")
    ap.add_argument("--layouts", default="0-24")
    ap.add_argument("--per_layout", type=int, default=8)
    ap.add_argument("--n_obs", default="1,2", help="choices for the number of obstacles per scene")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p_elev_I", type=float, default=0.2)
    ap.add_argument("--p_elev_II", type=float, default=0.6)
    ap.add_argument("--cycle_types", default="", help="comma list of canonical obstacle types; one obstacle per scene, "
                    "cycling through the list (balanced coverage for the paired seen/novel test)")
    a = ap.parse_args()
    cyc = [x for x in a.cycle_types.split(",") if x]
    sl_env.set_suite(a.suite)
    rng = np.random.default_rng(a.seed)
    lo, hi = map(int, a.layouts.split("-"))
    nchoice = [int(x) for x in a.n_obs.split(",")]
    states, meta, tries = [], [], 0
    for lv in a.levels.split(","):
        for t in [int(x) for x in a.tasks.split(",")]:
            env, lang, inits = sl_env.make_env(lv, t)
            scene_obs = sl_env.scene_obstacles(env.reset()) if cyc else []
            for e in range(lo, hi + 1):
                made = 0
                tries_here = 0
                while made < a.per_layout and tries_here < 60:
                    tries_here += 1
                    tries += 1
                    ty = None
                    if cyc:
                        want = cyc[(made + e) % len(cyc)]
                        ty = [n for n in scene_obs if sl_env.canon(n) == want][:1]
                        if not ty:
                            made += 1
                            continue
                    r = make_scene(env, inits[e], rng, 1 if ty else int(rng.choice(nchoice)), types=ty,
                                   p_elevated=a.p_elev_II if lv == "II" else a.p_elev_I)
                    if r is None:
                        continue
                    states.append(r[0]); meta.append(dict(suite=a.suite, level=lv, task=t, layout=e, obstacles=r[1], elevated=bool(r[2]), lang=lang))
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
