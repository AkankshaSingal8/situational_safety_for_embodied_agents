"""Scenes from the official SafeLIBERO init states (obstacles where the benchmark put them).

Used for cells where scenes.py could not place random obstacles (Goal I0, I1, II0, II1, II2).

    python scenes_official.py --suite safelibero_goal --cells I/0,I/1,II/0,II/1,II/2 --layouts 0-39 --copies 3 \
        --out data_v2/goal/scenes_official.npz

Each layout is repeated --copies times; collect_v2 draws the driving sentence per scene index, so copies differ in
whether and which obstacle is named, and pi0.5's sampling noise differs between rollouts.
Also prints the table height scenes.py would compute next to the real table top, to diagnose its rejections.
"""
import argparse
import json
import os

import numpy as np

import scenes as SC
import sl_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="safelibero_goal")
    ap.add_argument("--cells", required=True)
    ap.add_argument("--layouts", default="0-39")
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sl_env.set_suite(a.suite)
    lo, hi = map(int, a.layouts.split("-"))
    states, meta = [], []
    for cell in a.cells.split(","):
        lv, t = cell.split("/")
        t = int(t)
        env, lang, inits = sl_env.make_env(lv, t)
        # diagnostic: what scenes.py thinks the table height is, vs the table geom top
        env.reset()
        obs = env.set_init_state(inits[lo])
        obs = sl_env.settle(env, 5)
        objs = [n for n in SC.task_objects(env, obs) if sl_env.on_table(obs[n + "_pos"])]
        bottoms = {n: round(float(sl_env.object_aabb(env.sim, n)[0][2]), 3) for n in objs}
        obst = {n: round(float(sl_env.object_aabb(env.sim, n)[0][2]), 3) for n in sl_env.active_obstacles(obs)}
        print(cell, repr(lang), "task-object bottoms", bottoms, "obstacle bottoms", obst, flush=True)
        n_obs = []
        for e in range(lo, hi + 1):
            env.reset()
            obs = env.set_init_state(inits[e])
            obs = sl_env.settle(env)
            present = sl_env.active_obstacles(obs)
            n_obs.append(len(present))
            for c in range(a.copies):
                states.append(np.asarray(inits[e]))
                meta.append(dict(suite=a.suite, level=lv, task=t, layout=e, obstacles=present, elevated=False,
                                 lang=lang, official=True, copy=c))
        env.close()
        print(cell, "layouts", hi - lo + 1, "obstacles per layout", dict(zip(*np.unique(n_obs, return_counts=True))),
              flush=True)
    arr = np.empty(len(states), dtype=object)
    for i, st in enumerate(states):
        arr[i] = st
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, states=arr, meta=json.dumps(meta))
    print("wrote", a.out, len(states))


if __name__ == "__main__":
    main()
