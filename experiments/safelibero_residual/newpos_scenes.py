"""Table 3 (new obstacle positions): training layouts, task objects untouched, obstacle placed somewhere new.

    python newpos_scenes.py gen --suite safelibero_spatial --out data_v2/spatial/scenes_newpos_cand.npz
    python newpos_scenes.py pick --suite safelibero_spatial --cand data_v2/spatial/scenes_newpos_cand.npz \
        --logs data_v2/spatial/newpos_check --out data_v2/spatial/scenes_newpos.npz

gen: for every training layout (0-39), task and level, clears the benchmark obstacles and places one obstacle with
  scenes.make_scene (task objects and other LIBERO objects stay exactly at the layout's positions; scenes where they
  move are rejected). A placement is kept only if it is at least --min_dist from every placement of the same obstacle
  in the same layout/task/level in the training and validation scene files, and from the other candidates.
pick: keeps candidates where the shield (pi0.5 + lookahead shield, collect_v2 --p_drive 1.0) finished the task
  without a collision, so every test scene is completable while avoiding the obstacle; then takes --per_level scenes
  per level, balanced over tasks and layouts with a fixed seed. Model outcomes are never used.
New training placements (DAgger round 2) use gen with a different --seed, --extra_refs set to the Table 3 files and
a separate --out, so they stay --min_dist from every train, val and Table 3 placement (run_dag2.slurm).
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

import scenes
import sl_env

REFS = ["scenes_train.npz", "scenes_train_r1_I.npz", "scenes_train_r1_II.npz", "scenes_train_b.npz",
        "scenes_train_c.npz", "scenes_val_sl.npz"]


def load(path):
    d = np.load(path, allow_pickle=True)
    return list(d["states"]), json.loads(str(d["meta"]))


def obstacle_xy(env, state, names):
    obs = env.set_init_state(state)
    return {n: np.asarray(obs[n + "_pos"][:2]) for n in names if n + "_pos" in obs}


def gen(a):
    sl_env.set_suite(a.suite)
    s = a.suite.replace("safelibero_", "")
    refs = defaultdict(list)  # (level, task, layout) -> [(state, names)]
    for f in REFS + [x for x in a.extra_refs.split(",") if x]:
        p = f"data_v2/{s}/{f}"
        if os.path.exists(p):
            st, me = load(p)
            for x, m in zip(st, me):
                refs[(m["level"], m["task"], m["layout"])].append((x, m["obstacles"]))
            print("ref", p, len(st), flush=True)
    rng = np.random.default_rng(a.seed)
    lo, hi = map(int, a.layouts.split("-"))
    states, meta = [], []
    for lv in a.levels.split(","):
        for t in [int(x) for x in a.tasks.split(",")]:
            env, lang, inits = sl_env.make_env(lv, t)
            n_ref = n_rej = 0
            for e in range(lo, hi + 1):
                old = defaultdict(list)
                for x, names in refs[(lv, t, e)]:
                    for n, xy in obstacle_xy(env, x, names).items():
                        old[n].append(xy)
                n_ref += sum(len(v) for v in old.values())
                made = tries = 0
                while made < a.per_layout and tries < 80:
                    tries += 1
                    r = scenes.make_scene(env, inits[e], rng, 1,
                                          p_elevated=a.p_elev_II if lv == "II" else a.p_elev_I)
                    if r is None:
                        continue
                    n = r[1][0]
                    xy = obstacle_xy(env, r[0], [n]).get(n)
                    if xy is None or any(np.linalg.norm(xy - q) < a.min_dist for q in old[n]):
                        n_rej += 1
                        continue
                    old[n].append(xy)
                    states.append(r[0])
                    meta.append(dict(suite=a.suite, level=lv, task=t, layout=e, obstacles=r[1], elevated=bool(r[2]),
                                     lang=lang, xy=[float(v) for v in xy]))
                    made += 1
            env.close()
            print(lv, t, len(states), "candidates; ref placements", n_ref, "too close", n_rej,
                  dict(scenes.REJECT), flush=True)
    arr = np.empty(len(states), dtype=object)
    for i, st in enumerate(states):
        arr[i] = st
    np.savez_compressed(a.out, states=arr, meta=json.dumps(meta))
    print("wrote", a.out, len(states))


def pick(a):
    st, me = load(a.cand)
    fin = {}
    for f in glob.glob(os.path.join(a.logs, "log_*.jsonl")):
        for line in open(f):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("final"):
                fin[d["scene"]] = bool(d.get("keep"))
    ok = [i for i in range(len(st)) if fin.get(i)]
    print(f"candidates {len(st)}, checked {len(fin)}, shield safe success {len(ok)}")
    rng = np.random.default_rng(a.seed)
    out = []
    for lv in sorted({me[i]["level"] for i in ok}):
        by_task = defaultdict(list)
        for i in ok:
            if me[i]["level"] == lv:
                by_task[me[i]["task"]].append(i)
        # round-robin over tasks, and within a task over layouts, so no task or layout dominates
        queues = {}
        for t, ids in by_task.items():
            by_lay = defaultdict(list)
            for i in rng.permutation(ids):
                by_lay[me[i]["layout"]].append(int(i))
            order, lays = [], list(rng.permutation(sorted(by_lay)))
            while any(by_lay[l] for l in lays):
                order += [by_lay[l].pop() for l in lays if by_lay[l]]
            queues[t] = order
        got = []
        while len(got) < a.per_level and any(queues.values()):
            for t in sorted(queues):
                if queues[t] and len(got) < a.per_level:
                    got.append(queues[t].pop(0))
        print(lv, "picked", len(got), {t: sum(me[i]["task"] == t for i in got) for t in sorted(queues)})
        out += sorted(got)
    arr = np.empty(len(out), dtype=object)
    for k, i in enumerate(out):
        arr[k] = st[i]
    np.savez_compressed(a.out, states=arr, meta=json.dumps([dict(me[i], cand_id=i) for i in out]))
    print("wrote", a.out, len(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "pick"])
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="I,II")
    ap.add_argument("--tasks", default="0,1,2,3")
    ap.add_argument("--layouts", default="0-39")
    ap.add_argument("--per_layout", type=int, default=3)
    ap.add_argument("--min_dist", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--p_elev_I", type=float, default=0.2)
    ap.add_argument("--p_elev_II", type=float, default=0.6)
    ap.add_argument("--extra_refs", default="", help="more scene files (under data_v2/<suite>/) to keep --min_dist "
                    "from, e.g. scenes_newpos_cand.npz,scenes_newpos.npz when generating new TRAINING placements")
    ap.add_argument("--cand", default="")
    ap.add_argument("--logs", default="")
    ap.add_argument("--per_level", type=int, default=200)
    a = ap.parse_args()
    gen(a) if a.cmd == "gen" else pick(a)
