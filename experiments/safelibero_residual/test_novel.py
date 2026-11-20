"""Check the novel-obstacle envs: every cell builds, the swapped obstacle stands where the seen one stood, it stays
upright through the settle steps, and no seen obstacle remains in the workspace. Saves agentview images.

    ./pleiades_run.sh python test_novel.py --out runs_v2/novel_check
"""
import argparse
import json
import os

import numpy as np


def run_cell(suite, level, task, ep, res):
    import sl_env
    sl_env.set_suite(suite)
    sl_env.ENV_SEED, sl_env.SETTLE = 0, 20
    import novel_objects
    novel_objects.enable()
    out = {}
    for novel in (False, True):
        mk = sl_env.make_env if novel else ORIG_MAKE_ENV
        env, lang, inits = mk(level, task, res=res)
        env.reset(); obs = env.set_init_state(inits[ep])
        pres0 = sl_env.active_obstacles(obs)
        p0 = {n: obs[n + "_pos"].copy() for n in pres0}
        obs = sl_env.settle(env)
        pres = sl_env.active_obstacles(obs)
        out["novel" if novel else "seen"] = dict(
            lang=lang, present=pres,
            pos={n: np.round(obs[n + "_pos"], 3).tolist() for n in pres},
            settle_drift={n: round(float(np.abs(obs[n + "_pos"] - p0[n]).sum()), 4) for n in pres if n in p0},
            quat={n: np.round(obs[n + "_quat"], 3).tolist() for n in pres if n + "_quat" in obs},
            img=np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs_v2/novel_check")
    ap.add_argument("--ep", type=int, default=40)
    ap.add_argument("--res", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import imageio
    import sl_env
    global ORIG_MAKE_ENV
    ORIG_MAKE_ENV = sl_env.make_env
    from novel_objects import SEEN_OF
    rep = []
    for suite in sl_env.SUITES:
        for level in ("I", "II"):
            for task in range(4):
                try:
                    r = run_cell(suite, level, task, a.ep, a.res)
                except Exception as e:  # noqa: BLE001
                    rep.append(dict(suite=suite, level=level, task=task, error=repr(e)[:300]))
                    print(suite, level, task, "ERROR", repr(e)[:300], flush=True)
                    continue
                s, n = r["seen"], r["novel"]
                swaps = []
                for nn in n["present"]:
                    seen = SEEN_OF.get(sl_env.canon(nn))
                    if seen:
                        sn = sl_env.scene_name({k + "_pos": 0 for k in s["present"]}, seen)
                        d = (np.abs(np.array(n["pos"][nn][:2]) - np.array(s["pos"][sn][:2])).sum()
                             if sn in s["pos"] else None)
                        swaps.append(dict(novel=nn, seen=sn, xy_dist=None if d is None else round(float(d), 4),
                                          z_novel=n["pos"][nn][2], z_seen=s["pos"][sn][2] if sn in s["pos"] else None,
                                          settle_drift=n["settle_drift"].get(nn)))
                leftover = [x for x in n["present"] if sl_env.canon(x) in SEEN_OF.values()]
                row = dict(suite=suite, level=level, task=task, seen=s["present"], novel=n["present"], swaps=swaps,
                           leftover_seen=leftover, seen_drift=s["settle_drift"], novel_drift=n["settle_drift"],
                           seen_quat=s["quat"], novel_quat=n["quat"])
                rep.append(row)
                print(json.dumps(row), flush=True)
                tag = f"{suite.split('_')[1]}_{level}{task}"
                imageio.imwrite(f"{a.out}/{tag}.png", np.concatenate([s["img"], n["img"]], 1))
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
