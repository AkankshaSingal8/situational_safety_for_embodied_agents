"""Smoke test of GSO obstacles: env builds, upright placement, stability; saves agentview images.

    ./pleiades_run.sh python test_gso.py --out runs_v2/gso_check --n 6
"""
import argparse
import collections
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs_v2/gso_check")
    ap.add_argument("--n", type=int, default=6, help="objects per suite")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import imageio
    import gso_objects as G
    import scenes as SC
    import sl_env
    G.ensure_local()
    objs = G.objects("train")
    rng = np.random.default_rng(0)
    rep = []
    for si, suite in enumerate(sl_env.SUITES):
        sl_env.set_suite(suite)
        sl_env.ENV_SEED, sl_env.SETTLE = 0, 20
        rej = collections.Counter()
        for j in range(a.n):
            obj = objs[(si * 37 + j * 53) % len(objs)]
            lv, t, e = ["I", "II"][j % 2], j % 4, (j * 7) % 40
            t0 = time.time()
            try:
                env, lang, inits = G.make_env(lv, t, obj, res=256)
                tb = time.time() - t0
                sc = None
                for k in range(8):
                    sc = SC.make_scene(env, inits[e], rng, 1, types=[G.NAME], p_elevated=0.6 if lv == "II" else 0.2,
                                       quats={G.NAME: G.yaw_quat(rng)})
                    if sc is not None:
                        break
                ok = sc is not None
                row = dict(suite=suite, level=lv, task=t, layout=e, gso=obj, noun=G.load_index()[obj]["noun"],
                           placed=ok, tries=k + 1, build_s=round(tb, 1), reject=dict(SC.REJECT))
                if ok:
                    env.reset(); obs = env.set_init_state(sc[0]); obs = sl_env.settle(env)
                    act = sl_env.active_obstacles(obs)
                    p = obs[G.NAME + "_pos"]
                    lo, hi = sl_env.object_aabb(env.sim, G.NAME)
                    row.update(active=act, pos=np.round(p, 3).tolist(), aabb_h=round(float(hi[2] - lo[2]), 3),
                               elevated=bool(sc[2]))
                    img = obs["agentview_image"][::-1, ::-1]
                    imageio.imwrite(f"{a.out}/{suite.split('_')[1]}_{j}_{obj[:30]}.png", img)
                env.close()
            except Exception as ex:  # noqa: BLE001
                import traceback
                row = dict(suite=suite, gso=obj, error=repr(ex)[:300], tb=traceback.format_exc()[-1500:])
            print(json.dumps(row), flush=True)
            rep.append(row)
            SC.REJECT.clear()
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
