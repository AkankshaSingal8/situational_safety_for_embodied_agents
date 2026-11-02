"""Does the obstacle move >1 mm with the robot held still? (official settle = 10 steps)"""
import numpy as np
import sl_env
for lv in ["I", "II"]:
    for t in range(4):
        env, lang, inits = sl_env.make_env(lv, t)
        out = []
        for e in range(0, 50, 7):
            env.reset(); obs = env.set_init_state(inits[e]); obs = sl_env.settle(env)
            n = sl_env.active_obstacles(obs)[0]; p0 = obs[n + "_pos"].copy()
            first = None
            for k in range(60):
                obs, *_ = env.step(sl_env.DUMMY)
                d = np.abs(obs[n + "_pos"] - p0).sum()
                if first is None and d > 0.001:
                    first = k
            out.append((e, first, round(float(np.abs(obs[n + "_pos"] - p0).sum() * 1000), 2)))
        print(lv, t, n, "(ep, first step >1mm, drift mm after 60 still steps):", out, flush=True)
        env.close()
