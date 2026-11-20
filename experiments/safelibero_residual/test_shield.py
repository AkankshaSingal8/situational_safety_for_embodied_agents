"""Lookahead + restore must not change the real rollout. Runs one Spatial and one Long env."""
import numpy as np
import sl_env
from shield import Lookahead

rng = np.random.default_rng(0)
for suite, lv, t in [("safelibero_spatial", "II", 1), ("safelibero_long", "I", 0)]:
    sl_env.set_suite(suite)
    sl_env.ENV_SEED, sl_env.SETTLE = 0, 20
    acts = np.clip(rng.normal(0, 0.5, (40, 7)), -1, 1); acts[:, 6] = np.sign(acts[:, 6])
    trajs = []
    for use_la in (False, True):
        env, lang, inits = sl_env.make_env(lv, t, res=128)
        env.reset(); obs = env.set_init_state(inits[0]); obs = sl_env.settle(env)
        la = Lookahead(env)
        tr = []
        for i, a in enumerate(acts):
            if use_la and i % 5 == 0:
                s = la.save()
                for b in np.clip(rng.normal(0, 1, (10, 7)), -1, 1):
                    la.step(b)
                la.restore(s)
            obs, _, done, _ = env.step(a.tolist())
            tr.append(np.concatenate([obs["robot0_eef_pos"], obs["robot0_gripper_qpos"], env.sim.data.qpos[:]]))
        env.close()
        trajs.append(np.array(tr))
    print(suite, "max |diff| with vs without lookahead:", float(np.abs(trajs[0] - trajs[1]).max()))
