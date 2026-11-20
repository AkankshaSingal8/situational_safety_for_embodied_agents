"""Why does the milk -> orange juice swap fail novel_scenes.py's stability check? Print positions, bounding boxes
and per-geom info for a few milk scenes, and save seen | novel agentview images."""
import sys, numpy as np, imageio, sl_env, novel_objects
from eval_sl import load_scenes
from novel_scenes import zaxis
suite = sys.argv[1] if len(sys.argv) > 1 else "spatial"
sl_env.set_suite("safelibero_" + suite); sl_env.ENV_SEED, sl_env.SETTLE = 0, 20
om = sl_env.make_env; novel_objects.enable(); nm = sl_env.make_env
st, meta = load_scenes(f"data_v2/{suite}/scenes_pair.npz")
idx = [i for i, m in enumerate(meta) if any("milk" in o for o in m["obstacles"])][:3]
for i in idx:
    m = meta[i]; imgs = []
    for k, mk in (("seen", om), ("novel", nm)):
        e, _, _ = mk(m["level"], m["task"], res=256)
        e.reset(); e.set_init_state(st[i]); o = sl_env.settle(e)
        n = sl_env.active_obstacles(o)[0]
        lo, hi = sl_env.object_aabb(e.sim, n)
        print(i, k, n, "pos", np.round(o[n + "_pos"], 3), "aabb", np.round(lo, 3), np.round(hi, 3),
              "z-axis", np.round(zaxis(o[n + "_quat"]), 2), flush=True)
        mdl = e.sim.model
        for b in sl_env.body_tree_ids(e.sim, n):
            gs = [g for g in range(mdl.ngeom) if mdl.geom_bodyid[g] == b]
            print("   body", mdl.body_id2name(b), "geoms", [(mdl.geom_type[g], np.round(e.sim.data.geom_xpos[g], 3).tolist(), int(mdl.geom_contype[g])) for g in gs][:6])
        imgs.append(o["agentview_image"][::-1, ::-1]); e.close()
    imageio.imwrite(f"runs_v2/novel_check/oj_{suite}_{i}.png", np.concatenate(imgs, 1))
