import os, time, json
import numpy as np
print("STAGE import", flush=True)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
print("STAGE imported", flush=True)

out = {}
bm = benchmark.get_benchmark_dict()["safelibero_spatial"]()
for tid in range(bm.n_tasks):
    t = bm.get_task(tid)
    out.setdefault("tasks", []).append({"id": tid, "name": t.name, "lang": t.language})
task = bm.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print("STAGE bddl", bddl, flush=True)
t0 = time.time()
env = ControlEnv(bddl, use_camera_obs=False, has_offscreen_renderer=False,
                 has_renderer=False, camera_names=[], control_freq=20,
                 hard_reset=False, horizon=100000, ignore_done=True)
print("STAGE env created %.1fs" % (time.time()-t0), flush=True)
env.seed(0); obs = env.reset()
print("STAGE reset", flush=True)
out["obs_keys"] = sorted(obs.keys())
out["objpos"] = {k: np.asarray(obs[k]).tolist() for k in sorted(obs) if k.endswith("_pos")}
out["eef"] = np.asarray(obs["robot0_eef_pos"]).tolist()
out["action_dim"] = int(env.env.action_dim)
sim = env.env.sim
out["bodies"] = [sim.model.body_id2name(i) for i in range(sim.model.nbody)]
t1 = time.time(); n = 100
for _ in range(n):
    obs, _, _, _ = env.step(np.zeros(7))
out["step_ms"] = (time.time()-t1)/n*1000
# delta-eef responsiveness on each axis
resp = {}
for ax, name in enumerate("xyz"):
    env.seed(0); obs = env.reset()
    p0 = np.asarray(obs["robot0_eef_pos"]).copy()
    a = np.zeros(7); a[ax] = 1.0; a[6] = -1
    for _ in range(20):
        obs, _, _, _ = env.step(a)
    resp[name] = (np.asarray(obs["robot0_eef_pos"]) - p0).tolist()
out["resp_20steps"] = resp
# reset determinism
env.seed(5); o1 = env.reset(); p1 = np.asarray(o1["robot0_eef_pos"]).copy()
env.seed(5); o2 = env.reset(); p2 = np.asarray(o2["robot0_eef_pos"]).copy()
out["reset_determinism_eef_delta"] = float(np.abs(p1-p2).max())
# pose variation across seeds (does reset move objects?)
poses = {}
for s in (1, 2, 3):
    env.seed(s); o = env.reset()
    for k in out["objpos"]:
        poses.setdefault(k, []).append(np.asarray(o[k]).tolist())
out["pose_spread"] = {k: float(np.ptp(np.array(v)[:, :3], axis=0).max()) for k, v in poses.items()}
with open("probe_result.json", "w") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "obs_keys"}, indent=2)[:4000], flush=True)
print("PROBEDONE", flush=True)
env.close()
