"""One-off: which obstacle is active per SafeLIBERO-Spatial scenario, object names, obs keys."""
import os, json, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
out = {}
for level in ["I", "II"]:
    suite = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level=level)
    for t in range(suite.n_tasks):
        task = suite.get_task(t)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=["agentview", "robot0_eye_in_hand"],
                                 camera_heights=128, camera_widths=128, hard_reset=False)
        env.seed(0)
        inits = suite.get_task_init_states(t)
        env.reset(); obs = env.set_init_state(inits[0])
        for _ in range(10): obs, *_ = env.step([0]*6 + [-1])
        names = [n.replace("_joint0", "") for n in env.sim.model.joint_names if n.endswith("_joint0")]
        pos = {n: np.round(obs[n + "_pos"], 3).tolist() for n in names if n + "_pos" in obs}
        act = [n for n in names if "obstacle" in n and abs(pos.get(n, [9])[0]) < 0.5 and abs(pos[n][1]) < 0.5 and pos[n][2] > 0]
        out[f"{level}/{t}"] = dict(lang=task.language, n_init=len(inits), active=act, pos=pos,
                                   eef=np.round(obs["robot0_eef_pos"], 3).tolist(),
                                   keys=sorted(k for k in obs if not k.endswith("_pos") and not k.endswith("_quat") and not "to_robot0" in k))
        print(level, t, act, len(inits), flush=True)
        env.close()
json.dump(out, open("scene_inspect.json", "w"), indent=1)
