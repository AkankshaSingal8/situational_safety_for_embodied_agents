"""Probe every SafeLIBERO suite (all levels/tasks, official episodes 0 and 25): object positions,
which obstacle is active, where inactive obstacles are parked, eef start, objects of interest, and
whether each task's language matches a LIBERO demo file.  Output: probe_suites.json

    python probe_suites.py --demos /data/asingal2/data_libero
"""
import argparse
import glob
import json
import os
import re

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="/data/asingal2/data_libero")
    ap.add_argument("--out", default="probe_suites.json")
    a = ap.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    demo_dir = {"safelibero_spatial": "libero_spatial", "safelibero_object": "libero_object",
                "safelibero_goal": "libero_goal", "safelibero_long": "libero_10"}
    out = {}
    for suite_name, dd in demo_dir.items():
        files = sorted(glob.glob(os.path.join(a.demos, dd, "*.hdf5")))
        demo_langs = [re.sub(r"^[A-Z_0-9]+SCENE[0-9]+_", "", os.path.basename(f).replace("_demo.hdf5", "")).replace("_", " ")
                      for f in files]
        for level in ["I", "II"]:
            suite = benchmark.get_benchmark_dict()[suite_name](safety_level=level)
            for t in range(suite.n_tasks):
                task = suite.get_task(t)
                bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
                txt = open(bddl).read()
                ooi = re.search(r"\(:obj_of_interest(.*?)\)", txt, re.S).group(1).split()
                import sl_env
                sl_env.set_suite(suite_name)
                env, _, inits = sl_env.make_env(level, t)  # retries robosuite's random-placement failures
                rec = dict(lang=task.language, bddl=task.bddl_file, n_init=len(inits), obj_of_interest=ooi,
                           demo_match=task.language in demo_langs,
                           demo_idx=demo_langs.index(task.language) if task.language in demo_langs else None)
                for ep in (0, 25):
                    env.reset(); obs = env.set_init_state(inits[ep])
                    for _ in range(20):
                        obs, *_ = env.step([0] * 6 + [-1])
                    names = [n.replace("_joint0", "") for n in env.sim.model.joint_names if n.endswith("_joint0")]
                    pos = {n: np.round(obs[n + "_pos"], 3).tolist() for n in names if n + "_pos" in obs}
                    obst = [n for n in names if "obstacle" in n]
                    rec[f"ep{ep}"] = dict(pos=pos, obstacles=obst, eef=np.round(obs["robot0_eef_pos"], 3).tolist(),
                                          box_base=pos.get("box_base_1"))
                env.close()
                out[f"{suite_name}/{level}/{t}"] = rec
                print(suite_name, level, t, task.language, "| demo", rec["demo_match"], "| ooi", ooi, flush=True)
                for ep in (0, 25):
                    e = rec[f"ep{ep}"]
                    print("   ep", ep, "eef", e["eef"], "obstacles", {n: e["pos"].get(n) for n in e["obstacles"]},
                          "box_base", e["box_base"], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
