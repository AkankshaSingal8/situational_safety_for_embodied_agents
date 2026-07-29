"""Dump the keep-out geometry actually used by the sphere and superquadric
arms, per SafeLIBERO scene, for offline visualisation.

No policy server needed: this only resets each task to its stored init state,
settles, and records the same quantities the eval client computes at episode
start (obstacle AABB, sym fit, mid fit, sphere radius, EEF, corridor entities).
"""

import argparse
import json
import pathlib

import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from run_guided_safelibero_pi05_eval import (
    DUMMY_ACTION,
    NUM_STEPS_WAIT,
    obstacle_half_extents,
    obstacle_radius,
    parse_entities,
)


def geom_aabb(sim, obstacle_name):
    """Raw world AABB (lo, hi) over the obstacle's geoms — the ground truth
    the two fits approximate."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for gid in range(sim.model.ngeom):
        if obstacle_name not in (sim.model.geom_id2name(gid) or ""):
            continue
        pos = sim.data.geom_xpos[gid]
        rot = sim.data.geom_xmat[gid].reshape(3, 3)
        size = sim.model.geom_size[gid]
        gtype = int(sim.model.geom_type[gid])
        if gtype == 6:
            half = np.abs(rot) @ size
        elif gtype == 2:
            half = np.full(3, size[0])
        elif gtype == 5:
            half = np.abs(rot) @ np.array([size[0], size[0], size[1]])
        elif gtype == 7:
            mid = int(sim.model.geom_dataid[gid])
            adr, num = int(sim.model.mesh_vertadr[mid]), int(sim.model.mesh_vertnum[mid])
            verts = np.asarray(sim.model.mesh_vert[adr:adr + num]).reshape(-1, 3)
            world = verts @ rot.T + pos
            lo = np.minimum(lo, world.min(axis=0))
            hi = np.maximum(hi, world.max(axis=0))
            continue
        else:
            half = np.full(3, float(sim.model.geom_rbound[gid]))
        lo = np.minimum(lo, pos - half)
        hi = np.maximum(hi, pos + half)
    if not np.all(np.isfinite(lo)):
        return None, None
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_suite_name", default="safelibero_spatial")
    ap.add_argument("--safety_level", default="I")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--eef_radius", type=float, default=0.09)
    ap.add_argument("--d_safe", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    suite = benchmark.get_benchmark_dict()[args.task_suite_name](safety_level=args.safety_level)
    scenes = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128,
                                 camera_widths=128, camera_depths=False)
        env.seed(0)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(task_id)[args.episode])
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(DUMMY_ACTION)

        names = [n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n]
        obstacle = None
        for n in names:
            p = obs.get(f"{n}_pos")
            if p is not None and p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
                obstacle = n
                break
        if obstacle is None:
            env.close()
            continue

        center = np.asarray(obs[f"{obstacle}_pos"], dtype=float)
        sym = obstacle_half_extents(env.sim, obstacle, center)
        mid_fit = obstacle_half_extents(env.sim, obstacle, center, return_mid=True)
        lo, hi = geom_aabb(env.sim, obstacle)
        ents = parse_entities(task.language, obs, env.sim)

        scenes.append({
            "task_id": task_id,
            "description": task.language,
            "obstacle": obstacle,
            "center": center.tolist(),
            "aabb_lo": lo.tolist(),
            "aabb_hi": hi.tolist(),
            "sym_extents": np.asarray(sym).tolist(),
            "mid_extents": np.asarray(mid_fit[0]).tolist(),
            "mid_center": np.asarray(mid_fit[1]).tolist(),
            "sphere_r_obs": obstacle_radius(obstacle),
            "eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=float).tolist(),
            "target_pos": ents.get("target_pos", np.zeros(3)).tolist() if "target_pos" in ents else None,
            "dest_pos": ents.get("dest_pos", np.zeros(3)).tolist() if "dest_pos" in ents else None,
            "eef_radius": args.eef_radius,
            "d_safe": args.d_safe,
        })
        print(f"[dump] task {task_id} {obstacle} sym={np.round(sym,3).tolist()} "
              f"mid={np.round(mid_fit[0],3).tolist()} sphere_r={obstacle_radius(obstacle):.3f}")
        env.close()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"suite": args.task_suite_name,
                               "level": args.safety_level,
                               "scenes": scenes}, indent=2))
    print(f"[dump] wrote {out} ({len(scenes)} scenes)")


if __name__ == "__main__":
    main()
