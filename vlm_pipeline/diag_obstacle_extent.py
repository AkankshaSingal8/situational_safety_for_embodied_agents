"""Per-geom audit of the obstacle extent used by the superquadric fit.

Prints every geom the fit's substring match picks up, its type, its own AABB,
and how much it contributes to the union — so an inflating geom (visual proxy,
unscaled mesh, rotated part) is visible instead of being hidden inside one
half-extent triple. Also reports the oriented (per-geom-frame) extent for
comparison against the world-axis-aligned box the barrier actually uses.
"""

import argparse
import pathlib

import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from run_guided_safelibero_pi05_eval import DUMMY_ACTION, NUM_STEPS_WAIT

GEOM_TYPE = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid",
             5: "cylinder", 6: "box", 7: "mesh"}


def geom_world_aabb(sim, gid):
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
        return world.min(axis=0), world.max(axis=0), verts
    else:
        half = np.full(3, float(sim.model.geom_rbound[gid]))
    return pos - half, pos + half, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_suite_name", default="safelibero_spatial")
    ap.add_argument("--safety_level", default="I")
    ap.add_argument("--tasks", type=int, nargs="*", default=[1, 3])
    args = ap.parse_args()

    suite = benchmark.get_benchmark_dict()[args.task_suite_name](safety_level=args.safety_level)
    for task_id in args.tasks:
        task = suite.get_task(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128,
                                 camera_widths=128, camera_depths=False)
        env.seed(0)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(task_id)[0])
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(DUMMY_ACTION)
        sim = env.sim

        names = [n.replace("_joint0", "") for n in sim.model.joint_names if "obstacle" in n]
        obstacle = None
        for n in names:
            p = obs.get(f"{n}_pos")
            if p is not None and p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
                obstacle = n
                break

        print(f"\n{'=' * 78}\nTASK {task_id}  obstacle={obstacle}\n{'=' * 78}")
        center = np.asarray(obs[f"{obstacle}_pos"])
        print(f"body/guidance center (obs `{obstacle}_pos`): {np.round(center, 4).tolist()}")

        # every joint-name match vs every geom substring match
        gids = [g for g in range(sim.model.ngeom)
                if obstacle in (sim.model.geom_id2name(g) or "")]
        print(f"substring '{obstacle}' matches {len(gids)} geoms\n")

        lo_all = np.full(3, np.inf)
        hi_all = np.full(3, -np.inf)
        print(f"{'geom name':44s} {'type':9s} {'contype':>7s} {'grp':>3s} "
              f"{'world half-size (x,y,z)':>30s}")
        for gid in gids:
            lo, hi, verts = geom_world_aabb(sim, gid)
            lo_all = np.minimum(lo_all, lo)
            hi_all = np.maximum(hi_all, hi)
            half = (hi - lo) / 2.0
            nm = (sim.model.geom_id2name(gid) or "?")[:44]
            print(f"{nm:44s} {GEOM_TYPE.get(int(sim.model.geom_type[gid]), '?'):9s} "
                  f"{int(sim.model.geom_contype[gid]):>7d} {int(sim.model.geom_group[gid]):>3d} "
                  f"{str(np.round(half, 4).tolist()):>30s}")
            if verts is not None:
                print(f"{'':44s} mesh verts={len(verts)} "
                      f"local |v|max={np.abs(verts).max(axis=0).round(4).tolist()} "
                      f"rbound={float(sim.model.geom_rbound[gid]):.4f}")

        print(f"\nUNION world AABB   lo={np.round(lo_all, 4).tolist()}  hi={np.round(hi_all, 4).tolist()}")
        print(f"UNION full size    {np.round(hi_all - lo_all, 4).tolist()}")
        print(f"mid half-extents   {np.round((hi_all - lo_all) / 2, 4).tolist()}")
        print(f"sym half-extents   {np.round(np.maximum(np.abs(hi_all - center), np.abs(center - lo_all)), 4).tolist()}")

        # collision-only union (contype != 0), i.e. what can actually be hit
        lo_c = np.full(3, np.inf)
        hi_c = np.full(3, -np.inf)
        n_c = 0
        for gid in gids:
            if int(sim.model.geom_contype[gid]) == 0 and int(sim.model.geom_conaffinity[gid]) == 0:
                continue
            lo, hi, _ = geom_world_aabb(sim, gid)
            lo_c = np.minimum(lo_c, lo)
            hi_c = np.maximum(hi_c, hi)
            n_c += 1
        if n_c:
            print(f"\nCOLLIDABLE-ONLY ({n_c} geoms) full size {np.round(hi_c - lo_c, 4).tolist()}"
                  f"  half {np.round((hi_c - lo_c) / 2, 4).tolist()}")

        # body-frame (oriented) extent of the whole obstacle, for comparison
        try:
            bid = sim.model.body_name2id(f"{obstacle}_main")
        except Exception:
            bid = None
            for b in sim.model.body_names:
                if obstacle in b:
                    bid = sim.model.body_name2id(b)
                    break
        if bid is not None:
            R = sim.data.body_xmat[bid].reshape(3, 3)
            bp = sim.data.body_xpos[bid]
            pts = []
            for gid in gids:
                lo, hi, verts = geom_world_aabb(sim, gid)
                if verts is not None:
                    grot = sim.data.geom_xmat[gid].reshape(3, 3)
                    pts.append(verts @ grot.T + sim.data.geom_xpos[gid])
                else:
                    c = np.array([[x, y, z] for x in (lo[0], hi[0])
                                  for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                    pts.append(c)
            P = np.concatenate(pts, axis=0)
            local = (P - bp) @ R
            print(f"body '{sim.model.body_id2name(bid)}' xpos={np.round(bp, 4).tolist()}")
            print(f"ORIENTED (body-frame) full size {np.round(local.max(0) - local.min(0), 4).tolist()}"
                  f"  half {np.round((local.max(0) - local.min(0)) / 2, 4).tolist()}")
            euler = np.degrees(np.array([
                np.arctan2(R[2, 1], R[2, 2]),
                np.arcsin(-np.clip(R[2, 0], -1, 1)),
                np.arctan2(R[1, 0], R[0, 0])]))
            print(f"body orientation rpy(deg) {np.round(euler, 1).tolist()}")
        env.close()


if __name__ == "__main__":
    main()
