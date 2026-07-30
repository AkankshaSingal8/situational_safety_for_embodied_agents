"""GT superquadric fitting from simulator point clouds — offline validator.

Replaces the AABB pseudo-fit with an actual fit, following the paper's recipe
(union of superquadrics per part, oriented frames, point-cloud based):

  1. sample surface points from every COLLIDABLE geom of the obstacle
  2. split into parts (2-means on geom centers; a moka pot separates into
     body vs handle, a bottle collapses to one part)
  3. per part: PCA frame -> fit ONE superquadric by minimizing volume with a
     coverage penalty (all points inside), eps free in [0.2, 1.0]
  4. inflate SEMI-AXES by eef_radius + d_safe (Minkowski-safe — the C2 fix;
     never inflate the radius)

Reports, per Spatial L1 task: coverage, effective keep-out volume vs the
production sphere / AABB arms, and the C4 target-clearance test (extent along
the obstacle->grasp-target ray vs the target's distance).
"""

import argparse
import json
import pathlib

import numpy as np
from scipy.optimize import minimize

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from run_guided_safelibero_pi05_eval import DUMMY_ACTION, NUM_STEPS_WAIT

OFF = 0.10  # eef_radius 0.09 + d_safe 0.01
N_SURF = 400  # surface samples per geom


def sample_geom_surface(sim, gid, rng):
    """Surface point sample of one geom in world frame."""
    pos = sim.data.geom_xpos[gid]
    rot = sim.data.geom_xmat[gid].reshape(3, 3)
    size = sim.model.geom_size[gid]
    gtype = int(sim.model.geom_type[gid])
    if gtype == 7:  # mesh: vertices ARE a surface sample
        mid = int(sim.model.geom_dataid[gid])
        adr, num = int(sim.model.mesh_vertadr[mid]), int(sim.model.mesh_vertnum[mid])
        verts = np.asarray(sim.model.mesh_vert[adr:adr + num]).reshape(-1, 3)
        if len(verts) > N_SURF:
            verts = verts[rng.choice(len(verts), N_SURF, replace=False)]
        return verts @ rot.T + pos
    if gtype == 6:  # box: uniform over the 6 faces
        pts = rng.uniform(-1, 1, size=(N_SURF, 3))
        face = rng.integers(0, 3, size=N_SURF)
        sign = rng.choice([-1.0, 1.0], size=N_SURF)
        pts[np.arange(N_SURF), face] = sign
        return (pts * size) @ rot.T + pos
    if gtype == 5:  # cylinder
        th = rng.uniform(0, 2 * np.pi, N_SURF)
        z = rng.uniform(-size[1], size[1], N_SURF)
        local = np.stack([size[0] * np.cos(th), size[0] * np.sin(th), z], -1)
        return local @ rot.T + pos
    if gtype == 2:  # sphere
        u = rng.normal(size=(N_SURF, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        return (u * size[0]) @ rot.T + pos
    return None


def sq_implicit(local, s, e1, e2):
    """Paper-form inside-outside: F <= 1 is inside. local: (N,3) in SQ frame."""
    x, y, z = np.abs(local[:, 0] / s[0]), np.abs(local[:, 1] / s[1]), np.abs(local[:, 2] / s[2])
    return ((x ** (2 / e2) + y ** (2 / e2)) ** (e2 / e1) + z ** (2 / e1)) ** e1


def sq_volume(s, e1, e2):
    """Closed-form superquadric volume (beta-function form)."""
    from scipy.special import beta
    return 2 * s[0] * s[1] * s[2] * e1 * e2 * beta(e1 / 2 + 1, e1) * beta(e2 / 2, e2 / 2)


def fit_part(points, center=None):
    """Fit one SQ deterministically: PCA frame; grid over (e1, e2); for each
    shape, the minimal CONTAINING scale via bisection (containment is monotone
    in a uniform scale factor); keep the minimum-volume containing fit."""
    c = points.mean(0) if center is None else center
    X = points - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    R = Vt
    L = X @ R.T
    base = np.maximum(np.abs(L).max(0), 5e-3)  # per-axis containing box = ratio prior

    best = None
    for e1 in (0.2, 0.4, 0.6, 0.8, 1.0):
        for e2 in (0.2, 0.4, 0.6, 0.8, 1.0):
            lo, hi = 0.5, 2.5  # uniform multiplier on `base`
            if sq_implicit(L, base * hi, e1, e2).max() > 1.0:
                continue  # even 2.5x cannot contain (degenerate) — skip
            for _ in range(30):
                mid = (lo + hi) / 2
                if sq_implicit(L, base * mid, e1, e2).max() <= 1.0:
                    hi = mid
                else:
                    lo = mid
            s = base * hi
            # Selection criterion = the ENFORCED volume (semi-axes + pad), not
            # the raw fit volume: after +0.10 m inflation a boxy exponent fills
            # its corners solid, so minimizing raw volume picks exactly the
            # shapes that cost most once inflated.
            v = sq_volume(s + OFF, e1, e2)
            if best is None or v < best["volume"]:
                best = {"center": c, "R": R, "scales": s, "e1": e1, "e2": e2,
                        "coverage": float((sq_implicit(L, s, e1, e2) <= 1.0).mean()),
                        "volume": float(v)}
    return best


def split_parts(sim, gids):
    """2-means on geom centers; collapse to 1 part if clusters are close."""
    centers = np.array([sim.data.geom_xpos[g] for g in gids])
    if len(gids) < 4:
        return [gids]
    c0, c1 = centers[0], centers[-1]
    for _ in range(12):
        d0 = np.linalg.norm(centers - c0, axis=1)
        d1 = np.linalg.norm(centers - c1, axis=1)
        m = d0 <= d1
        if m.all() or (~m).all():
            return [gids]
        c0n, c1n = centers[m].mean(0), centers[~m].mean(0)
        if np.linalg.norm(c0n - c0) + np.linalg.norm(c1n - c1) < 1e-6:
            break
        c0, c1 = c0n, c1n
    if np.linalg.norm(c0 - c1) < 0.04:  # parts not meaningfully separated
        return [gids]
    m = np.linalg.norm(centers - c0, axis=1) <= np.linalg.norm(centers - c1, axis=1)
    return [[g for g, k in zip(gids, m) if k], [g for g, k in zip(gids, m) if not k]]


def union_extent_along(parts, origin, u, offset=OFF):
    """Max extent of the INFLATED union surface from `origin` along ray u
    (bisection on the implicit functions with semi-axes + offset)."""
    def inside(p):
        for pt in parts:
            local = (p - pt["center"]) @ pt["R"].T
            s = pt["scales"] + offset
            if sq_implicit(local[None, :], s, pt["e1"], pt["e2"])[0] <= 1.0:
                return True
        return False
    lo, hi = 0.0, 0.8
    if not inside(origin + u * 1e-4):
        # origin outside the union: walk in to find first crossing, then out
        pass
    for _ in range(40):
        mid = (lo + hi) / 2
        if inside(origin + u * mid):
            lo = mid
        else:
            hi = mid
    return lo


def union_volume_mc(parts, offset=OFF, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    los, his = [], []
    for pt in parts:
        s = pt["scales"] + offset
        r = np.linalg.norm(s) * 1.05
        los.append(pt["center"] - r)
        his.append(pt["center"] + r)
    lo = np.min(los, axis=0)
    hi = np.max(his, axis=0)
    p = rng.uniform(lo, hi, size=(n, 3))
    hit = np.zeros(n, dtype=bool)
    for pt in parts:
        local = (p - pt["center"]) @ pt["R"].T
        s = pt["scales"] + offset
        hit |= sq_implicit(local, s, pt["e1"], pt["e2"]) <= 1.0
    return float(np.prod(hi - lo) * hit.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_suite_name", default="safelibero_spatial")
    ap.add_argument("--safety_level", default="I")
    ap.add_argument("--out", default="sq_viz/gt_pointcloud_fits.json")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name](safety_level=args.safety_level)
    results = []
    for task_id in range(suite.n_tasks):
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

        obstacle = None
        for n in [j.replace("_joint0", "") for j in sim.model.joint_names if "obstacle" in j]:
            p = obs.get(f"{n}_pos")
            if p is not None and p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
                obstacle = n
                break
        if obstacle is None:
            env.close()
            continue

        gids = [g for g in range(sim.model.ngeom)
                if obstacle in (sim.model.geom_id2name(g) or "")
                and (int(sim.model.geom_contype[g]) or int(sim.model.geom_conaffinity[g]))]
        part_gids = split_parts(sim, gids)
        parts = []
        all_pts = []
        for pg in part_gids:
            pts = [sample_geom_surface(sim, g, rng) for g in pg]
            pts = np.concatenate([p for p in pts if p is not None], axis=0)
            all_pts.append(pts)
            parts.append(fit_part(pts))
        pts_all = np.concatenate(all_pts, axis=0)

        # union coverage of ALL points (each point inside >= 1 part)
        covered = np.zeros(len(pts_all), dtype=bool)
        for pt in parts:
            local = (pts_all - pt["center"]) @ pt["R"].T
            covered |= sq_implicit(local, pt["scales"] * 1.02, pt["e1"], pt["e2"]) <= 1.05

        center = np.asarray(obs[f"{obstacle}_pos"], dtype=float)
        from run_guided_safelibero_pi05_eval import parse_entities
        ents = parse_entities(task.language, obs, sim)
        target = ents.get("target_pos")

        vol = union_volume_mc(parts)
        row = {
            "task_id": task_id, "obstacle": obstacle,
            "n_parts": len(parts),
            "parts": [{"scales": p["scales"].tolist(), "e1": p["e1"], "e2": p["e2"],
                       "center": p["center"].tolist(), "R": p["R"].tolist(),
                       "coverage": p["coverage"], "volume": p["volume"]} for p in parts],
            "union_coverage": float(covered.mean()),
            "union_volume_inflated": vol,
        }
        if target is not None:
            v = target - center
            dist = float(np.linalg.norm(v))
            ext = union_extent_along(parts, center, v / dist)
            row["target_dist"] = dist
            row["extent_along_target"] = ext
            row["target_inside"] = bool(ext > dist)
        results.append(row)
        print(f"t{task_id} {obstacle}: {len(parts)} part(s), coverage {covered.mean():.3f}, "
              f"inflated union vol {vol*1e3:.1f} L"
              + (f", target {row['target_dist']*1000:.0f} mm vs extent {row['extent_along_target']*1000:.0f} mm "
                 f"-> {'INSIDE' if row['target_inside'] else 'clear'}" if target is not None else ""))
        for i, p in enumerate(parts):
            print(f"   part {i}: scales {np.round(p['scales'],3).tolist()} "
                  f"e1 {p['e1']:.2f} e2 {p['e2']:.2f} cov {p['coverage']:.3f}")
        env.close()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
