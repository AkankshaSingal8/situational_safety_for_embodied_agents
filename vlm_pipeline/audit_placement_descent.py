"""Placement-descent audit for the Long timeout cells (offline, no policy).

Replays the exact barrier + corridor arithmetic the server enforces (numpy
mirror of `pi0_guided._dcbf_repair`'s geometry) along a synthetic vertical
descent from 15 cm above the destination down to place height, with the
production no-GT config numbers, and reports PER TERM why the descent is
blocked: raw barrier, corridor validity gate, relaxed margin, inflation,
payload companion. The output is the design input for the placement
exemption — it names the term to fix instead of guessing.
"""

import argparse
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from run_guided_safelibero_pi05_eval import (
    DUMMY_ACTION, NUM_STEPS_WAIT, OBSTACLE_RADII, DEFAULT_OBSTACLE_RADIUS,
    obstacle_radius, parse_entities,
)

# Production composed config (server defaults + client no-GT board settings)
EEF_R = 0.09
D_SAFE = 0.01
INFL = 0.004
CORRIDOR_RADIUS = 0.07
CORRIDOR_RELAX = 0.6
GAMMA = 0.9
COMPANIONS = np.array([[0, 0, 0.0], [0, 0, 0.08], [0, 0, -0.06]])


def corridor_scale(p, eef, target, dest):
    """Numpy mirror of pi0_guided._corridor_scale (incl. the t_raw gate)."""
    def seg(q):
        v = q - eef
        vv = max(float(v @ v), 1e-8)
        t_raw = float((p - eef) @ v) / vv
        t = np.clip(t_raw, 0.0, 1.0)
        proj = eef + t * v
        d = float(np.linalg.norm(p - proj))
        valid = (np.linalg.norm(q - eef) < 2.0) and (t_raw > 0.15)
        return d if valid else np.inf, t_raw
    d1, t1 = seg(target) if target is not None else (np.inf, None)
    d2, t2 = seg(dest) if dest is not None else (np.inf, None)
    inside = min(d1, d2) < CORRIDOR_RADIUS
    return (1.0 - CORRIDOR_RELAX * float(inside)), inside, t2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["long:I:1", "long:II:2"])
    args = ap.parse_args()

    for cell in args.cells:
        suite_key, lvl, tid = cell.split(":")
        tid = int(tid)
        suite = benchmark.get_benchmark_dict()[f"safelibero_{suite_key}"](safety_level=lvl)
        task = suite.get_task(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128,
                                 camera_widths=128, camera_depths=False)
        env.seed(0)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(tid)[0])
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(DUMMY_ACTION)

        obstacle = None
        for n in [j.replace("_joint0", "") for j in env.sim.model.joint_names if "obstacle" in j]:
            p = obs.get(f"{n}_pos")
            if p is not None and p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
                obstacle = n
                break
        o = np.asarray(obs[f"{obstacle}_pos"], dtype=float)
        r_obs = obstacle_radius(obstacle)
        r_eff = r_obs + EEF_R + D_SAFE
        ents = parse_entities(task.language, obs, env.sim)
        dest = ents.get("dest_pos")
        target = ents.get("target_pos")
        if dest is None:
            print(f"{cell}: NO DEST parsed — corridor never granted at the destination. That alone blocks placement.")
            env.close()
            continue
        dest = np.asarray(dest, dtype=float)

        print("=" * 78)
        print(f"{cell}  '{task.language}'")
        print(f"  obstacle {obstacle} at {np.round(o,3).tolist()}  r_obs {r_obs:.3f}  r_eff {r_eff:.3f}")
        print(f"  dest {np.round(dest,3).tolist()}  |dest - obstacle| = {np.linalg.norm(dest - o):.3f} m")
        print(f"  relaxed margin at c=0.4: {0.4*r_eff:.3f}  (+infl j=10: {0.4*(r_eff+10*INFL):.3f})")
        # Descent: EEF from 15 cm above dest straight down to 3 cm above dest.
        print(f"  {'z_above':>8s} {'B_raw':>8s} {'B_relax':>9s} {'c':>5s} {'gate_t':>7s} "
              f"{'inside':>7s} {'argmin_comp':>11s} {'verdict':>9s}")
        blocked_at = None
        for dz in np.linspace(0.15, 0.03, 13):
            eef = dest + np.array([0.0, 0.0, dz])
            # chunk rollout point = one step further down (j=1); payload low point governs
            p_j = dest + np.array([0.0, 0.0, max(dz - 0.02, 0.02)])
            c, inside, t2 = corridor_scale(p_j, eef, None if target is None else np.asarray(target), dest)
            dists = [np.linalg.norm((p_j + q) - o) for q in COMPANIONS]
            k = int(np.argmin(dists))
            r_j = (r_eff + INFL * 1.0) * c
            b_raw = min(dists) - (r_eff + INFL * 1.0)
            b_rel = min(dists) - r_j
            verdict = "OK" if b_rel > 0 else "BLOCKED"
            if verdict == "BLOCKED" and blocked_at is None:
                blocked_at = dz
            print(f"  {dz:8.3f} {b_raw:8.3f} {b_rel:9.3f} {c:5.2f} "
                  f"{'-' if t2 is None else f'{t2:7.2f}'} {str(inside):>7s} "
                  f"{['eef','wrist','payload'][k]:>11s} {verdict:>9s}")
        if blocked_at is None:
            print("  -> descent never blocked by the sphere+corridor arithmetic at this init;")
            print("     the dither must come from chain dynamics (gamma starvation) or K-selection —")
            print("     next check: B-chain feasibility along the descent (approach-speed cap).")
        else:
            print(f"  -> descent BLOCKED below z={blocked_at:.3f} m above dest.")
            # attribution: which change unblocks?
            for tag, args2 in [("corridor gate removed (t_raw>0 valid)", dict(gate=0.0)),
                               ("relax 0.8 instead of 0.6", dict(relax=0.8)),
                               ("payload companion off", dict(nopayload=True))]:
                dz = 0.04
                eef = dest + np.array([0, 0, dz + 0.02])
                p_j = dest + np.array([0, 0, dz])
                comps = COMPANIONS[:2] if args2.get("nopayload") else COMPANIONS
                relax = args2.get("relax", CORRIDOR_RELAX)
                # recompute with modified gate/relax
                def cs(p):
                    v = dest - eef
                    vv = max(float(v @ v), 1e-8)
                    t_raw = float((p - eef) @ v) / vv
                    proj = eef + np.clip(t_raw, 0, 1) * v
                    d = float(np.linalg.norm(p - proj))
                    valid = t_raw > args2.get("gate", 0.15)
                    return 1.0 - relax * float(valid and d < CORRIDOR_RADIUS)
                c = cs(p_j)
                b = min(np.linalg.norm((p_j + q) - o) for q in comps) - (r_eff + INFL) * c
                print(f"     [{tag:38s}] b at z=0.04: {b:+.3f}  {'UNBLOCKS' if b > 0 else 'still blocked'}")
        env.close()


if __name__ == "__main__":
    main()
