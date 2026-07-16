"""
Throwaway diagnostic (NOT a deliverable, not committed) for task 4's
"energy-injection bug" investigation on scenario 3a-ii. Runs the burst-pack
technique under a single (pack_spacing, dt, substeps, seed) config per
process invocation, logging max per-step candy speed and final xy spread,
to determine whether the original pilot's overlap-burst approach produces
implausible (explosive / barely-moving) results, and if so, which knob
fixes it.

NOTE: earlier version of this script looped over multiple configs calling
gs.init() once per config inside build_scene() -- died on the 2nd config
with `genesis.GenesisException: Genesis already initialized` (job 42161326).
gs.init() is process-global and may only be called once per process. Fixed
by making this script single-config-per-process (called N times as separate
subprocess invocations from the SLURM script), matching the existing
convention elsewhere in this repo (every scenario_*.py script is also a
single build_scene()-per-process design).
"""
import argparse

import numpy as np


TABLE_HEIGHT = 0.75
BAG_CENTER = (0.35, 0.0, TABLE_HEIGHT + 0.25)
N_CANDIES = 20
CANDY_RADIUS = 0.008


def build_scene(pack_spacing, dt, substeps, seed=0):
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(pos=(0.35, 0.0, TABLE_HEIGHT / 2), size=(0.9, 0.7, TABLE_HEIGHT), fixed=True))

    rng = np.random.default_rng(seed)
    grid_n = int(np.ceil(N_CANDIES ** (1 / 3)))
    candies = []
    count = 0
    for ix in range(grid_n):
        for iy in range(grid_n):
            for iz in range(grid_n):
                if count >= N_CANDIES:
                    break
                jitter = rng.uniform(-0.0005, 0.0005, size=3)
                offset = (
                    (ix - grid_n / 2) * pack_spacing + jitter[0],
                    (iy - grid_n / 2) * pack_spacing + jitter[1],
                    iz * pack_spacing + jitter[2],
                )
                pos = (BAG_CENTER[0] + offset[0], BAG_CENTER[1] + offset[1], BAG_CENTER[2] + offset[2])
                candy = scene.add_entity(gs.morphs.Sphere(pos=pos, radius=CANDY_RADIUS))
                candies.append(candy)
                count += 1
    scene.build()
    return scene, candies


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def run_config(pack_spacing, dt, substeps, seed, steps=200):
    scene, candies = build_scene(pack_spacing, dt, substeps, seed=seed)
    max_speed = 0.0
    prev_pos = np.stack([_np(c.get_pos()) for c in candies])
    pos = prev_pos
    for i in range(steps):
        scene.step()
        pos = np.stack([_np(c.get_pos()) for c in candies])
        speed = np.linalg.norm(pos - prev_pos, axis=1) / dt
        max_speed = max(max_speed, float(speed.max()))
        prev_pos = pos
    final = pos
    xy_spread = final[:, :2].max(axis=0) - final[:, :2].min(axis=0)
    finite = np.isfinite(final).all()
    z_min, z_max = final[:, 2].min(), final[:, 2].max()
    off_table = int(np.sum((np.abs(final[:, 0] - 0.35) > 0.45) | (np.abs(final[:, 1] - 0.0) > 0.35)))
    print(
        f"cfg spacing={pack_spacing} dt={dt} substeps={substeps} seed={seed}: "
        f"finite={finite} max_speed={max_speed:.3f} m/s xy_spread={xy_spread} "
        f"z_range=({z_min:.3f},{z_max:.3f}) n_off_table_footprint={off_table}"
    )
    return dict(max_speed=max_speed, xy_spread=xy_spread.tolist(), finite=bool(finite), off_table=off_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack_spacing", type=float, default=0.012)
    parser.add_argument("--dt", type=float, default=5e-3)
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    run_config(args.pack_spacing, args.dt, args.substeps, args.seed, steps=args.steps)
    print("DIAG_3A_II_DONE")
