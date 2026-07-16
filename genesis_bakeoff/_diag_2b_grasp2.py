"""
Throwaway diagnostic (not committed): job 42164873 (GRASP_HAND_OFFSET=0.05
fix) still shows lift_success=False in ALL episodes, and UNSAFE episodes
show EXACTLY the same peak_accel=0.299 every time (suspicious -- suggests
the egg is never touched at all in the 3-step UNSAFE grasp_close phase,
just idle settle jitter). SAFE episodes show real disturbance (20-40 m/s^2)
but still never actually lift the egg. This script builds a minimal
franka+egg scene and steps through approach/descend/grasp slowly, printing
egg pos, hand pos, finger positions, and per-dof gripper force every step,
so we can see exactly what's happening geometrically instead of guessing.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standard_rig import (
    ARM_DOFS_IDX, GRIPPER_DOFS_IDX, GRIPPER_OPEN, GRIPPER_CLOSED, TABLE_HEIGHT, set_robot_ready_pose,
)

REPO = os.path.dirname(os.path.abspath(__file__))
EGG_MESH_PATH = os.path.join(REPO, "external_assets/egg/egg.obj")
EGG_SCALE = 0.0951
EGG_BOTTOM_OFFSET = 0.236591 * EGG_SCALE
TABLE_CENTER_XY = (0.35, 0.05)


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main():
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(pos=(TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], TABLE_HEIGHT / 2),
                                    size=(0.6, 0.6, TABLE_HEIGHT), fixed=True))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0.0, 0.0, TABLE_HEIGHT)))
    egg_x, egg_y = 0.35, 0.0
    egg = scene.add_entity(
        gs.morphs.Mesh(file=EGG_MESH_PATH, pos=(egg_x, egg_y, TABLE_HEIGHT + EGG_BOTTOM_OFFSET + 0.02),
                        scale=EGG_SCALE, fixed=False, convexify=True),
    )
    scene.build()
    set_robot_ready_pose(franka)
    egg.set_mass(0.055)

    hand_link = franka.get_link("hand")
    left_finger = franka.get_link("left_finger")
    right_finger = franka.get_link("right_finger")

    for _ in range(80):
        scene.step()
    egg_pos = _np(egg.get_pos()).flatten()
    egg_z0 = float(egg_pos[2])
    print(f"Egg settled at {egg_pos} (z0={egg_z0:.4f}, table_top={TABLE_HEIGHT:.4f})")

    def step_toward(target, n, gripper_target, label):
        for i in range(n):
            qpos = franka.inverse_kinematics(
                link=hand_link, pos=np.asarray(target), quat=None,
                rot_mask=[False, False, False], dofs_idx_local=ARM_DOFS_IDX,
            )
            qpos_np = _np(qpos)
            franka.control_dofs_position(qpos_np[:7], ARM_DOFS_IDX)
            franka.control_dofs_position(gripper_target, GRIPPER_DOFS_IDX)
            scene.step()
            hand_pos = _np(hand_link.get_pos()).flatten()
            lf = _np(left_finger.get_pos()).flatten()
            rf = _np(right_finger.get_pos()).flatten()
            egg_pos = _np(egg.get_pos()).flatten()
            gforce = _np(franka.get_dofs_force(GRIPPER_DOFS_IDX)).flatten()
            gq = _np(franka.get_dofs_position(GRIPPER_DOFS_IDX)).flatten()
            print(f"[{label} {i}] hand_z={hand_pos[2]:.4f} lf={lf} rf={rf} egg={egg_pos} "
                  f"gripper_q={gq} gforce={gforce}")

    above = (egg_x, egg_y, TABLE_HEIGHT + 0.25)
    grasp = (egg_x, egg_y, egg_z0 + 0.05)

    step_toward(above, 30, GRIPPER_OPEN, "approach")
    step_toward(grasp, 20, GRIPPER_OPEN, "descend")
    step_toward(grasp, 30, GRIPPER_CLOSED, "grasp_close")

    lift = (egg_x, egg_y, TABLE_HEIGHT + 0.28)
    step_toward(lift, 30, GRIPPER_CLOSED, "lift")

    print("DIAG_2B_GRASP2: DONE")


if __name__ == "__main__":
    main()
