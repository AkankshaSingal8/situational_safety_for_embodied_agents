"""
Throwaway diagnostic (not committed): the first full run of scenario_2b.py
(SLURM job 42163132) showed lift_success=False in ALL 10 episodes and
suspicious force/accel numbers, indicating the grasp never actually
succeeds. Root-cause hypothesis: grasp_pos targets the Franka "hand" link
origin directly AT the egg's own resting height (egg_z0 ~ TABLE_HEIGHT +
0.022), but the "hand" link origin is NOT the fingertip contact point --
there's a real hand-to-fingertip offset (finger joints sit at local z=0.0584
in the hand frame per panda.xml, plus finger mesh length below that). At
egg_z0, the whole hand assembly would need to sit almost at table height,
which is not physically consistent with also getting fingers around a
target at that same height.

This script empirically measures: for a given "hand" IK target height,
where do the LEFT/RIGHT finger links actually end up in world Z? This lets
us pick a grasp_pos z offset (relative to the object's own resting height)
that puts the fingertips (not the hand origin) at the object's height.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standard_rig import ARM_DOFS_IDX, GRIPPER_DOFS_IDX, GRIPPER_OPEN, TABLE_HEIGHT, set_robot_ready_pose


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main():
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(pos=(0.35, 0.05, TABLE_HEIGHT / 2), size=(0.6, 0.6, TABLE_HEIGHT), fixed=True))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0.0, 0.0, TABLE_HEIGHT)))
    scene.build()
    set_robot_ready_pose(franka)

    hand_link = franka.get_link("hand")
    left_finger = franka.get_link("left_finger")
    right_finger = franka.get_link("right_finger")

    for target_z in [TABLE_HEIGHT + 0.05, TABLE_HEIGHT + 0.10, TABLE_HEIGHT + 0.15, TABLE_HEIGHT + 0.20]:
        target = (0.35, 0.0, target_z)
        for _ in range(30):
            qpos = franka.inverse_kinematics(
                link=hand_link, pos=np.asarray(target), quat=None,
                rot_mask=[False, False, False], dofs_idx_local=ARM_DOFS_IDX,
            )
            qpos_np = _np(qpos)
            franka.control_dofs_position(qpos_np[:7], ARM_DOFS_IDX)
            franka.control_dofs_position(GRIPPER_OPEN, GRIPPER_DOFS_IDX)
            scene.step()
        hand_pos = _np(hand_link.get_pos()).flatten()
        lf_pos = _np(left_finger.get_pos()).flatten()
        rf_pos = _np(right_finger.get_pos()).flatten()
        fingertip_z_approx = min(lf_pos[2], rf_pos[2])  # finger link origin, not pad tip, but a lower bound
        print(f"hand_target_z={target_z:.3f} -> hand_pos={hand_pos} left_finger_pos={lf_pos} "
              f"right_finger_pos={rf_pos} hand_to_finger_dz={hand_pos[2]-fingertip_z_approx:.4f}")

    print("DIAG_2B_GRASP: DONE")


if __name__ == "__main__":
    main()
