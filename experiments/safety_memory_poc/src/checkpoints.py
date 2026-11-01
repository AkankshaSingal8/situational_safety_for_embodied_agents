"""Robot body checkpoints for the barrier.

A barrier on the end-effector point alone does not describe the robot. In the
SafeLIBERO scene the forearm and wrist sweep through a protected object while
the end effector is still 0.2 m away, so an end-effector-only filter never
engages and the object is displaced in every arm, including the one that
believes the hazard. The repo's existing FOL filter guards elbow and wrist
checkpoints for the same reason.

Checkpoint positions are read from the simulator body frames; each is treated as
translating with the commanded end-effector delta, which is the same first-order
approximation the existing filter makes.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

# name -> margin scale applied to the base keep-away margin
DEFAULT_CHECKPOINTS = {
    "gripper0_right_gripper": 1.0,
    "gripper0_leftfinger": 1.0,
    "gripper0_rightfinger": 1.0,
    "robot0_link6": 0.75,   # wrist
    "robot0_link5": 0.65,
    "robot0_link4": 0.65,   # elbow
}


class CheckpointReader:
    """Resolves body names to ids once, then reads world positions per step."""

    def __init__(self, sim, names: Sequence[str] = None):
        self.sim = sim
        self.names: List[str] = []
        self.ids: List[int] = []
        wanted = list(names) if names is not None else list(DEFAULT_CHECKPOINTS)
        for n in wanted:
            try:
                bid = sim.model.body_name2id(n)
            except Exception:
                continue
            self.names.append(n)
            self.ids.append(bid)

    def positions(self) -> Dict[str, np.ndarray]:
        return {n: np.array(self.sim.data.body_xpos[i], dtype=float)
                for n, i in zip(self.names, self.ids)}
