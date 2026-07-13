"""Regression test for the critical build_env_action key bug (see
robocasa_action_utils.py's docstring): the arm's key in the dict passed to
`robot.create_action_vector()` must be the plain arm name (e.g. "right"),
NOT "{arm}_delta" -- the latter silently drops every arm command, since
unrecognized action_dict keys are ignored rather than raising an error.
This was confirmed empirically (forced max +x delta for 50 steps moved the
robot by ~0.6m after the fix vs ~0.000005m before it) and cost every
full-suite eval run before this test existed.
"""

import numpy as np
import pytest

from robocasa_action_utils import build_env_action


class _FakeGripper:
    dof = 1


class _FakeRobot:
    """Minimal stand-in for a robosuite robot, capturing the action_dict
    `create_action_vector` was called with instead of actually running any
    controller logic."""

    def __init__(self, is_mobile: bool):
        self.arms = ["right"]
        self.gripper = {"right": _FakeGripper()}
        self.is_mobile = is_mobile
        self.last_action_dict = None

    def create_action_vector(self, action_dict):
        self.last_action_dict = action_dict
        return np.zeros(12 if self.is_mobile else 7)


@pytest.mark.parametrize("is_mobile", [False, True])
def test_arm_key_is_plain_arm_name_not_suffixed(is_mobile):
    robot = _FakeRobot(is_mobile=is_mobile)
    arm_delta = np.array([0.5, 0.1, -0.2, 0.0, 0.0, 0.3])

    build_env_action(robot, arm_delta, gripper_value=-1.0)

    action_dict = robot.last_action_dict
    # The bug used "right_delta" as the key -- create_action_vector never
    # recognizes that as a valid part name, so it silently drops the arm
    # command. The correct key is just the arm name itself.
    assert "right" in action_dict, (
        f"Expected plain arm-name key 'right' in action_dict, got keys {list(action_dict.keys())} -- "
        "if this is '{arm}_delta' instead, the critical action-dropping bug has regressed."
    )
    assert "right_delta" not in action_dict
    np.testing.assert_array_equal(action_dict["right"], np.clip(arm_delta, -1.0, 1.0))


def test_gripper_key_is_arm_name_plus_gripper_suffix():
    robot = _FakeRobot(is_mobile=False)
    build_env_action(robot, np.zeros(6), gripper_value=1.0)
    assert "right_gripper" in robot.last_action_dict
    np.testing.assert_array_equal(robot.last_action_dict["right_gripper"], np.array([1.0]))


def test_mobile_robot_base_held_still():
    robot = _FakeRobot(is_mobile=True)
    build_env_action(robot, np.zeros(6), gripper_value=-1.0)
    action_dict = robot.last_action_dict
    np.testing.assert_array_equal(action_dict["base"], np.zeros(3))
    assert action_dict["base_mode"] == -1
