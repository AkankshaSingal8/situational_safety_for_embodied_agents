"""Unit tests for RoboCasaAdapter.

These tests do NOT require the real `oopsieverse_robocasa` conda env
(robosuite/mujoco/RoboCasa). They stub out `oopsiebench.envs.registry` and
`robosuite.controllers` in `sys.modules` so `RoboCasaAdapter.__init__` can run
against a fake env object, and exercise the observation-building / health /
close logic directly. Real end-to-end verification (constructing an actual
robosuite env) is Task 4's job, not this one.
"""

import importlib
import sys
import types
import unittest
from dataclasses import dataclass, field
from typing import List, Optional
from unittest import mock

import numpy as np


def _install_stub_oopsiebench_registry(env_config_by_name):
    """Install a stub `oopsiebench.envs.registry` module in sys.modules.

    `env_config_by_name` maps task_name -> EnvConfig-like object. Looking up
    a name not in this dict raises ValueError, matching the real registry.
    """

    @dataclass
    class _StubEnvConfig:
        damageable_class: object
        camera_name: str = "robot0_agentview_right"
        robot: str = "PandaOmron"
        control_freq: int = 20

    class _StubRegistry:
        @classmethod
        def get(cls, name):
            if name not in env_config_by_name:
                raise ValueError(f"Unknown environment: {name}")
            return env_config_by_name[name]

    oopsiebench_pkg = types.ModuleType("oopsiebench")
    envs_pkg = types.ModuleType("oopsiebench.envs")
    registry_mod = types.ModuleType("oopsiebench.envs.registry")
    registry_mod.EnvironmentRegistry = _StubRegistry
    registry_mod.EnvConfig = _StubEnvConfig

    sys.modules["oopsiebench"] = oopsiebench_pkg
    sys.modules["oopsiebench.envs"] = envs_pkg
    sys.modules["oopsiebench.envs.registry"] = registry_mod
    return _StubEnvConfig, _StubRegistry


def _install_stub_robosuite_controllers():
    robosuite_pkg = sys.modules.get("robosuite", types.ModuleType("robosuite"))
    controllers_mod = types.ModuleType("robosuite.controllers")
    controllers_mod.load_composite_controller_config = mock.MagicMock(return_value={})
    sys.modules["robosuite"] = robosuite_pkg
    sys.modules["robosuite.controllers"] = controllers_mod
    return controllers_mod


class _FakeEnv:
    """Minimal stand-in for a RoboCasa Damageable* env instance."""

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs
        self.health_list_link_names = ["obj_a@link1", "obj_b@link2"]
        self._success = False
        self.closed = False

    def reset(self):
        obs = {
            "robot0_agentview_right_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.ones((256, 256, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([0.02, -0.02]),
            "health": np.array([1.0, 1.0], dtype=np.float32),
        }
        info = {"damage_info": {}}
        return obs, info

    def step(self, action):
        obs = {
            "robot0_agentview_right_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.ones((256, 256, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([0.02, -0.02]),
            "health": np.array([0.9, 1.0], dtype=np.float32),
        }
        reward = 0.0
        done = False
        info = {"damage_info": {"obj_a": {"link1": 0.1}}}
        return obs, reward, done, info

    def _check_success(self):
        return self._success

    def close(self):
        self.closed = True


class _FakeEnvNoWristNoHealthAttr(_FakeEnv):
    """Simulates an env whose obs lacks the wrist-cam key, and which has no
    `health_list_link_names` attribute (to exercise the fallback paths)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.health_list_link_names

    def reset(self):
        obs, info = super().reset()
        del obs["robot0_eye_in_hand_image"]
        return obs, info

    def step(self, action):
        obs, reward, done, info = super().step(action)
        del obs["robot0_eye_in_hand_image"]
        return obs, reward, done, info


class _FakeEnvNoClose:
    """Env object without a close() method at all."""

    def __init__(self, *args, **kwargs):
        self.health_list_link_names = ["obj_a@link1"]

    def reset(self):
        return {
            "robot0_agentview_right_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.zeros(4),
            "robot0_gripper_qpos": np.zeros(2),
            "health": np.array([1.0], dtype=np.float32),
        }, {"damage_info": {}}

    def step(self, action):
        return self.reset()[0], 0.0, False, {"damage_info": {}}

    def _check_success(self):
        return True


class RoboCasaAdapterTestBase(unittest.TestCase):
    """Base class that (re)installs stub modules and imports a fresh copy of
    robocasa_adapter for each test, so tests don't leak state via sys.modules
    caching across test cases with different fake env classes."""

    fake_env_cls = _FakeEnv

    def setUp(self):
        self._orig_modules = {
            name: sys.modules.get(name)
            for name in (
                "oopsiebench",
                "oopsiebench.envs",
                "oopsiebench.envs.registry",
                "robosuite",
                "robosuite.controllers",
                "oopsieverse_eval.eval.robocasa_adapter",
            )
        }

        self._stub_env_config_cls, self._stub_registry_cls = (
            _install_stub_oopsiebench_registry(
                {"pick_egg": self._make_env_config()}
            )
        )
        _install_stub_robosuite_controllers()

        sys.modules.pop("oopsieverse_eval.eval.robocasa_adapter", None)
        self.adapter_module = importlib.import_module(
            "oopsieverse_eval.eval.robocasa_adapter"
        )

    def tearDown(self):
        for name, mod in self._orig_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    def _make_env_config(self):
        return self._env_config_for(self.fake_env_cls)

    def _env_config_for(self, fake_cls):
        @dataclass
        class _Cfg:
            damageable_class: object = fake_cls
            camera_name: str = "robot0_agentview_right"
            robot: str = "PandaOmron"
            control_freq: int = 20

        return _Cfg()

    def make_adapter(self, **kwargs):
        return self.adapter_module.RoboCasaAdapter(task_name="pick_egg", **kwargs)


class TestRegistryLookup(RoboCasaAdapterTestBase):
    def test_invalid_task_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.adapter_module.RoboCasaAdapter(task_name="not_a_real_task")

    def test_valid_task_name_constructs(self):
        adapter = self.make_adapter()
        self.assertIsInstance(adapter.env, _FakeEnv)


class TestCameraNamesDefaulting(RoboCasaAdapterTestBase):
    def test_default_camera_names_dedup(self):
        adapter = self.make_adapter()
        self.assertEqual(
            adapter.camera_names, ["robot0_agentview_right", "robot0_eye_in_hand"]
        )

    def test_explicit_camera_names_passed_through(self):
        adapter = self.make_adapter(camera_names=["custom_cam"])
        self.assertEqual(adapter.camera_names, ["custom_cam"])


class TestObsContract(RoboCasaAdapterTestBase):
    def test_reset_obs_keys(self):
        adapter = self.make_adapter()
        obs = adapter.reset()
        self.assertIn("agentview_image", obs)
        self.assertIn("wrist_image", obs)
        self.assertIn("eef_pos", obs)
        self.assertIn("eef_quat", obs)
        self.assertIn("gripper_qpos", obs)
        self.assertIn("health", obs)
        self.assertIn("raw", obs)
        np.testing.assert_array_equal(obs["eef_pos"], np.array([0.1, 0.2, 0.3]))
        self.assertEqual(obs["agentview_image"].shape, (256, 256, 3))

    def test_step_returns_expected_tuple(self):
        adapter = self.make_adapter(max_episode_steps=10)
        adapter.reset()
        obs, success, episode_done, info = adapter.step(np.zeros(7))
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(success, bool)
        self.assertFalse(success)
        self.assertFalse(episode_done)
        self.assertIn("damage_info", info)
        self.assertIn("health", info)
        np.testing.assert_array_equal(info["health"], obs["health"])

    def test_episode_done_on_max_steps(self):
        adapter = self.make_adapter(max_episode_steps=2)
        adapter.reset()
        _, _, done1, _ = adapter.step(np.zeros(7))
        self.assertFalse(done1)
        _, _, done2, _ = adapter.step(np.zeros(7))
        self.assertTrue(done2)

    def test_success_reflects_check_success(self):
        adapter = self.make_adapter()
        adapter.reset()
        adapter.env._success = True
        _, success, _, _ = adapter.step(np.zeros(7))
        self.assertTrue(success)


class TestMissingWristCameraAndHealthAttr(RoboCasaAdapterTestBase):
    fake_env_cls = _FakeEnvNoWristNoHealthAttr

    def test_wrist_image_key_omitted_when_missing(self):
        adapter = self.make_adapter()
        obs = adapter.reset()
        self.assertNotIn("wrist_image", obs)

    def test_get_health_summary_empty_without_attr(self):
        adapter = self.make_adapter()
        adapter.reset()
        self.assertEqual(adapter.get_health_summary(), {})


class TestHealthSummary(RoboCasaAdapterTestBase):
    def test_get_health_summary_zips_names_and_values(self):
        adapter = self.make_adapter()
        adapter.reset()
        summary = adapter.get_health_summary()
        self.assertEqual(summary, {"obj_a@link1": 1.0, "obj_b@link2": 1.0})

    def test_get_health_summary_updates_after_step(self):
        adapter = self.make_adapter()
        adapter.reset()
        adapter.step(np.zeros(7))
        summary = adapter.get_health_summary()
        self.assertAlmostEqual(summary["obj_a@link1"], 0.9, places=5)


class TestClose(RoboCasaAdapterTestBase):
    def test_close_calls_env_close(self):
        adapter = self.make_adapter()
        adapter.close()
        self.assertTrue(adapter.env.closed)

    def test_close_noop_when_env_has_no_close(self):
        registry_env_config = self._env_config_for(_FakeEnvNoClose)
        _install_stub_oopsiebench_registry({"pick_egg": registry_env_config})
        adapter = self.adapter_module.RoboCasaAdapter(task_name="pick_egg")
        # Should not raise even though _FakeEnvNoClose has no close().
        adapter.close()


class TestRobotOverridePassedThrough(RoboCasaAdapterTestBase):
    def test_robot_read_off_env_config_not_hardcoded(self):
        panda_cfg = self._env_config_for(_FakeEnv)
        panda_cfg.robot = "Panda"
        _install_stub_oopsiebench_registry({"shelve_item": panda_cfg})
        adapter = self.adapter_module.RoboCasaAdapter(task_name="shelve_item")
        self.assertEqual(adapter.env.init_kwargs["robots"], "Panda")


if __name__ == "__main__":
    unittest.main()
