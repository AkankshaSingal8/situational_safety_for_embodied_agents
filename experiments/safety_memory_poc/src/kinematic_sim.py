"""A minimal kinematic stand-in with the same interface as LIBERO's ControlEnv.

Purpose: validate the full experimental machinery (runner, filter, ledger,
memory, analysis) end to end without a simulator dependency, and provide a
fallback board if the MuJoCo path is unavailable.

It is NOT a substitute for the physics result. It deliberately includes
first-order actuator lag and noise so the filter is not exact by construction.
Object layout is randomised per seed, which is what makes pose-invariance of the
memory a real property rather than an artefact of a fixed scene.
"""
from __future__ import annotations

import numpy as np


class KinematicEnv:
    OBJECTS = ("plate", "raw_source", "lookalike_source", "sink")

    def __init__(self, action_scale: float = 0.05, alpha: float = 0.6,
                 noise: float = 0.0015, workspace: float = 0.35):
        self.action_scale = action_scale
        self.alpha = alpha
        self.noise = noise
        self.workspace = workspace
        self._seed = 0
        self._rng = np.random.default_rng(0)
        self._pos = np.zeros(3)
        self._u_prev = np.zeros(3)
        self._obj = {}

    def seed(self, s: int):
        self._seed = int(s)
        return [s]

    def reset(self):
        rng = np.random.default_rng(self._seed)
        self._rng = np.random.default_rng(self._seed + 99991)
        w = self.workspace
        # Objects are placed on a table plane with per-seed jitter.
        base = {
            "plate":            np.array([0.10, 0.00, 0.85]),
            "raw_source":       np.array([-0.05, 0.22, 0.85]),
            "lookalike_source": np.array([-0.05, -0.22, 0.85]),
            "sink":             np.array([0.22, 0.20, 0.85]),
        }
        self._obj = {k: v + rng.uniform(-0.05, 0.05, 3) * np.array([1, 1, 0.2])
                     for k, v in base.items()}
        self._pos = np.array([-0.20, 0.0, 1.00]) + rng.uniform(-0.03, 0.03, 3)
        self._u_prev = np.zeros(3)
        return self._obs()

    def _obs(self):
        o = {"robot0_eef_pos": self._pos.copy()}
        for k, v in self._obj.items():
            o[f"{k}_pos"] = v.copy()
        return o

    def step(self, action):
        u = np.clip(np.asarray(action, dtype=float)[:3], -1.0, 1.0)
        eff = self.alpha * u + (1.0 - self.alpha) * self._u_prev
        self._u_prev = u
        self._pos = self._pos + self.action_scale * eff
        if self.noise > 0:
            self._pos = self._pos + self._rng.normal(0, self.noise, 3)
        return self._obs(), 0.0, False, {}

    def close(self):
        pass
