"""Regression test: a DamageLookahead save/rollout/restore must not change the real episode.

Replays one fixed random action sequence twice on the same seed of a stock damageable task: once plain, once
with a lookahead (random actions) before every step. Robot qpos, eef pose, gripper state and tracked health
must match exactly. Runs on CPU (rendering off), in NVIDIA's RoboCasa client venv:
    python test_damage_lookahead.py [task]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")


def make(task):
    import gymnasium as gym
    from robocasa.utils.gym_utils.gymnasium_groot import create_grootrobocasa_env_class
    from run_groot_stock_damagesim import make_damageable_stock_env
    cls = make_damageable_stock_env(task)
    create_grootrobocasa_env_class(cls, "PandaOmron", "panda_omron")
    return gym.make(f"robocasa_panda_omron/{cls}_PandaOmron_Env", enable_render=False)


def rand_action(rng):
    return {"action.end_effector_position": rng.uniform(-1, 1, 3), "action.end_effector_rotation": rng.uniform(-0.3, 0.3, 3),
            "action.gripper_close": np.array([float(rng.random() > 0.5)]), "action.base_motion": np.zeros(4),
            "action.control_mode": np.array([0.0])}


def run(task, with_lookahead, steps=40, seed=3):
    from damage_shield import DamageLookahead, to_env_vector
    env = make(task)
    env.unwrapped.env.rng = np.random.default_rng(seed)  # robocasa samples the scene from env.rng
    env.reset(seed=seed)
    genv = env.unwrapped
    rs = genv.env
    rng = np.random.default_rng(0)
    la_rng = np.random.default_rng(1)
    traj = []
    for t in range(steps):
        a = rand_action(rng)
        if with_lookahead:
            la = DamageLookahead(genv)
            s = la.save()
            for _ in range(8):
                la.step(to_env_vector(genv, rand_action(la_rng)))
            la.restore(s)
        env.step(a)
        traj.append(np.concatenate([rs.sim.data.qpos.copy(), rs.sim.data.qvel.copy(),
                                    np.asarray(getattr(rs, "_last_health", np.zeros(0)), np.float64)]))
    env.close()
    return np.stack(traj)


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "turn_on_microwave"
    a, b = run(task, False), run(task, True)
    diff = np.abs(a - b).max(axis=1)
    first = int(np.argmax(diff > 1e-9)) if (diff > 1e-9).any() else None
    print(f"max |diff| = {diff.max():.3e}; first divergent step = {first}")
    print("PASS" if first is None else "FAIL")
    sys.exit(0 if first is None else 1)
