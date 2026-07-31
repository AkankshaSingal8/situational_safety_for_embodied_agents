#!/usr/bin/env python3
"""
BEHAVIOR-1K / OmniGibson headless-rendering risk-gate spike (Task 1 of the
enchanted-finding-piglet plan, Phase 5).

Purpose: prove -- under `sbatch` on a Bridges2 GPU-shared node, with NO
interactive display attached -- that OmniGibson (Isaac Sim's Kit-based
renderer) can:
  1. boot headless,
  2. instantiate a real OopsieVerse BEHAVIOR-1K task
     (`OGDamageableEnvironment`, same construction path as
     `scripts/teleop_b1k.py`),
  3. `env.reset()` and run a short zero/no-op action loop via `env.step()`,
  4. render and save a real camera RGB frame to disk.

This is a GO/NO-GO gate, not a policy/adapter integration -- no VLA, no
learned policy, zero action every step.

Headless mechanism (verified directly against
UT-Austin-RobIn/BEHAVIOR-1K@proj/safemanibench's
`OmniGibson/omnigibson/macros.py`, not guessed):

    gm.HEADLESS = os.getenv("OMNIGIBSON_HEADLESS", "False").lower() in ("true", "1", "t")

`gm` (`omnigibson.macros.gm`) is read once at `import omnigibson` time
(before the Kit `SimulationApp` boots), so the `OMNIGIBSON_HEADLESS` env
var MUST be set before `import omnigibson` -- see this script's imports
below. `gm` is a `MacroDict` that locks a key after its first read, so
re-assigning `gm.HEADLESS` after import raises `AttributeError` -- do not
try to set it programmatically after import.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

os.environ.setdefault("CARB_LOG_CHANNELS", "omni.physx.plugin=off")
# Headless mechanism (see module docstring + report for the confirmed
# citation from the installed omnigibson/macros.py): set before `import
# omnigibson` so gm.HEADLESS is already True when the Kit SimulationApp is
# constructed.
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "oopsieverse"))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="Zero-action BEHAVIOR-1K headless spike.")
    p.add_argument("--task_name", type=str, default="pick_egg",
                    help="Task module name under oopsiebench/envs/behavior1k/ (default: pick_egg).")
    p.add_argument("--n_steps", type=int, default=30,
                    help="Number of zero-action env.step() calls (default: 30, brief asks for 20-50).")
    p.add_argument("--out_dir", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "results", "b1k_headless_spike"),
                    help="Directory to save the camera RGB PNG + summary into.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[spike] CARB_LOG_CHANNELS={os.environ.get('CARB_LOG_CHANNELS')}")
    print(f"[spike] OMNIGIBSON_HEADLESS={os.environ.get('OMNIGIBSON_HEADLESS')}")
    print(f"[spike] task_name={args.task_name} n_steps={args.n_steps} out_dir={out_dir}")

    import importlib
    import numpy as np
    import torch as th

    import omnigibson as og
    from omnigibson.macros import gm

    # Confirmed mechanism (OmniGibson/omnigibson/macros.py, ~line 133 in the
    # UT-Austin-RobIn/BEHAVIOR-1K@proj/safemanibench fork):
    #   gm.HEADLESS = os.getenv("OMNIGIBSON_HEADLESS", "False").lower() in ("true", "1", "t")
    # gm is a MacroDict that locks each key after its first *read* -- and
    # `import omnigibson` itself reads gm.HEADLESS while booting the Kit
    # SimulationApp, so by the time we get here the value is already locked
    # in from the OMNIGIBSON_HEADLESS env var set above (before import).
    # Do NOT try to re-set gm.HEADLESS here -- MacroDict raises
    # AttributeError on writes to an already-read, locked key.
    print(f"[spike] omnigibson.macros.gm.HEADLESS = {gm.HEADLESS}")

    from damagesim.omnigibson.damageable_env import OGDamageableEnvironment

    # Reuse teleop_b1k.py's exact task-config -> env-config construction
    # pattern (imported, not reimplemented, to avoid guessing constructor args).
    sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
    import teleop_b1k as tb1k  # noqa: E402

    task_cfg, task_mod = tb1k.load_task_config(args.task_name)
    gm.USE_GPU_DYNAMICS = task_cfg.use_gpu_dynamics
    gm.ENABLE_TRANSITION_RULES = task_cfg.enable_transition_rules

    env_config = tb1k.build_env_config(task_cfg)
    # Always include external cameras for this spike -- we need a real RGB
    # frame regardless of whether HDF5 saving is requested.
    if getattr(task_cfg, "external_camera_configs", None):
        env_config["env"]["external_sensors"] = tb1k.build_external_sensors_config(
            task_cfg, task_cfg.robot_name, task_cfg.robot_type,
            image_height=512, image_width=512,
        )

    print("[spike] Constructing OGDamageableEnvironment...")
    env = OGDamageableEnvironment(configs=env_config)
    print("[spike] Environment constructed OK.")

    robot = env.robots[0]

    print("[spike] Calling env.reset()...")
    obs, info = env.reset()
    print("[spike] env.reset() OK.")

    print("[spike] obs dict keys (top level):", list(obs.keys()))

    def summarize(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                summarize(f"{prefix}{k}.", v)
            else:
                shape = getattr(v, "shape", None)
                dtype = getattr(v, "dtype", type(v).__name__)
                print(f"[spike]   {prefix}{k}: shape={shape} dtype={dtype}")

    summarize("", obs)

    zero_action = th.zeros(robot.action_dim, dtype=th.float32)
    print(f"[spike] Running {args.n_steps} zero-action steps (action_dim={robot.action_dim})...")
    last_obs = obs
    for i in range(args.n_steps):
        last_obs, reward, terminated, truncated, step_info = env.step(zero_action.clone())
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[spike]   step {i + 1}/{args.n_steps} OK "
                  f"(terminated={terminated}, truncated={truncated})")

    print("[spike] Zero-action loop completed with no exception.")

    # ── Extract + save a real camera RGB frame ──────────────────────────
    frame = tb1k.capture_rgb(camera=tb1k.VIDEO_CAMERA_NAME, env=env)
    print(f"[spike] Captured frame from '{tb1k.VIDEO_CAMERA_NAME}': "
          f"shape={frame.shape} dtype={frame.dtype} "
          f"min={frame.min()} max={frame.max()} mean={frame.mean():.2f}")

    from PIL import Image
    png_path = os.path.join(out_dir, f"{args.task_name}_spike_frame.png")
    Image.fromarray(frame).save(png_path)
    print(f"[spike] Saved camera RGB frame to: {png_path}")

    is_blank = bool(frame.std() < 1.0)
    print(f"[spike] Frame std-dev={frame.std():.3f} -> "
          f"{'LOOKS BLANK/UNIFORM (suspicious)' if is_blank else 'has real variance (looks like a render)'}")

    summary_path = os.path.join(out_dir, f"{args.task_name}_spike_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"task_name={args.task_name}\n")
        f.write(f"n_steps={args.n_steps}\n")
        f.write(f"obs_top_level_keys={list(obs.keys())}\n")
        f.write(f"frame_shape={frame.shape}\n")
        f.write(f"frame_dtype={frame.dtype}\n")
        f.write(f"frame_min={frame.min()} frame_max={frame.max()} frame_mean={frame.mean():.3f} frame_std={frame.std():.3f}\n")
        f.write(f"frame_looks_blank={is_blank}\n")
        f.write("NO EXCEPTION RAISED -- reset() + step() x n_steps + camera capture all succeeded.\n")
    print(f"[spike] Wrote summary to: {summary_path}")

    og.shutdown()
    print("[spike] og.shutdown() OK. SPIKE PASSED.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[spike] SPIKE FAILED WITH EXCEPTION:")
        traceback.print_exc()
        sys.exit(1)
