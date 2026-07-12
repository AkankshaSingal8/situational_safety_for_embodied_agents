"""Random-action smoke test for RoboCasaAdapter against the real OopsieVerse RoboCasa envs.

Run inside the oopsieverse_robocasa conda env:
    conda activate oopsieverse_robocasa
    MUJOCO_GL=egl python oopsieverse_eval/eval/smoke_test_robocasa.py
"""
import os
import sys
import traceback

import imageio.v2 as imageio
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from robocasa_adapter import RoboCasaAdapter  # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "..", "results", "smoke_test")

# Representative subset covering different damageable-object profiles.
#
# NOTE: shelve_item (the one task with robot/camera overrides - single-arm Panda +
# world-fixed cameras) is intentionally excluded here. Its world-fixed cameras require a
# one-time interactive calibration file (resources/camera_states/shelve_item_1.json,
# saved via teleop_robocasa.py's "P" key) that is gitignored and not shipped with the
# OopsieVerse repo - see oopsiebench/envs/robocasa/shelve_item.py:36. This is a real
# per-installation setup step for that specific task, not a RoboCasaAdapter bug; its
# robot/camera-override handling is still covered by RoboCasaAdapter's unit tests
# (Task 3) even though it isn't exercised end-to-end here.
TASKS = ["pick_egg", "place_plate", "turn_on_faucet", "nav_lift_bowl"]

NUM_STEPS = 30


def run_task(task_name: str) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {"task_name": task_name, "error": None}

    adapter = RoboCasaAdapter(task_name=task_name, max_episode_steps=NUM_STEPS)
    try:
        obs = adapter.reset()
        result["obs_keys"] = sorted(k for k, v in obs.items() if k != "raw")

        action_dim = adapter.env.action_dim
        frames = []
        any_success = False

        for _ in range(NUM_STEPS):
            action = np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
            obs, success, episode_done, info = adapter.step(action)
            any_success = any_success or success
            frame = obs.get("agentview_image")
            if frame is not None:
                frames.append(frame)
            if episode_done:
                break

        video_path = os.path.join(RESULTS_DIR, f"{task_name}.mp4")
        if frames:
            imageio.mimsave(video_path, frames, fps=10)
            result["video_path"] = video_path
            result["video_bytes"] = os.path.getsize(video_path)
        else:
            result["video_path"] = None
            result["video_bytes"] = 0

        result["any_success"] = any_success
        result["health_summary"] = adapter.get_health_summary()
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        adapter.close()

    return result


def main():
    results = []
    for task_name in TASKS:
        print(f"=== {task_name} ===")
        r = run_task(task_name)
        results.append(r)
        if r["error"]:
            print(f"FAIL: {task_name}\n{r['error']}")
        else:
            print(f"obs keys: {r['obs_keys']}")
            print(f"any_success during rollout: {r['any_success']}")
            print(f"video: {r['video_path']} ({r['video_bytes']} bytes)")
            print(f"health summary (sample): {dict(list(r['health_summary'].items())[:3])}")

    print("\n=== SUMMARY ===")
    all_pass = True
    for r in results:
        status = "PASS" if (r["error"] is None and r.get("video_bytes", 0) > 1000) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"{r['task_name']}: {status}")

    print("OVERALL:", "PASS" if all_pass else "FAIL")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
