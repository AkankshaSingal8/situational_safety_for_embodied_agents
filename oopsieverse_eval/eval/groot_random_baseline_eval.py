"""Random-action baseline for the GR00T-N1.6 risk-gate spike's motion-
degeneracy check (task step 6) -- see run_robocasa_groot_eval.py and
motion_stats.py. Same task/seed/episode-length/action-chunking shape as the
real GR00T client, but samples `arm_delta`/`gripper` uniformly at random
instead of querying a policy. No GR00T server needed.

Run inside the `oopsieverse_robocasa` conda env:

    conda activate oopsieverse_robocasa
    export MUJOCO_GL=egl
    python oopsieverse_eval/eval/groot_random_baseline_eval.py \
        --tasks pick_egg --num_trials 3 \
        --results_dir oopsieverse_eval/results/groot_robocasa_random_baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import imageio.v2 as imageio
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from robocasa_adapter import RoboCasaAdapter  # noqa: E402
from robocasa_action_utils import build_env_action, json_default  # noqa: E402
from motion_stats import eef_total_displacement, mean_frame_diff  # noqa: E402
from run_robocasa_groot_eval import CAMERA_NAMES, ALL_TASKS  # noqa: E402


def run_random_episode(adapter: RoboCasaAdapter, rng: np.random.Generator, save_video_path: str) -> dict:
    obs = adapter.reset()
    instruction = adapter.get_instruction()
    robot = adapter.env.robots[0]
    frames = []
    eef_positions = [np.asarray(obs["eef_pos"], dtype=np.float64)]
    success = False
    n_steps = 0

    for _ in range(adapter.max_episode_steps):
        # Same magnitude range as a plausible per-step OSC delta command --
        # uniform in [-1, 1] for the 6-dim arm_delta, gripper in {-1, +1}.
        arm_delta = rng.uniform(-1.0, 1.0, size=6)
        gripper = float(rng.choice([-1.0, 1.0]))
        env_action = build_env_action(robot, arm_delta, gripper)
        obs, success, episode_done, info = adapter.step(env_action)
        n_steps += 1
        eef_positions.append(np.asarray(obs["eef_pos"], dtype=np.float64))

        frame = obs.get("agentview_image")
        if frame is not None:
            frames.append(frame)
        if episode_done:
            break

    if frames:
        os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
        imageio.mimsave(save_video_path, frames, fps=10)

    return {
        "success": bool(success),
        "num_steps": n_steps,
        "instruction": instruction,
        "health_summary": adapter.get_health_summary(),
        "video_path": save_video_path if frames else None,
        "eef_total_displacement": eef_total_displacement(eef_positions),
        "mean_frame_diff": mean_frame_diff(frames),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--max_episode_steps", type=int, default=400)
    parser.add_argument(
        "--results_dir",
        default=os.path.join(_HERE, "..", "results", "groot_robocasa_random_baseline"),
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    all_results = []
    for task_name in args.tasks:
        for trial in range(args.num_trials):
            episode_seed = args.seed + trial
            print(f"=== [RANDOM BASELINE] {task_name} trial {trial} (seed={episode_seed}) ===")
            record = {"task_name": task_name, "trial": trial, "seed": episode_seed, "error": None}
            adapter = RoboCasaAdapter(
                task_name=task_name,
                camera_names=CAMERA_NAMES,
                max_episode_steps=args.max_episode_steps,
                seed=episode_seed,
            )
            rng = np.random.default_rng(episode_seed)
            try:
                video_path = os.path.join(args.results_dir, f"{task_name}_trial{trial}.mp4")
                record.update(run_random_episode(adapter, rng, video_path))
            except Exception:
                record["error"] = traceback.format_exc()
                print(record["error"])
            finally:
                adapter.close()
            all_results.append(record)

    results_path = os.path.join(args.results_dir, "episodes.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=json_default)

    n_ok = sum(1 for r in all_results if r["error"] is None)
    print(f"\n=== SUMMARY: {n_ok}/{len(all_results)} random-baseline episodes ran cleanly ===")
    print(f"Results written to {results_path}")
    sys.exit(0 if n_ok == len(all_results) else 1)


if __name__ == "__main__":
    main()
