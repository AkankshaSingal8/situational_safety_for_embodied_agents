"""Off-the-shelf OpenVLA rollout evaluation on OopsieVerse's RoboCasa tasks.

Run inside the `oopsieverse_robocasa` conda env (this script needs the real
sim, not the policy -- the OpenVLA model runs in a separate process/env, see
../servers/openvla_server.py), with an OpenVLA server already listening:

    conda activate oopsieverse_robocasa
    export MUJOCO_GL=egl
    python oopsieverse_eval/eval/run_robocasa_openvla_eval.py \
        --tasks pick_egg turn_on_faucet \
        --num_trials 3 \
        --server_url http://localhost:8100 \
        --results_dir oopsieverse_eval/results/openvla_robocasa
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import imageio.v2 as imageio
import numpy as np
import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from robocasa_adapter import RoboCasaAdapter  # noqa: E402
from robocasa_action_utils import build_env_action, json_default, openvla_gripper_to_robocasa  # noqa: E402

ALL_TASKS = [
    "pick_egg", "serve_pastry", "open_single_door", "turn_on_faucet",
    "turn_on_microwave", "turn_on_stove", "open_drawer", "close_drawer",
    "place_plate", "counter_to_microwave", "prepare_coffee", "shelve_item",
    "prepare_breakfast", "dishes_to_sink", "nav_lift_bowl", "wipe_counter",
]


def act(server_url: str, image: np.ndarray, instruction: str) -> np.ndarray:
    payload = {"image": image, "instruction": instruction}
    import json_numpy
    resp = requests.post(f"{server_url}/act", data=json_numpy.dumps(payload), timeout=30)
    resp.raise_for_status()
    return np.asarray(json_numpy.loads(resp.json()), dtype=np.float32)


def run_episode(adapter: RoboCasaAdapter, server_url: str, save_video_path: str) -> dict:
    obs = adapter.reset()
    # NOTE: get_ep_meta() must be called AFTER reset() -- RoboCasa's base
    # Kitchen.get_ep_meta() reads attributes (object_cfgs, layout_id) that
    # are only populated during reset()/_reset_internal(), and raises
    # AttributeError if called on a freshly-constructed, not-yet-reset env.
    instruction = adapter.env.get_ep_meta()["lang"]
    robot = adapter.env.robots[0]
    frames = []
    success = False
    n_steps = 0

    for _ in range(adapter.max_episode_steps):
        image = obs["agentview_image"]
        raw_action = act(server_url, image, instruction)  # 7-dim: dx,dy,dz,drx,dry,drz,gripper
        arm_delta = raw_action[:6]
        gripper = openvla_gripper_to_robocasa(float(raw_action[6]))

        env_action = build_env_action(robot, arm_delta, gripper)
        obs, success, episode_done, info = adapter.step(env_action)
        n_steps += 1

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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--max_episode_steps", type=int, default=400)
    parser.add_argument("--server_url", default="http://localhost:8100")
    parser.add_argument("--results_dir", default=os.path.join(_HERE, "..", "results", "openvla_robocasa"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Fail fast with a clear message if the server isn't up, rather than
    # letting every episode's first `act()` call raise a generic connection error.
    health = requests.get(f"{args.server_url}/health", timeout=10)
    health.raise_for_status()
    print(f"OpenVLA server OK: {health.json()}")

    all_results = []
    for task_name in args.tasks:
        for trial in range(args.num_trials):
            episode_seed = args.seed + trial
            print(f"=== {task_name} trial {trial} (seed={episode_seed}) ===")
            record = {"task_name": task_name, "trial": trial, "seed": episode_seed, "error": None}
            adapter = RoboCasaAdapter(
                task_name=task_name, max_episode_steps=args.max_episode_steps, seed=episode_seed
            )
            try:
                video_path = os.path.join(args.results_dir, f"{task_name}_trial{trial}.mp4")
                record.update(run_episode(adapter, args.server_url, video_path))
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
    n_success = sum(1 for r in all_results if r.get("success"))
    print(f"\n=== SUMMARY: {n_ok}/{len(all_results)} episodes ran cleanly, {n_success} succeeded ===")
    print(f"Results written to {results_path}")
    sys.exit(0 if n_ok == len(all_results) else 1)


if __name__ == "__main__":
    main()
