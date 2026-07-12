"""Off-the-shelf Fast-WAM (libero_uncond_2cam224) rollout evaluation on
OopsieVerse's RoboCasa tasks.

Run inside the `oopsieverse_robocasa` conda env (this script needs the real
sim; Fast-WAM runs in a separate process/env, see ../servers/fastwam_server.py),
with a Fast-WAM server already listening:

    conda activate oopsieverse_robocasa
    export MUJOCO_GL=egl
    python oopsieverse_eval/eval/run_robocasa_fastwam_eval.py \
        --tasks pick_egg turn_on_faucet \
        --num_trials 3 \
        --server_url http://localhost:8300 \
        --results_dir oopsieverse_eval/results/fastwam_robocasa

Action-chunk execution (replan every `replan_steps` out of `action_horizon`
predicted) mirrors vlm_pipeline/run_safelibero_fastwam_eval.py's own
episode loop. The server already returns denormalized, gripper-flipped
(robosuite convention) actions -- this client just chunks/replans and wires
them into `build_env_action`.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import traceback

import imageio.v2 as imageio
import numpy as np
import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from robocasa_adapter import RoboCasaAdapter  # noqa: E402
from robocasa_action_utils import build_env_action  # noqa: E402


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Match Fast-WAM's own training-time proprio convention (see
    vlm_pipeline/run_safelibero_fastwam_eval.py's `_quat2axisangle`),
    reimplemented here (not imported) to keep this build independent."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den

ALL_TASKS = [
    "pick_egg", "serve_pastry", "open_single_door", "turn_on_faucet",
    "turn_on_microwave", "turn_on_stove", "open_drawer", "close_drawer",
    "place_plate", "counter_to_microwave", "prepare_coffee", "shelve_item",
    "prepare_breakfast", "dishes_to_sink", "nav_lift_bowl", "wipe_counter",
]


def act(server_url: str, primary_image, wrist_image, proprio, instruction: str,
        action_horizon: int, num_inference_steps: int) -> np.ndarray:
    import json_numpy
    payload = {
        "primary_image": primary_image,
        "wrist_image": wrist_image,
        "proprio": proprio,
        "instruction": instruction,
        "action_horizon": action_horizon,
        "num_inference_steps": num_inference_steps,
    }
    resp = requests.post(f"{server_url}/act", data=json_numpy.dumps(payload), timeout=60)
    resp.raise_for_status()
    return np.asarray(json_numpy.loads(resp.json()), dtype=np.float32)  # [T, 7]


def run_episode(
    adapter: RoboCasaAdapter, server_url: str, instruction: str,
    action_horizon: int, replan_steps: int, num_inference_steps: int, save_video_path: str,
) -> dict:
    obs = adapter.reset()
    robot = adapter.env.robots[0]
    action_plan: collections.deque = collections.deque()
    frames = []
    success = False
    n_steps = 0

    for _ in range(adapter.max_episode_steps):
        frame = obs.get("agentview_image")
        if frame is not None:
            frames.append(frame)

        if not action_plan:
            proprio = np.concatenate((
                obs["eef_pos"], quat2axisangle(obs["eef_quat"]), obs["gripper_qpos"][:2]
            )).astype(np.float32)
            action_chunk = act(
                server_url,
                primary_image=obs["agentview_image"],
                wrist_image=obs["wrist_image"],
                proprio=proprio,
                instruction=instruction,
                action_horizon=action_horizon,
                num_inference_steps=num_inference_steps,
            )
            action_plan.extend(action_chunk[:replan_steps])

        raw_action = action_plan.popleft()
        env_action = build_env_action(robot, raw_action[:6], float(raw_action[6]))
        obs, success, episode_done, info = adapter.step(env_action)
        n_steps += 1
        if episode_done:
            break

    if frames:
        os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
        imageio.mimsave(save_video_path, frames, fps=10)

    return {
        "success": bool(success),
        "num_steps": n_steps,
        "health_summary": adapter.get_health_summary(),
        "video_path": save_video_path if frames else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--max_episode_steps", type=int, default=400)
    parser.add_argument("--server_url", default="http://localhost:8300")
    parser.add_argument("--action_horizon", type=int, default=32)
    parser.add_argument("--replan_steps", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--results_dir", default=os.path.join(_HERE, "..", "results", "fastwam_robocasa"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    health = requests.get(f"{args.server_url}/health", timeout=10)
    health.raise_for_status()
    print(f"Fast-WAM server OK: {health.json()}")

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
                instruction = adapter.env.get_ep_meta()["lang"]
                video_path = os.path.join(args.results_dir, f"{task_name}_trial{trial}.mp4")
                record.update(run_episode(
                    adapter, args.server_url, instruction,
                    args.action_horizon, args.replan_steps, args.num_inference_steps, video_path,
                ))
                record["instruction"] = instruction
            except Exception:
                record["error"] = traceback.format_exc()
                print(record["error"])
            finally:
                adapter.close()
            all_results.append(record)

    results_path = os.path.join(args.results_dir, "episodes.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    n_ok = sum(1 for r in all_results if r["error"] is None)
    n_success = sum(1 for r in all_results if r.get("success"))
    print(f"\n=== SUMMARY: {n_ok}/{len(all_results)} episodes ran cleanly, {n_success} succeeded ===")
    print(f"Results written to {results_path}")
    sys.exit(0 if n_ok == len(all_results) else 1)


if __name__ == "__main__":
    main()
