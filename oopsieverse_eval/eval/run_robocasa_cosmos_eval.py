"""Off-the-shelf Cosmos-Policy-RoboCasa rollout evaluation on OopsieVerse's
RoboCasa tasks.

Run inside the `oopsieverse_robocasa` conda env (this script needs the real
sim; Cosmos-Policy runs in a separate process inside its own Apptainer
container, see ../servers/launch_cosmos_server.sh), with a Cosmos server
already listening:

    conda activate oopsieverse_robocasa
    export MUJOCO_GL=egl
    python oopsieverse_eval/eval/run_robocasa_cosmos_eval.py \
        --tasks pick_egg turn_on_faucet \
        --num_trials 3 \
        --host localhost --port 8400 \
        --results_dir oopsieverse_eval/results/cosmos_robocasa

Uses the same stdlib-only pickle-over-TCP protocol as cosmos_server.py (no
new client-side deps needed either). Cosmos-Policy-RoboCasa was trained
directly on RoboCasa/robosuite action data, so -- unlike OpenVLA -- its
output actions are already in robosuite's native -1=open/+1=close, delta-
pose convention; no gripper conversion needed, same as Fast-WAM/pi0.5.

CAVEAT (see plan's "Extension: Cosmos-Policy + Fast-WAM" section): this
checkpoint's 24 training tasks are standard RoboCasa gym-registered tasks,
which mostly do NOT correspond to OopsieVerse's custom Kitchen-subclass
tasks (even same-named ones differ, e.g. Cosmos's generic OpenSingleDoor vs
OopsieVerse's microwave-door task). This is a fair off-the-shelf, same-
simulator-family baseline -- NOT an in-distribution one. Report results
accordingly.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import socket
import struct
import sys
import traceback

import imageio.v2 as imageio
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from robocasa_adapter import RoboCasaAdapter  # noqa: E402
from robocasa_action_utils import build_env_action, json_default  # noqa: E402

# NOTE: shelve_item excluded -- requires a one-time interactive camera
# calibration file (resources/camera_states/shelve_item_1.json) that
# OopsieVerse does not ship (see smoke_test_robocasa.py's original
# exclusion of it, and the FileNotFoundError this produces if attempted).
ALL_TASKS = [
    "pick_egg", "serve_pastry", "open_single_door", "turn_on_faucet",
    "turn_on_microwave", "turn_on_stove", "open_drawer", "close_drawer",
    "place_plate", "counter_to_microwave", "prepare_coffee",
    "prepare_breakfast", "dishes_to_sink", "nav_lift_bowl", "wipe_counter",
]

# Cosmos-Policy-RoboCasa hard-requires both third-person cameras
# (agentview_left=primary, agentview_right=secondary) plus the wrist cam.
CAMERA_NAMES = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed while reading")
        buf += chunk
    return buf


def _recv_msg(sock: socket.socket) -> dict:
    (length,) = struct.unpack(">I", _recv_exact(sock, 4))
    return pickle.loads(_recv_exact(sock, length))


def _send_msg(sock: socket.socket, obj: dict) -> None:
    payload = pickle.dumps(obj)
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def act(host: str, port: int, primary_image, secondary_image, wrist_image, proprio, instruction: str) -> np.ndarray:
    with socket.create_connection((host, port), timeout=120) as sock:
        _send_msg(sock, {
            "primary_image": primary_image,
            "secondary_image": secondary_image,
            "wrist_image": wrist_image,
            "proprio": proprio,
            "instruction": instruction,
        })
        response = _recv_msg(sock)
    if "error" in response:
        raise RuntimeError(f"Cosmos server error:\n{response['error']}")
    return np.asarray(response["action"], dtype=np.float32)  # [T, 7]


def health_check(host: str, port: int) -> dict:
    with socket.create_connection((host, port), timeout=30) as sock:
        _send_msg(sock, {"_health_check": True})
        return _recv_msg(sock)


def build_proprio(obs: dict) -> np.ndarray:
    """9-dim proprio Cosmos-Policy-RoboCasa expects: gripper_qpos(2) + eef_pos(3) + eef_quat(4)."""
    return np.concatenate((obs["gripper_qpos"][:2], obs["eef_pos"], obs["eef_quat"])).astype(np.float32)


def run_episode(
    adapter: RoboCasaAdapter, host: str, port: int, num_open_loop_steps: int, save_video_path: str,
) -> dict:
    obs = adapter.reset()
    instruction = adapter.get_instruction()
    robot = adapter.env.robots[0]
    action_plan: collections.deque = collections.deque()
    frames = []
    success = False
    n_steps = 0

    for _ in range(adapter.max_episode_steps):
        frame = obs.get("agentview_image")  # agentview_right (RoboCasaAdapter's default camera_name)
        if frame is not None:
            frames.append(frame)

        if not action_plan:
            # NOTE: Cosmos-Policy-RoboCasa's own convention is
            # primary=agentview_left, secondary=agentview_right (per
            # ROBOCASA.md) -- but RoboCasaAdapter's "agentview_image" key
            # maps to the registry's default camera_name (agentview_RIGHT),
            # and "secondary_image" maps to agentview_LEFT. Swap here so the
            # server sees the labels it actually expects.
            action_chunk = act(
                host, port,
                primary_image=obs["secondary_image"],   # agentview_left
                secondary_image=obs["agentview_image"],  # agentview_right
                wrist_image=obs["wrist_image"],
                proprio=build_proprio(obs),
                instruction=instruction,
            )
            action_plan.extend(action_chunk[:num_open_loop_steps])

        raw_action = action_plan.popleft()  # already in robosuite convention, no gripper flip
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
        "instruction": instruction,
        "health_summary": adapter.get_health_summary(),
        "video_path": save_video_path if frames else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--max_episode_steps", type=int, default=400)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8400)
    parser.add_argument("--num_open_loop_steps", type=int, default=16)
    parser.add_argument("--results_dir", default=os.path.join(_HERE, "..", "results", "cosmos_robocasa"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    health = health_check(args.host, args.port)
    print(f"Cosmos server OK: {health}")

    all_results = []
    for task_name in args.tasks:
        for trial in range(args.num_trials):
            episode_seed = args.seed + trial
            print(f"=== {task_name} trial {trial} (seed={episode_seed}) ===")
            record = {"task_name": task_name, "trial": trial, "seed": episode_seed, "error": None}
            adapter = RoboCasaAdapter(
                task_name=task_name, camera_names=CAMERA_NAMES,
                max_episode_steps=args.max_episode_steps, seed=episode_seed,
            )
            try:
                video_path = os.path.join(args.results_dir, f"{task_name}_trial{trial}.mp4")
                record.update(run_episode(adapter, args.host, args.port, args.num_open_loop_steps, video_path))
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
