"""Off-the-shelf pi0.5 (pi05_libero) rollout evaluation on OopsieVerse's
RoboCasa tasks.

Run inside the `oopsieverse_robocasa` conda env (this script needs the real
sim; pi0.5 runs in a separate process/env via openpi's own serve_policy.py,
see ../servers/launch_pi05_server.sh), with a pi0.5 server already listening:

    conda activate oopsieverse_robocasa
    export MUJOCO_GL=egl
    python oopsieverse_eval/eval/run_robocasa_pi05_eval.py \
        --tasks pick_egg turn_on_faucet \
        --num_trials 3 \
        --host localhost --port 8200 \
        --results_dir oopsieverse_eval/results/pi05_robocasa

Observation schema and action-chunk execution (replan every `replan_steps`
out of the predicted chunk) mirror openpi's own examples/libero/main.py
client, since pi05_libero was trained on that exact schema:
  "observation/image", "observation/wrist_image", "observation/state"
  (eef_pos(3) + axis-angle(quat)(3) + gripper_qpos(2), 8-dim), "prompt".

Unlike OpenVLA's raw output (BridgeData/RLDS gripper convention, needs a
flip), pi05_libero was trained directly on LIBERO/robosuite action data, so
its output actions are already in the -1=open/+1=close, normalized-delta
robosuite convention `build_env_action` expects -- no gripper conversion
needed here.

NOTE on image orientation: openpi's own LIBERO example applies a 180-degree
rotation (`obs[...][::-1, ::-1]`) to compensate for a LIBERO-specific
render/training-data quirk. OopsieVerse's RoboCasaAdapter already applies
its own single vertical flip internally to match "standard robosuite
convention" (see damagesim/robosuite/damageable_env.py), so this script does
NOT apply LIBERO's extra flip by default -- applying an unverified
transform on top of an already-corrected image is more likely to hurt than
help. If rollouts look systematically confused in ways that suggest an
orientation mismatch, inspect saved videos and reconsider.
"""

from __future__ import annotations

import argparse
import collections
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

from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy as _websocket_client_policy  # noqa: E402
from openpi_client import msgpack_numpy  # noqa: E402
import websockets.sync.client  # noqa: E402


class NoPingWebsocketClientPolicy(_websocket_client_policy.WebsocketClientPolicy):
    """WebsocketClientPolicy, but with keepalive pings disabled.

    pi0.5's server does JAX/XLA JIT compilation on its first inference call,
    which routinely takes well over `websockets`' default 20s ping_timeout
    (observed: connections were force-closed with "keepalive ping timeout"
    on every single episode's first infer() call). Since this is a
    long-lived local script talking to a single dedicated server process
    (not a shared production service), disabling pings entirely is safe and
    avoids re-implementing the whole class just to change one constructor
    kwarg -- `_wait_for_server` is the only method that constructs the
    connection, so only it needs overriding.
    """

    def _wait_for_server(self):
        import logging
        import time

        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=None,
                    open_timeout=None,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)


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

RESIZE_SIZE = 224


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert a wxyz/xyzw-ambiguous robosuite quat to an axis-angle vector.

    Reimplemented (not imported) from the standard LIBERO/openvla-oft
    utility to keep this build independent; robosuite quats are (x,y,z,w).
    """
    quat = np.asarray(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def build_observation(obs: dict, instruction: str) -> dict:
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(obs["agentview_image"], RESIZE_SIZE, RESIZE_SIZE)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(obs["wrist_image"], RESIZE_SIZE, RESIZE_SIZE)
    )
    state = np.concatenate(
        (obs["eef_pos"], quat2axisangle(obs["eef_quat"]), obs["gripper_qpos"])
    )
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(instruction),
    }


def run_episode(
    adapter: RoboCasaAdapter, client, replan_steps: int, save_video_path: str
) -> dict:
    obs = adapter.reset()
    # NOTE: get_ep_meta() must be called AFTER reset() -- RoboCasa's base
    # Kitchen.get_ep_meta() reads attributes (object_cfgs, layout_id) only
    # populated during reset()/_reset_internal().
    instruction = adapter.get_instruction()
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
            element = build_observation(obs, instruction)
            action_chunk = client.infer(element)["actions"]
            action_plan.extend(action_chunk[:replan_steps])

        raw_action = np.asarray(action_plan.popleft(), dtype=np.float32)  # 7-dim, robosuite convention
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
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--results_dir", default=os.path.join(_HERE, "..", "results", "pi05_robocasa"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    client = NoPingWebsocketClientPolicy(args.host, args.port)
    print(f"pi0.5 server metadata: {client.get_server_metadata()}")

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
                record.update(run_episode(adapter, client, args.replan_steps, video_path))
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
