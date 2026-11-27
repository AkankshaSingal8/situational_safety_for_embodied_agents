"""Off-the-shelf GR00T-N1.6 rollout evaluation on OopsieVerse's RoboCasa
tasks (high-risk risk-gate spike -- see ../results/groot_risk_gate_report.md).

Run inside the `oopsieverse_robocasa` conda env (this script needs the real
sim; GR00T runs in a separate process, see ../servers/groot_server.py), with
a groot_server.py already listening:

    conda activate oopsieverse_robocasa
    export MUJOCO_GL=egl
    python oopsieverse_eval/eval/run_robocasa_groot_eval.py \
        --tasks pick_egg --num_trials 5 \
        --server_host 127.0.0.1 --server_port 5555 \
        --results_dir oopsieverse_eval/results/groot_robocasa

ACTION-SPACE ADAPTATION (the core research risk of this spike -- see the
report for the full primary-source trail): GR00T's `ROBOCASA_PANDA_OMRON`
embodiment predicts `end_effector_position` (3) + `end_effector_rotation`
(3, axis-angle-shaped) + `gripper_close` (1) + `base_motion` (4) +
`control_mode` (1). Despite `processor_config.json` tagging these
`rep: "ABSOLUTE"`, `statistics.json` shows `action.end_effector_position`
has mean~0, std~0.42 (i.e. small per-step deltas, NOT raw workspace
coordinates like `state.end_effector_position_relative`'s mean~[0.27,
-0.04, 0.54]) -- consistent with OSC-delta-style per-step commands, the
same convention `robocasa_action_utils.build_env_action()` expects for its
6-dim `arm_delta`. `base_motion`/`control_mode` have mean=std=0 in training
statistics -- i.e. NVIDIA's own panda_omron training data never moved the
base either, so zeroing them here (mirroring this benchmark's existing
OpenVLA/pi0.5 "base held still" convention) matches the training
distribution rather than fighting it.

This mapping is a best-effort reading of primary sources (processor_config
.json + statistics.json + gymnasium_groot.py), NOT verified against a real
GR00T rollout in this spike (checkpoint load / degeneracy check was done
via groot_risk_gate_spike.py without a sim in the loop) -- treat the exact
scale factor between GR00T's normalized output and robosuite's [-1,1] OSC
range as the first thing to empirically check if actions look too small
or too large.
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
from groot_zmq_client import GrootZmqClient  # noqa: E402
from run_robocasa_pi05_eval import robocasa365_to_env_action  # noqa: E402
from motion_stats import eef_total_displacement, mean_frame_diff  # noqa: E402

# 3 cameras required (see groot_server.py docstring): agentview_right (the
# adapter's default primary), agentview_left (secondary -- must be
# explicitly requested, same accommodation already used by Cosmos-Policy),
# and eye_in_hand (always populated by the adapter when present).
CAMERA_NAMES = ["robot0_agentview_right", "robot0_agentview_left", "robot0_eye_in_hand"]

ALL_TASKS = [
    "pick_egg", "serve_pastry", "open_single_door", "turn_on_faucet",
    "turn_on_microwave", "turn_on_stove", "open_drawer", "close_drawer",
    "place_plate", "counter_to_microwave", "prepare_coffee",
    "prepare_breakfast", "dishes_to_sink", "nav_lift_bowl", "wipe_counter",
]


def build_groot_obs(obs: dict, instruction: str) -> dict:
    """Build a `Gr00tPolicy.get_action()`-shaped observation, batch=1, time=1."""

    # NVIDIA's eval env (squarefk/robocasa@d89d481, robocasa/models/robots/__init__.py
    # PandaOmronKeyConverter + gym_utils/gymnasium_groot.py): side_0 = robot0_agentview_LEFT,
    # side_1 = robot0_agentview_RIGHT, wrist_0 = eye_in_hand; each rendered at 512x512, flipped
    # once (DamageableEnv applies the same [::-1]), then cv2.resize(..., (256, 256), INTER_AREA).
    # The previous mapping had side_0/side_1 swapped and rendered at 256 directly.
    import cv2

    raw = obs["raw"]

    def img(cam):
        arr = raw.get(f"{cam}_image")
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.uint8)
        if arr.shape[:2] != (256, 256):
            arr = cv2.resize(arr, (256, 256))  # NVIDIA passes INTER_AREA as cv2's 3rd positional arg (dst) -> bilinear
        return np.copy(arr)[None, None]

    video = {
        "res256_image_side_0": img("robot0_agentview_left"),
        "res256_image_side_1": img("robot0_agentview_right"),
        "res256_image_wrist_0": img("robot0_eye_in_hand"),
    }
    missing = [k for k, v in video.items() if v is None]
    if missing:
        raise RuntimeError(
            f"Missing camera(s) {missing} in adapter obs -- did you pass "
            f"camera_names={CAMERA_NAMES} to RoboCasaAdapter?"
        )

    # BUG FIX (found post-papermatch: 86.7% robot-self-damage, 0/120 success --
    # see groot_risk_gate_report.md's "papermatch postmortem" section): this
    # used to read `obs["eef_pos"]`/`obs["eef_quat"]`, which RoboCasaAdapter
    # sets from raw `robot0_eef_pos`/`robot0_eef_quat` -- the WORLD-frame eef
    # pose. GR00T's `state.end_effector_position_relative` key name, and its
    # training statistics (mean ~[0.27, -0.04, 0.54], std ~[0.19, 0.26, 0.24]
    # -- meters-scale, arm's-workspace-scale numbers), both indicate this must
    # be BASE-relative, not world-frame. Confirmed directly: for PandaOmron
    # (mobile base placed far from world origin in RoboCasa's large scene
    # layouts), `robot0_eef_pos` was observed at ~[-2.33, -3.57, 1.30] while
    # `robot0_base_to_eef_pos` (robosuite's own base-relative observable) was
    # ~[0.23, -0.00, 0.60] -- matching the training statistics almost
    # exactly, while the world-frame value is off by several METERS. Feeding
    # the wrong frame gave the policy a catastrophically wrong belief about
    # where its own arm currently is, which plausibly explains both 0%
    # success and the smooth-but-wrong self-colliding motion observed in
    # saved rollout videos (arm progressively curling into the cabinet/itself
    # over an episode, not jerky/random -- consistent with confidently acting
    # on a wrong but internally-consistent state estimate).
    #
    # Also fixed: `base_position`/`base_rotation` were hardcoded to zero/
    # identity ("base held still"), but training statistics show
    # `state.base_position` has mean ~[2.52, -1.52, 0.69] (NOT zero) --
    # holding the base *physically* still is still correct (matches
    # action.base_motion's zero training statistics), but the *reported*
    # base pose should be the real (fixed) value, not a zero placeholder,
    # to stay in-distribution for the state encoder.
    eef_pos = np.asarray(obs["raw"]["robot0_base_to_eef_pos"], dtype=np.float32)[None, None]  # (1,1,3)
    eef_quat = np.asarray(obs["raw"]["robot0_base_to_eef_quat"], dtype=np.float32)[None, None]  # (1,1,4)
    gripper_qpos = np.asarray(obs["gripper_qpos"], dtype=np.float32)[None, None]  # (1,1,2)
    base_pos = np.asarray(obs["raw"]["robot0_base_pos"], dtype=np.float32)[None, None]  # (1,1,3)
    base_quat = np.asarray(obs["raw"]["robot0_base_quat"], dtype=np.float32)[None, None]  # (1,1,4)

    return {
        "video": video,
        "state": {
            "end_effector_position_relative": eef_pos,
            "end_effector_rotation_relative": eef_quat,
            "gripper_qpos": gripper_qpos,
            "base_position": base_pos,
            "base_rotation": base_quat,
        },
        # Key confirmed against nvidia/GR00T-N1.6-3B's processor_config.json
        # modality_configs["robocasa_panda_omron"]["language"]["modality_keys"]
        # (see groot_risk_gate_spike.py's same fix -- job 42911087 first hit
        # this as a KeyError with the simplified "task_description" key).
        "language": {"annotation.human.action.task_description": [[instruction]]},
    }


def groot_action_to_arm_delta_and_gripper(action_chunk: dict, step: int):
    """Slice step `step` out of a GR00T action-chunk response and convert to
    `build_env_action()`'s (arm_delta[6], gripper_value) contract.

    `action_chunk` is the raw dict returned by `Gr00tPolicy.get_action()` /
    `PolicyClient.get_action()`: {"end_effector_position": (1,T,3), ...}.
    """
    pos = np.asarray(action_chunk["end_effector_position"])[0, step]  # (3,)
    rot = np.asarray(action_chunk["end_effector_rotation"])[0, step]  # (3,)
    arm_delta = np.concatenate([pos, rot]).astype(np.float64)

    gripper_close = float(np.asarray(action_chunk["gripper_close"])[0, step, 0])
    # gripper_close in GR00T's own training convention (see gymnasium_groot.py:
    # < 0.5 -> open (-1.0), else closed (+1.0)) -- already robosuite sign
    # convention, no flip needed (unlike OpenVLA's 0=close/1=open).
    gripper_value = -1.0 if gripper_close < 0.5 else 1.0

    return arm_delta, gripper_value


def act(client, obs: dict, instruction: str):
    groot_obs = build_groot_obs(obs, instruction)
    action_chunk, _info = client.get_action(groot_obs)
    return action_chunk


def groot_action_12d(action_chunk: dict, step: int) -> np.ndarray:
    """GR00T robocasa_panda_omron action at `step` as the 12-D vector RoboCasa's gym wrapper
    consumes (eef pos 3, eef rot 3, gripper_close 1, base_motion 4, control_mode 1)."""
    g = lambda k: np.asarray(action_chunk[k])[0, step].reshape(-1)
    return np.concatenate([g("end_effector_position"), g("end_effector_rotation"), g("gripper_close"),
                           g("base_motion"), g("control_mode")]).astype(np.float64)


def run_episode(adapter: RoboCasaAdapter, client, save_video_path: str, n_action_steps: int = 8,
                paper_protocol: bool = False) -> dict:
    obs = adapter.reset()
    instruction = adapter.get_instruction()
    if paper_protocol:
        # GR00T's own RoboCasa training prompts (nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
        # single_panda_gripper.*/meta/tasks.jsonl) are lowercase without a period, e.g.
        # "open the microwave door", "turn on the front left burner of the stove" -- the same style
        # as OopsieVerse's own prompts. Only fill in the stove knob OopsieVerse drops.
        knob = getattr(adapter.env, "knob", None)
        if adapter.task_name == "turn_on_stove" and knob:
            instruction = f"turn on the {knob.replace('_', ' ')} burner of the stove"
    robot = adapter.env.robots[0]
    frames = []
    eef_positions = [np.asarray(obs["eef_pos"], dtype=np.float64)]
    success = False
    n_steps = 0
    first_success_step = None
    health_trace = []

    while n_steps < adapter.max_episode_steps:
        action_chunk = act(client, obs, instruction)
        for step in range(min(n_action_steps, 16)):
            if paper_protocol:
                # NVIDIA's gymnasium_groot.py -> PandaOmronKeyConverter.unmap_action (base/torso/mode passed through)
                env_action = robocasa365_to_env_action(robot, groot_action_12d(action_chunk, step))
            else:
                arm_delta, gripper = groot_action_to_arm_delta_and_gripper(action_chunk, step)
                env_action = build_env_action(robot, arm_delta, gripper)
            obs, success, episode_done, info = adapter.step(env_action)
            n_steps += 1
            if success and first_success_step is None:
                first_success_step = n_steps
            h = obs.get("health")
            if h is not None and len(h):
                health_trace.append(float(np.mean(h)))
            eef_positions.append(np.asarray(obs["eef_pos"], dtype=np.float64))

            frame = obs.get("agentview_image")
            if frame is not None:
                frames.append(frame)
            if episode_done or n_steps >= adapter.max_episode_steps:
                break
        if episode_done:
            break

    if frames:
        os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
        imageio.mimsave(save_video_path, frames, fps=10)

    return {
        "success": bool(success),
        "success_any": first_success_step is not None,
        "first_success_step": first_success_step,
        "avg_env_health": float(np.mean(health_trace)) if health_trace else None,
        "num_steps": n_steps,
        "instruction": instruction,
        "health_summary": adapter.get_health_summary(),
        "video_path": save_video_path if frames else None,
        # Motion-degeneracy check (task step 6): compare against
        # groot_random_baseline_eval.py's numbers for the same task/seed --
        # see motion_stats.py.
        "eef_total_displacement": eef_total_displacement(eef_positions),
        "mean_frame_diff": mean_frame_diff(frames),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--max_episode_steps", type=int, default=720)
    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=5555)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument(
        "--results_dir", default=os.path.join(_HERE, "..", "results", "groot_robocasa")
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--paper_protocol", action="store_true",
                        help="RoboCasa-style prompts (stove names its knob), full 12-D action, health trace")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Deliberately NOT `from gr00t.policy.server_client import PolicyClient`
    # -- see groot_zmq_client.py's docstring: importing the real gr00t
    # package here would pull torch/transformers/deepspeed into this sim
    # env. GrootZmqClient reimplements just the wire protocol.
    client = GrootZmqClient(host=args.server_host, port=args.server_port)
    if not client.ping():
        raise RuntimeError(
            f"Could not reach groot_server.py at {args.server_host}:{args.server_port} "
            "-- start it first (see ../servers/groot_server.py)."
        )
    print(f"GR00T server OK at {args.server_host}:{args.server_port}")

    all_results = []
    for task_name in args.tasks:
        for trial in range(args.num_trials):
            episode_seed = args.seed + trial
            print(f"=== {task_name} trial {trial} (seed={episode_seed}) ===")
            record = {"task_name": task_name, "trial": trial, "seed": episode_seed, "error": None}
            adapter = RoboCasaAdapter(
                task_name=task_name,
                camera_names=CAMERA_NAMES,
                # NVIDIA's PandaOmronKeyConverter renders all three cameras at 512x512
                camera_width=512 if args.paper_protocol else 256,
                camera_height=512 if args.paper_protocol else 256,
                max_episode_steps=args.max_episode_steps,
                seed=episode_seed,
            )
            try:
                video_path = os.path.join(args.results_dir, f"{task_name}_trial{trial}.mp4")
                record.update(
                    run_episode(adapter, client, video_path, n_action_steps=args.n_action_steps,
                                paper_protocol=args.paper_protocol)
                )
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
