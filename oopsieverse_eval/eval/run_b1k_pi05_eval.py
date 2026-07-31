"""Off-the-shelf pi0.5 (pi05_libero) rollout evaluation on OopsieVerse's
BEHAVIOR-1K (OmniGibson) tasks.

Run inside the `oopsieverse_b1k` conda env, INSIDE the
`b1k_container.sif` Apptainer container (see
`oopsieverse_eval/slurm/spike_b1k_headless.slurm` for the mandatory
`apptainer exec --nv` invocation -- LD_LIBRARY_PATH additions + Vulkan ICD
binds -- bare metal cannot run Isaac Sim on this cluster at all). pi0.5
runs in a separate process/env via openpi's own serve_policy.py, exactly
as for the RoboCasa side (see ../servers/launch_pi05_server.sh) -- REUSE
THE EXACT SAME SERVER, unchanged. With a pi0.5 server already listening:

    export OMNIGIBSON_HEADLESS=1
    conda activate oopsieverse_b1k
    python oopsieverse_eval/eval/run_b1k_pi05_eval.py \\
        --tasks pick_egg wipe_counter add_firewood \\
        --num_trials 3 \\
        --host localhost --port 8200 \\
        --results_dir oopsieverse_eval/results/pi05_b1k

Deviation from `run_robocasa_pi05_eval.py`: builds ONE `B1KAdapter` per
task (not one per trial) and resets it between trials, because Isaac
Sim/Kit's cold boot took ~25 minutes in the confirmed-working headless
spike -- rebuilding the env per trial would make even a 3-trial smoke test
impractically slow. See `b1k_adapter.py`'s docstring for the full
reasoning (including why `reset()` calls `teleop_b1k.reset_env`, not a
bare `env.reset()`).

Observation schema mirrors the RoboCasa pi0.5 client's, adapted to B1K's
actual obs keys (see `b1k_adapter.py._build_obs`): "observation/image"
(from `external_sensor0`, the primary third-person view -- `external_sensor1`
is captured into `obs["secondary_image"]` but unused here, matching pi0.5's
2-camera-input training schema), "observation/wrist_image" (from the
robot's own EEF camera), "observation/state" (eef_pos(3) + axis-angle(3) +
gripper_qpos(2), 8-dim -- gripper_qpos read directly off the robot's own
gripper joint positions via `B1KAdapter.get_gripper_qpos()`, not guessed
from a flat proprio slice), "prompt" (task name with underscores replaced
by spaces -- B1K's `TaskConfig` has no language-instruction field at all).

Action-space conversion is NOT a pass-through unlike the RoboCasa client:
B1K's IK controller (`command_input_limits=None`) and gripper controller
(`smooth` mode, `[0,1]` range) are both genuinely different conventions
from RoboCasa's composite controller -- see `b1k_action_utils.py`'s
extensive docstring for the confirmed-from-source reasoning and the
assumptions that still need empirical confirmation (gripper open/close
direction) before results are fully trustworthy.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import traceback

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from b1k_adapter import B1KAdapter  # noqa: E402
from b1k_action_utils import (  # noqa: E402
    build_b1k_env_action,
    robosuite_gripper_to_b1k_smooth,
    json_default,
)

from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy as _websocket_client_policy  # noqa: E402
from openpi_client import msgpack_numpy  # noqa: E402
import websockets.sync.client  # noqa: E402

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    imageio = None


class NoPingWebsocketClientPolicy(_websocket_client_policy.WebsocketClientPolicy):
    """Identical to the RoboCasa client's copy of this class -- pi0.5's
    server does JAX/XLA JIT compilation on its first inference call, which
    routinely exceeds `websockets`' default 20s ping_timeout. See
    `run_robocasa_pi05_eval.py` for the full rationale (unchanged here)."""

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


# Paper's B1K subset (OopsieVerse, arXiv:2606.31993, Table I: "(B1K-OG)
# attach camera", "(B1K-OG) open microwave door", "(B1K-OG) pick up
# scrubber", "(B1K-OG) ignite wood") matched against THIS repo's actual
# `oopsiebench/envs/behavior1k/` task registry and its own published
# task-description table (`external/oopsieverse/docs/oopsiebench.md`):
#   ignite wood        -> add_firewood      (fireplace + logs, thermal+mechanical)
#   pick up scrubber    -> wipe_counter      (docs: "wipe dirt... with the sponge")
#   open microwave door -> food_in_microwave (closest match; docs describe
#                          it as "place/heat food with the microwave", which
#                          requires opening the microwave door as a step --
#                          no task isolates "open microwave door" alone)
#   attach camera       -> NO MATCH. No task in this registry's 13 modules
#                          (confirmed against oopsiebench.md's own OmniGibson
#                          task table) involves a camera-type object at all;
#                          cameras only appear as VIEWER_CAMERA/EXTERNAL_
#                          CAMERA_CONFIGS (sensors), never as a `task_objects`
#                          category. This task is simply absent from this
#                          fork's B1K registry -- not guessed, not run.
PAPER_SUBSET_TASKS = ["add_firewood", "wipe_counter", "food_in_microwave"]

ALL_TASKS = [
    "add_firewood", "fill_bowl", "food_in_microwave", "heat_saucepot",
    "nav_to_table", "open_drawer", "open_single_door", "pick_egg",
    "place_plate", "pour_water", "shelve_item", "turn_on_faucet",
    "wipe_counter",
]

RESIZE_SIZE = 224


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """(x,y,z,w) quat -> axis-angle. Reimplemented, not imported, matching
    `run_robocasa_pi05_eval.py`'s own copy (kept build-independent)."""
    quat = np.asarray(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def _safe_image(obs: dict, key: str) -> np.ndarray:
    """Return obs[key] resized/padded to RESIZE_SIZE, or a black frame if
    missing. Missing is expected for e.g. `nav_to_table`'s Tiago robot,
    whose wrist-camera key convention differs from the single-Franka tasks
    (`{robot_name}:eef_link:Camera:0` -- see `b1k_adapter.py`); rather than
    hard-crashing every trial of that one out-of-scope (mobile, dual-arm)
    task, substitute a black frame and let it show up as a real 0% success
    signal instead of a Python exception."""
    frame = obs.get(key)
    if frame is None:
        frame = np.zeros((RESIZE_SIZE, RESIZE_SIZE, 3), dtype=np.uint8)
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(frame, RESIZE_SIZE, RESIZE_SIZE))


def build_observation(obs: dict, instruction: str) -> dict:
    img = _safe_image(obs, "agentview_image")
    wrist_img = _safe_image(obs, "wrist_image")
    gripper_qpos = obs.get("gripper_qpos")
    if gripper_qpos is None or len(gripper_qpos) < 2:
        gripper_qpos = np.zeros(2)
    state = np.concatenate(
        (obs["eef_pos"], quat2axisangle(obs["eef_ori"]), np.asarray(gripper_qpos)[:2])
    )
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(instruction),
    }


def diagnose_controllers_and_motion(adapter: B1KAdapter, results_dir: str) -> dict:
    """One-time, policy-free diagnostic: print each controller's spec, then
    empirically determine (a) real eef displacement under a forced large
    arm delta, and (b) which end of the gripper's [0,1] `smooth` range is
    open vs closed -- per this phase's explicit requirement to verify the
    real convention at runtime rather than assume it from memory/source
    reading alone (source reading gave us `command_input_limits=None` for
    the arm and the `smooth`/[0,1] gripper convention, but NOT which
    physical direction is open vs closed, nor whether `ARM_POS_SCALE_M`
    produces sane, non-degenerate motion for this specific robot/task).
    """
    import torch as th

    robot = adapter.robot
    print(f"[diagnose] robot.action_dim={robot.action_dim} controller_order={robot.controller_order}")
    if robot.action_dim > 7:
        print(f"[diagnose] WARNING: action_dim={robot.action_dim} > 7 -- this robot has controllers "
              f"beyond a single arm+gripper (e.g. Tiago's second arm/base for nav_to_table). Zero-fill "
              f"for those extra controllers is a true no-op ONLY for delta-mode controllers; an "
              f"absolute-position controller (e.g. a trunk/head joint controller) would get slammed to "
              f"joint-zero every step. Treat this task's results as degenerate-by-construction, not a "
              f"clean single-arm-policy evaluation.")
    for name, ctrl in robot.controllers.items():
        print(
            f"[diagnose] controller '{name}': class={type(ctrl).__name__} "
            f"command_dim={ctrl.command_dim} "
            f"command_input_limits={getattr(ctrl, '_command_input_limits', 'N/A')} "
            f"command_output_limits={getattr(ctrl, '_command_output_limits', 'N/A')} "
            f"mode={getattr(ctrl, '_mode', 'N/A')}"
        )

    from b1k_action_utils import build_b1k_env_action, ARM_POS_SCALE_M

    obs = adapter.reset()
    eef_before = np.array(obs["eef_pos"])
    print(f"[diagnose] eef_pos before forced-motion test: {eef_before}")

    # Forced HALF-scale +x arm delta (0.5 -> 0.025 m/step commanded via
    # ARM_POS_SCALE_M), gripper held open, for 10 steps. Comparing OBSERVED
    # per-step displacement against the COMMANDED per-step delta (not just
    # checking displacement > some small epsilon) can actually distinguish
    # "scale is roughly right" from "scale is off by 10x" -- a full-scale
    # forced command can look like "real motion" even with a badly wrong
    # scale, since the arm moves *something* either way.
    commanded_per_step = 0.5 * ARM_POS_SCALE_M
    per_step_positions = [eef_before.tolist()]
    per_step_deltas = []
    for _ in range(10):
        action = build_b1k_env_action(robot, np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]), gripper_value=1.0)
        prev_pos = np.array(obs["eef_pos"])
        obs, _success, _done, _info = adapter.step(action)
        new_pos = np.array(obs["eef_pos"])
        per_step_positions.append(new_pos.tolist())
        per_step_deltas.append(float(np.linalg.norm(new_pos - prev_pos)))
    eef_after_motion = np.array(obs["eef_pos"])
    displacement = float(np.linalg.norm(eef_after_motion - eef_before))
    # Report the MEDIAN per-step delta, not total/10 -- if the arm drives
    # into the table/a joint limit partway through the 10 forced steps,
    # later per-step deltas collapse toward zero and the aggregate
    # (total/10) would misleadingly read as "roughly right" or "too small"
    # depending on when the limit was hit, rather than showing the actual
    # per-step pattern (see per_step_deltas in diagnostic.json).
    observed_per_step = float(np.median(per_step_deltas))
    ratio = observed_per_step / commanded_per_step if commanded_per_step > 0 else float("nan")
    print(f"[diagnose] eef_pos after 10x forced +0.5x-delta steps: {eef_after_motion} "
          f"(total displacement={displacement:.4f} m, per-step deltas={['%.4f' % d for d in per_step_deltas]}, "
          f"median/step={observed_per_step:.4f} m, commanded/step={commanded_per_step:.4f} m, ratio={ratio:.2f})")
    if max(per_step_deltas) > 0 and min(per_step_deltas[-3:]) < 0.1 * max(per_step_deltas):
        print("[diagnose] -> per-step deltas COLLAPSE toward the end of the forced sequence: likely "
              "hit a joint/workspace limit partway through, not a scale or drop issue -- read "
              "per_step_deltas in diagnostic.json directly rather than trusting the aggregate ratio.")
    if observed_per_step < 0.2 * commanded_per_step:
        print("[diagnose] -> OBSERVED << COMMANDED: IK likely not tracking (command may be slamming "
              "into joint limits, or action is being silently dropped -- see build_env_action's "
              "RoboCasa-side history of exactly this failure mode).")
    elif ratio > 3.0 or ratio < 0.33:
        print(f"[diagnose] -> ratio={ratio:.2f} is far from 1.0: ARM_POS_SCALE_M is likely wrong "
              f"(also recall LIBERO trained pi0.5 at 20Hz vs B1K's 30Hz action_frequency -- same "
              f"per-step delta is 1.5x the velocity pi0.5 was trained to produce).")
    else:
        print("[diagnose] -> observed roughly tracks commanded: scale looks reasonable.")

    # Gripper direction: command 0.0 (assumed closed) then 1.0 (assumed open),
    # holding arm still, and read back finger joint positions.
    for _ in range(15):
        action = build_b1k_env_action(robot, np.zeros(6), gripper_value=0.0)
        obs, _s, _d, _i = adapter.step(action)
    qpos_at_0 = adapter.get_gripper_qpos()
    for _ in range(15):
        action = build_b1k_env_action(robot, np.zeros(6), gripper_value=1.0)
        obs, _s, _d, _i = adapter.step(action)
    qpos_at_1 = adapter.get_gripper_qpos()
    print(f"[diagnose] gripper qpos at command=0.0: {qpos_at_0}")
    print(f"[diagnose] gripper qpos at command=1.0: {qpos_at_1}")

    diag = {
        "action_dim": int(robot.action_dim),
        "controller_order": list(robot.controller_order),
        "eef_displacement_m_total": displacement,
        "eef_displacement_m_per_step_median_observed": observed_per_step,
        "eef_displacement_m_per_step_commanded": commanded_per_step,
        "eef_per_step_deltas": per_step_deltas,
        "eef_per_step_positions": per_step_positions,
        "gripper_qpos_at_0": None if qpos_at_0 is None else qpos_at_0.tolist(),
        "gripper_qpos_at_1": None if qpos_at_1 is None else qpos_at_1.tolist(),
    }
    # Write BEFORE the assert below -- a failed assert should still leave the
    # controller specs + scale measurement on disk, not just in the SLURM
    # stdout log (a 25-minute cold start is too expensive to lose the
    # diagnostic data along with the process on failure).
    with open(os.path.join(results_dir, "diagnostic.json"), "w") as f:
        json.dump(diag, f, indent=2, default=json_default)

    assert qpos_at_0 is not None and qpos_at_1 is not None, (
        "get_gripper_qpos() returned None -- see its own printed warning above. Refusing to proceed "
        "with rollouts (every episode would silently see a wrong/zeroed gripper state otherwise)."
    )
    mean0, mean1 = float(np.mean(qpos_at_0)), float(np.mean(qpos_at_1))
    direction = (
        "0=closed/1=open (ASSUMPTION CONFIRMED)" if mean1 > mean0
        else "0=open/1=closed (ASSUMPTION WRONG -- must flip robosuite_gripper_to_b1k_smooth)"
    )
    print(f"[diagnose] gripper direction: mean(qpos@0)={mean0:.4f} mean(qpos@1)={mean1:.4f} -> {direction}")
    diag["gripper_direction"] = direction
    with open(os.path.join(results_dir, "diagnostic.json"), "w") as f:
        json.dump(diag, f, indent=2, default=json_default)

    return diag


def run_episode(
    adapter: B1KAdapter, client, replan_steps: int, save_video_path: str
) -> dict:
    import time

    t_start = time.perf_counter()
    obs = adapter.reset()
    instruction = adapter.get_instruction()
    robot = adapter.robot
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
        gripper_b1k = robosuite_gripper_to_b1k_smooth(float(raw_action[6]))
        env_action = build_b1k_env_action(robot, raw_action[:6], gripper_b1k)
        obs, success, episode_done, info = adapter.step(env_action)
        n_steps += 1
        if episode_done:
            break

    if frames and imageio is not None:
        os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
        imageio.mimsave(save_video_path, frames, fps=10)

    elapsed_s = time.perf_counter() - t_start
    return {
        "success": bool(success),
        "num_steps": n_steps,
        "elapsed_s": elapsed_s,
        "instruction": instruction,
        "health_summary": adapter.get_health_summary(),
        "video_path": save_video_path if (frames and imageio is not None) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--max_episode_steps", type=int, default=700)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--results_dir", default=os.path.join(_HERE, "..", "results", "pi05_b1k"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--diagnose", action="store_true",
                         help="Run controller-introspection + forced-motion/gripper-direction diagnostic "
                              "on the FIRST task before any policy rollouts (no server call needed for this "
                              "part, but the server connection is still opened first so it's included in "
                              "episodes.json / doesn't require a second cold Kit boot).")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    client = NoPingWebsocketClientPolicy(args.host, args.port)
    print(f"pi0.5 server metadata: {client.get_server_metadata()}")

    import random
    import torch as th

    results_path = os.path.join(args.results_dir, "episodes.json")

    def _flush(results):
        # Rewritten after every trial (not just once at the end) -- a 30-trial
        # shard that hits the 4hr SLURM time cap partway through would
        # otherwise lose every already-completed episode's results.
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=json_default)

    all_results = []
    for i, task_name in enumerate(args.tasks):
        print(f"=== Building B1KAdapter for task={task_name} (one env for all {args.num_trials} trials) ===")
        adapter = B1KAdapter(task_name=task_name, max_episode_steps=args.max_episode_steps)
        try:
            if args.diagnose and i == 0:
                diagnose_controllers_and_motion(adapter, args.results_dir)
            for trial in range(args.num_trials):
                episode_seed = args.seed + trial
                print(f"=== {task_name} trial {trial} (seed={episode_seed}) ===")
                random.seed(episode_seed)
                np.random.seed(episode_seed)
                th.manual_seed(episode_seed)  # pick_egg.reset() (and likely others) use th.rand for jitter
                record = {"task_name": task_name, "trial": trial, "seed": episode_seed, "error": None}
                try:
                    video_path = os.path.join(args.results_dir, f"{task_name}_trial{trial}.mp4")
                    record.update(run_episode(adapter, client, args.replan_steps, video_path))
                except Exception:
                    record["error"] = traceback.format_exc()
                    print(record["error"])
                all_results.append(record)
                _flush(all_results)
        finally:
            adapter.close()

    n_ok = sum(1 for r in all_results if r["error"] is None)
    n_success = sum(1 for r in all_results if r.get("success"))
    print(f"\n=== SUMMARY: {n_ok}/{len(all_results)} episodes ran cleanly, {n_success} succeeded ===")
    print(f"Results written to {results_path}")
    sys.exit(0 if n_ok == len(all_results) else 1)


if __name__ == "__main__":
    main()
