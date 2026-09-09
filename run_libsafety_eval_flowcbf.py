"""
run_libsafety_eval_flowcbf.py

Neurosymbolic constrained-flow-matching baseline (English et al.,
arXiv:2607.01378) on LIBERO-Safety. Uses pi05_libero_safety (the
fine-tuned checkpoint) IN-PROCESS -- no websocket server, since sampling
needs direct access to the model's sample_actions internals
(flow_cbf_sample.py).

Ground-truth obstacle position, matching both the paper's own privileged-
tier setup and this project's AEGIS port (run_libsafety_eval_aegis_gt.py):
hazard object identity from the task's BDDL CheckRobotContact predicate,
position from obs[f"{hazard_name}_pos"].

Correction scope: the paper's Algorithm 1 -- the FULL H-step predicted
trajectory corrected jointly (one SLSQP solve over delta in R^{H x 3}, all H
consecutive-point barrier constraints enforced simultaneously), not just the
next action. It is applied to the DECODED, UNNORMALISED action chunk, which is
the only representation where the barrier's metre-valued geometry is
commensurable with the command; see flow_cbf_barrier.py.

NOT YET REVALIDATED ON GPU: the barrier's space and its dynamics gain were both
corrected without a rollout (no jax/GPU in the fixing environment). Re-run this
arm before reporting any number from it.

Run inside the libsafety_openpi conda env (needs jax/openpi in-process,
not the lighter libsafety_client env used for websocket-only drivers):
  python run_libsafety_eval_flowcbf.py --task_suite_name obstacle_avoidance \
      --task_index 0 --num_trials_per_task 1
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib
import sys
import time

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import tyro
from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from scipy.spatial.transform import Rotation as R

from flow_cbf_sample import eager_sample_actions
from flow_cbf_barrier import (
    assert_barrier_is_active,
    barrier_values,
    correct_action_chunk,
    predict_trajectory,
)
from libsafety_env_utils import select_initial_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
CKPT_DIR = "LIBERO-Safety/checkpoints/pi05_libero_safety"
OBSTACLE_RADIUS = 0.06  # same unvalidated default as run_libsafety_eval_aegis_gt.py
# 20 Hz control, per SafeLIBERO (arXiv:2512.11891 Sec. V.A).
DT_CONTROL = 0.05


@dataclasses.dataclass
class Args:
    resize_size: int = 224
    replan_steps: int = 5
    num_denoise_steps: int = 10

    task_suite_name: str = ""
    task_index: int = 0
    all_tasks: bool = False
    num_trials_per_task: int = 1
    num_steps_wait: int = 10
    max_steps: int = 300
    checkpoint_name: str = "flowcbf_pi05_libero_safety"

    # Seeds torch/numpy/random only. The simulator seed is env_seed below:
    # it moves OBJECT POSITIONS even with a fixed initial state, so it must
    # match across every method in a comparison table. This driver passed
    # `seed` (7) to env.seed() while Cosmos/Fast-WAM/OpenVLA got 0 from the
    # shared helper, so the rows were not layout-matched.
    seed: int = 7
    env_seed: int = 0
    video_out_path: str = "LIBERO-Safety/rollouts"
    results_out_path: str = "LIBERO-Safety/results"


def _get_env(task_suite, task, resolution, env_seed):
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_file = task_suite.get_task_bddl_file_path_by_level_id(task.level, task.level_id)
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(env_seed)
    return env, task.language


def _find_hazard_object_name(env):
    """Same convention as run_libsafety_eval_aegis_gt.py -- see that file
    for the full explanation of the lowercase predicate-name gotcha."""
    parsed = getattr(env, "parsed_problem", None)
    if parsed is None:
        parsed = getattr(getattr(env, "env", None), "parsed_problem", None)
    if parsed is None:
        return None
    for constraint in parsed.get("constraints", []):
        if isinstance(constraint, (list, tuple)) and len(constraint) >= 2 and constraint[0].lower() == "checkrobotcontact":
            return constraint[1]
    return None


def _quat2axisangle(quat):
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def save_rollout_video(replay_images, video_dir, task_suite_name, task_index, episode_idx, success, task_description):
    video_dir.mkdir(parents=True, exist_ok=True)
    processed = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    path = (
        video_dir
        / f"{DATE_TIME}--{task_suite_name}--task{task_index}--episode={episode_idx}--success={success}--task={processed}.mp4"
    )
    imageio.mimwrite(path, [np.asarray(x) for x in replay_images], fps=30)
    return str(path)


def _infer_with_flow_cbf(policy, model, rng, obs_dict, eef_pos_world, p_obs, r_obs, num_denoise_steps):
    transformed = policy._input_transform(dict(obs_dict))
    if "image_mask" in transformed:
        transformed["image_mask"] = {k: np.asarray(v, dtype=bool) for k, v in transformed["image_mask"].items()}
    transformed = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transformed)
    observation = _model.Observation.from_dict(transformed)

    # Sample WITHOUT any in-loop correction: v_t is the velocity field, not an
    # action, and Unnormalize has not run yet.
    actions = eager_sample_actions(
        model, rng, observation,
        eef_pos_world=eef_pos_world, p_obs=p_obs, r_obs=r_obs,
        num_steps=num_denoise_steps, apply_correction=False,
    )
    actions = np.asarray(actions[0, ...])
    outputs = {"state": np.asarray(transformed["state"][0, ...]), "actions": actions}
    result = policy._output_transform(outputs)

    # Correct the DECODED, UNNORMALISED chunk -- the only place the barrier's
    # metre-valued geometry is commensurable with the commanded action.
    solver_ok = True
    if p_obs is not None:
        chunk = np.asarray(result["actions"], dtype=np.float64)
        before = barrier_values(
            predict_trajectory(eef_pos_world, chunk[:, :3], DT_CONTROL), p_obs, r_obs
        )
        corrected_xyz, _, solver_ok = correct_action_chunk(
            chunk[:, :3], eef_pos_world, DT_CONTROL, p_obs, r_obs
        )
        if not solver_ok:
            logger.warning(
                "flow-CBF SLSQP failed -- chunk returned UNCORRECTED, so the "
                "filter is off for this step"
            )
        complaint = assert_barrier_is_active(corrected_xyz, chunk[:, :3], before)
        if complaint:
            logger.warning("flow-CBF: %s", complaint)
        chunk[:, :3] = corrected_xyz
        result["actions"] = chunk

    return result, solver_ok


def eval_one_task(args, policy, model, task_suite, task_index, video_dir, results_dir):
    task = task_suite.get_task(task_index)
    initial_states = task_suite.get_task_init_states(task.level, task.level_id)
    env, task_description = _get_env(task_suite, task, LIBERO_ENV_RESOLUTION, args.env_seed)

    hazard_name = None
    rng = jax.random.key(args.seed)
    task_results = []
    for episode_idx in range(args.num_trials_per_task):
        env.reset()
        action_plan = collections.deque()
        obs = env.set_init_state(select_initial_state(initial_states, episode_idx, task_index))

        t = 0
        while t < args.num_steps_wait:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1

        if hazard_name is None:
            hazard_name = _find_hazard_object_name(env)
        has_hazard = hazard_name is not None and f"{hazard_name}_pos" in obs
        if not has_hazard:
            # The affordance suite has no :constraints block in any of its 15
            # BDDL files, so no hazard resolves and this "safety filter" arm is
            # byte-identical to the unfiltered policy. Say so loudly: a silent
            # inert arm reported as a filtered arm is a false claim.
            logger.warning(
                "SAFETY FILTER INERT: no hazard resolved for suite=%s task=%s "
                "(hazard_name=%r). This arm is identical to the unfiltered "
                "policy for this task.",
                args.task_suite_name, task.name, hazard_name,
            )

        t = 0
        collide_flag = False
        replay_images = []
        # Steps where SLSQP failed and the chunk went through UNCORRECTED,
        # i.e. the filter was off. A 'filter on' episode with a high count
        # here is not the method it claims to be.
        solver_failures = 0
        done_flag = False

        while t < args.max_steps:
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            replay_images.append(img)

            if not action_plan:
                eef_pos_world = np.array(obs["robot0_eef_pos"], dtype=np.float64)
                p_obs = np.array(obs[f"{hazard_name}_pos"], dtype=np.float64) if has_hazard else None
                obs_dict = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ).astype(np.float32),
                    "prompt": str(task_description),
                }
                rng, sample_rng = jax.random.split(rng)
                result, solver_ok = _infer_with_flow_cbf(
                    policy, model, sample_rng, obs_dict, eef_pos_world, p_obs, OBSTACLE_RADIUS, args.num_denoise_steps
                )
                if not solver_ok:
                    solver_failures += 1
                action_chunk = result["actions"]
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()
            obs, reward, done, info = env.step(np.asarray(action).tolist())

            if not collide_flag:
                cost = info.get("cost", {}) if info else {}
                if any(v for v in cost.values()):
                    collide_flag = True

            if done:
                done_flag = True
                break
            t += 1

        video_path = save_rollout_video(
            replay_images, video_dir, args.task_suite_name, task_index, episode_idx, done_flag, task_description
        )
        result = {
            "task": task_description,
            "task_suite": args.task_suite_name,
            "task_index": task_index,
            "episode": episode_idx,
            "success": bool(done_flag),
            "collision": bool(collide_flag),
            "flow_cbf_solver_failures": solver_failures,
            "hazard_object": hazard_name,
            "num_steps": t,
            "video": video_path,
            "checkpoint": args.checkpoint_name,
        }
        task_results.append(result)
        logger.info(f"task={task_index} episode={episode_idx}: success={done_flag} collision={collide_flag}")

    out_path = results_dir / f"{args.task_suite_name}_task{task_index}_{args.checkpoint_name}.json"
    out_path.write_text(json.dumps(task_results, indent=2))
    return task_results


def eval_libsafety(args: Args) -> None:
    from libero.libero import benchmark

    assert args.task_suite_name, "task_suite_name must be set"
    np.random.seed(args.seed)

    train_config = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(train_config, CKPT_DIR)
    model = policy._model

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    video_dir = pathlib.Path(args.video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)
    results_dir = pathlib.Path(args.results_out_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    task_indices = list(range(task_suite.n_tasks)) if args.all_tasks else [args.task_index]
    failed_tasks = []
    for task_index in task_indices:
        logger.info(f"=== Starting suite='{args.task_suite_name}' task_index={task_index} ===")
        # The other four drivers guard this. Without it, one bad task -- e.g. a
        # missing .pruned_init making get_task_init_states_by_level_id return
        # None, or a robosuite placement RandomizationError -- aborts the whole
        # --all_tasks sweep and discards every task already completed.
        try:
            eval_one_task(args, policy, model, task_suite, task_index, video_dir, results_dir)
        except Exception:
            logger.exception(
                "TASK %d FAILED -- recording and continuing to the next task", task_index
            )
            failed_tasks.append(task_index)
    if failed_tasks:
        logger.warning(
            "SWEEP INCOMPLETE: %d of %d tasks failed: %s",
            len(failed_tasks), len(task_indices), failed_tasks,
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    eval_libsafety(args)
