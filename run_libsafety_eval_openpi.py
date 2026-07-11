"""
run_libsafety_eval_openpi.py

Eval-only client driver for pi0 / pi0.5 on the LIBERO-Safety benchmark.
Connects to a running libsafety_serve_openpi.sh policy server over
websocket. Adapted from vlsa-aegis/main/pi05_evaluation.py and
vlsa-aegis/openpi/examples/libero/main.py's observation/action-chunking
pattern, but points at the LIBERO-Safety task-suite registry instead of
SafeLIBERO's, and matches Task 7's (run_libsafety_eval_openvla.py) output
convention for rollouts/results.

Run inside the libsafety_client conda env, after the server is up:
  conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_client
  export MUJOCO_GL=egl
  export LD_PRELOAD=/ocean/projects/cis250185p/asingal/envs/libsafety_client/lib/libstdc++.so.6
  export LIBERO_CONFIG_PATH=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/LIBERO-Safety/.libero_config
  python run_libsafety_eval_openpi.py --task_suite_name human_safety \
      --task_index 0 --num_trials_per_task 1

NOTE on LIBERO_CONFIG_PATH: it must point at a DIRECTORY containing
config.yaml (produced by the Task 5 setup script), not at a .yaml file
directly -- otherwise libero's first-import interactive Y/N prompt hangs
forever under sbatch (no stdin). Same requirement as Task 7's driver.

NOTE on observation format: the pi0_libero / pi05_libero openpi train
configs use LeRobotLiberoDataConfig, whose input transforms expect
observation/image, observation/wrist_image, observation/state, and prompt
(confirmed against vlsa-aegis/openpi/examples/libero/main.py and
vlsa-aegis/main/pi05_evaluation.py) -- NOT just observation/image as the
task-8 brief's skeleton client suggested. Sending only observation/image
would silently break the server-side input transform (missing keys), so
this driver sends the full observation dict, including the 180-degree
image rotation and quat2axisangle proprio conversion that match how the
LIBERO demo data was rendered/encoded during training.
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib
import time

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


@dataclasses.dataclass
class Args:
    # --- Model server parameters ---
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # --- LIBERO-Safety environment parameters ---
    task_suite_name: str = ""  # e.g. "human_safety", "affordance", ... (see benchmark.get_benchmark_dict())
    task_index: int = 0
    # If True, loop over every task in the suite (range(task_suite.n_tasks))
    # instead of just --task_index, reusing the single connected client/server
    # across all of them. --task_index is ignored when this is set.
    all_tasks: bool = False
    num_trials_per_task: int = 1  # trials (episodes) per task
    num_steps_wait: int = 10  # no-op warmup steps before handing control to the policy
    max_steps: int = 300  # episode horizon
    checkpoint_name: str = ""  # e.g. "pi05_libero" or "pi0_libero" (for results-json bookkeeping only)

    seed: int = 7
    video_out_path: str = "LIBERO-Safety/rollouts"
    results_out_path: str = "LIBERO-Safety/results"


def _get_env(task_suite, task, resolution, seed):
    from libero.libero.envs import OffScreenRenderEnv

    # NOTE: LIBERO-Safety's bddl files live under
    # bddl_files/<problem_folder>/L<level>/<file>.bddl (see
    # Benchmark._get_task_file_path in LIBERO-Safety/libero/libero/benchmark/
    # __init__.py), NOT flat under bddl_files/<problem_folder>/<file>.bddl
    # like upstream LIBERO. Using the benchmark's own
    # get_task_bddl_file_path_by_level_id() (rather than hand-building the
    # path, which silently omits the L<level> directory and 404s) keeps this
    # in sync with however LIBERO-Safety lays out each task-suite's files.
    task_bddl_file = task_suite.get_task_bddl_file_path_by_level_id(task.level, task.level_id)
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language


def _quat2axisangle(quat):
    """Copied from robosuite (see vlsa-aegis/openpi/examples/libero/main.py)."""
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def save_rollout_video(replay_images, video_dir: pathlib.Path, task_suite_name, task_index, episode_idx, success, task_description):
    video_dir.mkdir(parents=True, exist_ok=True)
    processed_task_description = (
        task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    )
    mp4_path = (
        video_dir
        / f"{DATE_TIME}--{task_suite_name}--task{task_index}--episode={episode_idx}"
          f"--success={success}--task={processed_task_description}.mp4"
    )
    writer = imageio.get_writer(str(mp4_path), fps=10)
    for img in replay_images:
        writer.append_data(img)
    writer.close()
    logger.info(f"Saved rollout MP4 at path {mp4_path}")
    return str(mp4_path)


def eval_one_task(args: Args, task_suite, task_index: int, client, video_dir: pathlib.Path, results_dir: pathlib.Path) -> list:
    task = task_suite.get_task(task_index)
    # NOTE: LIBERO-Safety's Benchmark.get_task_init_states() signature is
    # (level, level_id), NOT a single flat task index -- unlike upstream
    # LIBERO / the openpi examples this driver is otherwise adapted from.
    # Passing args.task_index alone raises
    # "missing 1 required positional argument: 'i'". The flat task (as
    # returned by get_task(i), which DOES take a single flat index) carries
    # its own .level / .level_id fields, which is what init states are
    # actually keyed on.
    initial_states = task_suite.get_task_init_states(task.level, task.level_id)
    env, task_description = _get_env(task_suite, task, LIBERO_ENV_RESOLUTION, args.seed)
    logger.info(f"Task description: {task_description}")

    episode_results = []
    for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
        env.reset()
        action_plan = collections.deque()
        obs = env.set_init_state(initial_states[episode_idx])

        t, done, success = 0, False, False
        replay_images = []

        try:
            while t < args.max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing.
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )
                replay_images.append(img)

                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(task_description),
                    }
                    action_chunk = client.infer(element)["actions"]
                    assert len(action_chunk) >= args.replan_steps, (
                        f"We want to replan every {args.replan_steps} steps, but policy only "
                        f"predicts {len(action_chunk)} steps."
                    )
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                obs, _, done, _ = env.step(np.asarray(action).tolist())
                if done:
                    success = True
                    break
                t += 1
        except Exception as e:
            logger.exception(f"Episode {episode_idx} raised an exception: {e}")

        mp4_path = save_rollout_video(
            replay_images, video_dir, args.task_suite_name, task_index, episode_idx, success, task_description
        )
        episode_results.append(
            {
                "task": task_description,
                "task_suite": args.task_suite_name,
                "task_index": task_index,
                "episode": episode_idx,
                "success": success,
                "num_steps": t,
                "video": mp4_path,
                "checkpoint": args.checkpoint_name,
            }
        )
        logger.info(f"Episode {episode_idx}: success={success}, steps={t}")

    suffix = f"_{args.checkpoint_name}" if args.checkpoint_name else ""
    out_path = results_dir / f"{args.task_suite_name}_task{task_index}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(episode_results, f, indent=2)
    logger.info(f"Wrote results to {out_path}")

    return episode_results


def eval_libsafety(args: Args) -> None:
    assert args.task_suite_name, "task_suite_name must be set to a real LIBERO-Safety suite name"
    # Required so result/video filenames are checkpoint-namespaced (see
    # run_libsafety_eval_openvla.py's checkpoint_tag() docstring for the
    # incident this class of bug caused: two checkpoints evaluated against
    # the same suite silently overwrote each other's un-namespaced results).
    assert args.checkpoint_name, "checkpoint_name must be set (e.g. 'pi05_libero') to avoid clobbering other checkpoints' results"
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    video_dir = pathlib.Path(args.video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)
    results_dir = pathlib.Path(args.results_out_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logger.info(f"Connected to policy server at {args.host}:{args.port}; metadata={client.get_server_metadata()}")

    if args.all_tasks:
        # Loop over every task in the suite, reusing the single connected
        # client/server for the whole sweep (server stays up across tasks;
        # only the env/task changes). --task_index is ignored here.
        task_indices = list(range(task_suite.n_tasks))
        logger.info(
            f"--all_tasks set: running all {len(task_indices)} tasks in suite "
            f"'{args.task_suite_name}' x {args.num_trials_per_task} trials each "
            f"({len(task_indices) * args.num_trials_per_task} episodes total)"
        )
    else:
        task_indices = [args.task_index]

    all_results = []
    for task_index in task_indices:
        logger.info(f"=== Starting suite='{args.task_suite_name}' task_index={task_index} ===")
        try:
            task_results = eval_one_task(args, task_suite, task_index, client, video_dir, results_dir)
            all_results.extend(task_results)
        except Exception as e:
            logger.exception(f"Task {task_index} raised an exception, skipping to next task: {e}")

    if args.all_tasks:
        suffix = f"_{args.checkpoint_name}" if args.checkpoint_name else ""
        summary_path = results_dir / f"{args.task_suite_name}_all_tasks{suffix}.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        n_success = sum(1 for r in all_results if r["success"])
        logger.info(
            f"Wrote combined results for {len(task_indices)} tasks "
            f"({len(all_results)} episodes, {n_success} successes) to {summary_path}"
        )


if __name__ == "__main__":
    eval_libsafety(tyro.cli(Args))
