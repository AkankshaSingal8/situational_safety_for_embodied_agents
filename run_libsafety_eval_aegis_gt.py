"""
run_libsafety_eval_aegis_gt.py

AEGIS baseline on LIBERO-Safety, ground-truth-obstacle variant -- matches
how English et al. (arXiv:2607.01378) actually benchmarked AEGIS: "Obstacle
positions are obtained from the simulator, isolating the control
contribution from perception challenges" (their Sec 5.1). This applies to
BOTH their own method and the AEGIS baseline in their Table 1.

This is NOT a rewrite of AEGIS's method -- the CBF construction and QP
correction (compute_h_coeffs_3d, project_matrix, from
vlsa-aegis/main/utils.py) are imported UNMODIFIED and drive the actual
safety intervention, exactly as in vlsa-aegis/main/main_aegis.py. What's
swapped is only the obstacle *perception* step:

  - vlsa-aegis/main/main_aegis.py's original path: GroundingDINO detection
    (obstacle_detection(), which calls a hardcoded-empty ZhipuAI API key --
    broken as checked in) -> two-camera point cloud -> mvee_cvxpy ellipsoid
    fit (fit_ellipse()). Also hardcodes SafeLIBERO-only scene-geometry
    bounds in filtering_points() (crashes via UnboundLocalError on any
    suite name outside "spatial"/"object"/"goal"/"long").
  - This script's path (matching the paper's own privileged-tier setup):
    obstacle identity from the task's own BDDL :constraints block (the
    CheckRobotContact predicate's object argument -- same ground-truth
    hazard-object convention as the CAR fix in run_libsafety_eval_openpi.py
    / run_libsafety_cosmos_policy_eval.py / run_libsafety_fastwam_eval.py),
    position from obs[f"{hazard_name}_pos"], and an isotropic-sphere
    ellipsoid (R=identity, Q_diag=[r,r,r]) in place of the perceived/fit
    ellipsoid -- matching English et al.'s own "ellipsoid eef vs sphere
    obstacle" SDF convention (their Sec on collision detection).

Uses the OFF-THE-SHELF pi05_libero checkpoint (not the LIBERO-Safety
fine-tune) -- AEGIS's premise is a post-hoc filter with NO retraining
required, matching how English et al.'s Table 1 sets up the AEGIS baseline
against the same base VLA as their own method.

Run inside the libsafety_client conda env, after a pi05_libero server is
up (same server infra as run_libsafety_eval_openpi.py):
  python run_libsafety_eval_aegis_gt.py --task_suite_name obstacle_avoidance \
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

import cvxpy as cp
import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from scipy.spatial.transform import Rotation as R

_AEGIS_MAIN_DIR = str(
    pathlib.Path(__file__).resolve().parent / "vlsa-aegis" / "main"
)
if _AEGIS_MAIN_DIR not in sys.path:
    sys.path.insert(0, _AEGIS_MAIN_DIR)
from utils import compute_h_coeffs_3d, project_matrix  # noqa: E402  (AEGIS's own CBF math, unmodified)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

OBSTACLE_RADIUS = 0.06  # isotropic sphere semi-axis, matches English et al.'s obstacle SDF radius scale
EEF_Q_DIAG = np.array([0.06, 0.12, 0.11])  # same eef ellipsoid semi-axes as vlsa-aegis/main/main_aegis.py


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = ""
    task_index: int = 0
    all_tasks: bool = False
    num_trials_per_task: int = 1
    episode_offset: int = 0
    num_steps_wait: int = 10
    max_steps: int = 300
    checkpoint_name: str = "aegis_gt_pi05_libero"

    seed: int = 7
    video_out_path: str = "LIBERO-Safety/rollouts"
    results_out_path: str = "LIBERO-Safety/results"


def _get_env(task_suite, task, resolution, seed):
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_file = task_suite.get_task_bddl_file_path_by_level_id(task.level, task.level_id)
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language


def _find_hazard_object_name(env):
    """Extract the CheckRobotContact predicate's object argument from the
    task's own BDDL :constraints block -- the ground-truth hazard object,
    same convention as the CAR fix (info['cost'] keying).

    NOTE: LIBERO-Safety's parsed predicate names are lowercased
    ('checkrobotcontact', not 'CheckRobotContact') -- confirmed via direct
    inspection of env.env.parsed_problem['constraints'], e.g.
    [['checkrobotcontact', 'bottle_of_perfume_1'],
     ['checkrobotcontact', 'white_place_box_1'],
     ['checkcontact', 'akita_black_bowl_1', 'bottle_of_perfume_1'], ...].
    A task can have MULTIPLE checkrobotcontact hazards (matching AEGIS's
    single-obstacle design, we take the first one, same as
    vlsa-aegis/main/main_aegis.py's obstacle_names loop, which also stops
    at its first in-bounds match).
    """
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
    processed = (
        task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    )
    path = (
        video_dir
        / f"{DATE_TIME}--{task_suite_name}--task{task_index}--episode={episode_idx}--success={success}--task={processed}.mp4"
    )
    imageio.mimwrite(path, [np.asarray(x) for x in replay_images], fps=30)
    return str(path)


def eval_one_task(args, task_suite, task_index, client, video_dir, results_dir):
    task = task_suite.get_task(task_index)
    # LIBERO-Safety's get_task_init_states() takes (level, level_id), not a
    # flat task index (see run_libsafety_eval_openpi.py's identical note).
    initial_states = task_suite.get_task_init_states(task.level, task.level_id)
    env, task_description = _get_env(task_suite, task, LIBERO_ENV_RESOLUTION, args.seed)

    hazard_name = None
    task_results = []
    for episode_idx in range(args.episode_offset, args.episode_offset + args.num_trials_per_task):
        env.reset()
        action_plan = collections.deque()
        obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])

        t = 0
        while t < args.num_steps_wait:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1

        if hazard_name is None:
            hazard_name = _find_hazard_object_name(env)
        flag_safety_control = hazard_name is not None and f"{hazard_name}_pos" in obs

        eef_pos = obs["robot0_eef_pos"]
        eef_quat = obs["robot0_eef_quat"]
        R1 = R.from_quat(eef_quat).as_matrix()
        offset_world = R1 @ np.array([0, 0, -0.08])
        p1 = eef_pos + offset_world

        if flag_safety_control:
            p2 = np.array(obs[f"{hazard_name}_pos"])
            R2 = np.eye(3)
            Q2_diag = np.array([OBSTACLE_RADIUS, OBSTACLE_RADIUS, OBSTACLE_RADIUS])
            z_fixed = (p2 - p1)
            z_fixed = z_fixed / (np.linalg.norm(z_fixed) + 1e-10)

        t = 0
        collide_flag = False
        replay_images = []
        done_flag = False
        max_steps = args.max_steps

        while t < max_steps:
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            replay_images.append(img)
            img_r = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
            wrist_r = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))

            if not action_plan:
                element = {
                    "observation/image": img_r,
                    "observation/wrist_image": wrist_r,
                    "observation/state": np.concatenate(
                        (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                    "prompt": str(task_description),
                }
                action_chunk = client.infer(element)["actions"]
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()

            if flag_safety_control:
                v_ref = R1.T @ action[:3]
                u_v_ref = 5 * v_ref
                u_omega_ref = 5 * action[3:6]

                a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(p1, EEF_Q_DIAG, R1, p2, Q2_diag, R2, z_fixed)
                a_u_v = 0.2 * a_v
                a_u_omega = 0.2 * a_omega
                u_z_nom = 10 * mu_row

                u = cp.Variable(9)
                W = np.diag([1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25, 1.0, 1.0, 1.0])
                u_ref_vec = np.hstack([u_v_ref, u_omega_ref, u_z_nom])
                objective = cp.Minimize(cp.quad_form(u - u_ref_vec, W))
                constraints = [a_u_v @ u[:3] + a_u_omega @ u[3:6] + a_uz @ u[6:] + 10 * h >= 0]
                prob = cp.Problem(objective, constraints)
                prob.solve(solver=cp.OSQP)

                if u.value is not None:
                    u_v, u_omega_val, u_z = u.value[:3], u.value[3:6], u.value[6:]
                else:
                    u_v, u_omega_val, u_z = action[:3], action[3:6], u_z_nom

                Id = np.eye(len(z_fixed))
                dz = (Id - np.outer(z_fixed, z_fixed)) @ u_z
                z_fixed = z_fixed + dz * 0.05
                z_fixed = z_fixed / (np.linalg.norm(z_fixed) + 1e-10)

                action_input = np.zeros(7)
                action_input[:3] = 0.2 * R1 @ u_v
                action_input[3:6] = 0.2 * u_omega_val
                action_input[6] = action[6]
                obs, reward, done, info = env.step(action_input.tolist())
            else:
                obs, reward, done, info = env.step(action.tolist())

            if not collide_flag:
                cost = info.get("cost", {}) if info else {}
                if any(v for v in cost.values()):
                    collide_flag = True

            eef_pos = obs["robot0_eef_pos"]
            eef_quat = obs["robot0_eef_quat"]
            R1 = R.from_quat(eef_quat).as_matrix()
            p1 = eef_pos + R1 @ np.array([0, 0, -0.08])

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
            "hazard_object": hazard_name,
            "num_steps": t,
            "video": video_path,
            "checkpoint": args.checkpoint_name,
        }
        task_results.append(result)
        logger.info(f"task={task_index} episode={episode_idx}: success={done_flag} collision={collide_flag}")

    ep_tag = f"_ep{args.episode_offset}" if args.episode_offset > 0 else ""
    out_path = results_dir / f"{args.task_suite_name}_task{task_index}{ep_tag}_{args.checkpoint_name}.json"
    out_path.write_text(json.dumps(task_results, indent=2))
    return task_results


def eval_libsafety(args: Args) -> None:
    assert args.task_suite_name, "task_suite_name must be set"
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    video_dir = pathlib.Path(args.video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)
    results_dir = pathlib.Path(args.results_out_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logger.info(f"Connected to policy server at {args.host}:{args.port}; metadata={client.get_server_metadata()}")

    task_indices = list(range(task_suite.n_tasks)) if args.all_tasks else [args.task_index]
    for task_index in task_indices:
        logger.info(f"=== Starting suite='{args.task_suite_name}' task_index={task_index} ===")
        eval_one_task(args, task_suite, task_index, client, video_dir, results_dir)


if __name__ == "__main__":
    args = tyro.cli(Args)
    eval_libsafety(args)
