"""
run_libsafety_cosmos_policy_eval.py

Evaluates NVIDIA Cosmos Policy on the LIBERO-Safety benchmark, reporting:
  - Task Success Rate (TSR): fraction of episodes where the task goal is achieved
  - Collision Avoidance Rate (CAR): fraction of episodes with no obstacle collision
  - Execution Time Steps (ETS): mean/median steps per episode

Structurally mirrors vlm_pipeline/run_safelibero_cosmos_policy_eval.py (main
repo) but points at the LIBERO-Safety task-suite registry instead of
SafeLIBERO's (get_libsafety_env / get_task_bddl_file_path_by_level_id /
get_task_init_states(level, level_id), per run_libsafety_eval_openvla.py's
get_libsafety_env() pattern) and hardcodes absolute paths into the MAIN repo
for the already-built Cosmos Policy container/venv/checkpoint cache, which
are NOT duplicated into this worktree (read-only reuse — see
docs/superpowers/specs/2026-07-12-cosmos-fastwam-libsafety-design.md).

No --safety_level flag: LIBERO-Safety encodes safety level per-task
(task.level / task.level_id), unlike SafeLIBERO's global L1/L2 argument.

Run inside the Cosmos Policy apptainer container (see
slurm/smoke_cosmos_libsafety.slurm for the full invocation):
  apptainer exec --nv \\
      --bind $REPO:$REPO \\
      /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/cosmos-policy/cosmos_policy.sif \\
      bash -c "/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/cosmos-policy/.venv_rhel8/bin/python \\
          run_libsafety_cosmos_policy_eval.py \\
              --task_suite_name human_safety --task_index 0 --num_trials_per_task 1"
"""

import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# ── Path setup: main-repo Cosmos Policy source (read-only reuse) ──────────────
_WORKTREE_ROOT = Path(__file__).resolve().parent
_MAIN_REPO_ROOT = Path("/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents")
COSMOS_POLICY_PATH = str(_MAIN_REPO_ROOT / "cosmos-policy")

# IMPORTANT: cosmos-policy's .venv_rhel8 already has SafeLIBERO's `libero`
# package pip-installed (for the SafeLIBERO cosmos-policy eval, which we must
# not break). That package lacks LIBERO-Safety's per-level API
# (get_task_bddl_file_path_by_level_id, Task.level/.level_id). Insert this
# worktree's LIBERO-Safety/ source at the FRONT of sys.path so `import libero`
# resolves to LIBERO-Safety's fork instead, without touching the shared env.
_LIBSAFETY_LIBERO_PATH = str(_WORKTREE_ROOT / "LIBERO-Safety")

for _p in [_LIBSAFETY_LIBERO_PATH, str(_WORKTREE_ROOT), COSMOS_POLICY_PATH]:
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import draccus
import imageio
import numpy as np
from libero.libero import benchmark

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.utils.utils import set_seed_everywhere

from libsafety_env_utils import get_libsafety_env

# ── Constants ──────────────────────────────────────────────────────────────────

COSMOS_LIBERO_CKPT = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"

# Dummy no-op action used during stabilisation warm-up
_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    # fmt: off

    # ── Cosmos Policy model ────────────────────────────────────────────────────
    model_family:   str = "cosmos"
    ckpt_path:      str = COSMOS_LIBERO_CKPT
    config:         str = "cosmos_predict2_2b_480p_libero__inference_only"
    config_file:    str = "cosmos_policy/config/config.py"

    # Cosmos inference knobs (match the reported 98.5% checkpoint)
    chunk_size:                    int  = 16
    num_open_loop_steps:           int  = 16
    num_denoising_steps_action:    int  = 5
    num_denoising_steps_future:    int  = 1
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value:     int  = 1
    use_third_person_image:        bool = True
    num_third_person_images:       int  = 1
    use_wrist_image:               bool = True
    num_wrist_images:              int  = 1
    use_proprio:                   bool = True
    normalize_proprio:             bool = True
    unnormalize_actions:           bool = True
    trained_with_image_aug:        bool = True
    use_jpeg_compression:          bool = True
    use_variance_scale:            bool = False
    ar_future_prediction:          bool = False
    ar_value_prediction:           bool = False
    ar_qvalue_prediction:          bool = False
    # NOTE: flip_images=False because _cosmos_flip() below already applies the
    # correct vertical-only flip (np.flipud) for Cosmos's training convention.
    flip_images:                   bool = False
    deterministic:                 bool = True
    deterministic_reset:           bool = False
    deterministic_reset_seed:      Optional[int] = None
    seed:                          int  = 195
    randomize_seed:                bool = False

    # Planning / best-of-N
    planning_model_config_name:                   str  = ""
    planning_model_ckpt_path:                     str  = ""
    use_ensemble_future_state_predictions:        bool = False
    num_future_state_predictions_in_ensemble:     int  = 3
    future_state_ensemble_aggregation_scheme:     str  = "average"
    use_ensemble_value_predictions:               bool = False
    num_value_predictions_in_ensemble:            int  = 5
    value_ensemble_aggregation_scheme:            str  = "average"
    search_depth:                                 int  = 1
    mask_current_state_action_for_value_prediction: bool = False
    mask_future_state_for_qvalue_prediction:      bool = False
    num_queries_best_of_n:                        int  = 1
    use_parallel_inference:                       bool = False
    available_gpus:                               str  = "0,1,2,3,4,5,6,7"
    parallel_timeout:                             int  = 15

    # HuggingFace paths (auto-downloaded from ckpt_path when using HF repo)
    dataset_stats_path:    str = f"{COSMOS_LIBERO_CKPT}/libero_dataset_statistics.json"
    t5_text_embeddings_path: str = f"{COSMOS_LIBERO_CKPT}/libero_t5_embeddings.pkl"

    # Cosmos model-specific (must be "libero" so get_action uses the LIBERO image key layout)
    suite: str = "libero"

    # ── Environment ────────────────────────────────────────────────────────────
    task_suite_name:     str = ""  # e.g. "human_safety", "affordance", ...
    task_index:           int = 0
    all_tasks:            bool = False
    num_steps_wait:      int = 20
    num_trials_per_task: int = 1
    max_steps:           int = 300
    env_img_res:         int = 256

    # ── Output ─────────────────────────────────────────────────────────────────
    video_output_dir:   str = "cosmos_video_libsafety"
    results_output_dir: str = "cosmos_benchmark_libsafety"
    local_log_dir:      str = "cosmos_benchmark_libsafety/logs"
    run_id_note:         Optional[str] = None

    # fmt: on


# ── Helpers ────────────────────────────────────────────────────────────────────

def setup_logging(cfg: EvalConfig):
    run_id = f"EVAL-{cfg.task_suite_name}-cosmos_policy-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    log_path = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(log_path, "w")
    logger.info(f"Logging to: {log_path}")
    return log_file, log_path, run_id


def log_message(msg: str, log_file=None) -> None:
    logger.info(msg)
    if log_file:
        log_file.write(msg + "\n")
        log_file.flush()


def _cosmos_flip(img: np.ndarray) -> np.ndarray:
    """Vertical flip only (np.flipud) — matches Cosmos's training preprocessing."""
    return np.ascontiguousarray(img[::-1])


def _validate_image(img: np.ndarray, name: str = "image") -> None:
    if img is None or img.size == 0:
        raise ValueError(f"Invalid {name}: empty or None")
    if img.std() < 5.0:
        raise ValueError(f"Invalid {name}: solid color (std={img.std():.2f})")
    hist, _ = np.histogram(img.flatten(), bins=32, range=(0, 256))
    threshold = img.size / 300
    if hist.std() < threshold:
        raise ValueError(f"Invalid {name}: uniform noise (hist_std={hist.std():.0f} < {threshold:.0f})")


def build_cosmos_obs(obs) -> dict:
    primary_raw = obs["agentview_image"]
    wrist_raw = obs["robot0_eye_in_hand_image"]
    _validate_image(primary_raw, "agentview_image")
    _validate_image(wrist_raw, "wrist_image")
    primary_img = _cosmos_flip(primary_raw)
    wrist_img = _cosmos_flip(wrist_raw)

    # Order matches Cosmos Policy LIBERO training data construction:
    # np.concatenate((gripper_qpos, eef_pos, eef_quat))
    proprio = np.concatenate([
        obs["robot0_gripper_qpos"],
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
    ])

    return {"primary_image": primary_img, "wrist_image": wrist_img, "proprio": proprio}


def build_cosmos_obs_fallback(obs, last_primary, last_wrist) -> dict:
    try:
        primary_img = _cosmos_flip(obs["agentview_image"])
    except Exception:
        primary_img = last_primary
    try:
        wrist_img = _cosmos_flip(obs["robot0_eye_in_hand_image"])
    except Exception:
        wrist_img = last_wrist

    proprio = np.concatenate([
        obs["robot0_gripper_qpos"],
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
    ])
    return {"primary_image": primary_img, "wrist_image": wrist_img, "proprio": proprio}


def save_episode_video(images: List[np.ndarray], path: str, fps: int = 30) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    for img in images:
        writer.append_data(img)
    writer.close()
    logger.info(f"Saved video: {path}")


# ── Episode / task loops ───────────────────────────────────────────────────────

def run_episode(cfg: EvalConfig, env, task_description: str, model, dataset_stats: dict,
                 log_file=None, initial_state=None):
    """Run one episode. Returns (success, collide_flag, replay_images, steps)."""
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()

    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(_DUMMY_ACTION)

    # Safety-violation detection: LIBERO-Safety's own BDDL `:constraints` block
    # (e.g. `CheckRobotContact`/`CheckContact` against a moving hazard object
    # like "bottle_of_shampoo_1_with_hand_1") is evaluated every step by
    # bddl_base_domain.py's _check_constraint() and returned as info['cost']
    # (a dict of predicate_name -> 1 if violated, else 0). This REPLACES the
    # SafeLIBERO-style "static obstacle displaced >1mm" heuristic, which is a
    # silent no-op on LIBERO-Safety: its hazard objects are dynamic (moving
    # via a scripted trajectory, not bumped by the robot) and never have
    # "obstacle" in their joint name, so `obstacle_name` was always None and
    # collide_flag could never become True (confirmed via LIBERO-Safety BDDL
    # inspection and full-eval logs never printing "Active obstacle:").
    collide_flag = False

    action_queue: deque = deque(maxlen=cfg.num_open_loop_steps)
    replay_images: List[np.ndarray] = []
    last_primary = np.zeros((cfg.env_img_res, cfg.env_img_res, 3), dtype=np.uint8)
    last_wrist = np.zeros((cfg.env_img_res, cfg.env_img_res, 3), dtype=np.uint8)
    success = False
    t = 0

    try:
        while t < cfg.max_steps:
            if os.environ.get("DETERMINISTIC", "").lower() == "true":
                set_seed_everywhere(0)

            try:
                cosmos_obs = build_cosmos_obs(obs)
                last_primary = cosmos_obs["primary_image"]
                last_wrist = cosmos_obs["wrist_image"]
                replay_images.append(np.ascontiguousarray(last_primary.copy()))
            except ValueError as exc:
                logger.warning(f"t={t}: corrupted frame ({exc}); using previous frame")
                cosmos_obs = build_cosmos_obs_fallback(obs, last_primary, last_wrist)
                replay_images.append(replay_images[-1].copy() if replay_images else last_primary.copy())

            if not action_queue:
                action_dict = get_action(
                    cfg, model, dataset_stats, cosmos_obs, task_description,
                    seed=cfg.seed, randomize_seed=not cfg.deterministic,
                    num_denoising_steps_action=cfg.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=False,
                )
                actions = action_dict["actions"]
                action_queue.extend(actions)

            action = action_queue.popleft()
            obs, _, done, info = env.step(action.tolist())

            if not collide_flag:
                cost = info.get("cost", {}) if info else {}
                if any(v for v in cost.values()):
                    collide_flag = True
                    log_message(f"Safety constraint violated at t={t}: {cost}", log_file)

            if done:
                success = True
                break
            t += 1

    except Exception as exc:
        log_message(f"Episode error: {exc}", log_file)

    return success, collide_flag, replay_images, t


def run_task(cfg: EvalConfig, task_suite, task_index: int, model, dataset_stats: dict, log_file=None):
    task = task_suite.get_task(task_index)
    # LIBERO-Safety's get_task_init_states() takes (level, level_id), not a
    # single global task index — each Task carries its own .level/.level_id.
    initial_states = task_suite.get_task_init_states(task.level, task.level_id)
    env, task_description = get_libsafety_env(task_suite, task, resolution=cfg.env_img_res)

    log_message(f"\n{'='*60}", log_file)
    log_message(f"Task {task_index}: {task_description}", log_file)
    log_message(f"{'='*60}", log_file)

    task_episodes = task_successes = task_collides = 0
    timesteps_list: List[int] = []

    try:
        for episode_idx in range(cfg.num_trials_per_task):
            initial_state = initial_states[episode_idx]
            log_message(f"  Episode {episode_idx + 1}/{cfg.num_trials_per_task}", log_file)

            success, collide, replay_images, steps = run_episode(
                cfg, env, task_description, model, dataset_stats, log_file, initial_state
            )

            task_episodes += 1
            task_successes += int(success)
            task_collides += int(collide)
            timesteps_list.append(steps)
            log_message(
                f"  success={success}  collide={collide}  safe_success={success and not collide}  steps={steps}",
                log_file,
            )

            if episode_idx < 10 or (episode_idx + 1) % 10 == 0:
                short_desc = task_description.lower().replace(" ", "_")[:50]
                video_path = os.path.join(
                    cfg.video_output_dir, cfg.task_suite_name, str(task_index),
                    f"episode_{episode_idx}_{short_desc}.mp4",
                )
                save_episode_video(replay_images, video_path)
    finally:
        try:
            env.close()
        except Exception:
            logger.warning("env.close() failed/unavailable; continuing anyway", exc_info=True)

    tsr = task_successes / task_episodes if task_episodes > 0 else 0.0
    car = (task_episodes - task_collides) / task_episodes if task_episodes > 0 else 0.0
    log_message(
        f"  Task {task_index} summary  TSR={tsr:.3f}  CAR={car:.3f}  "
        f"ETS(mean)={np.mean(timesteps_list):.1f}",
        log_file,
    )
    return {
        "task_index": task_index,
        "task_description": task_description,
        "episodes": task_episodes,
        "successes": task_successes,
        "collisions": task_collides,
        "TSR": round(tsr, 4),
        "CAR": round(car, 4),
        "ETS_mean": round(float(np.mean(timesteps_list)) if timesteps_list else 0.0, 2),
        "ETS_median": round(float(np.median(timesteps_list)) if timesteps_list else 0.0, 2),
    }


def checkpoint_tag(ckpt_path: str) -> str:
    return str(ckpt_path).rstrip("/").split("/")[-1]


# ── Main entry point ───────────────────────────────────────────────────────────

@draccus.wrap()
def eval_libsafety(cfg: EvalConfig) -> dict:
    """Evaluate Cosmos Policy on LIBERO-Safety tasks."""
    assert cfg.task_suite_name, "task_suite_name must be set to a real LIBERO-Safety suite name"
    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    set_seed_everywhere(cfg.seed)

    logger.info("Loading Cosmos Policy model…")
    model, cosmos_config = get_model(cfg)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

    log_file, log_path, run_id = setup_logging(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()

    task_indices = [cfg.task_index] if not cfg.all_tasks else list(range(task_suite.n_tasks))
    log_message(
        f"Suite: {cfg.task_suite_name}  Tasks: {len(task_indices)}  Trials/task: {cfg.num_trials_per_task}",
        log_file,
    )

    results: dict = {}
    all_episodes = all_successes = all_collides = 0
    all_timesteps: List[int] = []
    task_errors = {}

    for task_index in task_indices:
        try:
            r = run_task(cfg, task_suite, task_index, model, dataset_stats, log_file)
        except Exception as e:
            logger.exception(f"Task {task_index} raised an exception, skipping to next task: {e}")
            task_errors[task_index] = repr(e)
            continue

        results[f"task_{task_index}"] = r
        all_episodes += r["episodes"]
        all_successes += r["successes"]
        all_collides += r["collisions"]
        all_timesteps.extend([r["ETS_mean"]] * r["episodes"])

    overall_tsr = all_successes / all_episodes if all_episodes > 0 else 0.0
    overall_car = (all_episodes - all_collides) / all_episodes if all_episodes > 0 else 0.0

    results["overall"] = {
        "model": cfg.ckpt_path,
        "suite": cfg.task_suite_name,
        "total_episodes": all_episodes,
        "total_successes": all_successes,
        "total_collisions": all_collides,
        "TSR": round(overall_tsr, 4),
        "CAR": round(overall_car, 4),
        "ETS_mean": round(float(np.mean(all_timesteps)) if all_timesteps else 0.0, 2),
        "ETS_median": round(float(np.median(all_timesteps)) if all_timesteps else 0.0, 2),
        "task_errors": task_errors,
    }

    results_dir = os.path.join(cfg.results_output_dir, cfg.task_suite_name)
    os.makedirs(results_dir, exist_ok=True)
    suffix = "all_tasks" if cfg.all_tasks else f"task{cfg.task_index}"
    results_path = os.path.join(
        results_dir, f"{cfg.task_suite_name}_{checkpoint_tag(cfg.ckpt_path)}_{suffix}.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Results saved to: {results_path}", log_file)

    log_message(
        f"\n{'='*60}\nLIBERO-Safety Evaluation Summary — Cosmos Policy\n"
        f"Suite: {cfg.task_suite_name}  Total episodes: {all_episodes}\n"
        f"TSR: {overall_tsr:.3f}  CAR: {overall_car:.3f}\n{'='*60}",
        log_file,
    )

    if log_file:
        log_file.close()

    if task_errors and len(task_errors) == len(task_indices):
        logger.error(f"ALL {len(task_indices)} tasks raised exceptions (0 episodes ran): {task_errors}")
        sys.exit(1)

    return results


if __name__ == "__main__":
    eval_libsafety()
