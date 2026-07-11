"""
run_libsafety_eval_openvla.py

Eval-only driver for OpenVLA / OpenVLA-OFT on the LIBERO-Safety benchmark,
using standard off-the-shelf LIBERO-pretrained checkpoints (not the paper's
own fine-tuned weights). Structurally mirrors run_libero_eval.py (in the
main repo) but points at the LIBERO-Safety task-suite registry instead of
SafeLIBERO's, and supports both base OpenVLA and OpenVLA-OFT checkpoints
via config flags (num_images_in_input, use_proprio, use_l1_regression, ...).

Run inside the libsafety_openvla conda env:
  conda activate /ocean/projects/cis250185p/asingal/envs/libsafety_openvla
  export MUJOCO_GL=egl
  export LIBERO_CONFIG_PATH=/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark/LIBERO-Safety/.libero_config
  cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/libero-safety-benchmark
  python run_libsafety_eval_openvla.py \
      --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
      --task_suite_name human_safety --task_index 0 --num_trials_per_task 1

NOTE on LIBERO_CONFIG_PATH: it must point at a DIRECTORY containing
config.yaml (produced by the Task 5 setup script), not at a .yaml file
directly.

NOTE on openvla-oft location: openvla-oft/ is only checked out as a
populated submodule in the MAIN repo
(/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/openvla-oft),
not inside this worktree, so we hardcode that absolute path below rather
than deriving it from __file__.
"""

import json
import logging
import pathlib
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import imageio
import numpy as np
import tqdm

from libero.libero import benchmark

# openvla-oft only exists (populated) in the MAIN repo, not this worktree.
OPENVLA_OFT_PATH = "/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/openvla-oft"
sys.path.insert(0, OPENVLA_OFT_PATH)

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from experiments.robot.openvla_utils import prepare_images_for_vla, resize_image_for_policy  # noqa: E402
from experiments.robot.robot_utils import (  # noqa: E402
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

import torch  # noqa: E402

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

try:
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK
except ImportError:
    NUM_ACTIONS_CHUNK = 8

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

# ---------------------------------------------------------------------------
# Compatibility shim: base (non-OFT) OpenVLA checkpoints on HF Hub are loaded
# with trust_remote_code=True, which pulls in THAT repo's own bundled
# modeling_prismatic.py (not openvla-oft's local prismatic package). The
# base-OpenVLA-era PrismaticVisionBackbone class predates OFT's multi-image
# support and has no set_num_images_in_input() method, but
# experiments.robot.openvla_utils.get_vla() (in openvla-oft, which we must
# not modify) unconditionally calls
# vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input).
# Patch a no-op fallback onto torch.nn.Module so that call degrades
# gracefully for single-image (num_images_in_input=1) base-OpenVLA usage,
# where the model's native default is already single-image and there is
# nothing to configure.
# ---------------------------------------------------------------------------
import torch.nn as _nn  # noqa: E402

if not hasattr(_nn.Module, "set_num_images_in_input"):
    def _noop_set_num_images_in_input(self, num_images_in_input):
        logger.warning(
            "vision_backbone has no set_num_images_in_input() (base OpenVLA "
            f"modeling code) — ignoring request to set num_images_in_input="
            f"{num_images_in_input}. This is expected/harmless for "
            "num_images_in_input=1 (base OpenVLA); if you need >1 images "
            "with a checkpoint whose bundled modeling code lacks this "
            "method, this shim will silently no-op instead of configuring it."
        )

    _nn.Module.set_num_images_in_input = _noop_set_num_images_in_input


@dataclass
class LibSafetyEvalConfig:
    # --- Model-specific parameters ---
    model_family: str = "openvla"
    pretrained_checkpoint: str = "openvla/openvla-7b-finetuned-libero-spatial"

    # Base OpenVLA (off-the-shelf) defaults: single image, no proprio, no
    # action head, discrete/autoregressive action decoding, unchunked
    # (1 action per VLA query). Override these on the CLI for OFT
    # checkpoints, e.g.:
    #   --use_l1_regression true --use_proprio true --num_images_in_input 2
    #   --num_open_loop_steps 8
    use_l1_regression: bool = False
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = False
    center_crop: bool = True
    num_open_loop_steps: int = 1
    lora_rank: int = 32
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Action un-normalization key into the VLA's norm_stats dict. If left
    # empty, we auto-detect: if the loaded checkpoint's norm_stats has
    # exactly one key, use it; otherwise try task_suite_name (with a
    # "_no_noops" fallback), and fail loudly if neither resolves.
    unnorm_key: Union[str, Path] = ""

    # --- LIBERO-Safety environment parameters ---
    task_suite_name: str = ""  # e.g. "human_safety", "affordance", ...
    task_index: int = 0
    num_trials_per_task: int = 1
    num_steps_wait: int = 10
    max_steps: int = 300
    env_img_res: int = 256

    seed: int = 7
    video_out_path: str = "LIBERO-Safety/rollouts"
    results_out_path: str = "LIBERO-Safety/results"


def resolve_unnorm_key(cfg: LibSafetyEvalConfig, model) -> str:
    """Pick a valid action-normalization key present in model.norm_stats."""
    if cfg.unnorm_key:
        if cfg.unnorm_key not in model.norm_stats:
            raise ValueError(
                f"Requested unnorm_key='{cfg.unnorm_key}' not found in model.norm_stats "
                f"(available: {list(model.norm_stats.keys())})"
            )
        return cfg.unnorm_key

    available = list(model.norm_stats.keys())
    if len(available) == 1:
        logger.info(f"Auto-selected unnorm_key='{available[0]}' (only key in model.norm_stats)")
        return available[0]

    candidate = cfg.task_suite_name
    if candidate in model.norm_stats:
        return candidate
    if f"{candidate}_no_noops" in model.norm_stats:
        return f"{candidate}_no_noops"

    raise ValueError(
        f"Could not auto-resolve unnorm_key. task_suite_name='{cfg.task_suite_name}' not found "
        f"(available keys: {available}). Pass --unnorm_key explicitly."
    )


def prepare_observation(obs, resize_size, cfg: LibSafetyEvalConfig):
    """Prepare observation dict for policy input (mirrors run_libero_eval.py)."""
    img = get_libero_image(obs)
    img_resized = resize_image_for_policy(img, resize_size)

    observation = {"full_image": img_resized}

    if cfg.num_images_in_input > 1:
        wrist_img = get_libero_wrist_image(obs)
        observation["wrist_image"] = resize_image_for_policy(wrist_img, resize_size)

    if cfg.use_proprio:
        observation["state"] = np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        )

    return observation, img


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def get_base_openvla_action(cfg, model, processor, observation, task_description):
    """Query a base (non-OFT) OpenVLA checkpoint for a single action.

    NOTE: We cannot use experiments.robot.robot_utils.get_action /
    openvla_utils.get_vla_action (openvla-oft, which we must not modify)
    for base OpenVLA checkpoints: get_vla_action() unconditionally does
    `action, _ = vla.predict_action(...)`, assuming a 2-tuple return like
    OFT's custom action-head predict_action(). Base OpenVLA checkpoints
    loaded from HF Hub run THEIR OWN bundled modeling_prismatic.py, whose
    predict_action() returns a single (action_dim,) np.ndarray (one
    autoregressively-decoded action, no chunking) — so the tuple-unpack
    raises "too many values to unpack". This function reimplements the
    single-image, no-proprio inference path directly against that
    contract, returning a length-1 list of actions.
    """
    all_images = prepare_images_for_vla([observation["full_image"]], cfg)
    primary_image = all_images[0]

    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)

    action = model.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)
    return [action]


def get_libsafety_env(task_suite, task, resolution=256):
    """Build the LIBERO env for a LIBERO-Safety task.

    NOTE: We cannot reuse experiments.robot.libero.libero_utils.get_libero_env
    (from openvla-oft, which we must not modify) because it assumes the
    upstream-LIBERO/SafeLIBERO bddl-file layout
    (<bddl_files>/<problem_folder>/<bddl_file>). LIBERO-Safety nests an
    additional per-difficulty-level directory
    (<bddl_files>/<problem_folder>/L{level}/<bddl_file>), addressable only
    via Benchmark.get_task_bddl_file_path_by_level_id(level, level_id).
    """
    task_description = task.language
    task_bddl_file = task_suite.get_task_bddl_file_path_by_level_id(task.level, task.level_id)
    if task_bddl_file is None or not pathlib.Path(task_bddl_file).exists():
        raise FileNotFoundError(f"LIBERO-Safety bddl file not found for task {task}: {task_bddl_file}")
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


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
    writer = imageio.get_writer(str(mp4_path), fps=30)
    for img in replay_images:
        writer.append_data(img)
    writer.close()
    logger.info(f"Saved rollout MP4 at path {mp4_path}")
    return str(mp4_path)


@draccus.wrap()
def eval_libsafety(cfg: LibSafetyEvalConfig) -> None:
    assert cfg.task_suite_name, "task_suite_name must be set to a real LIBERO-Safety suite name"
    set_seed_everywhere(cfg.seed)

    model = get_model(cfg)
    resize_size = get_image_resize_size(cfg)
    cfg.unnorm_key = resolve_unnorm_key(cfg, model)
    logger.info(f"Using unnorm_key='{cfg.unnorm_key}'")

    processor = None
    if cfg.model_family == "openvla":
        from experiments.robot.openvla_utils import get_processor

        processor = get_processor(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task = task_suite.get_task(cfg.task_index)
    # NOTE: LIBERO-Safety's Benchmark.get_task_init_states() takes (level, level_id),
    # not a single global task index — unlike upstream LIBERO/SafeLIBERO. Each Task
    # carries its own .level/.level_id, which we must look up explicitly.
    initial_states = task_suite.get_task_init_states(task.level, task.level_id)

    env, task_description = get_libsafety_env(task_suite, task, resolution=cfg.env_img_res)
    logger.info(f"Task description: {task_description}")

    video_dir = pathlib.Path(cfg.video_out_path)
    results_dir = pathlib.Path(cfg.results_out_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    episode_results = []
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])

        action_queue = deque(maxlen=cfg.num_open_loop_steps)
        replay_images = []
        t, done, success = 0, False, False

        while t < cfg.num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1

        t = 0
        try:
            while t < cfg.max_steps:
                observation, img = prepare_observation(obs, resize_size, cfg)
                replay_images.append(img)

                if len(action_queue) == 0:
                    if cfg.model_family == "openvla" and not cfg.use_l1_regression and not cfg.use_diffusion:
                        # Base (non-OFT) OpenVLA: bypass openvla-oft's get_action(), which
                        # assumes a (action, aux) tuple return incompatible with base
                        # OpenVLA's predict_action(). See get_base_openvla_action() docstring.
                        actions = get_base_openvla_action(cfg, model, processor, observation, task_description)
                    else:
                        actions = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                            use_film=cfg.use_film,
                        )
                    action_queue.extend(actions)

                action = action_queue.popleft()
                action = process_action(action, cfg.model_family)

                obs, _, done, _ = env.step(action.tolist() if hasattr(action, "tolist") else list(action))
                if done:
                    success = True
                    break
                t += 1
        except Exception as e:
            logger.exception(f"Episode {episode_idx} raised an exception: {e}")

        mp4_path = save_rollout_video(
            replay_images, video_dir, cfg.task_suite_name, cfg.task_index, episode_idx, success, task_description
        )
        episode_results.append(
            {
                "task": task_description,
                "task_suite": cfg.task_suite_name,
                "task_index": cfg.task_index,
                "episode": episode_idx,
                "success": success,
                "num_steps": t,
                "video": mp4_path,
                "pretrained_checkpoint": str(cfg.pretrained_checkpoint),
            }
        )
        logger.info(f"Episode {episode_idx}: success={success}, steps={t}")

    out_path = results_dir / f"{cfg.task_suite_name}_task{cfg.task_index}.json"
    with open(out_path, "w") as f:
        json.dump(episode_results, f, indent=2)
    logger.info(f"Wrote results to {out_path}")


if __name__ == "__main__":
    eval_libsafety()
