"""
run_safelibero_fastwam_eval.py

Evaluates the Fast-WAM policy on SafeLIBERO benchmark tasks, reporting:
  - Task Success Rate (TSR): fraction of episodes where task goal is achieved
  - Collision Avoidance Rate (CAR): fraction of episodes with no obstacle collision
  - Execution Time Steps (ETS): mean/median steps per successful/complete episode

Fast-WAM is a video-diffusion-based visuomotor policy (built on Wan2.2-TI2V-5B)
that jointly predicts future video frames and action sequences via a shared MoT
(Motion of Things) backbone.

This script mirrors the structure of run_safelibero_cosmos_policy_eval.py, using
argparse (not draccus/Hydra) and loading the Hydra config programmatically so
CLI args can override checkpoint path and dataset stats.

Usage (run from project root under fastwam_env):
  export MUJOCO_GL=egl
  export HF_HOME=/ocean/projects/cis250185p/asingal/.hf_cache
  python vlm_pipeline/run_safelibero_fastwam_eval.py \\
    --task_suite_name safelibero_spatial \\
    --safety_level I \\
    --num_trials_per_task 10 \\
    --ckpt_path /path/to/fastwam_checkpoint.pt \\
    --stats_path /path/to/dataset_stats.json
"""

import argparse
import inspect
import json
import logging
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import imageio
import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
_VLM_PIPELINE = _REPO_ROOT / "vlm_pipeline"
_SAFELIBERO = _REPO_ROOT / "SafeLIBERO" / "safelibero"
_FASTWAM_SRC = _REPO_ROOT / "fast-wam" / "FastWAM" / "src"
_FASTWAM_EXPERIMENTS = _REPO_ROOT / "fast-wam" / "FastWAM" / "experiments"
_FASTWAM_CONFIGS = _REPO_ROOT / "fast-wam" / "FastWAM" / "configs"

for _p in [
    str(_VLM_PIPELINE),
    str(_SAFELIBERO),
    str(_FASTWAM_SRC),
    str(_FASTWAM_EXPERIMENTS),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Hydra programmatic config loading ────────────────────────────────────────
# Register Hydra resolvers used by Fast-WAM configs BEFORE any config compose call.
from omegaconf import OmegaConf, DictConfig
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("max", lambda x: max(x), replace=True)
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)], replace=True)

# ── Fast-WAM imports ──────────────────────────────────────────────────────────
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.utils.pytorch_utils import set_global_seed

from PIL import Image as PILImage

# ── SafeLIBERO + libero imports ───────────────────────────────────────────────
from libero.libero import benchmark

from safelibero_utils import (
    get_safelibero_env,
    get_safelibero_image,
    get_safelibero_wrist_image,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

TASK_MAX_STEPS = {
    "safelibero_spatial": 300,
    "safelibero_object":  300,
    "safelibero_goal":    300,
    "safelibero_long":    600,
}

_DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ── Image helpers (mirror eval_libero_single.py) ──────────────────────────────

def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = PILImage.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize(
        (round(src_w * scale), round(src_h * scale)), resample=PILImage.BILINEAR
    )
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


# ── Proprioception helpers ────────────────────────────────────────────────────

def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x,y,z,w] to axis-angle (3,)."""
    if quat[3] > 1.0:
        quat = quat.copy(); quat[3] = 1.0
    elif quat[3] < -1.0:
        quat = quat.copy(); quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _extract_proprio(obs: dict) -> np.ndarray:
    """Build 8-dim proprioception: [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]."""
    return np.concatenate([
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"][:2],
    ]).astype(np.float32)


def _normalize_proprio(proprio: np.ndarray, processor: FastWAMProcessor) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]
    state_batch = {
        "state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}
    }
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    """Denormalize action tensor [T, D] or [1, T, D] → numpy [T, D]."""
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval expects a single merged action key in shape_meta['action']."
        )
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_fastwam_model(ckpt_path: str, stats_path: str, device: str, args):
    """Load Fast-WAM model using Hydra programmatic config compose.

    Mirrors eval_libero_single.py:
        model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
        model.load_checkpoint(ckpt)
        processor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(dataset_stats)
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(_FASTWAM_CONFIGS),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="sim_libero",
            overrides=[
                f"ckpt={ckpt_path}",
                f"gpu_id={args.gpu_id}",
                f"EVALUATION.num_steps_wait=30",
                f"EVALUATION.replan_steps={args.replan_steps}",
                f"EVALUATION.action_horizon={args.action_horizon}",
                f"EVALUATION.num_inference_steps={args.num_inference_steps}",
                f"EVALUATION.rand_device=cpu",
                f"EVALUATION.text_cfg_scale=1.0",
                "EVALUATION.binarize_gripper=false",
                "model.load_text_encoder=true",
                "model.skip_dit_load_from_pretrain=true",
                "model.action_dit_pretrained_path=null",
            ],
        )

    model_dtype = torch.bfloat16
    logger.info("Instantiating Fast-WAM model on %s (dtype=%s)...", device, model_dtype)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)

    logger.info("Loading checkpoint: %s", ckpt_path)
    model.load_checkpoint(ckpt_path)
    model = model.to(device).eval()

    logger.info("Instantiating processor...")
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()

    logger.info("Loading dataset stats: %s", stats_path)
    dataset_stats = load_dataset_stats_from_json(str(stats_path))
    processor.set_normalizer_from_stats(dataset_stats)

    # Derive input image dimensions from config
    video_size = cfg.data.train.get("video_size", [224, 448])
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    num_cameras = int(processor.num_output_cameras)

    logger.info(
        "Model ready. video_size=%s  num_cameras=%d  action_dim=%d  proprio_dim=%d",
        list(video_size), num_cameras, processor.action_output_dim, processor.proprio_output_dim,
    )

    return model, processor, cfg, input_h, input_w, num_cameras


# ── Observation preprocessing ─────────────────────────────────────────────────

def build_fastwam_obs(
    obs: dict,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    input_h: int,
    input_w: int,
    num_cameras: int,
    device: str,
    dtype: torch.dtype,
):
    """Build model input tensor and normalized proprio from SafeLIBERO observation.

    Images are 180°-flipped by get_safelibero_image / get_safelibero_wrist_image
    (matching training preprocessing). Then center-crop-resize and horizontal
    concat (for 2-camera models), followed by [-1, 1] normalization.

    Returns:
        img_tensor: [1, 3, H, W*num_cams] float tensor in [-1, 1]
        proprio_tensor: normalized proprio as returned by _normalize_proprio
    """
    image_meta = processor.shape_meta["images"]
    concat_mode = cfg.data.train.get("concat_multi_camera", "horizontal")

    def _meta_hw(idx: int):
        shape = image_meta[idx]["shape"]  # [C, H, W]
        return int(shape[1]), int(shape[2])

    primary_img = get_safelibero_image(obs, validate=True)

    if num_cameras == 1:
        ph, pw = _meta_hw(0)
        rgb = _center_crop_resize(primary_img, width=pw, height=ph)
    elif num_cameras == 2:
        ph, pw = _meta_hw(0)
        wh, ww = _meta_hw(1)
        wrist_img = get_safelibero_wrist_image(obs, validate=True)
        primary_resized = _center_crop_resize(primary_img, width=pw, height=ph)
        wrist_resized   = _center_crop_resize(wrist_img,   width=ww, height=wh)
        if concat_mode == "horizontal":
            rgb = np.concatenate([primary_resized, wrist_resized], axis=1)
        else:
            rgb = np.concatenate([primary_resized, wrist_resized], axis=0)
    else:
        raise ValueError(f"Unsupported num_output_cameras={num_cameras}")

    img_tensor = (
        torch.tensor(rgb, dtype=dtype)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device)
    )
    img_tensor = img_tensor * (2.0 / 255.0) - 1.0

    proprio = _extract_proprio(obs)
    proprio_tensor = _normalize_proprio(proprio, processor)

    return img_tensor, proprio_tensor, primary_img


# ── Video saving ──────────────────────────────────────────────────────────────

def save_episode_video(images: List[np.ndarray], path: str, fps: int = 30) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    for img in images:
        writer.append_data(img)
    writer.close()
    logger.info("Saved video: %s", path)


# ── Episode loop ──────────────────────────────────────────────────────────────

def run_episode(
    env,
    task_description: str,
    model,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    args: argparse.Namespace,
    device: str,
    dtype: torch.dtype,
    input_h: int,
    input_w: int,
    num_cameras: int,
    initial_state=None,
    images_out: Optional[List[np.ndarray]] = None,
):
    """Run one episode. Returns (success, collided, steps).

    Args:
        env: SafeLIBERO environment
        task_description: natural-language task goal
        model: Fast-WAM model
        processor: FastWAMProcessor with normalizer loaded
        cfg: Hydra DictConfig (used for infer_action kwargs)
        args: CLI args namespace
        device: torch device string
        dtype: model dtype
        input_h, input_w: image input dimensions (H, W*num_cams)
        num_cameras: number of camera views
        initial_state: optional environment initial state
        images_out: if provided, primary-view frames are appended here
    """
    obs = env.reset()
    if initial_state is not None and hasattr(env, "set_init_state"):
        obs = env.set_init_state(initial_state)

    # Stabilisation warm-up (30 dummy steps)
    for _ in range(30):
        obs, _, _, _ = env.step(_DUMMY_ACTION)

    # Identify obstacle once after stabilisation
    obstacle_names = [
        n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n
    ]
    obstacle_name = None
    for name in obstacle_names:
        p = obs.get(f"{name}_pos", np.zeros(3))
        if p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
            obstacle_name = name
            logger.info("Active obstacle: %s", obstacle_name)
            break
    if obstacle_name is None and obstacle_names:
        obstacle_name = obstacle_names[0]
        logger.info("No obstacle in bounds; defaulting to %s", obstacle_name)

    initial_obstacle_pos = (
        obs.get(f"{obstacle_name}_pos", np.zeros(3)) if obstacle_name else np.zeros(3)
    )
    collide_flag = False

    action_queue: deque = deque()
    max_steps = TASK_MAX_STEPS[args.task_suite_name]
    last_primary = np.zeros((256, 256, 3), dtype=np.uint8)
    success = False
    t = 0

    try:
        while t < max_steps:
            # Build observation; fall back to last valid frame on corruption
            try:
                img_tensor, proprio_tensor, primary_img = build_fastwam_obs(
                    obs, processor, cfg, input_h, input_w, num_cameras, device, dtype
                )
                last_primary = primary_img
            except (ValueError, KeyError) as exc:
                logger.warning("t=%d: corrupted frame (%s); using previous frame", t, exc)
                # Re-use last valid image: re-encode from last_primary
                ph = input_h
                pw_single = input_w // max(num_cameras, 1)
                try:
                    resized = _center_crop_resize(last_primary, width=pw_single, height=ph)
                    if num_cameras == 2:
                        rgb = np.concatenate([resized, resized], axis=1)
                    else:
                        rgb = resized
                    img_tensor = (
                        torch.tensor(rgb, dtype=dtype)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(device=device)
                    ) * (2.0 / 255.0) - 1.0
                    proprio = _extract_proprio(obs)
                    proprio_tensor = _normalize_proprio(proprio, processor)
                except Exception:
                    # If fallback also fails, skip this step
                    obs, _, done, _ = env.step(_DUMMY_ACTION)
                    t += 1
                    continue

            if images_out is not None:
                images_out.append(np.ascontiguousarray(last_primary.copy()))

            # Replan when action queue is empty
            if not action_queue:
                formatted_prompt = DEFAULT_PROMPT.format(task=task_description)
                num_video_frames = (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1

                _infer_sig = inspect.signature(model.infer_action)
                _infer_kwargs = dict(
                    prompt=formatted_prompt,
                    input_image=img_tensor,
                    action_horizon=args.action_horizon,
                    proprio=proprio_tensor,
                    num_inference_steps=args.num_inference_steps,
                    negative_prompt="",
                    text_cfg_scale=1.0,
                    rand_device="cpu",
                )
                if "num_video_frames" in _infer_sig.parameters:
                    _infer_kwargs["num_video_frames"] = num_video_frames

                with torch.no_grad():
                    pred = model.infer_action(**_infer_kwargs)

                action_chunk_norm = pred["action"]  # [T, D], normalized, on CPU

                # Denormalize
                action_chunk = _denormalize_action(action_chunk_norm, processor)[0]  # [T, D]

                # Gripper convention: undo [0,1] encoding, then flip sign for robosuite
                action_chunk[..., -1] = action_chunk[..., -1] * 2 - 1
                action_chunk[..., -1] = action_chunk[..., -1] * -1.0

                # Enqueue replan_steps worth of actions
                for a in action_chunk[:args.replan_steps]:
                    action_queue.append(a)

            action = action_queue.popleft()
            obs, _, done, _ = env.step(action.tolist())

            # Collision detection: obstacle displacement > 1 mm
            if not collide_flag and obstacle_name:
                cur_pos = obs.get(f"{obstacle_name}_pos", initial_obstacle_pos)
                displacement = np.sum(np.abs(cur_pos - initial_obstacle_pos))
                if displacement > 0.001:
                    collide_flag = True
                    logger.info("Collision detected at t=%d (disp=%.4f m)", t, displacement)

            if done:
                success = True
                break

            t += 1

    except Exception as exc:
        logger.error("Episode error at t=%d: %s", t, exc, exc_info=True)

    return success, collide_flag, t


# ── Task loop ─────────────────────────────────────────────────────────────────

def run_task(
    task_id: int,
    task_suite,
    model,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    args: argparse.Namespace,
    device: str,
    dtype: torch.dtype,
    input_h: int,
    input_w: int,
    num_cameras: int,
):
    """Run num_trials_per_task episodes for one task. Returns per-task result dict."""
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)

    # Pad initial_states if fewer than requested trials
    while len(initial_states) < args.num_trials_per_task:
        initial_states = list(initial_states) + list(initial_states)
    initial_states = initial_states[:args.num_trials_per_task]

    include_wrist = num_cameras >= 2
    env, task_description = get_safelibero_env(
        task,
        model_family="fastwam",
        resolution=256,
        include_wrist_camera=include_wrist,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Task %d: %s", task_id, task_description)
    logger.info("=" * 60)

    task_episodes = task_successes = task_collides = 0
    timesteps_list: List[int] = []

    for episode_idx in range(args.num_trials_per_task):
        # Save video for the first 10 episodes and every 5th episode thereafter (1-indexed).
        episode_num = episode_idx + 1
        capture_video = args.save_videos and (episode_num <= 10 or episode_num % 5 == 0)
        images_out: List[np.ndarray] = [] if capture_video else None
        logger.info("  Episode %d/%d", episode_idx + 1, args.num_trials_per_task)

        success, collide, steps = run_episode(
            env=env,
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            args=args,
            device=device,
            dtype=dtype,
            input_h=input_h,
            input_w=input_w,
            num_cameras=num_cameras,
            initial_state=initial_states[episode_idx],
            images_out=images_out,
        )

        task_episodes  += 1
        task_successes += int(success)
        task_collides  += int(collide)
        timesteps_list.append(steps)

        logger.info(
            "  success=%s  collide=%s  safe_success=%s  steps=%d",
            success, collide, (success and not collide), steps,
        )

        if args.save_videos and images_out:
            short_desc = task_description.lower().replace(" ", "_")[:50]
            video_path = os.path.join(
                args.video_output_dir,
                args.task_suite_name,
                args.safety_level,
                str(task_id),
                f"episode_{episode_idx}_{short_desc}.mp4",
            )
            save_episode_video(images_out, video_path)

    env.close()

    tsr = task_successes / task_episodes if task_episodes > 0 else 0.0
    car = (task_episodes - task_collides) / task_episodes if task_episodes > 0 else 0.0
    ets_mean   = float(np.mean(timesteps_list))   if timesteps_list else 0.0
    ets_median = float(np.median(timesteps_list)) if timesteps_list else 0.0

    logger.info(
        "  Task %d summary  TSR=%.3f  CAR=%.3f  ETS(mean)=%.1f",
        task_id, tsr, car, ets_mean,
    )

    return {
        "task_id":          task_id,
        "task_description": task_description,
        "episodes":         task_episodes,
        "successes":        task_successes,
        "collisions":       task_collides,
        "TSR":              round(tsr, 4),
        "CAR":              round(car, 4),
        "ETS_mean":         round(ets_mean, 2),
        "ETS_median":       round(ets_median, 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Fast-WAM policy on SafeLIBERO benchmark."
    )
    parser.add_argument(
        "--task_suite_name",
        choices=["safelibero_spatial", "safelibero_object", "safelibero_goal", "safelibero_long"],
        default="safelibero_spatial",
        help="SafeLIBERO task suite to evaluate.",
    )
    parser.add_argument(
        "--safety_level",
        choices=["I", "II"],
        default="I",
        help="SafeLIBERO safety level.",
    )
    parser.add_argument(
        "--num_trials_per_task",
        type=int,
        default=10,
        help="Number of episodes to run per task.",
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Path to Fast-WAM checkpoint (.pt file).",
    )
    parser.add_argument(
        "--stats_path",
        required=True,
        help="Path to dataset_stats.json for action/state normalization.",
    )
    parser.add_argument(
        "--results_output_dir",
        default="fastwam_benchmark",
        help="Directory to save JSON results.",
    )
    parser.add_argument(
        "--video_output_dir",
        default="fastwam_video",
        help="Directory to save episode videos (only if --save_videos).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=195,
        help="Global random seed.",
    )
    parser.add_argument(
        "--task_indices",
        nargs="+",
        type=int,
        default=None,
        help="Subset of task IDs to evaluate (default: all tasks in suite).",
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Save per-episode rollout videos.",
    )
    parser.add_argument(
        "--replan_steps",
        type=int,
        default=10,
        help="Number of environment steps to execute from each action chunk before replanning.",
    )
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=32,
        help="Number of action steps to predict per infer_action call.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of diffusion denoising steps.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="CUDA device index to use.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    set_global_seed(args.seed, get_worker_init_fn=False)

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Load model once
    logger.info("Loading Fast-WAM model...")
    model, processor, cfg, input_h, input_w, num_cameras = load_fastwam_model(
        ckpt_path=args.ckpt_path,
        stats_path=args.stats_path,
        device=device,
        args=args,
    )
    dtype = model.torch_dtype

    # Load SafeLIBERO benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name](safety_level=args.safety_level)
    num_tasks = task_suite.n_tasks
    assert num_tasks > 0, f"Task suite '{args.task_suite_name}' returned 0 tasks."

    task_ids = args.task_indices if args.task_indices is not None else list(range(num_tasks))
    logger.info(
        "Suite: %s  Level: %s  Tasks: %d (running %d)  Trials/task: %d",
        args.task_suite_name, args.safety_level, num_tasks, len(task_ids),
        args.num_trials_per_task,
    )

    all_episodes = all_successes = all_collides = 0
    all_timesteps: List[int] = []
    per_task_results = []

    for task_id in task_ids:
        result = run_task(
            task_id=task_id,
            task_suite=task_suite,
            model=model,
            processor=processor,
            cfg=cfg,
            args=args,
            device=device,
            dtype=dtype,
            input_h=input_h,
            input_w=input_w,
            num_cameras=num_cameras,
        )
        per_task_results.append(result)
        all_episodes  += result["episodes"]
        all_successes += result["successes"]
        all_collides  += result["collisions"]
        all_timesteps.extend([result["ETS_mean"]] * result["episodes"])

    overall_tsr    = all_successes / all_episodes if all_episodes > 0 else 0.0
    overall_car    = (all_episodes - all_collides) / all_episodes if all_episodes > 0 else 0.0
    overall_ets_m  = float(np.mean(all_timesteps))   if all_timesteps else 0.0

    output = {
        "policy":            "Fast-WAM (libero_uncond_2cam224)",
        "task_suite":        args.task_suite_name,
        "safety_level":      args.safety_level,
        "total_episodes":    all_episodes,
        "total_successes":   all_successes,
        "total_collisions":  all_collides,
        "overall_TSR":       round(overall_tsr, 4),
        "overall_CAR":       round(overall_car, 4),
        "overall_ETS_mean":  round(overall_ets_m, 2),
        "per_task":          per_task_results,
    }

    results_dir = os.path.join(args.results_output_dir, args.task_suite_name, args.safety_level)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"results_{_DATE_TIME}.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to: %s", results_path)

    # Print summary
    lines = [
        "",
        "=" * 60,
        "SafeLIBERO Evaluation Summary — Fast-WAM",
        f"Suite: {args.task_suite_name}  |  Level: {args.safety_level}",
        f"Total episodes: {all_episodes}",
        "-" * 60,
        f"  Task Success Rate (TSR):        {overall_tsr:.3f}  ({overall_tsr*100:.1f}%)",
        f"  Collision Avoidance Rate (CAR): {overall_car:.3f}  ({overall_car*100:.1f}%)",
        f"  Execution Time Steps (mean):    {overall_ets_m:.1f}",
        "-" * 60,
    ]
    for tr in per_task_results:
        lines.append(
            f"  Task {tr['task_id']}  TSR={tr['TSR']:.3f}  CAR={tr['CAR']:.3f}"
            f"  ETS={tr['ETS_mean']:.1f}  | {tr['task_description'][:60]}"
        )
    lines.append("=" * 60)
    logger.info("\n".join(lines))

    return output


if __name__ == "__main__":
    main()
