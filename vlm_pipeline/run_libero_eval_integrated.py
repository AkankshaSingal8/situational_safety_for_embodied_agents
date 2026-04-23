"""
run_libero_eval_integrated.py

Integrated SafeLIBERO evaluation with live VLM-based CBF safety filtering.

Pipeline per episode:
  1. OpenVLA generates nominal actions (run in openvla_libero_merged env)
  2. Qwen VLM (M1) extracts semantic safety constraints at episode start
     via subprocess IPC into the 'qwen' conda env
  3. cbf_construction.build_constraints() fits ellipsoid CBFs
  4. certify_action_simple() certifies each action at 20 Hz

Environment: openvla_libero_merged (main), qwen (VLM subprocess)
"""

import json
import logging
import os
import re
import subprocess
import sys
import time as _time_module
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple

import draccus
import imageio
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Append openvla-oft repo so that interpreter can find experiments.robot and prismatic
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "openvla-oft"))
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    quat2axisangle,
)

# Use SafeLIBERO-specific functions to avoid camera view contamination
from safelibero_utils import (
    get_safelibero_env,
    get_safelibero_image,
    get_safelibero_wrist_image,
    create_vlm_obs_env,
)
from robosuite.utils.camera_utils import (
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix,
    get_real_depth_map,
)
from save_vlm_inputs import (
    get_camera_specs,
    save_rgb,
    save_depth,
    save_segmentation,
    save_camera_params,
    save_metadata,
)
from cbf_construction import build_constraints
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "safelibero_spatial"
    LIBERO_OBJECT = "safelibero_object"
    LIBERO_GOAL = "safelibero_goal"
    LIBERO_LONG = "safelibero_long"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 300,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 300,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_LONG: 550,  # longest training demo has 505 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: str = "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"     # Pretrained checkpoint path (HuggingFace model ID or local path)

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_LONG  # Task suite
    safety_level: str = "I"
    num_steps_wait: int = 20                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 1024                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    results_output_dir: str = "openvla_cbf_benchmark" # Directory for JSON results
    video_output_dir: str = "openvla_cbf_video"       # Directory for rollout videos

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "akankshasingal08"          # Name of WandB entity
    wandb_project: str = "SafeLibero"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    use_safety_filter: bool = True                   # Whether to apply CBF safety filtering

    #################################################################################################################
    # VLM + CBF integrated pipeline
    #################################################################################################################
    vlm_method: str = "m1"                           # VLM method: m1, m2, m3
    qwen_conda_env: str = "qwen"                     # Conda env for Qwen VLM subprocess
    vlm_model: str = "qwen2.5-vl-7b"                # Qwen model key (see QWEN_MODELS in qwen_vlm_worker.py)
    num_vlm_votes: int = 1                           # Majority voting rounds per VLM query
    vlm_dry_run: bool = False                        # True = placeholder VLM responses (no GPU, for testing)
    vlm_tmp_dir: str = "/tmp/vlm_obs"                # Temp dir for per-episode obs folders
    vlm_resolution: int = 512                        # Camera resolution for VLM observation capture
    vlm_server_url: str = "http://localhost:5001"    # URL of persistent Qwen HTTP server

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"

    if cfg.safety_level not in ("I", "II"):
        raise ValueError(f"Invalid safety_level '{cfg.safety_level}'. Must be 'I' or 'II'.")


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor

def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    # 增加一个映射逻辑
    mapping = {
        "safelibero_spatial": "libero_spatial",
        "safelibero_object": "libero_object",
        "safelibero_goal": "libero_goal",
        "safelibero_long": "libero_10"
    }
    
    # 获取原始模型认识的名字
    unnorm_key = mapping.get(cfg.task_suite_name, cfg.task_suite_name)

    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
    cfg.unnorm_key = unnorm_key
def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-level{cfg.safety_level}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size, validate_images=True):
    """Prepare observation for policy input.

    Args:
        obs: Observation from environment
        resize_size: Size to resize images to
        validate_images: If True, validates images for corruption (default: True)
    """
    # Get preprocessed images
    img = get_safelibero_image(obs, validate=validate_images)
    wrist_img = get_safelibero_wrist_image(obs, validate=validate_images)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def save_episode_video(images: list, path: str, fps: int = 30) -> None:
    """Write replay images to an MP4 at the given absolute path."""
    if not images:
        logger.warning(f"save_episode_video: no frames to write, skipping {path}")
        return
    dir_part = os.path.dirname(path)
    if dir_part:
        os.makedirs(dir_part, exist_ok=True)
    with imageio.get_writer(path, fps=fps) as writer:
        for img in images:
            writer.append_data(img)
    logger.info(f"Saved video: {path}")


def check_vlm_server(server_url: str, timeout: int = 5) -> bool:
    """Return True if the Qwen VLM HTTP server is reachable at server_url/health."""
    try:
        import requests as _requests
    except ImportError:
        logger.warning("requests not installed; pip install requests")
        return False
    try:
        resp = _requests.get(f"{server_url}/health", timeout=timeout)
        return resp.ok
    except Exception as exc:
        logger.debug(f"check_vlm_server: {exc}")
        return False


# ═══════════════════════════════════════════════════════════════════════
# VLM + CBF INTEGRATED PIPELINE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

_DEFAULT_BEHAVIORAL = {"caution": False, "alpha_scale": 1.0}
_DEFAULT_POSE = {"rotation_lock": False, "w_rot": 0.0}


def capture_episode_obs(
    task,
    initial_state: np.ndarray,
    obs_folder: str,
    task_description: str,
    task_id: int,
    episode_idx: int,
    safety_level: str = "I",
    resolution: int = 512,
    num_steps_wait: int = 20,
) -> None:
    """Capture VLM observation data to obs_folder using a secondary env.

    Creates a 512×512 env with depth+seg, resets to initial_state,
    waits num_steps_wait steps, saves all obs files, then closes the env.

    Saves: agentview_rgb.png, eye_in_hand_rgb.png,
           agentview_depth.npy, agentview_seg.npy,
           eye_in_hand_depth.npy, eye_in_hand_seg.npy,
           camera_params.json, metadata.json
    """
    obs_path = Path(obs_folder)
    obs_path.mkdir(parents=True, exist_ok=True)

    camera_specs = get_camera_specs(["agentview", "robot0_eye_in_hand"])

    vlm_env = None
    try:
        vlm_env, _ = create_vlm_obs_env(task, resolution=resolution)
        vlm_env.reset()
        obs = vlm_env.set_init_state(initial_state)

        # Wait for physics to settle (same count as policy env)
        for _ in range(num_steps_wait):
            obs, _, _, _ = vlm_env.step([0.0] * 7)

        save_rgb(obs, obs_path, camera_specs)
        save_depth(obs, vlm_env.sim, obs_path, camera_specs)
        save_segmentation(obs, obs_path, camera_specs)
        save_camera_params(vlm_env.sim, obs_path, resolution, camera_specs)
        save_metadata(
            obs, vlm_env, task_id, episode_idx, safety_level, task_description, obs_path
        )
    finally:
        if vlm_env is not None:
            vlm_env.close()


def run_vlm_subprocess(
    obs_folder: str,
    output_json_path: str,
    method: str = "m1",
    qwen_conda_env: str = "qwen",
    model: str = "qwen2.5-vl-7b",
    num_votes: int = 1,
    timeout: int = 600,
    dry_run: bool = False,
) -> Optional[Dict]:
    """Run qwen_vlm_worker.py in the qwen conda env via subprocess.

    Returns the parsed output JSON dict on success, or None on failure.
    The returned dict has key "single" (because --input_folder is used).
    """
    worker_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "qwen_vlm_worker.py"
    )
    cmd = [
        "conda", "run", "-n", qwen_conda_env, "--no-capture-output",
        "python", worker_script,
        "--method", method,
        "--input_folder", obs_folder,
        "--output_json", output_json_path,
        "--model", model,
        "--num_votes", str(num_votes),
    ]
    if dry_run:
        cmd.append("--dry_run")

    logger.info(f"VLM subprocess: conda run -n {qwen_conda_env} qwen_vlm_worker.py --method {method} (dry_run={dry_run})")
    t0 = _time_module.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(f"VLM subprocess timed out after {timeout}s")
        return None
    except FileNotFoundError:
        logger.warning("VLM subprocess failed: 'conda' not found in PATH")
        return None

    elapsed = _time_module.time() - t0
    if result.returncode != 0:
        logger.warning(f"VLM subprocess failed (rc={result.returncode}) in {elapsed:.1f}s")
        logger.warning(f"stderr: {result.stderr[-500:]}")
        return None

    logger.info(f"VLM subprocess done in {elapsed:.1f}s")
    try:
        with open(output_json_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"VLM output parse error: {e}")
        return None


def build_episode_ellipsoids(
    vlm_json: Dict,
    obs_folder: str,
) -> Tuple[List[Dict], Dict, Dict]:
    """Convert build_constraints() output to certify_action_simple() format.

    build_constraints() returns {center, scales, ...}.
    certify_action_simple() requires {center, semi_axes}.
    Conversion: semi_axes = scales (direct rename).

    Returns:
        ellipsoids: List of {object, relationship, center, semi_axes}
        behavioral: {"caution": bool, "alpha_scale": float}
        pose: {"rotation_lock": bool, "w_rot": float}
    """
    constraints, behavioral, pose, eef_pos = build_constraints(vlm_json, obs_folder)
    ellipsoids = []
    for c in constraints:
        ellipsoids.append({
            "object": c["object"],
            "relationship": c["relationship"],
            "center": c["center"],
            "semi_axes": c["scales"],   # KEY RENAME: certify_action_simple() expects "semi_axes"
        })
    logger.info(
        f"Built {len(ellipsoids)} ellipsoids | "
        f"caution={behavioral['caution']} | "
        f"rotation_lock={pose['rotation_lock']}"
    )
    return ellipsoids, behavioral, pose


def call_vlm_server(
    obs_folder: str,
    output_json_path: str,
    server_url: str = "http://localhost:5001",
    method: str = "m1",
    timeout: int = 30,
    dry_run: bool = False,
) -> Optional[Dict]:
    """Call the persistent Qwen HTTP server for per-chunk VLM inference.

    Returns the constraint JSON dict on success, None on any failure.
    The server must be running (started via qwen_vlm_server.py) before eval.
    dry_run=True returns a placeholder without contacting the server.
    """
    if dry_run:
        result = {"single": {"description": "dry run", "end_object": "object", "objects": []}}
        with open(output_json_path, "w") as f:
            json.dump(result, f)
        return result
    try:
        import requests as _requests
    except ImportError:
        logger.warning("requests not installed; pip install requests")
        return None
    try:
        resp = _requests.post(
            f"{server_url}/infer",
            json={"obs_folder": obs_folder, "method": method},
            timeout=timeout,
        )
        if resp.ok:
            result = resp.json()
            with open(output_json_path, "w") as f:
                json.dump(result, f)
            logger.info(f"VLM server response received for {os.path.basename(obs_folder)}")
            return result
        logger.warning(f"VLM server returned {resp.status_code}: {resp.text[:200]}")
        return None
    except _requests.exceptions.Timeout:
        logger.warning(
            f"VLM server timed out after {timeout}s for {os.path.basename(obs_folder)}"
        )
        return None
    except _requests.exceptions.ConnectionError:
        logger.warning(
            f"VLM server not reachable at {server_url} — is qwen_vlm_server.py running?"
        )
        return None
    except Exception as exc:
        logger.warning(f"VLM server call failed: {exc}")
        return None


def capture_chunk_obs_from_env(
    obs: dict,
    sim,
    obs_folder: str,
    task_id: int,
    episode_idx: int,
    chunk_idx: int,
    safety_level: str,
    task_description: str,
    resolution: int,
    camera_specs,
) -> None:
    """Save depth+seg+RGB observations from the running policy env to obs_folder.

    Unlike capture_episode_obs(), this does NOT create a secondary env.
    The policy env must have been created with camera_depths=True and
    camera_segmentations='instance' (done in run_task() when use_safety_filter=True).
    Saves: agentview_rgb.png, agentview_depth.npy, agentview_seg.npy,
           camera_params.json, metadata.json
    """
    obs_path = Path(obs_folder)
    obs_path.mkdir(parents=True, exist_ok=True)
    save_rgb(obs, obs_path, camera_specs)
    save_depth(obs, sim, obs_path, camera_specs)
    save_segmentation(obs, obs_path, camera_specs)
    save_camera_params(sim, obs_path, resolution, camera_specs)

    # save_metadata expects an env-like object with a .sim attribute
    class _SimWrapper:
        def __init__(self, s):
            self.sim = s

    save_metadata(
        obs, _SimWrapper(sim), task_id, episode_idx, safety_level, task_description, obs_path
    )


def get_episode_cbf_constraints(
    task,
    initial_state: np.ndarray,
    task_id: int,
    episode_idx: int,
    cfg,
) -> Tuple[List[Dict], Dict, Dict]:
    """Full M1 pipeline: obs capture → VLM subprocess → CBF ellipsoids.

    Returns (ellipsoids, behavioral, pose). On any failure, returns
    ([], _DEFAULT_BEHAVIORAL, _DEFAULT_POSE) so the episode runs without CBF.
    """
    obs_folder = os.path.join(
        cfg.vlm_tmp_dir, f"task_{task_id}", f"episode_{episode_idx:02d}"
    )
    vlm_output_path = os.path.join(
        cfg.vlm_tmp_dir, f"vlm_task{task_id}_ep{episode_idx:02d}.json"
    )

    logger.info(f"[VLM] Capturing obs for task {task_id} episode {episode_idx} → {obs_folder}")
    try:
        capture_episode_obs(
            task=task,
            initial_state=initial_state,
            obs_folder=obs_folder,
            task_description=task.language,
            task_id=task_id,
            episode_idx=episode_idx,
            safety_level=cfg.safety_level,
            resolution=cfg.vlm_resolution,
            num_steps_wait=cfg.num_steps_wait,
        )
    except Exception as e:
        logger.warning(f"[VLM] Obs capture failed: {e}. CBF disabled for this episode.")
        return [], _DEFAULT_BEHAVIORAL.copy(), _DEFAULT_POSE.copy()

    logger.info(f"[VLM] Running M1 inference (dry_run={cfg.vlm_dry_run})")
    vlm_json = run_vlm_subprocess(
        obs_folder=obs_folder,
        output_json_path=vlm_output_path,
        method=cfg.vlm_method,
        qwen_conda_env=cfg.qwen_conda_env,
        model=cfg.vlm_model,
        num_votes=cfg.num_vlm_votes,
        dry_run=cfg.vlm_dry_run,
    )
    if vlm_json is None:
        logger.warning("[VLM] Subprocess returned None. CBF disabled for this episode.")
        return [], _DEFAULT_BEHAVIORAL.copy(), _DEFAULT_POSE.copy()

    try:
        ellipsoids, behavioral, pose = build_episode_ellipsoids(vlm_json, obs_folder)
    except Exception as e:
        logger.warning(f"[VLM] Ellipsoid build failed: {e}. CBF disabled for this episode.")
        return [], _DEFAULT_BEHAVIORAL.copy(), _DEFAULT_POSE.copy()

    return ellipsoids, behavioral, pose


# ═══════════════════════════════════════════════════════════════════════
# CBF SAFETY FILTER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def load_ellipsoids(task_id: int, cbf_outputs_dir: str = "cbf_outputs/gt") -> List[Dict]:
    """
    Load pre-computed CBF ellipsoids for a task.

    Args:
        task_id: Task identifier
        cbf_outputs_dir: Directory containing ellipsoid JSONs

    Returns:
        List of ellipsoid dicts with keys: object, relationship, center, semi_axes
    """
    ellipsoids_path = os.path.join(cbf_outputs_dir, f"task_{task_id}_ellipsoids.json")

    if not os.path.exists(ellipsoids_path):
        logger.warning(f"No ellipsoids found at {ellipsoids_path}, CBF disabled for task {task_id}")
        return []

    with open(ellipsoids_path, 'r') as f:
        data = json.load(f)

    ellipsoids = data.get("ellipsoids", [])
    logger.info(f"Loaded {len(ellipsoids)} CBF ellipsoids from {ellipsoids_path}")

    return ellipsoids


def evaluate_cbf(x_ee: np.ndarray, center: np.ndarray, semi_axes: np.ndarray) -> float:
    """
    Evaluate ellipsoid-based Control Barrier Function.

    Source: minimal_cbf_demo_interactive.py line 200-205

    Args:
        x_ee: (3,) array - end-effector position
        center: (3,) array - ellipsoid center
        semi_axes: (3,) array - ellipsoid semi-axes

    Returns:
        h: float - barrier function value
            h > 0: safe (outside unsafe region)
            h = 0: boundary
            h < 0: unsafe (inside unsafe region)
    """
    diff = x_ee - center
    normalized_sq = (diff / semi_axes) ** 2
    h = np.sum(normalized_sq) - 1.0
    return h


def certify_action_simple(
    u_cmd: np.ndarray,
    ee_pos: np.ndarray,
    ellipsoids: List[Dict],
    dt: float = 0.05,
    alpha: float = 1.0,
    max_iter: int = 10
) -> Tuple[np.ndarray, Dict]:
    """
    Certify action using gradient-based CBF-QP.

    Solves: min ||u - u_cmd||^2
            s.t. h_dot >= -alpha * h for all ellipsoids

    Args:
        u_cmd: (7,) commanded action [dx, dy, dz, drx, dry, drz, gripper]
        ee_pos: (3,) end-effector position
        ellipsoids: List of ellipsoid dicts
        dt: Timestep for forward prediction
        alpha: Class-K function scale
        max_iter: Max iterations for iterative projection

    Returns:
        u_cert: (7,) certified action
        info: dict with h_values, interventions, etc.
    """
    u_cert = u_cmd.copy()
    h_values = []
    interventions = []

    if len(ellipsoids) == 0:
        return u_cert, {"h_values": [], "interventions": [], "modified": False}

    # Only certify translational part (position), preserve rotation + gripper
    dp = u_cert[:3].copy()

    # Iterative gradient projection
    for iteration in range(max_iter):
        modified_this_iter = False

        for ellipsoid in ellipsoids:
            center = np.array(ellipsoid["center"])
            semi_axes = np.array(ellipsoid["semi_axes"])

            # Current h-value
            h_current = evaluate_cbf(ee_pos, center, semi_axes)

            # Predicted h-value after action
            ee_next = ee_pos + dp * dt
            h_next = evaluate_cbf(ee_next, center, semi_axes)

            # h_dot = (h_next - h_current) / dt
            h_dot = (h_next - h_current) / dt

            # CBF condition: h_dot >= -alpha * h
            cbf_threshold = -alpha * max(h_current, 0.01)  # Avoid division by zero

            if h_dot < cbf_threshold:
                # Constraint violated - project onto constraint boundary
                # Gradient: ∇h = 2 * (ee - center) / semi_axes^2
                grad_h = 2 * (ee_pos - center) / (semi_axes ** 2 + 1e-10)
                grad_norm_sq = np.dot(grad_h, grad_h)

                if grad_norm_sq > 1e-10:
                    # Correction: project dp to satisfy constraint
                    violation = cbf_threshold - h_dot
                    correction = (violation * dt) * grad_h / grad_norm_sq
                    dp = dp + correction
                    modified_this_iter = True

                    interventions.append({
                        "object": ellipsoid["object"],
                        "relationship": ellipsoid["relationship"],
                        "h_current": float(h_current),
                        "h_dot_before": float(h_dot),
                    })

        if not modified_this_iter:
            break

    # Capture final h-values once after convergence
    for ellipsoid in ellipsoids:
        center = np.array(ellipsoid["center"])
        semi_axes = np.array(ellipsoid["semi_axes"])
        h_values.append({
            "object": ellipsoid["object"],
            "relationship": ellipsoid["relationship"],
            "h": float(evaluate_cbf(ee_pos, center, semi_axes))
        })

    # Update certified action
    u_cert[:3] = dp

    modified = np.linalg.norm(u_cert[:3] - u_cmd[:3]) > 1e-5

    info = {
        "h_values": h_values,
        "interventions": interventions,
        "modified": modified,
        "num_iterations": iteration + 1,
        "num_interventions": len(interventions),  # Added for consistency
        "h_min": min([hv["h"] for hv in h_values]) if h_values else 0.0,
    }

    return u_cert, info


def save_trajectory(
    trajectory_data: List[Dict],
    task_id: int,
    episode_idx: int,
    run_dir: str
):
    """
    Save trajectory to compressed NPZ.

    Args:
        trajectory_data: List of per-timestep dicts
        task_id: Task identifier
        episode_idx: Episode index
        run_dir: Base directory for trajectories
    """
    if len(trajectory_data) == 0:
        return

    n_steps = len(trajectory_data)

    # Structured array for fixed-size fields
    dtype = [
        ('t', 'i4'),
        ('ee_pos', 'f4', (3,)),
        ('ee_quat', 'f4', (4,)),
        ('action_commanded', 'f4', (7,)),
        ('action_certified', 'f4', (7,)),
        ('action_executed', 'f4', (7,)),
        ('cbf_active', 'bool'),
        ('h_min', 'f4'),
        ('num_interventions', 'i4'),
    ]

    trajectory = np.zeros(n_steps, dtype=dtype)
    for i, step in enumerate(trajectory_data):
        trajectory[i]['t'] = step['t']
        trajectory[i]['ee_pos'] = step['ee_pos']
        trajectory[i]['ee_quat'] = step['ee_quat']
        trajectory[i]['action_commanded'] = step['action_commanded']
        trajectory[i]['action_certified'] = step['action_certified']
        trajectory[i]['action_executed'] = step['action_executed']
        trajectory[i]['cbf_active'] = step['cbf_active']
        trajectory[i]['h_min'] = step.get('h_min', 0.0)
        trajectory[i]['num_interventions'] = step.get('num_interventions', 0)

    # Write
    path = os.path.join(run_dir, f"task_{task_id}", f"episode_{episode_idx:02d}.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, trajectory=trajectory)

    logger.info(f"Saved trajectory: {path} ({n_steps} steps, {os.path.getsize(path)/1024:.1f} KB)")


# ═══════════════════════════════════════════════════════════════════════
# END CBF FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    task_id=None,
    trajectory_dir=None,
    episode_idx=None,
    task=None,  # LIBERO task object (kept for API compatibility)
    camera_specs=None,  # camera specs for per-chunk obs capture
):
    """Run a single episode in the environment with CBF safety filtering."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        logger.warning(
            f"num_open_loop_steps ({cfg.num_open_loop_steps}) != NUM_ACTIONS_CHUNK ({NUM_ACTIONS_CHUNK}). "
            "For best results both should match."
        )
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    # Wait for objects to stabilize (separate loop, matching run_libero_eval.py)
    while t < cfg.num_steps_wait:
        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
        t += 1

    replay_images = []
    trajectory_data = []  # NEW: trajectory logging
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    t = 0  # Reset timer after wait (matching run_libero_eval.py)

    # Per-chunk VLM constraints (updated at each action chunk boundary)
    ellipsoids: List[Dict] = []
    behavioral = _DEFAULT_BEHAVIORAL.copy()
    pose = _DEFAULT_POSE.copy()
    cbf_active = False

    # Obstacle detection (happens once after wait, outside main loop - matching run_libero_eval.py)
    # Extract all obstacle names from the joint list
    obstacle_names = [n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n]
    obstacle_name = None
    for name in obstacle_names:
        p = obs.get(f"{name}_pos", np.zeros(3))
        if p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
            obstacle_name = name
            logger.info(f"Active obstacle: {obstacle_name}")
            break
    if obstacle_name is None and obstacle_names:
        obstacle_name = obstacle_names[0]
        logger.warning(f"No obstacle in bounds; defaulting to {obstacle_name}")
    initial_obstacle_pos = obs.get(f"{obstacle_name}_pos", np.zeros(3)) if obstacle_name else np.zeros(3)
    collide_flag = False

    # Run episode (matching run_libero_eval.py structure)
    success = False
    try:
        while t < max_steps:

            # ── capture frame for video replay (every step) ───────────────────────
            try:
                observation, img = prepare_observation(obs, resize_size, validate_images=True)
                img_safe = np.ascontiguousarray(img.copy())
                assert img_safe.dtype == np.uint8
                assert img_safe.shape == (cfg.env_img_res, cfg.env_img_res, 3)
                assert img_safe.flags["C_CONTIGUOUS"]
                replay_images.append(img_safe)
            except ValueError as exc:
                logger.warning(f"t={t}: corrupted frame ({exc}); using previous")
                replay_images.append(
                    replay_images[-1].copy() if replay_images
                    else np.zeros((cfg.env_img_res, cfg.env_img_res, 3), dtype=np.uint8)
                )
                observation, _ = prepare_observation(obs, resize_size, validate_images=False)

            # ── ACTION CHUNK BOUNDARY ──────────────────────────────────────────
            if len(action_queue) == 0:
                # 1. Update VLM constraints from current observation
                if cfg.use_safety_filter and camera_specs is not None and task_id is not None and episode_idx is not None:
                    chunk_idx = t // cfg.num_open_loop_steps
                    chunk_folder = os.path.join(
                        cfg.vlm_tmp_dir,
                        f"task_{task_id}",
                        f"episode_{episode_idx:02d}",
                        f"chunk_{chunk_idx:04d}",
                    )
                    vlm_out = os.path.join(
                        cfg.vlm_tmp_dir,
                        f"vlm_t{task_id}_ep{episode_idx:02d}_chunk{chunk_idx:04d}.json",
                    )
                    try:
                        capture_chunk_obs_from_env(
                            obs, env.sim, chunk_folder,
                            task_id, episode_idx, chunk_idx,
                            cfg.safety_level, task_description,
                            cfg.vlm_resolution, camera_specs,
                        )
                        vlm_json = call_vlm_server(
                            obs_folder=chunk_folder,
                            output_json_path=vlm_out,
                            server_url=cfg.vlm_server_url,
                            method=cfg.vlm_method,
                            dry_run=cfg.vlm_dry_run,
                        )
                        if vlm_json is not None:
                            ellipsoids, behavioral, pose = build_episode_ellipsoids(
                                vlm_json, chunk_folder
                            )
                            cbf_active = len(ellipsoids) > 0
                            logger.info(
                                f"[VLM] t={t} chunk={chunk_idx}: "
                                f"{len(ellipsoids)} ellipsoids | "
                                f"caution={behavioral.get('caution', False)} | "
                                f"rotation_lock={pose.get('rotation_lock', False)}"
                            )
                        else:
                            logger.warning(
                                f"[VLM] t={t} chunk={chunk_idx}: server returned None; "
                                "keeping previous constraints"
                            )
                    except Exception as exc:
                        logger.warning(f"[VLM] t={t}: {exc}; keeping previous constraints")

                # 2. Query OpenVLA policy
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions[:cfg.num_open_loop_steps])

            # Get action from queue
            action = action_queue.popleft()
            action_commanded = action.copy()  # NEW: save original action

            # CBF certification with VLM-derived behavioral alpha
            cbf_info = {}
            action_certified = action.copy()
            if cbf_active:
                ee_pos = obs["robot0_eef_pos"]
                alpha = 0.25 if behavioral.get("caution", False) else 1.0
                action_certified, cbf_info = certify_action_simple(
                    u_cmd=action,
                    ee_pos=ee_pos,
                    ellipsoids=ellipsoids,
                    dt=0.05,
                    alpha=alpha,
                )
                action = action_certified
                # Rotation lock: zero out rotation delta if pose constraint active
                if pose.get("rotation_lock", False):
                    action[3:6] = np.zeros(3)

            # Process action
            action = process_action(action, cfg.model_family)
            action_executed = action.copy()  # NEW: save post-processed action

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())

            # NEW: Log trajectory data
            if trajectory_dir is not None:
                trajectory_data.append({
                    't': t,
                    'ee_pos': obs["robot0_eef_pos"].copy(),
                    'ee_quat': obs["robot0_eef_quat"].copy(),
                    'action_commanded': action_commanded,
                    'action_certified': action_certified if cbf_active else action_commanded,
                    'action_executed': action_executed,
                    'cbf_active': cbf_active,
                    'h_min': cbf_info.get('h_min', 0.0),
                    'num_interventions': len(cbf_info.get('interventions', [])),
                })

            # Collision detection (matching run_libero_eval.py exactly)
            if not collide_flag and obstacle_name is not None:
                then_pos = obs.get(f"{obstacle_name}_pos", initial_obstacle_pos)
                if np.sum(np.abs(then_pos - initial_obstacle_pos)) > 0.001:
                    collide_flag = True
                    logger.info(f"Collision at t={t}")

            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    # NEW: Save trajectory
    if trajectory_dir is not None and len(trajectory_data) > 0 and task_id is not None and episode_idx is not None:
        save_trajectory(trajectory_data, task_id, episode_idx, trajectory_dir)

    return success, collide_flag, replay_images, t


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    log_file=None,
    run_id: str = "",
    trajectory_dir=None,
):
    """Run evaluation for a single task with CBF safety filtering.

    Returns:
        task_episodes (int), task_successes (int), task_collides (int),
        timesteps_list (List[int]), task_description (str)
    """
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    # Use SafeLIBERO-specific function to avoid multi-camera buffer conflicts
    env, task_description = get_safelibero_env(
        task, cfg.model_family,
        resolution=cfg.env_img_res,
        include_wrist_camera=True,
        camera_depths=cfg.use_safety_filter,
        camera_segmentations="instance" if cfg.use_safety_filter else None,
    )
    camera_specs = get_camera_specs(["agentview", "robot0_eye_in_hand"]) if cfg.use_safety_filter else None

    # Start episodes
    task_episodes, task_successes, task_collides, task_safesuccesses = 0, 0, 0, 0
    timesteps_list: List[int] = []
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):

        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, collide, replay_images, total_time = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
            task_id=task_id,
            trajectory_dir=trajectory_dir,
            episode_idx=episode_idx,
            task=task,
            camera_specs=camera_specs,
        )
        # Update counters
        task_episodes += 1
        timesteps_list.append(total_time)
        if success:
            task_successes += 1
        if collide:
            task_collides += 1
        # Save replay video for first episode only (matches reference eval script)
        if episode_idx == 0:
            short_desc = re.sub(r'[^\w]', '_', task_description.lower())[:50]
            video_path = os.path.join(
                cfg.video_output_dir,
                cfg.task_suite_name,
                str(task_id),
                cfg.safety_level,
                f"episode_0_{short_desc}.mp4",
            )
            try:
                save_episode_video(replay_images, video_path)
            except Exception as exc:
                logger.warning(f"Video save failed for task {task_id}: {exc}")

        # Log results
        ss = success and not collide
        if ss:
            task_safesuccesses += 1
        log_message(
            f"  success={success}  collide={collide}  safe_success={ss}  steps={total_time}",
            log_file,
        )

    # Per-task metrics (matching run_safelibero_openvla_oft_eval.py)
    tsr = float(task_successes) / task_episodes if task_episodes > 0 else 0.0
    car = float(task_episodes - task_collides) / task_episodes if task_episodes > 0 else 0.0
    ets_mean = float(np.mean(timesteps_list)) if timesteps_list else 0.0

    log_message(
        f"  Task {task_id} summary  TSR={tsr:.3f}  CAR={car:.3f}  ETS(mean)={ets_mean:.1f}",
        log_file,
    )

    if cfg.use_wandb:
        wandb.log({
            f"TSR/task_{task_id}": tsr,
            f"CAR/task_{task_id}": car,
            f"ETS_mean/task_{task_id}": ets_mean,
        })

    return task_episodes, task_successes, task_collides, timesteps_list, task_description


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)
    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # NEW: Setup trajectory logging directory
    trajectory_dir = os.path.join(cfg.local_log_dir, "trajectories", run_id)
    os.makedirs(trajectory_dir, exist_ok=True)
    log_message(f"Trajectory directory: {trajectory_dir}", log_file)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name](safety_level=cfg.safety_level)
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Safety level: {cfg.safety_level}", log_file)

    # VLM server health check
    if cfg.use_safety_filter and not cfg.vlm_dry_run:
        if check_vlm_server(cfg.vlm_server_url):
            log_message(f"VLM server reachable at {cfg.vlm_server_url}", log_file)
        else:
            msg = (
                f"VLM server not reachable at {cfg.vlm_server_url}. "
                "CBF constraints will be skipped for all chunks. "
                "Start qwen_vlm_server.py or use --vlm_dry_run True."
            )
            logger.warning(msg)
            if log_file:
                log_file.write("WARNING: " + msg + "\n")
                log_file.flush()

    # Per-task results storage
    results: dict = {}
    all_episodes = all_successes = all_collides = 0
    all_timesteps: List[int] = []

    assert num_tasks > 0, f"Task suite '{cfg.task_suite_name}' returned 0 tasks."
    for task_id in range(num_tasks):
        task_episodes, task_successes, task_collides, timesteps_list, task_desc = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            log_file,
            run_id=run_id,
            trajectory_dir=trajectory_dir,
        )

        all_episodes += task_episodes
        all_successes += task_successes
        all_collides += task_collides
        all_timesteps.extend(timesteps_list)

        tsr = float(task_successes) / task_episodes if task_episodes > 0 else 0.0
        car = float(task_episodes - task_collides) / task_episodes if task_episodes > 0 else 0.0

        results[f"task_{task_id}"] = {
            "description": task_desc,
            "episodes": task_episodes,
            "successes": task_successes,
            "collisions": task_collides,
            "TSR": round(tsr, 4),
            "CAR": round(car, 4),
            "ETS_mean": round(float(np.mean(timesteps_list)) if timesteps_list else 0.0, 2),
            "ETS_median": round(float(np.median(timesteps_list)) if timesteps_list else 0.0, 2),
        }

    # Overall metrics
    overall_tsr = float(all_successes) / all_episodes if all_episodes > 0 else 0.0
    overall_car = float(all_episodes - all_collides) / all_episodes if all_episodes > 0 else 0.0
    overall_ets_mean = float(np.mean(all_timesteps)) if all_timesteps else 0.0
    overall_ets_median = float(np.median(all_timesteps)) if all_timesteps else 0.0

    results["overall"] = {
        "model": cfg.pretrained_checkpoint,
        "suite": cfg.task_suite_name,
        "safety_level": cfg.safety_level,
        "vlm_method": cfg.vlm_method,
        "vlm_dry_run": cfg.vlm_dry_run,
        "total_episodes": all_episodes,
        "total_successes": all_successes,
        "total_collisions": all_collides,
        "TSR": round(overall_tsr, 4),
        "CAR": round(overall_car, 4),
        "ETS_mean": round(overall_ets_mean, 2),
        "ETS_median": round(overall_ets_median, 2),
    }

    # Save JSON results
    results_dir = os.path.join(cfg.results_output_dir, cfg.task_suite_name)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"results_{run_id}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log_message(f"Results saved to: {results_path}", log_file)

    # Terminal summary
    summary_lines = [
        "",
        "=" * 60,
        "SafeLIBERO Integrated VLM+CBF Evaluation Summary",
        f"Suite: {cfg.task_suite_name}  |  Level: {cfg.safety_level}  |  VLM: {cfg.vlm_method}",
        f"Total episodes: {all_episodes}",
        "-" * 60,
        f"  Task Success Rate (TSR):       {overall_tsr:.3f}  ({overall_tsr*100:.1f}%)",
        f"  Collision Avoidance Rate (CAR):{overall_car:.3f}  ({overall_car*100:.1f}%)",
        f"  Execution Time Steps (mean):   {overall_ets_mean:.1f}",
        f"  Execution Time Steps (median): {overall_ets_median:.1f}",
        "-" * 60,
    ]
    for tid in range(num_tasks):
        tr = results[f"task_{tid}"]
        summary_lines.append(
            f"  Task {tid}  TSR={tr['TSR']:.3f}  CAR={tr['CAR']:.3f}"
            f"  ETS={tr['ETS_mean']:.1f}  | {tr['description'][:60]}"
        )
    summary_lines.append("=" * 60)
    summary = "\n".join(summary_lines)
    log_message(summary, log_file)

    if cfg.use_wandb:
        wandb.log({
            "TSR/overall": overall_tsr,
            "CAR/overall": overall_car,
            "ETS_mean/overall": overall_ets_mean,
            "total_episodes": all_episodes,
        })
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    return results


if __name__ == "__main__":
    eval_libero()
