"""
run_libero_eval_with_cbf.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite
with CBF-based semantic safety filtering.

Based on run_libero_eval.py with added:
- Pre-computed ellipsoid loading
- CBF-QP action certification
- Trajectory logging with CBF data
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple

import draccus
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
    save_rollout_video,
)
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

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "akankshasingal08"          # Name of WandB entity
    wandb_project: str = "SafeLibero"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    use_safety_filter: bool = True                   # Whether to apply CBF safety filtering

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


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
# def initialize_model(cfg: GenerateConfig):
#     """Initialize model and associated components (Mocked for testing)."""
#     # --- 注释掉实际加载逻辑 ---
#     # model = get_model(cfg)
    
#     # --- 创建一个 Mock 对象来替代 model，防止访问属性时报错 ---
#     class MockModel:
#         def __init__(self):
#             self.llm_dim = 1024  # 随便给一个维度，通常是1024或2048
#             self.norm_stats = {cfg.task_suite_name: {}} # 绕过 check_unnorm_key 的 assert
    
#     model = MockModel()
    
#     # 其他组件直接返回 None 即可，后续函数通常有 if 判断
#     proprio_projector = None
#     action_head = None
#     noisy_action_projector = None
#     processor = None

#     return model, action_head, proprio_projector, noisy_action_projector, processor

# def check_unnorm_key(cfg: GenerateConfig, model) -> None:
#     """Check that the model contains the action un-normalization key."""
#     # Initialize unnorm_key
#     unnorm_key = cfg.task_suite_name

#     # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
#     # with the suffix "_no_noops" in the dataset name)
#     if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
#         unnorm_key = f"{unnorm_key}_no_noops"

#     assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

#     # Set the unnorm_key in cfg
#     cfg.unnorm_key = unnorm_key

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
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + "_1.txt")
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
    task_id=None,  # NEW: for ellipsoid loading
    trajectory_dir=None,  # NEW: for trajectory saving
    episode_idx=None,  # NEW: for trajectory saving
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
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
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

    # NEW: Load pre-computed CBF ellipsoids
    ellipsoids = []
    if cfg.use_safety_filter and task_id is not None:
        ellipsoids = load_ellipsoids(task_id, cbf_outputs_dir="cbf_outputs")
    cbf_active = len(ellipsoids) > 0
    if cbf_active:
        log_message(f"  CBF: Loaded {len(ellipsoids)} ellipsoid constraints", log_file)

    # Obstacle detection (happens once after wait, outside main loop - matching run_libero_eval.py)
    # Extract all obstacle names from the joint list
    obstacle_names = [n.replace('_joint0', '') for n in env.sim.model.joint_names if 'obstacle' in n]

    # Identify the active obstacle within the workspace bounds
    obstacle_name = " "
    for i in obstacle_names:
        p = obs[f"{i}_pos"]  # Get position from observation
        # Check if the object is within the valid workspace range
        if p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
            obstacle_name = i
            print("Obstacle name:", i)
            break
    initial_obstacle_pos = obs[obstacle_name + "_pos"]
    collide_flag = False
    collide_time = 0

    # Run episode (matching run_libero_eval.py structure)
    success = False
    try:
        while t < max_steps:

            # Prepare observation
            try:
                observation, img = prepare_observation(obs, resize_size, validate_images=True)

                # BUGFIX: Ensure frame is fully independent and contiguous before storing
                # MuJoCo render buffers may be reused, causing corruption if we store references
                img_safe = np.ascontiguousarray(img.copy())

                # Validate frame properties to catch any buffer issues early
                assert img_safe.dtype == np.uint8, f"Frame dtype mismatch: {img_safe.dtype}"
                assert img_safe.shape == (cfg.env_img_res, cfg.env_img_res, 3), f"Frame shape mismatch: {img_safe.shape}"
                assert img_safe.flags['C_CONTIGUOUS'], "Frame is not C-contiguous"

                replay_images.append(img_safe)

            except ValueError as e:
                # Corrupted frame detected - use previous valid frame in replay video
                logger.warning(f"t={t}: Corrupted frame detected, using previous frame for video")
                if len(replay_images) > 0:
                    # BUGFIX: Copy the frame instead of appending a reference
                    replay_images.append(replay_images[-1].copy())
                else:
                    # First frame corrupted - append black frame
                    replay_images.append(np.zeros((cfg.env_img_res, cfg.env_img_res, 3), dtype=np.uint8))

                # For policy, bypass validation (policy uses lower 256x256 resolution which may be more robust)
                observation, _ = prepare_observation(obs, resize_size, validate_images=False)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                print(f"t={t}: before get_action")
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
                print(f"t={t}: after get_action, len={len(actions)}")
                action_queue.extend(actions[:cfg.num_open_loop_steps])

            # Get action from queue
            action = action_queue.popleft()
            action_commanded = action.copy()  # NEW: save original action

            # NEW: CBF certification
            cbf_info = {}
            if cbf_active:
                ee_pos = obs["robot0_eef_pos"]
                action_certified, cbf_info = certify_action_simple(
                    u_cmd=action,
                    ee_pos=ee_pos,
                    ellipsoids=ellipsoids,
                    dt=0.05,
                    alpha=1.0
                )
                action = action_certified

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
            if collide_flag == False:
                then_obstacle_pos = obs[obstacle_name + "_pos"]
                # print(np.sum(np.abs(then_obstacle_pos - initial_obstacle_pos)))
                if np.sum(np.abs(then_obstacle_pos - initial_obstacle_pos)) > 0.001:
                    print("obstacle collided")
                    collide_flag = True
                    collide_time = t
                    print("collide time:", collide_time)

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
    total_episodes=0,
    total_successes=0,
    log_file=None,
    trajectory_dir=None,  # NEW: for CBF trajectory logging
):
    """Run evaluation for a single task with CBF safety filtering."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    # Use SafeLIBERO-specific function to avoid multi-camera buffer conflicts
    env, task_description = get_safelibero_env(task, cfg.model_family, resolution=cfg.env_img_res, include_wrist_camera=True)

    # Start episodes
    task_episodes, task_successes, task_collides, task_safesuccesses = 0, 0, 0, 0
    total_time_task = []
    # for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
    for episode_idx in range(1):  # TESTING: Run just 1 episode

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
            task_id=task_id,  # NEW
            trajectory_dir=trajectory_dir,  # NEW
            episode_idx=episode_idx,  # NEW
        )
        total_time_task.append(total_time)
        # Update counters
        task_episodes += 1
        print("task_episodes", task_episodes)
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1
        if collide:
            task_collides += 1
        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"Collide: {collide}", log_file)
        ss = success and not collide
        if ss:
            task_safesuccesses += 1
        log_message(f"SS (Safe Success): {ss}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)
        log_message(f"# Collides: {task_collides} ({task_collides / total_episodes * 100:.1f}%)", log_file)
        log_message(f"# safesuccesses: {task_safesuccesses} ({task_safesuccesses / total_episodes * 100:.1f}%)", log_file)

        log_message(f"# Time step: {total_time_task}", log_file)


    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


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


    # Start evaluation
    total_episodes, total_successes = 0, 0
    # for task_id in tqdm.tqdm(range(num_tasks)):
    for task_id in [0]:  # CHANGED: Start with task 0 for testing
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
            trajectory_dir=trajectory_dir,  # NEW
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
