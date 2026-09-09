"""
run_vlm_eval.py

Evaluates OpenVLA-OFT policy on SafeLIBERO WITH Semantic CBF Safety Filter.

Pipeline:
  OpenVLA-OFT (main conda env) → u_nominal
  Qwen VLM ("qwen" conda env)  → semantic predicates (once per episode)
  CBF-QP                       → u_safe (every timestep)
  env.step(u_safe)

Conda environments:
  - Main env (e.g. "safelibero"):  OpenVLA-OFT, LIBERO, MuJoCo, this script
  - "qwen" env:                    Qwen2.5-VL / Qwen3-VL, transformers≥4.45
  The Qwen model is called via subprocess (qwen_vlm_worker.py) across envs.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action, get_libero_env, get_libero_image,
    get_libero_wrist_image, quat2axisangle, save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head, get_noisy_action_projector, get_processor,
    get_proprio_projector, resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME, get_action, get_image_resize_size, get_model,
    invert_gripper_action, normalize_gripper_action, set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

# --- Import Semantic CBF Safety Filter ---
from semantic_cbf_filter import SemanticCBFPipeline

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "safelibero_spatial"
    LIBERO_OBJECT = "safelibero_object"
    LIBERO_GOAL = "safelibero_goal"
    LIBERO_LONG = "safelibero_long"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 300,
    TaskSuite.LIBERO_OBJECT: 300,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_LONG: 550,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # ── OpenVLA-OFT model (runs in main conda env) ───────────────────────
    model_family: str = "openvla"
    pretrained_checkpoint: str = "checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
    config_json: str = "config_1.json"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # ── LIBERO environment ───────────────────────────────────────────────
    task_suite_name: str = TaskSuite.LIBERO_LONG
    safety_level: str = "I"
    num_steps_wait: int = 20
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 1024

    # ── Semantic CBF Safety Filter ───────────────────────────────────────
    use_cbf_safety_filter: bool = True
    cbf_use_vlm: bool = False                       # True → call Qwen via subprocess
    cbf_vlm_model: str = "qwen2.5-vl-7b"           # "qwen2.5-vl-7b" or "qwen3-vl-8b"
    cbf_vlm_conda_env: str = "qwen"                # Conda env where Qwen is installed
    cbf_vlm_device: str = "auto"                    # GPU for Qwen ("auto","cuda:0","cuda:1")
    cbf_vlm_load_in_4bit: bool = False              # 4-bit quantization for Qwen
    cbf_vlm_worker_script: str = "qwen_vlm_worker.py"
    cbf_vlm_num_votes: int = 3                      # Majority voting queries per constraint
    cbf_alpha_default: float = 1.0
    cbf_alpha_caution: float = 0.25
    cbf_safety_margin: float = 0.01
    cbf_workspace_z_max: float = 1.2
    cbf_dt: float = 0.05

    # ── Logging ──────────────────────────────────────────────────────────
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    seed: int = 7


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def validate_config(cfg):
    assert cfg.pretrained_checkpoint is not None
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop
    assert not (cfg.load_in_8bit and cfg.load_in_4bit)
    assert cfg.task_suite_name in [s.value for s in TaskSuite]


def initialize_model(cfg):
    model = get_model(cfg)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    action_head = get_action_head(cfg, model.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None
    noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim) if cfg.use_diffusion else None
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)
    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg, model):
    mapping = {"safelibero_spatial":"libero_spatial","safelibero_object":"libero_object",
               "safelibero_goal":"libero_goal","safelibero_long":"libero_10"}
    key = mapping.get(cfg.task_suite_name, cfg.task_suite_name)
    if key not in model.norm_stats and f"{key}_no_noops" in model.norm_stats:
        key = f"{key}_no_noops"
    assert key in model.norm_stats, f"Key {key} not found in norm_stats"
    cfg.unnorm_key = key


def setup_logging(cfg):
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.use_cbf_safety_filter: run_id += "-CBF"
    if cfg.cbf_use_vlm: run_id += f"-{cfg.cbf_vlm_model}"
    if cfg.run_id_note: run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    path = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(path, "w")
    logger.info(f"Log file: {path}")
    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)
    return log_file, path, run_id


def log_msg(msg, log_file=None):
    logger.info(msg)
    if log_file: log_file.write(msg + "\n"); log_file.flush()


def load_initial_states(cfg, task_suite, task_id, log_file=None):
    states = task_suite.get_task_init_states(task_id)
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path) as f: all_states = json.load(f)
        return states, all_states
    return states, None


def prepare_obs(obs, resize_size):
    img = get_libero_image(obs); wrist = get_libero_wrist_image(obs)
    return {
        "full_image": resize_image_for_policy(img, resize_size),
        "wrist_image": resize_image_for_policy(wrist, resize_size),
        "state": np.concatenate((obs["robot0_eef_pos"],
                                  quat2axisangle(obs["robot0_eef_quat"]),
                                  obs["robot0_gripper_qpos"])),
    }, img


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


# ═══════════════════════════════════════════════════════════════════════
# CBF Pipeline Init
# ═══════════════════════════════════════════════════════════════════════

def init_cbf(cfg, env):
    """Create the CBF pipeline. Qwen (if enabled) will be called via subprocess
    into the 'qwen' conda env when setup_scene() runs."""
    if not cfg.use_cbf_safety_filter:
        return None

    pipeline = SemanticCBFPipeline(
        env=env,
        use_vlm=cfg.cbf_use_vlm,
        vlm_model=cfg.cbf_vlm_model,
        vlm_conda_env=cfg.cbf_vlm_conda_env,
        vlm_device=cfg.cbf_vlm_device,
        vlm_load_in_4bit=cfg.cbf_vlm_load_in_4bit,
        vlm_worker_script=cfg.cbf_vlm_worker_script,
        vlm_num_votes=cfg.cbf_vlm_num_votes,
        dt=cfg.cbf_dt,
        alpha_default=cfg.cbf_alpha_default,
        alpha_caution=cfg.cbf_alpha_caution,
        safety_margin=cfg.cbf_safety_margin,
        workspace_z_max=cfg.cbf_workspace_z_max,
    )
    logger.info(f"CBF pipeline ready (VLM={'Qwen:'+cfg.cbf_vlm_model+'@'+cfg.cbf_vlm_conda_env if cfg.cbf_use_vlm else 'rule-based'})")
    return pipeline


# ═══════════════════════════════════════════════════════════════════════
# Episode Runner
# ═══════════════════════════════════════════════════════════════════════

def run_episode(cfg, env, task_desc, model, resize_size,
                processor=None, action_head=None, proprio_projector=None,
                noisy_action_projector=None, initial_state=None, log_file=None,
                cbf_pipeline=None):
    """Run one episode: OpenVLA → u_nominal → CBF → u_safe → env.step()"""

    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()

    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: num_open_loop_steps={cfg.num_open_loop_steps} != NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Wait for objects to settle
    t = 0
    while t < cfg.num_steps_wait:
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family)); t += 1

    # ── CBF SETUP (once per episode — static scene) ──
    cbf_interventions = 0; cbf_total = 0
    if cbf_pipeline is not None:
        cbf_pipeline.reset()
        cam_imgs = [get_libero_image(obs), get_libero_wrist_image(obs)]
        cbf_pipeline.setup_scene(obs=obs, task_description=task_desc, camera_images=cam_imgs)
        log_msg(f"[CBF] Scene setup complete: {task_desc}", log_file)

    # ── Obstacle tracking ──
    obs_names = [n.replace('_joint0','') for n in env.sim.model.joint_names if 'obstacle' in n]
    obstacle_name = " "
    for nm in obs_names:
        p = obs[f"{nm}_pos"]
        if p[2]>0 and abs(p[0])<0.5 and abs(p[1])<0.5:
            obstacle_name = nm; print(f"Obstacle: {nm}"); break
    init_obs_pos = obs[obstacle_name+"_pos"]
    collide_flag = False; collide_time = 0

    # ── Run episode ──
    replay_imgs = []; max_steps = TASK_MAX_STEPS[cfg.task_suite_name]; t = 0; success = False
    try:
        while t < max_steps:
            observation, img = prepare_obs(obs, resize_size)
            replay_imgs.append(img)

            # Query OpenVLA-OFT for u_nominal (action chunk)
            if len(action_queue) == 0:
                actions = get_action(cfg, model, observation, task_desc,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film)
                action_queue.extend(actions)

            # Get one u_nominal from chunk
            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)

            # ── CBF: u_nominal → u_safe ──
            if cbf_pipeline is not None:
                u_nominal = action.copy()
                action = cbf_pipeline.filter_action(action, obs)  # u_safe
                cbf_total += 1
                if np.linalg.norm(action[:6] - u_nominal[:6]) > 1e-4:
                    cbf_interventions += 1

            # Step environment with u_safe
            obs, reward, done, info = env.step(action.tolist())

            # Collision tracking
            if not collide_flag:
                if np.sum(np.abs(obs[obstacle_name+"_pos"] - init_obs_pos)) > 0.001:
                    collide_flag = True; collide_time = t
                    print(f"Collision at t={t}")

            if done: success = True; break
            t += 1

    except Exception as e:
        log_msg(f"Episode error: {e}", log_file)

    if cbf_pipeline and cbf_total > 0:
        rate = cbf_interventions / cbf_total * 100
        log_msg(f"[CBF] Interventions: {cbf_interventions}/{cbf_total} ({rate:.1f}%)", log_file)

    return success, collide_flag, replay_imgs, t


# ═══════════════════════════════════════════════════════════════════════
# Task Runner
# ═══════════════════════════════════════════════════════════════════════

def run_task(cfg, task_suite, task_id, model, resize_size,
             processor=None, action_head=None, proprio_projector=None,
             noisy_action_projector=None, total_ep=0, total_suc=0, log_file=None):

    task = task_suite.get_task(task_id)
    init_states, all_init = load_initial_states(cfg, task_suite, task_id, log_file)
    env, task_desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    cbf = init_cbf(cfg, env)

    t_ep, t_suc, t_col, t_ss = 0, 0, 0, 0; times = []
    for ep_idx in range(50):
        log_msg(f"\nTask: {task_desc}", log_file)

        if cfg.initial_states_path == "DEFAULT":
            init_state = init_states[ep_idx]
        else:
            tkey = task_desc.replace(" ","_"); ekey = f"demo_{ep_idx}"
            if not all_init[tkey][ekey]["success"]: continue
            init_state = np.array(all_init[tkey][ekey]["initial_state"])

        log_msg(f"Episode {t_ep+1}...", log_file)
        success, collide, replay, total_time = run_episode(
            cfg, env, task_desc, model, resize_size,
            processor, action_head, proprio_projector, noisy_action_projector,
            init_state, log_file, cbf_pipeline=cbf)
        times.append(total_time)

        t_ep += 1; total_ep += 1
        if success: t_suc += 1; total_suc += 1
        if collide: t_col += 1
        ss = success and not collide
        if ss: t_ss += 1

        save_rollout_video(replay, total_ep, success=success,
                           task_description=task_desc, safety_level=cfg.safety_level, log_file=log_file)

        log_msg(f"Success: {success} | Collide: {collide} | SS: {ss}", log_file)
        log_msg(f"Episodes: {total_ep} | Successes: {total_suc} ({total_suc/total_ep*100:.1f}%)", log_file)
        log_msg(f"Collides: {t_col} | SafeSuccesses: {t_ss}", log_file)

    sr = t_suc/t_ep if t_ep>0 else 0
    log_msg(f"Task SR: {sr:.3f} | Total SR: {total_suc/total_ep:.3f}", log_file)
    if cfg.use_wandb:
        wandb.log({f"success_rate/{task_desc}": sr, f"num_episodes/{task_desc}": t_ep})
    return total_ep, total_suc


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    print("=" * 60)
    print("Loading OpenVLA-OFT model (main conda env)...")
    print("=" * 60)
    model, action_head, proprio_proj, noisy_proj, processor = initialize_model(cfg)

    print("=" * 60)
    if cfg.use_cbf_safety_filter:
        print("Semantic CBF Safety Filter: ENABLED")
        print(f"  α_default={cfg.cbf_alpha_default}  α_caution={cfg.cbf_alpha_caution}")
        print(f"  safety_margin={cfg.cbf_safety_margin}")
        if cfg.cbf_use_vlm:
            print(f"  VLM: {cfg.cbf_vlm_model}  (conda env: {cfg.cbf_vlm_conda_env})")
            print(f"  VLM device: {cfg.cbf_vlm_device}  4bit: {cfg.cbf_vlm_load_in_4bit}")
            print(f"  VLM votes: {cfg.cbf_vlm_num_votes}")
            print(f"  Worker: {cfg.cbf_vlm_worker_script}")
        else:
            print("  VLM: disabled (rule-based constraints)")
    else:
        print("CBF Safety Filter: DISABLED")
    print("=" * 60)

    resize_size = get_image_resize_size(cfg)
    log_file, log_path, run_id = setup_logging(cfg)

    bench = benchmark.get_benchmark_dict()
    task_suite = bench[cfg.task_suite_name](safety_level=cfg.safety_level)

    log_msg(f"Suite: {cfg.task_suite_name} | Safety: {cfg.safety_level}", log_file)
    log_msg(f"CBF: {'ON ('+cfg.cbf_vlm_model+')' if cfg.use_cbf_safety_filter and cfg.cbf_use_vlm else 'ON (rules)' if cfg.use_cbf_safety_filter else 'OFF'}", log_file)

    total_ep, total_suc = 0, 0
    for task_id in [2]:
        total_ep, total_suc = run_task(
            cfg, task_suite, task_id, model, resize_size,
            processor, action_head, proprio_proj, noisy_proj,
            total_ep, total_suc, log_file)

    sr = total_suc / total_ep if total_ep > 0 else 0
    log_msg(f"\nFinal: {total_ep} episodes, {total_suc} successes, SR={sr:.4f} ({sr*100:.1f}%)", log_file)

    if cfg.use_wandb:
        wandb.log({"success_rate/total": sr, "num_episodes/total": total_ep})
        wandb.save(log_path)
    if log_file: log_file.close()
    return sr


if __name__ == "__main__":
    eval_libero()