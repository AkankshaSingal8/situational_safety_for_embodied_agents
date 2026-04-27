#!/usr/bin/env python3
"""
save_vlm_inputs.py

Captures all VLM-required inputs at the first timestep of every episode
for a configurable subset of tasks, safety levels, and episode count.

Usage:
    export MUJOCO_GL=egl   # if headless
    python save_vlm_inputs.py \
        --output_dir vlm_inputs/safelibero_spatial \
        --task_suite safelibero_spatial \
        --safety_levels I II \
        --task_ids 0 1 2 3 \
        --num_episodes 10 \
        --resolution 512
"""

import argparse
import json
import os

import numpy as np
from pathlib import Path
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.camera_utils import (
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix,
    get_real_depth_map,
)

# Canonical camera order for collection.
DEFAULT_CAMERA_NAMES = ["agentview", "robot0_eye_in_hand", "backview"]


# ─── Environment construction ────────────────────────────────────────────────

def camera_name_to_prefix(camera_name):
    """Map robosuite camera name to output file prefix."""
    return "eye_in_hand" if camera_name == "robot0_eye_in_hand" else camera_name


# Cameras that need 180° rotation (vertical + horizontal flip) to match policy preprocessing.
# Derived from safelibero_utils.py get_safelibero_image / get_safelibero_wrist_image.
# backview is not used by the policy so only vertical flip is applied there.
CAMERAS_NEEDING_HFLIP = {"agentview", "robot0_eye_in_hand"}


def get_camera_specs(camera_names):
    """Build camera specs used by save/load functions."""
    specs = []
    for cam_name in camera_names:
        prefix = camera_name_to_prefix(cam_name)
        specs.append(
            {
                "name": cam_name,
                "prefix": prefix,
                "image_key": f"{cam_name}_image",
                "depth_key": f"{cam_name}_depth",
                "seg_key": f"{cam_name}_segmentation_instance",
                "flip_horizontal": cam_name in CAMERAS_NEEDING_HFLIP,
            }
        )
    return specs


def create_observation_env(task, resolution=512, camera_names=None):
    """Create an OffScreenRenderEnv with depth and segmentation enabled.

    If a requested camera (e.g., backview) is unavailable in the current
    task XML, we gracefully fall back to the default two-view setup.
    """
    if camera_names is None:
        camera_names = list(DEFAULT_CAMERA_NAMES)
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": True,
        "camera_segmentations": "instance",
        "camera_names": camera_names,
    }
    try:
        env = OffScreenRenderEnv(**env_args)
    except Exception as exc:
        fallback = [c for c in camera_names if c != "backview"]
        if "backview" in camera_names and fallback:
            print(f"[warn] Failed to initialize with backview: {exc}")
            print(f"[warn] Falling back to cameras: {fallback}")
            env_args["camera_names"] = fallback
            env = OffScreenRenderEnv(**env_args)
            camera_names = fallback
        else:
            raise
    return env, task.language


# ─── Save functions ───────────────────────────────────────────────────────────

def _apply_flips(arr, flip_horizontal):
    """Vertical flip always (robosuite OpenGL origin); horizontal flip for cameras
    that need 180° rotation to match policy preprocessing."""
    arr = np.flip(arr, axis=0)
    if flip_horizontal:
        arr = np.flip(arr, axis=1)
    return arr.copy()


def save_rgb(obs, save_dir, camera_specs):
    """Save RGB images from all cameras."""
    for spec in camera_specs:
        key = spec["image_key"]
        if key not in obs:
            continue
        img = _apply_flips(obs[key], spec["flip_horizontal"])
        Image.fromarray(img).save(save_dir / f"{spec['prefix']}_rgb.png")


def save_depth(obs, sim, save_dir, camera_specs):
    """Convert raw z-buffer depth to metric depth and save."""
    for spec in camera_specs:
        key = spec["depth_key"]
        if key not in obs:
            continue
        real_depth = get_real_depth_map(sim, obs[key])
        real_depth = _apply_flips(real_depth, spec["flip_horizontal"])
        np.save(save_dir / f"{spec['prefix']}_depth.npy", real_depth.astype(np.float32))


def save_segmentation(obs, save_dir, camera_specs):
    """Save instance segmentation masks."""
    for spec in camera_specs:
        key = spec["seg_key"]
        if key not in obs:
            continue
        seg = obs[key].squeeze(-1)  # (H,W,1) → (H,W)
        seg = _apply_flips(seg, spec["flip_horizontal"])
        np.save(save_dir / f"{spec['prefix']}_seg.npy", seg.astype(np.int32))


def save_camera_params(sim, save_dir, resolution, camera_specs):
    """Save intrinsic and extrinsic camera matrices."""
    params = {}
    for spec in camera_specs:
        cam = spec["name"]
        intrinsic = get_camera_intrinsic_matrix(sim, cam, resolution, resolution)
        extrinsic = get_camera_extrinsic_matrix(sim, cam)
        params[cam] = {
            "intrinsic": intrinsic.tolist(),
            "extrinsic": extrinsic.tolist(),
        }
    with open(save_dir / "camera_params.json", "w") as f:
        json.dump(params, f, indent=2)


def save_obstacle(obs, env, save_dir):
    """Save active and all obstacle positions/quaternions to obstacle.json."""
    obstacle_names = [
        n.replace("_joint0", "")
        for n in env.sim.model.joint_names
        if "obstacle" in n
    ]

    all_obstacles = {}
    active_obstacle = None
    for name in obstacle_names:
        p = obs.get(f"{name}_pos")
        if p is None:
            continue
        q = obs.get(f"{name}_quat", np.zeros(4))
        all_obstacles[name] = {
            "position": p.tolist(),
            "quaternion": q.tolist(),
        }
        if active_obstacle is None and p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
            active_obstacle = {
                "name": name,
                "position": p.tolist(),
                "quaternion": q.tolist(),
            }

    obstacle_data = {
        "active_obstacle": active_obstacle,
        "all_obstacles": all_obstacles,
    }
    with open(save_dir / "obstacle.json", "w") as f:
        json.dump(obstacle_data, f, indent=2)


def save_metadata(obs, env, task_id, episode_idx, safety_level, task_description,
                  save_dir, task_suite="safelibero_spatial", resolution=512,
                  camera_specs=None):
    """Save robot state, object poses, obstacle info, and geom ID mapping."""
    # Robot state
    robot_state = {
        "eef_pos": obs["robot0_eef_pos"].tolist(),
        "eef_quat": obs["robot0_eef_quat"].tolist(),
        "joint_pos": obs["robot0_joint_pos"].tolist(),
        "joint_vel": obs["robot0_joint_vel"].tolist(),
        "gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
    }

    # Scene objects (all *_pos keys that aren't robot)
    objects = {}
    robot_keys = {"robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                  "robot0_joint_pos", "robot0_joint_vel"}
    for key in obs:
        if key.endswith("_pos") and key not in robot_keys and "to_robot0" not in key and obs[key].shape == (3,):
            name = key[:-4]  # strip "_pos"
            objects[name] = {
                "position": obs[key].tolist(),
                "quaternion": obs.get(f"{name}_quat", np.zeros(4)).tolist(),
            }

    # Obstacle identification
    obstacle_names = [
        n.replace("_joint0", "")
        for n in env.sim.model.joint_names
        if "obstacle" in n
    ]
    active_obstacle = {"name": None, "position": None}
    for name in obstacle_names:
        p = obs.get(f"{name}_pos")
        if p is not None and p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
            active_obstacle = {"name": name, "position": p.tolist()}
            break

    # Geom ID → name mapping (for interpreting segmentation masks)
    geom_id_to_name = {}
    for i in range(env.sim.model.ngeom):
        gname = env.sim.model.geom_id2name(i)
        if gname:
            geom_id_to_name[str(i)] = gname

    metadata = {
        "task_suite": task_suite,
        "safety_level": safety_level,
        "task_id": task_id,
        "episode_idx": episode_idx,
        "task_description": task_description,
        "image_resolution": resolution,
        "robot_state": robot_state,
        "objects": objects,
        "obstacle": active_obstacle,
        "geom_id_to_name": geom_id_to_name,
        # Per-camera flip state for downstream alignment.
        # vflip=True always (robosuite OpenGL); hflip=True for 180° rotation cameras.
        "image_alignment": {
            spec["prefix"]: {"vflip": True, "hflip": spec["flip_horizontal"]}
            for spec in (camera_specs or [])
        },
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# ─── Main loop ────────────────────────────────────────────────────────────────

def main(output_dir, resolution, task_suite, safety_levels, task_ids, num_episodes):
    output_root = Path(output_dir)
    benchmark_dict = benchmark.get_benchmark_dict()

    for safety_level in safety_levels:
        task_suite_obj = benchmark_dict[task_suite](safety_level=safety_level)
        resolved_task_ids = task_ids if task_ids is not None else range(task_suite_obj.n_tasks)

        for task_id in resolved_task_ids:
            task = task_suite_obj.get_task(task_id)
            env, task_description = create_observation_env(
                task,
                resolution,
                camera_names=list(DEFAULT_CAMERA_NAMES),
            )
            camera_names = list(getattr(env, "camera_names", DEFAULT_CAMERA_NAMES))
            camera_specs = get_camera_specs(camera_names)
            initial_states = task_suite_obj.get_task_init_states(task_id)

            for episode_idx in range(num_episodes):
                # Reset + init state
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

                # Wait steps to let physics settle
                for _ in range(20):
                    obs, _, _, _ = env.step([0.0] * 7)

                # Save directory
                save_dir = output_root / f"level_{safety_level}" / f"task_{task_id}" / f"episode_{episode_idx:02d}"
                save_dir.mkdir(parents=True, exist_ok=True)

                # Save all VLM inputs
                save_rgb(obs, save_dir, camera_specs)
                save_depth(obs, env.sim, save_dir, camera_specs)
                save_segmentation(obs, save_dir, camera_specs)
                save_camera_params(env.sim, save_dir, resolution, camera_specs)
                save_metadata(obs, env, task_id, episode_idx, safety_level, task_description,
                              save_dir, task_suite=task_suite, resolution=resolution,
                              camera_specs=camera_specs)
                save_obstacle(obs, env, save_dir)

                print(f"[{safety_level}] task {task_id} ep {episode_idx:02d} done")

            env.close()

    total = len(safety_levels) * len(list(resolved_task_ids)) * num_episodes
    print(f"Done. Saved {total} snapshots to {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save VLM inputs for SafeLIBERO")
    parser.add_argument("--output_dir", type=str, default="vlm_inputs/safelibero_spatial")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--task_suite", type=str, default="safelibero_spatial",
                        help="Benchmark suite name (default: safelibero_spatial)")
    parser.add_argument("--safety_levels", type=str, nargs="+", default=["I", "II"],
                        help="Safety levels to collect (default: I II)")
    parser.add_argument("--task_ids", type=int, nargs="*", default=None,
                        help="Task IDs to collect; omit for all tasks")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of episodes per task (episodes 0..N-1, default: 50)")
    args = parser.parse_args()
    main(
        output_dir=args.output_dir,
        resolution=args.resolution,
        task_suite=args.task_suite,
        safety_levels=args.safety_levels,
        task_ids=args.task_ids,
        num_episodes=args.num_episodes,
    )

'''
# Example: collect first 10 episodes, level I only, all 4 tasks
export MUJOCO_GL=egl
python save_vlm_inputs.py \
    --output_dir vlm_inputs/safelibero_spatial \
    --task_suite safelibero_spatial \
    --safety_levels I \
    --num_episodes 10

# Example: single task/level for quick testing
python save_vlm_inputs.py \
    --output_dir vlm_inputs/test \
    --safety_levels I \
    --task_ids 0 \
    --num_episodes 2
'''
