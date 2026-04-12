#!/usr/bin/env python3
"""
save_vlm_inputs.py

Captures all VLM-required inputs at the first timestep of every episode
across all 4 tasks × 2 safety levels × 50 episodes in SafeLIBERO-Spatial.

Usage:
    export MUJOCO_GL=egl   # if headless
    python save_vlm_inputs.py --output_dir vlm_inputs/safelibero_spatial
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
                "seg_instance_key": f"{cam_name}_segmentation_instance",
                "seg_element_key": f"{cam_name}_segmentation_element",
            }
        )
    return specs


def normalize_camera_segmentations(camera_segmentations, camera_names):
    """Normalize segmentation config for robosuite.

    Accepted forms:
      - "instance"
      - "instance,element"
      - ["instance", "element"]
      - [["instance", "element"], ...] (already camera-specific)
    Returns either a single string, None, or a per-camera nested list.
    """
    if camera_segmentations is None:
        return None

    # CLI string form, e.g. "instance,element"
    if isinstance(camera_segmentations, str):
        tokens = [t.strip() for t in camera_segmentations.split(",") if t.strip()]
        if len(tokens) <= 1:
            return tokens[0] if tokens else None
        return [tokens[:] for _ in camera_names]

    # Already list/tuple form.
    if isinstance(camera_segmentations, (list, tuple)):
        vals = list(camera_segmentations)
        if not vals:
            return None

        # Already nested per camera.
        if any(isinstance(v, (list, tuple)) for v in vals):
            return vals

        # Flat list of segmentation types -> apply same set to each camera.
        return [vals[:] for _ in camera_names]

    return camera_segmentations


def create_observation_env(task, resolution=512, camera_names=None,
                           camera_segmentations="instance"):
    """Create an OffScreenRenderEnv with depth and segmentation enabled.

    If a requested camera (e.g., backview) is unavailable in the current
    task XML, we gracefully fall back to the default two-view setup.
    """
    if camera_names is None:
        camera_names = list(DEFAULT_CAMERA_NAMES)
    camera_segmentations = normalize_camera_segmentations(camera_segmentations, camera_names)
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
        "camera_segmentations": camera_segmentations,
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

def save_rgb(obs, save_dir, camera_specs):
    """Save RGB images from both cameras."""
    for spec in camera_specs:
        key = spec["image_key"]
        if key not in obs:
            continue
        img = obs[key]
        # robosuite returns images upside-down; flip vertically
        img = np.flip(img, axis=0).copy()
        Image.fromarray(img).save(save_dir / f"{spec['prefix']}_rgb.png")


def save_depth(obs, sim, save_dir, camera_specs):
    """Convert raw z-buffer depth to metric depth and save."""
    for spec in camera_specs:
        key = spec["depth_key"]
        if key not in obs:
            continue
        raw = obs[key]
        real_depth = get_real_depth_map(sim, raw)
        # Keep pixel alignment consistent with flipped RGB.
        real_depth = np.flip(real_depth, axis=0).copy()
        np.save(save_dir / f"{spec['prefix']}_depth.npy", real_depth.astype(np.float32))


def save_segmentation(obs, save_dir, camera_specs):
    """Save segmentation masks.

    Writes:
      - <prefix>_seg_instance.npy  (if available)
      - <prefix>_seg_element.npy   (if available)
      - <prefix>_seg.npy           (canonical: prefer element, else instance)
    """
    for spec in camera_specs:
        seg_instance = None
        seg_element = None

        if spec["seg_instance_key"] in obs:
            seg_instance = obs[spec["seg_instance_key"]].squeeze(-1)
            seg_instance = np.flip(seg_instance, axis=0).copy()
            np.save(save_dir / f"{spec['prefix']}_seg_instance.npy",
                    seg_instance.astype(np.int32))

        if spec["seg_element_key"] in obs:
            seg_element = obs[spec["seg_element_key"]].squeeze(-1)
            seg_element = np.flip(seg_element, axis=0).copy()
            np.save(save_dir / f"{spec['prefix']}_seg_element.npy",
                    seg_element.astype(np.int32))

        # Canonical mask used by downstream scripts: prefer element IDs.
        if seg_element is not None:
            np.save(save_dir / f"{spec['prefix']}_seg.npy",
                    seg_element.astype(np.int32))
        elif seg_instance is not None:
            np.save(save_dir / f"{spec['prefix']}_seg.npy",
                    seg_instance.astype(np.int32))


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


def save_metadata(obs, env, task_id, episode_idx, safety_level, task_description,
                  save_dir, camera_segmentations):
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
        "task_suite": "safelibero_spatial",
        "safety_level": safety_level,
        "task_id": task_id,
        "episode_idx": episode_idx,
        "task_description": task_description,
        "image_resolution": 512,
        "robot_state": robot_state,
        "objects": objects,
        "obstacle": active_obstacle,
        "geom_id_to_name": geom_id_to_name,
        # Explicitly record saved image orientation policy so downstream
        # scripts can align legacy and new captures robustly.
        "image_alignment": {"rgb": True, "depth": True, "seg": True},
        "camera_segmentations": camera_segmentations,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# ─── Main loop ────────────────────────────────────────────────────────────────

def main(output_dir="vlm_inputs/safelibero_spatial", resolution=512,
         camera_segmentations="instance,element", safety_levels=None,
         task_ids=None, episode_indices=None):
    output_root = Path(output_dir)
    benchmark_dict = benchmark.get_benchmark_dict()

    if safety_levels is None:
        safety_levels = ["I", "II"]

    for safety_level in safety_levels:
        task_suite = benchmark_dict["safelibero_spatial"](safety_level=safety_level)

        cur_task_ids = list(range(task_suite.n_tasks)) if task_ids is None else task_ids
        for task_id in cur_task_ids:
            task = task_suite.get_task(task_id)
            env, task_description = create_observation_env(
                task,
                resolution,
                camera_names=list(DEFAULT_CAMERA_NAMES),
                camera_segmentations=camera_segmentations,
            )
            camera_names = list(getattr(env, "camera_names", DEFAULT_CAMERA_NAMES))
            camera_specs = get_camera_specs(camera_names)
            initial_states = task_suite.get_task_init_states(task_id)

            cur_episode_indices = list(range(50)) if episode_indices is None else episode_indices
            for episode_idx in cur_episode_indices:
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
                save_metadata(
                    obs, env, task_id, episode_idx, safety_level,
                    task_description, save_dir, camera_segmentations
                )

                print(f"[{safety_level}] task {task_id} ep {episode_idx:02d} done")

            env.close()

    print(f"Done. Saved 400 snapshots to {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save VLM inputs for SafeLIBERO-Spatial")
    parser.add_argument("--output_dir", type=str, default="vlm_inputs/safelibero_spatial")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--camera_segmentations", type=str, default="instance,element",
        help='robosuite camera_segmentations setting (e.g. "instance", "element", "instance,element")'
    )
    parser.add_argument(
        "--safety_levels", type=str, default="I,II",
        help='Comma-separated safety levels to export, e.g. "I" or "I,II"'
    )
    parser.add_argument(
        "--task_ids", type=str, default=None,
        help='Optional comma-separated task ids, e.g. "0,1"'
    )
    parser.add_argument(
        "--episode_indices", type=str, default=None,
        help='Optional comma-separated episode indices, e.g. "0,3,7"'
    )
    args = parser.parse_args()
    safety_levels = [s.strip() for s in args.safety_levels.split(",") if s.strip()]
    task_ids = None if args.task_ids is None else [int(x.strip()) for x in args.task_ids.split(",") if x.strip()]
    episode_indices = None if args.episode_indices is None else [int(x.strip()) for x in args.episode_indices.split(",") if x.strip()]
    main(
        output_dir=args.output_dir,
        resolution=args.resolution,
        camera_segmentations=args.camera_segmentations,
        safety_levels=safety_levels,
        task_ids=task_ids,
        episode_indices=episode_indices,
    )

'''
export MUJOCO_GL=egl
python save_vlm_inputs.py --output_dir vlm_inputs/safelibero_spatial
'''