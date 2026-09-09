# Implementation Plan: `save_vlm_inputs.py`

**Goal:** Create a new standalone script that captures all VLM-required inputs at the first timestep of every episode across all 4 tasks × 2 safety levels × 50 episodes in SafeLIBERO-Spatial.

We only need the LIBERO simulation environment to render observations.


## 1. SafeLIBERO-Spatial Scope

| Task ID | Task Description |
|---------|-----------------|
| 0 | Pick up the black bowl between the plate and the ramekin and place it on the plate |
| 1 | Pick up the black bowl on the ramekin and place it on the plate |
| 2 | Pick up the black bowl on the stove and place it on the plate |
| 3 | Pick up the black bowl on the wooden cabinet and place it on the plate |

**Safety levels:** I (obstacle near target), II (obstacle obstructing path)

**Total captures:** 4 tasks × 2 levels × 50 episodes = **400 snapshots**

---

## 2. Script Dependencies


```python
import os
import json
import argparse
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
```



---

## 3. Environment Construction

### 3.1 Custom Function (replaces `get_libero_env`)

The existing `get_libero_env()` from `libero_utils.py` does not enable depth or segmentation. Write a self-contained constructor:

```python
def create_observation_env(task, resolution=512):
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
        "camera_names": ["agentview", "robot0_eye_in_hand"],
    }
    env = OffScreenRenderEnv(**env_args)
    return env, task.language
```

### 3.2 Observation Keys Produced

After stepping the environment, `obs` will contain:

| Key | Shape | Type |
|-----|-------|------|
| `agentview_image` | (512, 512, 3) | uint8 |
| `robot0_eye_in_hand_image` | (512, 512, 3) | uint8 |
| `agentview_depth` | (512, 512, 1) | float32 (raw z-buffer) |
| `robot0_eye_in_hand_depth` | (512, 512, 1) | float32 (raw z-buffer) |
| `agentview_segmentation_instance` | (512, 512, 1) | int32 |
| `robot0_eye_in_hand_segmentation_instance` | (512, 512, 1) | int32 |
| `robot0_eef_pos` | (3,) | float64 |
| `robot0_eef_quat` | (4,) | float64 |
| `robot0_gripper_qpos` | (2,) | float64 |
| `robot0_joint_pos` | (7,) | float64 |
| `robot0_joint_vel` | (7,) | float64 |
| `{object_name}_pos` | (3,) | float64 |
| `{object_name}_quat` | (4,) | float64 |

### 3.3 First-Run Validation

Before writing any save logic, verify the obs keys exist:

```python
env, desc = create_observation_env(task, resolution=512)
env.reset()
obs = env.set_init_state(initial_states[0])
for _ in range(20):
    obs, _, _, _ = env.step([0.0] * 7)
print(sorted(obs.keys()))
```

If `*_depth` or `*_segmentation_*` keys are missing, try alternate values for `camera_segmentations` (`"element"`, `True`, or `"instance,element"`).

---

## 4. What to Save Per Episode

### 4.1 RGB Images

```python
def save_rgb(obs, save_dir):
    for cam, key in [("agentview", "agentview_image"),
                     ("eye_in_hand", "robot0_eye_in_hand_image")]:
        img = obs[key]
        img = np.flip(img, axis=0).copy()  # flip if robosuite returns upside-down
        Image.fromarray(img).save(save_dir / f"{cam}_rgb.png")
```

**Image flip check:** Save one image without flipping and compare against the SafeLIBERO benchmark screenshots. If it's upside-down, keep the flip. If it already looks correct, remove it.

### 4.2 Depth Maps (Metric)

```python
def save_depth(obs, sim, save_dir):
    for cam, key in [("agentview", "agentview_depth"),
                     ("eye_in_hand", "robot0_eye_in_hand_depth")]:
        raw = obs[key]
        real_depth = get_real_depth_map(sim, raw)  # z-buffer → meters
        np.save(save_dir / f"{cam}_depth.npy", real_depth.astype(np.float32))
```

### 4.3 Segmentation Masks

```python
def save_segmentation(obs, save_dir):
    for cam, key in [("agentview", "agentview_segmentation_instance"),
                     ("eye_in_hand", "robot0_eye_in_hand_segmentation_instance")]:
        seg = obs[key].squeeze(-1)  # (H,W,1) → (H,W)
        np.save(save_dir / f"{cam}_seg.npy", seg.astype(np.int32))
```

### 4.4 Camera Parameters

```python
def save_camera_params(sim, save_dir, resolution):
    params = {}
    for cam in ["agentview", "robot0_eye_in_hand"]:
        intrinsic = get_camera_intrinsic_matrix(sim, cam, resolution, resolution)
        extrinsic = get_camera_extrinsic_matrix(sim, cam)
        params[cam] = {
            "intrinsic": intrinsic.tolist(),
            "extrinsic": extrinsic.tolist(),
        }
    with open(save_dir / "camera_params.json", "w") as f:
        json.dump(params, f, indent=2)
```

### 4.5 Metadata

```python
def save_metadata(obs, env, task_id, episode_idx, safety_level, task_description, save_dir):
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
        if key.endswith("_pos") and key not in robot_keys and obs[key].shape == (3,):
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
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
```

---

## 5. Output Folder Structure

```
vlm_inputs/
└── safelibero_spatial/
    ├── level_I/
    │   ├── task_0/
    │   │   ├── episode_00/
    │   │   │   ├── agentview_rgb.png
    │   │   │   ├── eye_in_hand_rgb.png
    │   │   │   ├── agentview_depth.npy
    │   │   │   ├── eye_in_hand_depth.npy
    │   │   │   ├── agentview_seg.npy
    │   │   │   ├── eye_in_hand_seg.npy
    │   │   │   ├── camera_params.json
    │   │   │   └── metadata.json
    │   │   ├── episode_01/
    │   │   └── ...
    │   │   └── episode_49/
    │   ├── task_1/
    │   ├── task_2/
    │   └── task_3/
    └── level_II/
        ├── task_0/
        ├── task_1/
        ├── task_2/
        └── task_3/
```

**Estimated size:** ~1.2 GB total (depth and seg `.npy` files dominate at 512×512).

---

## 6. Main Loop (Pseudocode)

```python
def main(output_dir="vlm_inputs/safelibero_spatial", resolution=512):
    output_root = Path(output_dir)
    benchmark_dict = benchmark.get_benchmark_dict()
    
    for safety_level in ["I", "II"]:
        task_suite = benchmark_dict["safelibero_spatial"](safety_level=safety_level)
        
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            env, task_description = create_observation_env(task, resolution)
            initial_states = task_suite.get_task_init_states(task_id)
            
            for episode_idx in range(50):
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
                save_rgb(obs, save_dir)
                save_depth(obs, env.sim, save_dir)
                save_segmentation(obs, save_dir)
                save_camera_params(env.sim, save_dir, resolution)
                save_metadata(obs, env, task_id, episode_idx, safety_level, task_description, save_dir)
                
                print(f"[{safety_level}] task {task_id} ep {episode_idx:02d} ✓")
            
            env.close()
    
    print(f"Done. Saved 400 snapshots to {output_root}")
```

---

## 7. Implementation Steps

### Step 1: Scaffold the script

Create `save_vlm_inputs.py` in the repo root (sibling to `run_libero_eval.py`). Add imports (§2), `argparse` with `--output_dir` and `--resolution` flags, and an empty `main()`.

### Step 2: Implement `create_observation_env()`

As in §3.1. Test it standalone: create the env, reset, print `obs.keys()`, confirm depth and segmentation keys are present.

### Step 3: Implement the five save functions

`save_rgb`, `save_depth`, `save_segmentation`, `save_camera_params`, `save_metadata` — as detailed in §4.1–4.5.

### Step 4: Determine the image flip

Run a single episode, save the raw `agentview_image` both with and without `np.flip(img, axis=0)`. Compare against the benchmark screenshots at `https://vlsa-aegis.github.io/benchmark.html`. Keep whichever matches. Hardcode the decision.

### Step 5: Test on one episode

```bash
export MUJOCO_GL=egl  # if headless
python save_vlm_inputs.py --output_dir test_output --resolution 512
```

With the main loop hardcoded to `safety_level="I"`, `task_id=0`, `episode_idx=0` only. Verify:

- [ ] RGB PNGs are correct orientation and show the right scene
- [ ] Depth `.npy` has values in ~0.5–2.0 meter range for tabletop objects
- [ ] Segmentation `.npy` has unique integer IDs; IDs map to names via `geom_id_to_name` in metadata
- [ ] `metadata.json` object positions are consistent with what's visible in the RGB
- [ ] `camera_params.json` intrinsics have fx ≈ fy and cx ≈ cy ≈ 256 (half of 512)
- [ ] Obstacle is correctly identified and its position is reasonable

### Step 6: Run full capture

Restore the full nested loop. Run:

```bash
python save_vlm_inputs.py --output_dir vlm_inputs/safelibero_spatial
```

Estimated time: **2–3 minutes** (400 × ~0.3s). No GPU needed.

### Step 7: Verify completeness

```python
from pathlib import Path
root = Path("vlm_inputs/safelibero_spatial")
for level in ["I", "II"]:
    for task in range(4):
        d = root / f"level_{level}" / f"task_{task}"
        eps = sorted(d.glob("episode_*"))
        assert len(eps) == 50, f"MISSING: level={level} task={task} got {len(eps)}"
        for ep in eps:
            for f in ["agentview_rgb.png", "eye_in_hand_rgb.png",
                       "agentview_depth.npy", "eye_in_hand_depth.npy",
                       "agentview_seg.npy", "eye_in_hand_seg.npy",
                       "camera_params.json", "metadata.json"]:
                assert (ep / f).exists(), f"MISSING: {ep / f}"
print("All 400 episodes × 8 files verified.")
```

---

## 8. Known Issues

| Issue | Mitigation |
|-------|------------|
| `camera_segmentations="instance"` not recognized | Try `"element"` or `True`; print obs keys to verify |
| Depth is z-buffer [0,1] not meters | Always convert with `get_real_depth_map(sim, raw)` |
| Images upside-down | Check once visually (Step 4), apply `np.flip` if needed |
| `env_img_res=1024` OOMs with depth+seg | Use 512 — enough for VLM, avoids memory issues |
| `MUJOCO_GL` not set on Bridges2 | `export MUJOCO_GL=egl` in sbatch script |
| `geom_id2name` returns `None` for unnamed geoms | Filter with `if gname:` (already handled in §4.5) |
| `robot0_joint_pos` key might differ across robosuite versions | Print obs keys first; adapt if the key is `robot0_joint_pos_cos`/`sin` instead |

