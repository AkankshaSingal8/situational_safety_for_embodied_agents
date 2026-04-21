# Folder Notes (Jisu)

Project repo: `/ocean/projects/cis250185p/jqian8/situational_safety_for_embodied_agents`

## What This Folder Is Doing

SafeLIBERO + OpenVLA-OFT evaluation with an optional semantic CBF safety filter.

Pipeline, in the order things actually run:

```
env dump (RGBD + seg)         ->  per-object point clouds          ->  superquadric fit + CBF json
  save_vlm_inputs.py              cbf_superquadric.py (build_constraints)   cbf_superquadric.py (fit_superquadric)

                                                                                │
                                                                                ▼

VLA action  ->  CBF-QP safety filter  ->  u_safe  ->  env.step
 OpenVLA-OFT    semantic_cbf_filter.py           (run via run_vlm_eval.py)
```

## Important Environment Notes

- Main conda env for running anything in this repo: **`safelibero-vlm`** (not vanilla `libero`).
  - Activate with `conda activate safelibero-vlm`.
  - `libero` is editable-installed from `/ocean/projects/cis250185p/jqian8/src/vlsa-aegis/safelibero` but the `.pth` finder is sometimes broken (empty MAPPING / empty `top_level.txt`). Result: bare `python3 -c "import libero"` fails with `ModuleNotFoundError: No module named 'libero'`.
  - Workaround is baked into every top-level script as a `_prepend_safelibero_path()` helper that prepends the source tree to `sys.path` **before** the `from libero.libero import ...` line. It is ordered to prefer:
    1. `$SAFELIBERO_ROOT` (if set)
    2. `/ocean/projects/cis250185p/jqian8/src/vlsa-aegis/safelibero`
    3. `/ocean/projects/cis250185p/jqian8/src/vlsa-aegis/safelibero/libero`
    It only accepts a candidate if `libero/libero/benchmark/__init__.py` exists → SafeLIBERO, NOT vanilla LIBERO.
  - If a script is missing that helper and throws `ModuleNotFoundError: No module named 'libero'`, copy the helper from `vlm_pipeline/run_vlm_eval.py` / `save_vlm_inputs.py`.
- Rendering: always `export MUJOCO_GL=egl` before running, headless nodes crash otherwise.
- Compute: Bridges-2 GPU-shared node, typically `salloc -N 1 -p GPU-shared --gres=gpu:v100-32:1 -t 01:00:00`, then `ssh <v-node>`.

## Repo Layout

All pipeline scripts now live under `vlm_pipeline/` only (root-level duplicates were removed). Run everything with `python3 vlm_pipeline/<script>.py ...`.

| File | Role |
|---|---|
| `vlm_pipeline/save_vlm_inputs.py` | Dumps `{rgb, depth, seg_instance, seg_element, seg, camera_params, metadata}.{npy,png,json}` for each (safety_level, task, episode). This is the **data generator**. |
| `vlm_pipeline/cbf_superquadric.py` | Takes one episode's dump + a per-scene VLM JSON, builds per-object point clouds, fits superquadrics, writes `cbf_params.json` and 3D/2D visualizations. Gripper is approximated as a sphere (`GRIPPER_SPHERE_RADIUS = 0.12` m) via obstacle inflation. |
| `vlm_pipeline/semantic_cbf_filter.py` | Runtime CBF-QP filter (`SemanticCBFPipeline`). `setup_from_cbf_json(...)` loads a precomputed `cbf_params.json`. |
| `vlm_pipeline/run_libero_eval.py` | **Plain VLA eval, NO CBF.** |
| `vlm_pipeline/run_vlm_eval.py` | **VLA + Semantic CBF eval.** Exposes `--cbf_precomputed_json` CLI arg (accepts either a file path or a directory holding `cbf_params.json`; supports `{task_suite}/{safety_level}/{task_id}/{episode_idx}` templating). |
| `vlm_pipeline/qwen_vlm_worker.py` | Subprocess target for the live Qwen VLM path (only used when `--cbf_use_vlm True`). Not on the canonical flow. |
| `vlm_pipeline/vlm_outputs/` | VLM-produced relation JSONs (e.g. `vlm_manual_scene_safety_ep00_single.json`). |
| `vlm_pipeline/vlm_inputs/safelibero_spatial/level_I/task_X/episode_YY/` | Offline env dumps consumed by `cbf_superquadric.py`. |
| `vlm_pipeline/cbf_sq_outputs_mves_gripper_sphere/` | Canonical output of `cbf_superquadric.py` (`cbf_params.json` + visualizations). |

## Data File Contents (what's in each `episode_YY/` folder)

- `agentview_rgb.png`, `eye_in_hand_rgb.png`, `backview_rgb.png` (if backview renders) — 512×512 RGB.
- `agentview_depth.npy`, `eye_in_hand_depth.npy` — float32 metric depth (H,W).
- `agentview_seg_instance.npy`, `agentview_seg_element.npy` — robosuite instance / element segs.
- `agentview_seg.npy` — **canonical seg used by downstream scripts.** Prefers element IDs (geom IDs), falls back to instance IDs.
- `camera_params.json` — per-camera intrinsic + extrinsic (camera→world).
- `metadata.json` — robot state, object poses, active obstacle, `geom_id_to_name` map, `image_alignment` policy, and — after my edit — `robot_geoms` + `robot_links` for arm/gripper volume.

### `metadata.json` new fields (my addition)

- `robot_geoms`: list of `{name, body, part="arm"|"gripper", type (MuJoCo 2=sphere,3=capsule,4=ellipsoid,5=cylinder,6=box,7=mesh), type_name, size, pos_local, quat_local, pos_world, xmat_world}`. For mesh geoms (Panda links mostly are mesh), also `{mesh_id, mesh_num_verts, mesh_aabb_min, mesh_aabb_max}` in the mesh's local frame, so downstream CBF can fit a box/capsule without loading the raw mesh.
- `robot_links`: per-body world pose for `robot0_base`, `robot0_link0..7`, `gripper0_right_hand`, `gripper0_eef`, `gripper0_leftfinger`, `gripper0_rightfinger`. Any body not found is skipped (not all SafeLIBERO setups expose every name).

Only **new** episode dumps generated after my edit have these fields. If you see `camera_segmentations: None` / `image_alignment: None` in a `metadata.json`, that dump is stale (pre-Apr 12 layout) and you should regenerate it.

## Gotchas I've Already Hit

1. **`_seg.npy` silently held instance IDs on old dumps.** `cbf_superquadric.py::_geom_ids_for_object` matches against geom (element) IDs. If the dump was produced by the pre-Apr-12 `save_vlm_inputs.py` (`"camera_segmentations": "instance"` hardcoded), the mask ends up empty, `pc = None`, and the code **silently falls back** to `build_gt_point_cloud(...)`, which samples 200 points in a 0.05 m sphere around the GT object center — looks like a tiny sphere per object in `vis_3d.html`. Fix: regenerate the dumps with the current `save_vlm_inputs.py` (it defaults to `"instance,element"` and writes both `_seg_instance.npy` / `_seg_element.npy`, and canonical `_seg.npy = element`).
2. **CBF-for-one-episode vs VLA-over-all-episodes mismatch.** The precomputed `cbf_params.json` is **static** for the scene it was fit on (task X, episode Y, safety level L). If you run VLA eval over multiple episodes with a single hardcoded JSON, the constraints won't match the scene after episode 0. Either (a) restrict `--num_trials_per_task 1` so only ep 0 runs, or (b) precompute a JSON per episode and pass a templated `--cbf_precomputed_json` path.
3. ~~`run_task` hardcodes `task_id` iteration.~~ Fixed by the Apr 21 cleanup — both `vlm_pipeline/run_vlm_eval.py` and `vlm_pipeline/run_libero_eval.py` now take `--task_ids`. Accepts a single id (`--task_ids 0`), a comma-separated list (`--task_ids 0,1,2`), or the literal `all` for the whole suite. Defaults to `"0"`.
4. ~~Two copies of each script.~~ Fixed by the Apr 21 cleanup — only `vlm_pipeline/` copies remain. Working-tree versions of the deleted root copies are stashed under `.pre_cleanup_backup/` in case you need them.

## Commands That Work

### 0. Allocate GPU + set up env (on login node)

```bash
salloc -N 1 -p GPU-shared --gres=gpu:v100-32:1 --cpus-per-task=1 -t 01:00:00
# once allocated, ssh to the node, then:
conda activate safelibero-vlm
export MUJOCO_GL=egl
cd /ocean/projects/cis250185p/jqian8/situational_safety_for_embodied_agents
```

### 1. Regenerate env dumps for one episode (task 0, ep 0, level I)

```bash
python3 vlm_pipeline/save_vlm_inputs.py \
    --output_dir vlm_pipeline/vlm_inputs/safelibero_spatial \
    --safety_levels I \
    --task_ids 0 \
    --episode_indices 0
```

Sanity check that element seg is present (old dumps won't have these):

```bash
ls vlm_pipeline/vlm_inputs/safelibero_spatial/level_I/task_0/episode_00/*_seg_element.npy
```

To regenerate everything (all safety levels, all tasks, 50 eps each), drop the filter args and run the same command with defaults.

### 2. Fit superquadrics + write `cbf_params.json` for that episode

```bash
python3 vlm_pipeline/cbf_superquadric.py \
    --vlm_json vlm_pipeline/vlm_outputs/vlm_manual_scene_safety_ep00_single.json \
    --obs_folder vlm_pipeline/vlm_inputs/safelibero_spatial/level_I/task_0/episode_00 \
    --output_dir vlm_pipeline/cbf_sq_outputs_mves_gripper_sphere
```

In the log, `[build_constraints] <cam>: using <file>` should say `..._seg_element.npy`. If it says `..._seg.npy`, check whether the dump is stale and `_seg.npy` is really element-based. If point clouds end up empty, the pipeline falls back to the GT-center sphere (looks tiny in `vis_3d.html`).

Outputs land in `vlm_pipeline/cbf_sq_outputs_mves_gripper_sphere/`:
- `cbf_params.json` — per-constraint superquadric `{center, scales, ε1, ε2}`, `h_at_eef`, gradients, and gripper info (`{type: "sphere", radius: 0.12}`).
- `vis_3d.html`, `vis_pointcloud_only.html`, `vis_scene_3d.html`, `vis_{agentview,eye_in_hand,backview}_overlay.png`.

### 3. VLA + CBF eval using that CBF JSON

Use `vlm_pipeline/run_vlm_eval.py` (the one with `--cbf_precomputed_json`). Lock task / safety / episode count to match the JSON we just produced:

```bash
python3 vlm_pipeline/run_vlm_eval.py \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --num_trials_per_task 1 \
    --use_cbf_safety_filter True \
    --cbf_use_vlm False \
    --cbf_precomputed_json vlm_pipeline/cbf_sq_outputs_mves_gripper_sphere/cbf_params.json \
    --pretrained_checkpoint /ocean/projects/cis250185p/jqian8/checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
```

Key flags:
- `--cbf_use_vlm False` → rule-based CBF (no live Qwen VLM subprocess); since the JSON is already offline-produced, this is what you want.
- `--cbf_precomputed_json <file>` → a no-template path, so every episode reuses the same JSON. Combined with `--num_trials_per_task 1`, that's exactly "apply this CBF to ep 0 only".
- In the log, look for `[CBF] Loaded precomputed superquadrics: .../cbf_sq_outputs_mves_gripper_sphere/cbf_params.json` to confirm the JSON got picked up.
- Rollouts land in `./rollouts/<DATE>/*.mp4`; text logs in `./experiments/logs/EVAL-*-CBF*.txt`.

### 4. Baseline VLA eval without CBF (for comparison)

Cleanest path — `run_libero_eval.py` has no CBF wiring at all:

```bash
python3 vlm_pipeline/run_libero_eval.py \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --run_id_note no_cbf_spatial \
    --pretrained_checkpoint /ocean/projects/cis250185p/jqian8/checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
```

Equivalent — reuse the CBF eval script with the filter disabled:

```bash
python3 vlm_pipeline/run_vlm_eval.py \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --use_cbf_safety_filter False \
    --run_id_note no_cbf_spatial \
    --pretrained_checkpoint /ocean/projects/cis250185p/jqian8/checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
```

Note: the root-level `run_vlm_eval.py` / `run_libero_eval.py` duplicates no longer exist — use the `vlm_pipeline/` copies (see Repo Layout above).

## Quick Checks

Seg type in a given dump (element → max ≫ 20 matching geom IDs; instance → max typically < 20):

```bash
python3 -c "
import numpy as np
seg = np.load('vlm_pipeline/vlm_inputs/safelibero_spatial/level_I/task_0/episode_00/agentview_seg.npy')
print('dtype', seg.dtype, 'max', seg.max(), 'unique', len(np.unique(seg)))
"
```

Which CBF-JSON path a `run_vlm_eval.py` copy will actually use:

```bash
rg -n "HARDCODED_CBF_JSON|cbf_precomputed_json" run_vlm_eval.py vlm_pipeline/run_vlm_eval.py
```

Which task_id / episode range an eval iterates:

```bash
rg -n "for task_id in|num_trials_per_task" vlm_pipeline/run_vlm_eval.py
```

Most recent rollouts / logs:

```bash
find ./rollouts -type f -name "*.mp4" | xargs ls -lt | head
ls -lt experiments/logs | head
```
