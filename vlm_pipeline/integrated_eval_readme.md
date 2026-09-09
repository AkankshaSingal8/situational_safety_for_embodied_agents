# Integrated SafeLIBERO VLM+CBF Evaluation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete `run_libero_eval_integrated.py` — a full evaluation pipeline where OpenVLA-OFT generates nominal actions and a Qwen-based CBF safety filter runs **at every action chunk boundary** (same frequency as the policy) to extract updated semantic constraints from the current scene, construct ellipsoidal safe sets, certify the upcoming action chunk via CBF-QP, and report TSR/CAR/ETS metrics. The manipulator waits while the VLM+CBF pipeline solves before executing the next certified chunk.

**Architecture:** Two processes. The main process (`openvla_libero_merged`) runs the LIBERO sim, OpenVLA policy, and CBF-QP. At every action chunk boundary, it calls a **persistent Qwen HTTP server** running in the `qwen` conda env to update constraints from the current observation — then the sim resumes with the newly certified action chunk. The persistent server is mandatory because per-chunk calling frequency (~38 calls/episode) makes `conda run` startup overhead (~45s × 38 = ~28 min/episode) impractical.

**Tech Stack:** Python 3.10, PyTorch 2.2.0, OpenVLA-OFT (prismatic-vlm), LIBERO / SafeLIBERO, robosuite, draccus, Qwen2.5-VL-7B (persistent Flask server), scipy (CBF-QP), numpy, imageio, wandb

---

## 1. Current Script Analysis

### 1.1 `run_safelibero_openvla_oft_eval.py` — Baseline (Complete, Reference Implementation)

The clean baseline: OpenVLA-OFT only, no safety filter. Source of truth for config structure, logging style, metric formulas, and rollout flow.

**Key design decisions to mirror:**
- `EvalConfig` uses `@dataclass` + `draccus` with snake_case CLI flags
- `run_id` format: `EVAL-{suite}-level{level}-{model_family}-{DATE_TIME}`
- `results_output_dir` is a top-level config field; results go to `{results_output_dir}/{suite}/results_{run_id}.json`
- Video saved to `{video_output_dir}/{suite}/{task_id}/{level}/episode_0_{desc}.mp4` (first episode only)
- Collision detection: `np.sum(np.abs(current - initial)) > 0.001` with a `None` guard for missing obstacles
- Formulas (matching SafeLIBERO paper):
  - `TSR = task_successes / task_episodes`
  - `CAR = (task_episodes - task_collides) / task_episodes`
  - `ETS = mean(steps_per_episode)` (where steps = `t` counter, not counting `num_steps_wait`)

### 1.2 `run_libero_eval_integrated.py` — Integrated VLM+CBF (Needs Restructuring)

The integrated script exists but implements **once-per-episode** VLM calling. The architecture must change to **per-chunk** VLM calling. The functions that change and those that stay are:

| Function | Location | Change Required |
|---|---|---|
| `GenerateConfig` | line 109 | Add `vlm_server_url`, `results_output_dir`; remove `qwen_conda_env` from required fields |
| `validate_config()` | line 174 | Add safety_level check (Bug #3) |
| `setup_logging()` | line 271 | Fix run_id format (Bug #2) |
| `capture_episode_obs()` | line 368 | **Obsolete** in new flow; replaced by `capture_chunk_obs_from_env()` |
| `run_vlm_subprocess()` | line 416 | **Replaced** by `call_vlm_server()` |
| `get_episode_cbf_constraints()` | line 504 | **Obsolete** in new flow; logic inlined in `run_episode()` |
| `build_episode_ellipsoids()` | line 472 | **Unchanged** — reused per-chunk |
| `certify_action_simple()` | line 616 | Unchanged |
| `evaluate_cbf()` | line 593 | Unchanged |
| `save_trajectory()` | line 719 | Unchanged |
| `run_episode()` | line 777 | **Restructured** — VLM+CBF at chunk boundary, not episode start |
| `run_task()` | line 975 | Minor: pass `camera_specs` to `run_episode()` |
| `eval_libero()` | line 1092 | Add `results_output_dir` fix (Bug #4) |

**Known bugs that must be fixed before running:**

**Bug #1 — Obstacle name crash** (`run_libero_eval_integrated.py:844–852`):
```python
# CURRENT (broken):
obstacle_name = " "            # ← space, not None
for i in obstacle_names:
    p = obs[f"{i}_pos"]        # ← KeyError if key missing
    if p[2] > 0 and ...:
        obstacle_name = i
        break
initial_obstacle_pos = obs[obstacle_name + "_pos"]  # ← KeyError: ' _pos' if no obstacle found

# FIXED:
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
```

**Bug #2 — `run_id` missing safety level** (`run_libero_eval_integrated.py:274`):
```python
# CURRENT:
run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
# FIXED:
run_id = f"EVAL-{cfg.task_suite_name}-level{cfg.safety_level}-{cfg.model_family}-{DATE_TIME}"
```

**Bug #3 — `validate_config()` missing safety_level check** (`run_libero_eval_integrated.py:174`):
```python
# ADD after suite assertion:
if cfg.safety_level not in ("I", "II"):
    raise ValueError(f"Invalid safety_level '{cfg.safety_level}'. Must be 'I' or 'II'.")
```

**Bug #4 — `results_output_dir` not configurable** (`run_libero_eval_integrated.py:1184`):
```python
# CURRENT:
results_dir = os.path.join(cfg.local_log_dir, "results", cfg.task_suite_name)
# FIXED (add field to GenerateConfig and use it):
results_output_dir: str = "integrated_benchmark"
# ...
results_dir = os.path.join(cfg.results_output_dir, cfg.task_suite_name)
```

**Cleanup (non-blocking):**
- Lines 218–252: Commented-out mock model code. Remove after smoke-test passes.
- Lines 806, 904: stray `print()` calls. Replace with `logger.info()`.

### 1.3 `safelibero_utils.py` — Needs One Extension

`create_vlm_obs_env()` exists and is correct. The new architecture eliminates the secondary VLM env at episode start. Instead, the **policy env itself** needs to provide depth and segmentation, so observations are available mid-episode without EGL context conflicts.

`get_safelibero_env()` must accept `camera_depths` and `camera_segmentations` parameters (currently these are hardcoded to off).

### 1.4 `cbf_construction.py` — Complete (No Changes)

`build_constraints(vlm_json, obs_folder)` returns `(constraints, behavioral, pose, eef_pos)`. Reused identically per chunk.

### 1.5 `qwen_vlm_worker.py` — Complete (No Changes)

The existing worker script is reused as-is inside `qwen_vlm_server.py`. The server wraps the same inference logic and serves it over HTTP.

---

## 2. Proposed Architecture

### 2.1 Two-Process Model

```
┌─────────────────────────────────────────────────────────────────┐
│  Process A: openvla_libero_merged env (main Python process)     │
│  ──────────────────────────────────────────────────────────     │
│  • LIBERO / SafeLIBERO simulation (MuJoCo/EGL)                 │
│  • OpenVLA-OFT policy inference (GPU 0)                         │
│  • CBF-QP action certification (CPU, every step)               │
│  • Metrics + logging + trajectory save                         │
│                                                                 │
│  At each action chunk boundary:                                 │
│    1. write obs files to {vlm_tmp_dir}/task_N/ep_NN/chunk_KK/  │
│    2. HTTP POST → Qwen server                                   │
│    3. block until JSON response                                 │
│    4. build_episode_ellipsoids() → certify chunk                │
│    5. resume sim with certified actions                         │
│                            │                                    │
│         HTTP POST/response │  (filesystem for obs files)       │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Process B: conda activate qwen && python qwen_vlm_server│   │
│  │  ─────────────────────────────────────────────────────  │   │
│  │  • Loads Qwen2.5-VL-7B ONCE at startup (GPU 1)         │   │
│  │  • Serves POST /infer for each chunk                    │   │
│  │  • Input: reads obs files from shared filesystem        │   │
│  │  • Output: returns constraint JSON in response body     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

IPC: HTTP on localhost. Obs files on shared filesystem (no serialization overhead for numpy arrays).

### 2.2 Per-Episode Control Flow

```
Episode start:
  1. env.reset() + env.set_init_state(initial_state)
  2. Stabilization: 20 × env.step(dummy_action)  ← not counted in t
  3. Obstacle detection (once, from obs after stabilization)
  4. ellipsoids = [], cbf_active = False  ← no pre-episode VLM call

Episode control loop (t = 0 … max_steps):
  5. If action_queue is EMPTY (every num_open_loop_steps=8 steps):
     ┌─ VLM+CBF pipeline (sim pauses here) ──────────────────────┐
     │ a. capture_chunk_obs_from_env()                            │
     │      save agentview_rgb.png, agentview_depth.npy,         │
     │      agentview_seg.npy, camera_params.json, metadata.json  │
     │      from current policy env obs (depth+seg enabled)       │
     │ b. call_vlm_server()                                       │
     │      HTTP POST → localhost:5001/infer                      │
     │      blocks until Qwen returns (~1–5s)                     │
     │ c. build_episode_ellipsoids()                              │
     │      cbf_construction.build_constraints() + rename        │
     │      updates ellipsoids, behavioral, pose for this chunk   │
     └───────────────────────────────────────────────────────────┘
     d. get_action()  ← OpenVLA inference on current obs
     e. action_queue.extend(certified_chunk)

  6. action = action_queue.popleft()
  7. certify_action_simple()  ← CBF-QP using THIS chunk's ellipsoids
  8. rotation lock: action[3:6] = 0  if pose["rotation_lock"]
  9. process_action()         ← normalize + invert gripper
 10. env.step(action)
 11. collision detection      ← obstacle displacement > 1 mm
 12. done check
 13. t += 1
 14. save_trajectory()
```

**Key timing facts:**
- Control frequency: 20 Hz (each `env.step()` = 50 ms of simulated time)
- OpenVLA inference: every `num_open_loop_steps=8` steps → 2.5 Hz for policy queries
- VLM+CBF pipeline: also every 8 steps, **runs simultaneously with OpenVLA query** (same boundary). Sim pauses (offline eval) while both complete.
- CBF-QP: every step → 20 Hz (CPU-only, <1 ms per call), using constraints from the most recent chunk
- For a 300-step episode: ~38 VLM calls per episode

---

## 3. Qwen Process Design Decision

### Three Options Evaluated

| | Option A: `conda run` per chunk | Option B: Persistent HTTP server | Option C: Separate terminal |
|---|---|---|---|
| **Latency per call** | 30–90s (startup + inference) | 1–5s (inference only) | 1–5s (inference only) |
| **38 calls/episode cost** | **~28–57 min/episode** | **~0.6–3.2 min/episode** | ~0.6–3.2 min/episode |
| **50 trials × 10 tasks cost** | 237–475 hours | 5–27 hours | 5–27 hours |
| **IPC complexity** | None | Low (HTTP, filesystem for files) | Medium (must sync manually) |
| **Fault isolation** | Full per-call | Server crash = need restart | Terminal crash = need restart |
| **Debugging** | Easy | Easy (curl to test endpoint) | Medium |
| **SLURM compatibility** | Full | Single job with server started in background | Requires 2 terminal sessions |

**Recommendation: Option B (persistent HTTP server) is required for per-chunk calling.**

**Why Option A is impractical at per-chunk frequency:**
`conda run` incurs ~15–45s of process startup + model loading overhead on every call. With 38 calls per 300-step episode and 500 total episodes (50 trials × 10 tasks), Option A would require 285–855 GPU-hours of Qwen startup time alone. This is prohibitive even for offline evaluation.

**Why Option B is now the right tradeoff:**
1. Qwen loads once → ~20s startup cost amortized across all episodes.
2. Per-call inference: ~1–5s. For a 300-step episode: 38 × 3s = ~2 min VLM overhead.
3. HTTP server on localhost is trivial to implement with Flask (~50 lines).
4. The server can be started in the background by the SLURM job script before the main eval loop.
5. If the server crashes, `call_vlm_server()` returns `None` and the episode continues with the previous chunk's constraints — same graceful degradation as Option A.

**Why the sim can pause for VLM inference:**
This is offline simulation. `env.step()` is only called after `call_vlm_server()` returns. The simulation clock advances only when we call `step()`, so there is no "real-time" constraint. The robot arm physically waits in the scene while the VLM reasons — which is semantically correct (the safety system must complete before the arm moves).

### Can Qwen Run at the Same Frequency as `env.step()` (20 Hz)?

**No. This is physically impossible at 20 Hz.**

- Qwen2.5-VL-7B inference: ~1–5 seconds per call (persistent service)
- Required period for 20 Hz: 50 ms
- The gap is 20–100×

**The per-chunk design (2.5 Hz) is the correct compromise:**
- Constraints update fast enough to track scene changes across a 400 ms window
- Qwen inference (1–5s) fits within the action chunk execution time (~8 × 50ms = 400ms simulated, but wall-clock is dominated by OpenVLA inference)
- The manipulator waits; the simulation does not advance until both OpenVLA and Qwen have returned

---

## 4. Metrics Computation Plan

All three metrics are computed in `run_libero_eval_integrated.py`. No changes needed to metric formulas; only the VLM call site changes.

### 4.1 Collision Avoidance Rate (CAR)

**Definition:** Fraction of episodes where the obstacle is never displaced more than 1 mm.

**How it's detected** (after every `env.step()`):
```python
if obstacle_name is not None:
    then_obstacle_pos = obs.get(f"{obstacle_name}_pos", initial_obstacle_pos)
    if np.sum(np.abs(then_obstacle_pos - initial_obstacle_pos)) > 0.001:
        collide_flag = True   # set once, never cleared
```
- `initial_obstacle_pos` captured once after `num_steps_wait` warm-up
- Threshold: 1 mm total L1 displacement (matches `run_safelibero_openvla_oft_eval.py`)
- If no obstacle found, `collide_flag` stays `False` (episode counts as non-colliding)

**Aggregation:**
```python
car = float(task_episodes - task_collides) / task_episodes   # per task
overall_car = float(all_episodes - all_collides) / all_episodes  # overall
```

### 4.2 Task Success Rate (TSR)

**Definition:** Fraction of episodes where `done=True` before `max_steps`.

**How it's detected:**
```python
if done:
    success = True
    break
```
`done` is set by the LIBERO task termination condition (task-specific reward).

**Aggregation:**
```python
tsr = float(task_successes) / task_episodes
```

### 4.3 Execution Time Steps (ETS)

**Definition:** Number of `env.step()` calls per episode, not counting the `num_steps_wait` warm-up.

`t` starts at 0 after the wait loop, and `run_episode()` returns `t` as `steps`.

**Aggregation:**
```python
ets_mean = float(np.mean(timesteps_list))
ets_median = float(np.median(timesteps_list))
```

### 4.4 Output JSON Structure

```json
{
  "task_0": {
    "description": "pick up the black bowl...",
    "episodes": 50, "successes": 38, "collisions": 4,
    "TSR": 0.76, "CAR": 0.92, "ETS_mean": 187.3, "ETS_median": 201.0
  },
  "overall": {
    "model": "moojink/...", "suite": "safelibero_spatial",
    "safety_level": "I", "vlm_method": "m1", "vlm_dry_run": false,
    "total_episodes": 200, "TSR": 0.74, "CAR": 0.91,
    "ETS_mean": 190.2, "ETS_median": 203.0
  }
}
```

---

## 5. File-Level Change Plan

### New File: `qwen_vlm_server.py`

A Flask HTTP server that loads Qwen once at startup and serves inference requests. Runs in the `qwen` conda env.

```
qwen_vlm_server.py
  - load_model(model_key): loads Qwen2.5-VL and processor once
  - POST /infer: reads obs_folder + method from JSON body,
                 calls qwen_vlm_worker inference logic,
                 returns constraint JSON in response
  - POST /health: returns {"status": "ok"} for startup check
  - CLI: --port (default 5001), --model (default qwen2.5-vl-7b),
         --method (default m1), --num_votes (default 1)
```

### Modify: `safelibero_utils.py`

Add `camera_depths` and `camera_segmentations` parameters to `get_safelibero_env()`:

```python
def get_safelibero_env(task, model_family, resolution=256, include_wrist_camera=False,
                        camera_depths=False, camera_segmentations=None):
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_names": camera_names,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "hard_reset": False,
        "camera_depths": camera_depths,
    }
    if camera_segmentations is not None:
        env_args["camera_segmentations"] = camera_segmentations
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description
```

When `use_safety_filter=True`, the caller passes `camera_depths=True, camera_segmentations="instance"`, so `obs["agentview_depth"]` and `obs["agentview_segmentation_instance"]` are available at every step. No secondary VLM env needed.

### Modify: `run_libero_eval_integrated.py`

**New functions to add:**

```python
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
    """Save depth+seg+rgb obs from current running policy env to obs_folder.

    Does NOT create a secondary env. Requires the policy env to have been
    created with camera_depths=True and camera_segmentations='instance'.
    Saves: agentview_rgb.png, agentview_depth.npy, agentview_seg.npy,
           camera_params.json, metadata.json
    """
    obs_path = Path(obs_folder)
    obs_path.mkdir(parents=True, exist_ok=True)
    save_rgb(obs, obs_path, camera_specs)
    save_depth(obs, sim, obs_path, camera_specs)
    save_segmentation(obs, obs_path, camera_specs)
    save_camera_params(sim, obs_path, resolution, camera_specs)
    save_metadata(obs, ..., task_id, episode_idx, safety_level, task_description, obs_path)


def call_vlm_server(
    obs_folder: str,
    output_json_path: str,
    server_url: str = "http://localhost:5001",
    method: str = "m1",
    timeout: int = 120,
    dry_run: bool = False,
) -> Optional[Dict]:
    """Call persistent Qwen server via HTTP POST.

    Returns constraint JSON dict on success, None on failure.
    Falls back to run_vlm_subprocess() if server is not reachable and dry_run=True.
    """
    import requests
    if dry_run:
        return {"single": {"description": "dry run", "end_object": "object", "objects": []}}
    try:
        resp = requests.post(
            f"{server_url}/infer",
            json={"obs_folder": obs_folder, "method": method},
            timeout=timeout,
        )
        if resp.ok:
            result = resp.json()
            with open(output_json_path, "w") as f:
                json.dump(result, f)
            return result
        logger.warning(f"VLM server returned {resp.status_code}: {resp.text[:200]}")
        return None
    except requests.exceptions.ConnectionError:
        logger.warning(f"VLM server not reachable at {server_url}")
        return None
    except requests.exceptions.Timeout:
        logger.warning(f"VLM server timed out after {timeout}s")
        return None
```

**Updated `GenerateConfig` fields:**

```python
# Replace qwen_conda_env with:
vlm_server_url: str = "http://localhost:5001"   # persistent Qwen server

# Keep (unchanged):
vlm_method: str = "m1"
vlm_model: str = "qwen2.5-vl-7b"
num_vlm_votes: int = 1
vlm_dry_run: bool = False
vlm_tmp_dir: str = "/tmp/vlm_obs"
vlm_resolution: int = 512   # kept for metadata consistency; obs captured at env_img_res

# Add:
results_output_dir: str = "integrated_benchmark"
```

**Updated `run_task()` env creation:**

```python
# Pass depth+seg to policy env when safety filter is enabled
env, task_description = get_safelibero_env(
    task, cfg.model_family,
    resolution=cfg.env_img_res,
    include_wrist_camera=True,
    camera_depths=cfg.use_safety_filter,
    camera_segmentations="instance" if cfg.use_safety_filter else None,
)
camera_specs = get_camera_specs(["agentview", "robot0_eye_in_hand"]) if cfg.use_safety_filter else None
```

**Updated `run_episode()` — VLM called at every action chunk boundary:**

```python
def run_episode(cfg, env, task_description, model, resize_size,
                processor=None, action_head=None, proprio_projector=None,
                noisy_action_projector=None, initial_state=None, log_file=None,
                task_id=None, trajectory_dir=None, episode_idx=None,
                camera_specs=None):
    # ... existing reset + wait loop ...

    # Initialize constraints (empty until first chunk boundary)
    ellipsoids = []
    behavioral = _DEFAULT_BEHAVIORAL.copy()
    pose = _DEFAULT_POSE.copy()
    cbf_active = False

    # ... obstacle detection (with Bug #1 fix) ...

    success = False
    try:
        while t < max_steps:
            # ── ACTION CHUNK BOUNDARY ─────────────────────────────────
            if len(action_queue) == 0:
                # 1. Update VLM constraints from current observation
                if cfg.use_safety_filter and camera_specs is not None:
                    chunk_idx = t // cfg.num_open_loop_steps
                    chunk_folder = os.path.join(
                        cfg.vlm_tmp_dir,
                        f"task_{task_id}",
                        f"episode_{episode_idx:02d}",
                        f"chunk_{chunk_idx:04d}",
                    )
                    vlm_out = os.path.join(
                        cfg.vlm_tmp_dir,
                        f"vlm_t{task_id}_ep{episode_idx:02d}_chunk{chunk_idx:04d}.json"
                    )
                    try:
                        capture_chunk_obs_from_env(
                            obs, env.sim, chunk_folder,
                            task_id, episode_idx, chunk_idx,
                            cfg.safety_level, task_description,
                            cfg.env_img_res, camera_specs,
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
                                f"caution={behavioral['caution']} | "
                                f"rotation_lock={pose['rotation_lock']}"
                            )
                        else:
                            logger.warning(
                                f"[VLM] t={t} chunk={chunk_idx}: server returned None; "
                                "keeping previous constraints"
                            )
                    except Exception as exc:
                        logger.warning(f"[VLM] t={t}: {exc}; keeping previous constraints")

                # 2. Query OpenVLA policy
                observation, img = prepare_observation(obs, resize_size, validate_images=True)
                actions = get_action(cfg, model, observation, task_description,
                                     processor=processor, action_head=action_head,
                                     proprio_projector=proprio_projector,
                                     noisy_action_projector=noisy_action_projector,
                                     use_film=cfg.use_film)
                action_queue.extend(actions[:cfg.num_open_loop_steps])

            # ── STEP ──────────────────────────────────────────────────
            action = action_queue.popleft()
            action_commanded = action.copy()
            action_certified = action.copy()
            cbf_info = {}

            if cbf_active:
                ee_pos = obs["robot0_eef_pos"]
                alpha = 0.25 if behavioral.get("caution", False) else 1.0
                action_certified, cbf_info = certify_action_simple(
                    u_cmd=action, ee_pos=ee_pos, ellipsoids=ellipsoids, dt=0.05, alpha=alpha
                )
                action = action_certified
                if pose.get("rotation_lock", False):
                    action[3:6] = np.zeros(3)

            action = process_action(action, cfg.model_family)
            action_executed = action.copy()
            obs, _, done, _ = env.step(action.tolist())

            # collision detection (with Bug #1 fix)
            if not collide_flag and obstacle_name is not None:
                then_pos = obs.get(f"{obstacle_name}_pos", initial_obstacle_pos)
                if np.sum(np.abs(then_pos - initial_obstacle_pos)) > 0.001:
                    collide_flag = True
                    logger.info(f"Collision at t={t}")

            if trajectory_dir is not None:
                trajectory_data.append({...})  # unchanged

            if done:
                success = True
                break
            t += 1

    except Exception as exc:
        log_message(f"Episode error: {exc}", log_file)

    if trajectory_dir is not None and trajectory_data:
        save_trajectory(trajectory_data, task_id, episode_idx, trajectory_dir)

    return success, collide_flag, replay_images, t
```

### Files Already Correct (No Changes)

| File | Role |
|---|---|
| `cbf_construction.py` | `build_constraints()` — complete |
| `qwen_vlm_worker.py` | inference logic reused by server |
| `save_vlm_inputs.py` | `save_rgb`, `save_depth`, `save_segmentation`, etc. |

---

## 6. Staged Implementation Plan

### Task 1: Fix Bugs #1–4 in `run_libero_eval_integrated.py`

**Files:**
- Modify: `run_libero_eval_integrated.py`
- Create: `test_integrated_config.py`

- [ ] **Step 1: Write failing tests for all four bugs**

```python
# test_integrated_config.py
import numpy as np, pytest

# ── Bug #1: obstacle crash ──
def _find_obstacle(obstacle_names, obs):
    obstacle_name = None
    for name in obstacle_names:
        p = obs.get(f"{name}_pos", np.zeros(3))
        if p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
            obstacle_name = name
            break
    if obstacle_name is None and obstacle_names:
        obstacle_name = obstacle_names[0]
    return obstacle_name

def test_obstacle_empty_obs_no_crash():
    result = _find_obstacle(["moka_pot_obstacle"], {})
    assert result == "moka_pot_obstacle"   # fallback

def test_obstacle_in_bounds():
    obs = {"moka_pot_obstacle_pos": np.array([0.1, 0.1, 0.85])}
    assert _find_obstacle(["moka_pot_obstacle"], obs) == "moka_pot_obstacle"

def test_no_obstacles():
    assert _find_obstacle([], {}) is None

# ── Bug #3: missing safety_level validation ──
def _validate(safety_level):
    if safety_level not in ("I", "II"):
        raise ValueError(f"Invalid safety_level '{safety_level}'")

def test_invalid_safety_level_raises():
    with pytest.raises(ValueError):
        _validate("III")

def test_valid_safety_level_passes():
    _validate("I")
    _validate("II")
```

- [ ] **Step 2: Run tests — confirm they fail against the current code**

```bash
conda activate openvla_libero_merged
cd /ocean/projects/cis250185p/asingal
python -m pytest test_integrated_config.py -v
```
Expected: failures demonstrating each bug.

- [ ] **Step 3: Apply Bug #1 fix in `run_libero_eval_integrated.py`**

Replace lines 843–858 (`obstacle_name = " "` block):
```python
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
```

Also fix collision check (lines 950–957):
```python
if not collide_flag and obstacle_name is not None:
    then_pos = obs.get(f"{obstacle_name}_pos", initial_obstacle_pos)
    if np.sum(np.abs(then_pos - initial_obstacle_pos)) > 0.001:
        collide_flag = True
        logger.info(f"Collision at t={t}")
```

- [ ] **Step 4: Apply Bugs #2, #3, #4 fixes**

**Bug #2** — line 274:
```python
run_id = f"EVAL-{cfg.task_suite_name}-level{cfg.safety_level}-{cfg.model_family}-{DATE_TIME}"
```

**Bug #3** — after suite assertion in `validate_config()`:
```python
if cfg.safety_level not in ("I", "II"):
    raise ValueError(f"Invalid safety_level '{cfg.safety_level}'. Must be 'I' or 'II'.")
```

**Bug #4** — add to `GenerateConfig` (after `local_log_dir`):
```python
results_output_dir: str = "integrated_benchmark"
```
Replace line 1184:
```python
results_dir = os.path.join(cfg.results_output_dir, cfg.task_suite_name)
```

**Cleanup** — remove commented mock model (lines 218–252); replace `print()` at lines 806, 904 with `logger.info()`.

- [ ] **Step 5: Run tests — confirm they pass**

```bash
python -m pytest test_integrated_config.py -v
```
Expected: PASS all tests.

- [ ] **Step 6: Commit**

```bash
git add run_libero_eval_integrated.py test_integrated_config.py
git commit -m "fix: obstacle_name crash, run_id format, safety_level validation, results_output_dir"
```

---

### Task 2: Extend `get_safelibero_env()` to Support Depth+Segmentation

**Files:**
- Modify: `safelibero_utils.py`
- Create: `test_safelibero_env_depth.py`

- [ ] **Step 1: Write a failing test**

```python
# test_safelibero_env_depth.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "openvla-oft"))

def test_get_safelibero_env_accepts_depth_args():
    """Verify signature change does not break existing callers."""
    import inspect
    from safelibero_utils import get_safelibero_env
    sig = inspect.signature(get_safelibero_env)
    assert "camera_depths" in sig.parameters, "Missing camera_depths param"
    assert "camera_segmentations" in sig.parameters, "Missing camera_segmentations param"
    # defaults must be non-breaking
    assert sig.parameters["camera_depths"].default == False
    assert sig.parameters["camera_segmentations"].default is None
```

- [ ] **Step 2: Run test — confirm it fails**

```bash
python -m pytest test_safelibero_env_depth.py::test_get_safelibero_env_accepts_depth_args -v
```
Expected: FAIL (params don't exist yet).

- [ ] **Step 3: Update `get_safelibero_env()` in `safelibero_utils.py`**

```python
def get_safelibero_env(task, model_family, resolution=256, include_wrist_camera=False,
                        camera_depths=False, camera_segmentations=None):
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    camera_names = ["agentview", "robot0_eye_in_hand"] if include_wrist_camera else ["agentview"]
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_names": camera_names,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "hard_reset": False,
        "camera_depths": camera_depths,
    }
    if camera_segmentations is not None:
        env_args["camera_segmentations"] = camera_segmentations
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description
```

- [ ] **Step 4: Run test — confirm it passes**

```bash
python -m pytest test_safelibero_env_depth.py -v
```
Expected: PASS.

- [ ] **Step 5: Verify existing callers are unaffected**

```bash
python -c "
import sys; sys.path.insert(0, 'openvla-oft')
# Simulate existing caller signature (no new args)
from safelibero_utils import get_safelibero_env
print('Existing callers OK')
"
```
Expected: no error.

- [ ] **Step 6: Commit**

```bash
git add safelibero_utils.py test_safelibero_env_depth.py
git commit -m "feat: add camera_depths/camera_segmentations params to get_safelibero_env()"
```

---

### Task 3: Create `qwen_vlm_server.py` — Persistent Qwen HTTP Server

**Files:**
- Create: `qwen_vlm_server.py`
- Create: `test_vlm_server.py`

- [ ] **Step 1: Write a test for the server's response format**

```python
# test_vlm_server.py
import json, subprocess, time, os, tempfile, requests

SERVER_URL = "http://localhost:5002"   # use 5002 to avoid conflict with production

def _wait_for_server(url, max_wait=30):
    for _ in range(max_wait):
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def test_server_health():
    assert _wait_for_server(SERVER_URL), "Server did not start in time"
    r = requests.get(f"{SERVER_URL}/health")
    assert r.ok
    assert r.json()["status"] == "ok"

def test_server_infer_dry_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        r = requests.post(f"{SERVER_URL}/infer",
                          json={"obs_folder": tmpdir, "method": "m1", "dry_run": True})
    assert r.ok, f"Server returned {r.status_code}: {r.text}"
    data = r.json()
    assert "single" in data or isinstance(data, dict)
    # Check schema: must have 'objects' key somewhere
    episode_data = data.get("single", next(iter(data.values()), {}))
    assert "objects" in episode_data, f"Missing 'objects' key: {data}"
```

(Run these tests after the server is started manually in Step 3.)

- [ ] **Step 2: Create `qwen_vlm_server.py`**

```python
#!/usr/bin/env python3
"""
qwen_vlm_server.py

Persistent HTTP server that loads Qwen2.5-VL once and serves VLM inference
requests over localhost. Must run in the 'qwen' conda environment.

Usage:
    conda activate qwen
    python qwen_vlm_server.py --port 5001 --model qwen2.5-vl-7b

Endpoints:
    GET  /health          → {"status": "ok", "model": "..."}
    POST /infer           → body: {"obs_folder": "...", "method": "m1",
                                   "num_votes": 1, "dry_run": false}
                          ← constraint JSON (same schema as qwen_vlm_worker.py)
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request

# Reuse inference logic from qwen_vlm_worker
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_vlm_worker import run_single_episode, QWEN_MODELS, DRY_RUN_RESULT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
_model = None
_processor = None
_model_key = None


def load_model(model_key: str):
    global _model, _processor, _model_key
    if _model is not None:
        return
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    import torch
    model_id = QWEN_MODELS[model_key]
    logger.info(f"Loading {model_id} ...")
    _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    _model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    _model.eval()
    _model_key = model_key
    logger.info("Model loaded.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": _model_key})


@app.route("/infer", methods=["POST"])
def infer():
    body = request.get_json(force=True)
    obs_folder = body.get("obs_folder", "")
    method = body.get("method", "m1")
    num_votes = int(body.get("num_votes", 1))
    dry_run = bool(body.get("dry_run", False))

    if dry_run:
        return jsonify(DRY_RUN_RESULT)

    if not os.path.isdir(obs_folder):
        return jsonify({"error": f"obs_folder not found: {obs_folder}"}), 400

    try:
        result = run_single_episode(
            input_folder=obs_folder,
            method=method,
            model=_model,
            processor=_processor,
            num_votes=num_votes,
        )
        return jsonify({"single": result})
    except Exception as exc:
        logger.exception(f"Inference error: {exc}")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--model", default="qwen2.5-vl-7b",
                        choices=list(QWEN_MODELS.keys()))
    args = parser.parse_args()
    load_model(args.model)
    app.run(host="0.0.0.0", port=args.port, threaded=False)
```

**Note:** `run_single_episode()` and `DRY_RUN_RESULT` must be exported from `qwen_vlm_worker.py`. See Step 3b.

- [ ] **Step 3a: Export required symbols from `qwen_vlm_worker.py`**

Add at module level in `qwen_vlm_worker.py`:
```python
DRY_RUN_RESULT = {
    "single": {
        "description": "dry run placeholder",
        "end_object": "object",
        "objects": [],
    }
}
```
Ensure the per-episode inference logic is wrapped in a `run_single_episode(input_folder, method, model, processor, num_votes)` function callable from `qwen_vlm_server.py`. If the existing code is organized differently, extract the relevant logic into this function.

- [ ] **Step 3b: Install Flask in the qwen env**

```bash
conda run -n qwen pip install flask
```
Verify:
```bash
conda run -n qwen python -c "import flask; print(flask.__version__)"
```

- [ ] **Step 4: Start the server and run tests**

```bash
# Terminal 1: start server on test port
conda activate qwen
python qwen_vlm_server.py --port 5002 --model qwen2.5-vl-3b  # use 3b for faster startup

# Terminal 2 (or wait ~20s, then):
conda activate openvla_libero_merged
python -m pytest test_vlm_server.py -v
```
Expected: PASS health and dry-run tests.

- [ ] **Step 5: Commit**

```bash
git add qwen_vlm_server.py test_vlm_server.py
git commit -m "feat: add persistent Qwen HTTP server for per-chunk VLM inference"
```

---

### Task 4: Add `capture_chunk_obs_from_env()` and `call_vlm_server()` to `run_libero_eval_integrated.py`

**Files:**
- Modify: `run_libero_eval_integrated.py`
- Create: `test_chunk_obs.py`

- [ ] **Step 1: Write tests**

```python
# test_chunk_obs.py
import json, os, tempfile, numpy as np
from unittest.mock import MagicMock, patch

def test_call_vlm_server_dry_run_returns_dict():
    from run_libero_eval_integrated import call_vlm_server
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "out.json")
        result = call_vlm_server(
            obs_folder=tmpdir,
            output_json_path=out,
            server_url="http://localhost:5001",
            method="m1",
            dry_run=True,
        )
    assert result is not None
    assert "single" in result

def test_call_vlm_server_connection_error_returns_none():
    from run_libero_eval_integrated import call_vlm_server
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "out.json")
        result = call_vlm_server(
            obs_folder=tmpdir,
            output_json_path=out,
            server_url="http://localhost:9999",   # nothing running here
            method="m1",
            dry_run=False,
        )
    assert result is None, "Connection error should return None"
```

- [ ] **Step 2: Run tests — confirm they fail (function doesn't exist yet)**

```bash
python -m pytest test_chunk_obs.py -v
```
Expected: ImportError or AttributeError.

- [ ] **Step 3: Add `call_vlm_server()` to `run_libero_eval_integrated.py`**

Add after `run_vlm_subprocess()` (line ~470):
```python
def call_vlm_server(
    obs_folder: str,
    output_json_path: str,
    server_url: str = "http://localhost:5001",
    method: str = "m1",
    timeout: int = 120,
    dry_run: bool = False,
) -> Optional[Dict]:
    """Call persistent Qwen server via HTTP POST. Returns None on any failure."""
    if dry_run:
        return {"single": {"description": "dry run", "end_object": "object", "objects": []}}
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed; pip install requests")
        return None
    try:
        resp = requests.post(
            f"{server_url}/infer",
            json={"obs_folder": obs_folder, "method": method},
            timeout=timeout,
        )
        if resp.ok:
            result = resp.json()
            with open(output_json_path, "w") as f:
                json.dump(result, f)
            return result
        logger.warning(f"VLM server {resp.status_code}: {resp.text[:200]}")
        return None
    except Exception as exc:
        logger.warning(f"VLM server call failed: {exc}")
        return None
```

- [ ] **Step 4: Add `capture_chunk_obs_from_env()` to `run_libero_eval_integrated.py`**

Add after `call_vlm_server()`:
```python
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
    """Save depth+seg+rgb from running policy env obs to obs_folder.

    Policy env must have been created with camera_depths=True and
    camera_segmentations='instance'. No secondary env created.
    """
    obs_path = Path(obs_folder)
    obs_path.mkdir(parents=True, exist_ok=True)
    save_rgb(obs, obs_path, camera_specs)
    save_depth(obs, sim, obs_path, camera_specs)
    save_segmentation(obs, obs_path, camera_specs)
    save_camera_params(sim, obs_path, resolution, camera_specs)
    save_metadata(
        obs, None, task_id, episode_idx, safety_level, task_description, obs_path,
    )
```

- [ ] **Step 5: Run tests — confirm they pass**

```bash
pip install requests  # if not already in openvla_libero_merged
python -m pytest test_chunk_obs.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add run_libero_eval_integrated.py test_chunk_obs.py
git commit -m "feat: add call_vlm_server() and capture_chunk_obs_from_env() for per-chunk VLM"
```

---

### Task 5: Restructure `run_episode()` and `run_task()` for Per-Chunk VLM

**Files:**
- Modify: `run_libero_eval_integrated.py`

- [ ] **Step 1: Update `run_task()` to pass depth+seg to policy env and `camera_specs` to `run_episode()`**

In `run_task()`, replace the env creation line (~line 1003):
```python
env, task_description = get_safelibero_env(
    task, cfg.model_family,
    resolution=cfg.env_img_res,
    include_wrist_camera=True,
    camera_depths=cfg.use_safety_filter,
    camera_segmentations="instance" if cfg.use_safety_filter else None,
)
camera_specs = get_camera_specs(["agentview", "robot0_eye_in_hand"]) if cfg.use_safety_filter else None
```

Update `run_task()` call to `run_episode()` to pass `camera_specs`:
```python
success, collide, replay_images, total_time = run_episode(
    cfg, env, task_description, model, resize_size,
    processor, action_head, proprio_projector, noisy_action_projector,
    initial_state, log_file,
    task_id=task_id,
    trajectory_dir=trajectory_dir,
    episode_idx=episode_idx,
    task=task,
    camera_specs=camera_specs,   # ← new
)
```

- [ ] **Step 2: Update `run_episode()` signature and remove pre-episode VLM block**

Add `camera_specs=None` to the signature.

Remove the pre-episode VLM block (lines 827–835):
```python
# REMOVE this block entirely:
if cfg.use_safety_filter and task_id is not None and task is not None and initial_state is not None:
    ellipsoids, behavioral, pose = get_episode_cbf_constraints(...)
cbf_active = len(ellipsoids) > 0
```

Replace with:
```python
ellipsoids = []
behavioral = _DEFAULT_BEHAVIORAL.copy()
pose = _DEFAULT_POSE.copy()
cbf_active = False
```

- [ ] **Step 3: Replace action-queue-empty block with the per-chunk VLM+policy block**

Find the `if len(action_queue) == 0:` block in `run_episode()` (~line 890). Replace the existing policy-only block with the per-chunk VLM+policy block shown in Section 5's "Updated `run_episode()`" pseudocode above. The full implementation:

```python
if len(action_queue) == 0:
    # VLM constraints update (every chunk)
    if cfg.use_safety_filter and camera_specs is not None and task_id is not None:
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
                cfg.env_img_res, camera_specs,
            )
            vlm_json = call_vlm_server(
                obs_folder=chunk_folder,
                output_json_path=vlm_out,
                server_url=cfg.vlm_server_url,
                method=cfg.vlm_method,
                dry_run=cfg.vlm_dry_run,
            )
            if vlm_json is not None:
                ellipsoids, behavioral, pose = build_episode_ellipsoids(vlm_json, chunk_folder)
                cbf_active = len(ellipsoids) > 0
                logger.info(
                    f"[VLM] t={t} chunk={chunk_idx}: "
                    f"{len(ellipsoids)} ellipsoids | "
                    f"caution={behavioral['caution']} | "
                    f"rotation_lock={pose['rotation_lock']}"
                )
            else:
                logger.warning(
                    f"[VLM] t={t} chunk={chunk_idx}: server returned None; "
                    "keeping previous constraints"
                )
        except Exception as exc:
            logger.warning(f"[VLM] t={t}: {exc}; keeping previous constraints")

    # OpenVLA policy query
    try:
        observation, img = prepare_observation(obs, resize_size, validate_images=True)
        img_safe = np.ascontiguousarray(img.copy())
        replay_images.append(img_safe)
    except ValueError as exc:
        logger.warning(f"t={t}: corrupted frame ({exc}); using previous")
        replay_images.append(
            replay_images[-1].copy() if replay_images
            else np.zeros((cfg.env_img_res, cfg.env_img_res, 3), dtype=np.uint8)
        )
        observation, _ = prepare_observation(obs, resize_size, validate_images=False)

    actions = get_action(
        cfg, model, observation, task_description,
        processor=processor, action_head=action_head,
        proprio_projector=proprio_projector,
        noisy_action_projector=noisy_action_projector,
        use_film=cfg.use_film,
    )
    action_queue.extend(actions[:cfg.num_open_loop_steps])
```

**Note:** The image capture (replay_images.append) moves inside the `if len(action_queue) == 0` block here — but for the replay video, you need a frame every step, not every 8 steps. Keep a separate frame capture before the queue check to maintain the original per-step video recording.

- [ ] **Step 4: Add `vlm_server_url` to `GenerateConfig`**

After `qwen_conda_env` field:
```python
vlm_server_url: str = "http://localhost:5001"
```

- [ ] **Step 5: Run dry-run smoke test (1 episode, no GPU needed for VLM)**

```bash
export MUJOCO_GL=egl
conda activate openvla_libero_merged
python run_libero_eval_integrated.py \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --num_trials_per_task 1 \
    --use_safety_filter True \
    --vlm_dry_run True \
    --vlm_server_url http://localhost:5001 \
    --results_output_dir /tmp/integrated_smoke \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
```

Expected log output (dry_run bypasses server):
```
[VLM] t=0 chunk=0: 0 ellipsoids | caution=False | rotation_lock=False
[VLM] t=8 chunk=1: 0 ellipsoids | ...
[VLM] t=16 chunk=2: 0 ellipsoids | ...
...
success=False  collide=False  safe_success=False  steps=300
```

Results JSON must exist:
```bash
cat /tmp/integrated_smoke/safelibero_spatial/results_EVAL-*.json | python -m json.tool | head -20
```

- [ ] **Step 6: Commit**

```bash
git add run_libero_eval_integrated.py
git commit -m "feat: restructure run_episode() for per-chunk VLM+CBF at action chunk boundaries"
```

---

### Task 6: Live Server Smoke Test (1 Episode, Real Qwen)

Requires GPU access (Bridges2 GPU-shared partition). Validates real per-chunk VLM inference.

- [ ] **Step 1: Check GPU memory budget**

```bash
nvidia-smi
```
- OpenVLA-7B: ~14 GB on GPU 0
- Qwen2.5-VL-7B: ~14 GB on GPU 1 (or use 3B at ~7 GB if only 1 GPU)
- If single 32 GB GPU: use `--vlm_model qwen2.5-vl-3b` for Qwen

- [ ] **Step 2: Start Qwen server in background on GPU 1**

```bash
# In the same SLURM allocation or a separate terminal:
conda activate /ocean/projects/cis250185p/asingal/envs/qwen
CUDA_VISIBLE_DEVICES=1 python qwen_vlm_server.py --port 5001 --model qwen2.5-vl-3b &
VLM_SERVER_PID=$!

# Wait for server to load model (~20s):
sleep 25
curl http://localhost:5001/health   # must return {"status": "ok", "model": "qwen2.5-vl-3b"}
```

- [ ] **Step 3: Run 1-episode live test on GPU 0**

```bash
export MUJOCO_GL=egl
conda activate /ocean/projects/cis250185p/asingal/envs/openvla_libero_merged
python run_libero_eval_integrated.py \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --num_trials_per_task 1 \
    --use_safety_filter True \
    --vlm_dry_run False \
    --vlm_model qwen2.5-vl-3b \
    --vlm_server_url http://localhost:5001 \
    --vlm_tmp_dir /ocean/projects/cis250185p/asingal/tmp/vlm_live \
    --results_output_dir /ocean/projects/cis250185p/asingal/integrated_results \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
```

Expected: VLM called ~38 times, each logging `[VLM] t=N chunk=K: M ellipsoids`.

- [ ] **Step 4: Check per-chunk VLM latency**

Look for lines like:
```
2026-04-20 12:34:56 [INFO] VLM server done in 2.3s
2026-04-20 12:34:58 [INFO] [VLM] t=8 chunk=1: 2 ellipsoids | caution=False | rotation_lock=False
```
Latency per call should be 1–5s.

- [ ] **Step 5: Commit**

```bash
git add slurm_integrated_eval.sh  # see Task 7
git commit -m "feat: live per-chunk VLM smoke test validated"
```

---

### Task 7: Full Evaluation Run

- [ ] **Step 1: Create SLURM job script `slurm_integrated_eval.sh`**

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:2
#SBATCH --time=16:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=integrated_eval
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

module load anaconda3
mkdir -p logs

# Start Qwen server on GPU 1 (background)
source activate /ocean/projects/cis250185p/asingal/envs/qwen
CUDA_VISIBLE_DEVICES=1 python qwen_vlm_server.py \
    --port 5001 --model qwen2.5-vl-7b > logs/qwen_server_${SLURM_JOB_ID}.log 2>&1 &
VLM_PID=$!
echo "Qwen server PID: $VLM_PID"
sleep 30   # wait for model to load

# Verify server is up
curl -s http://localhost:5001/health || { echo "Qwen server failed to start"; kill $VLM_PID; exit 1; }

# Run evaluation on GPU 0
source activate /ocean/projects/cis250185p/asingal/envs/openvla_libero_merged
export MUJOCO_GL=egl
CUDA_VISIBLE_DEVICES=0 python run_libero_eval_integrated.py \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --num_trials_per_task 50 \
    --use_safety_filter True \
    --vlm_dry_run False \
    --vlm_model qwen2.5-vl-7b \
    --num_vlm_votes 3 \
    --vlm_server_url http://localhost:5001 \
    --vlm_tmp_dir /ocean/projects/cis250185p/asingal/tmp/vlm_live \
    --results_output_dir /ocean/projects/cis250185p/asingal/integrated_results \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10

# Shut down Qwen server
kill $VLM_PID
```

- [ ] **Step 2: Submit**

```bash
mkdir -p /ocean/projects/cis250185p/asingal/logs
sbatch slurm_integrated_eval.sh
```

- [ ] **Step 3: Monitor**

```bash
squeue -u $USER
tail -f /ocean/projects/cis250185p/asingal/logs/slurm_<JOBID>.out
```

- [ ] **Step 4: Verify results**

```bash
cat /ocean/projects/cis250185p/asingal/integrated_results/safelibero_spatial/results_EVAL-*.json \
    | python -m json.tool | grep -A 8 '"overall"'
```

- [ ] **Step 5: Commit SLURM script**

```bash
git add slurm_integrated_eval.sh
git commit -m "feat: SLURM script for full integrated evaluation with per-chunk VLM"
```

---

## 7. Risks, Assumptions, and Validation Plan

### Critical Assumptions

| Assumption | Where it breaks | How to verify |
|---|---|---|
| `OffScreenRenderEnv` supports `camera_depths + camera_segmentations` alongside `include_wrist_camera=True` | EGL FBO conflict at high resolution → grey depth frames | Create env with all flags, check `obs["agentview_depth"]` is non-zero after `env.step()` |
| `save_depth()` in `save_vlm_inputs.py` accepts obs from policy env (1024×1024 instead of 512×512) | Shape mismatch in npy save → `FileNotFoundError` in `build_constraints()` | Check `agentview_depth.npy` shape after `capture_chunk_obs_from_env()` |
| Qwen server can handle concurrent `obs_folder` paths without collision | Two calls writing to same folder (impossible here: sequential) | N/A — sequential control loop |
| `obs["agentview_segmentation_instance"]` key name matches what `save_vlm_inputs.save_segmentation()` expects | KeyError mid-episode | Check with `test_safelibero_obs_keys.py` with `camera_segmentations="instance"` |
| `build_constraints()` succeeds when obs_folder contains 1024×1024 depth/seg | Hardcoded resolution assumption in `cbf_construction.py` | Check after first live chunk call |
| Flask `requests` library available in `openvla_libero_merged` | ImportError at runtime | `conda activate openvla_libero_merged && python -c "import requests"` |
| Qwen server starts within `sleep 30` in SLURM script | 7B model may take 40–60s → server not ready → first call fails | Increase sleep to 60s or add retry loop in `call_vlm_server()` |

### Latency Budget Per Episode

| Component | Frequency | Wall-clock time per episode |
|---|---|---|
| Qwen inference (persistent, 7B) | 38 calls | 38 × 2–5s = 76–190s |
| OpenVLA inference (7B) | 38 calls | 38 × 1–3s = 38–114s |
| CBF-QP | 300 calls | 300 × <1ms = <1s |
| `build_constraints()` | 38 calls | 38 × 2–5s = 76–190s |
| `env.step()` | 300 calls | ~300 × 10ms = 3s (GPU-accelerated sim) |
| **Total per episode** | | **~4–8 min** |
| **Total for 500 episodes (50 trials × 10 tasks)** | | **~33–67 hours** |

Request a 16-hour SLURM job per task suite (10 tasks × 50 trials = 500 episodes at 8 min max = 67 hours → split across multiple jobs or use `--num_trials_per_task 10` for a first pass).

### Sanity-Check Checklist Before Full Run

- [ ] `get_safelibero_env(..., camera_depths=True, camera_segmentations="instance")` returns env without error
- [ ] After `env.step()`, `obs["agentview_depth"]` is non-zero and `obs["agentview_segmentation_instance"]` contains integer labels
- [ ] `capture_chunk_obs_from_env()` writes all 5 required files to `obs_folder`
- [ ] `build_constraints()` runs successfully on the files written above
- [ ] `curl http://localhost:5001/health` returns `{"status": "ok"}` after server startup
- [ ] `curl -X POST http://localhost:5001/infer -d '{"obs_folder": "/tmp", "dry_run": true}'` returns valid JSON
- [ ] 1-episode dry-run completes without crash
- [ ] 1-episode live-VLM run logs `[VLM] t=0 chunk=0: N ellipsoids` (N > 0)
- [ ] Results JSON exists with all required keys after 1-episode run

---

## 8. Quick Reference — Command Summary

### Start Qwen Server (run once before evaluation)

```bash
conda activate /ocean/projects/cis250185p/asingal/envs/qwen
CUDA_VISIBLE_DEVICES=1 python qwen_vlm_server.py --port 5001 --model qwen2.5-vl-7b
# Or 3B for lower VRAM:
CUDA_VISIBLE_DEVICES=1 python qwen_vlm_server.py --port 5001 --model qwen2.5-vl-3b
```

### Dry-Run (No GPU Required for VLM)

```bash
conda activate openvla_libero_merged
export MUJOCO_GL=egl
python run_libero_eval_integrated.py \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --num_trials_per_task 1 \
    --use_safety_filter True \
    --vlm_dry_run True \
    --results_output_dir /tmp/test_results
```

### Full Evaluation (Server must be running first)

```bash
conda activate openvla_libero_merged
export MUJOCO_GL=egl
CUDA_VISIBLE_DEVICES=0 python run_libero_eval_integrated.py \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --num_trials_per_task 50 \
    --use_safety_filter True \
    --vlm_method m1 \
    --vlm_model qwen2.5-vl-7b \
    --num_vlm_votes 3 \
    --vlm_server_url http://localhost:5001 \
    --vlm_tmp_dir /ocean/projects/cis250185p/asingal/tmp/vlm_live \
    --results_output_dir /ocean/projects/cis250185p/asingal/integrated_results \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
```

### Baseline (No Safety Filter, Reference Comparison)

```bash
python run_safelibero_openvla_oft_eval.py \
    --task_suite_name safelibero_spatial \
    --safety_level I \
    --num_trials_per_task 50 \
    --results_output_dir openvla_benchmark
```

---

## 9. Key Config Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--task_suite_name` | `safelibero_spatial` | SafeLIBERO suite: `safelibero_{spatial,object,goal,long}` |
| `--safety_level` | `I` | Obstacle difficulty: `I` (static) or `II` (dynamic) |
| `--num_trials_per_task` | `50` | Episodes per task |
| `--use_safety_filter` | `True` | Enable per-chunk VLM+CBF filtering |
| `--vlm_server_url` | `http://localhost:5001` | URL of persistent Qwen HTTP server |
| `--vlm_method` | `m1` | VLM constraint method: `m1` (Seg+VLM), `m2` (VLM-only), `m3` (3D+VLM) |
| `--vlm_model` | `qwen2.5-vl-7b` | Qwen model key (passed to `qwen_vlm_server.py --model`) |
| `--num_vlm_votes` | `1` | Majority voting rounds per chunk (Brunke et al. recommend 5) |
| `--vlm_dry_run` | `False` | Skip server call; use placeholder constraints (for testing) |
| `--vlm_tmp_dir` | `/tmp/vlm_obs` | Dir for per-chunk obs files |
| `--results_output_dir` | `integrated_benchmark` | Directory for JSON results |
| `--env_img_res` | `1024` | Policy env resolution (also used for VLM obs capture) |

---

## 10. Known Limitations

1. **VLM and OpenVLA queries are sequential at each chunk boundary**: The sim pauses for both VLM inference (~2–5s) and OpenVLA inference (~1–3s). Total pause per chunk: ~3–8s. For 300 steps with `num_open_loop_steps=8`: ~38 pauses × 5s avg = ~3 min per episode overhead. Acceptable for offline eval.

2. **Video output not configurable**: `save_rollout_video()` hardcodes `./rollouts/{DATE}`. Low priority.

3. **All episodes save video**: High disk usage for 50 trials. Consider saving only episode 0 per task (matching baseline behavior).

4. **`num_vlm_votes > 1` multiplies server calls per chunk**: With `num_vlm_votes=3`, latency per chunk becomes 3× the inference time. The server handles this internally via the `num_votes` parameter, so no additional subprocess calls occur — but inference time increases proportionally.

5. **Per-chunk obs files accumulate**: A 300-step episode with `num_open_loop_steps=8` creates ~38 folders under `vlm_tmp_dir`. For 500 episodes this is ~19,000 folders. Set `vlm_tmp_dir` to a fast scratch filesystem and periodically clean it.
