# Project Progress: Situational Safety for Embodied Agents

**Last Updated**: 2026-04-22  
**Project**: VLM-based Semantic Safety Filter for SafeLIBERO Benchmark  
**Goal**: Use Vision-Language Models (Qwen2.5-VL) to extract semantic safety constraints from robot workspace observations and enforce them at runtime via Control Barrier Functions (CBFs), improving safety without sacrificing task success on the SafeLIBERO benchmark.

---

## Research Summary

This project addresses a fundamental limitation of imitation-learned robot policies: they optimize for task success but ignore safety constraints that are obvious to a human observer (e.g., "don't swing the arm over the fragile object"). The approach extracts these constraints automatically at the start of each episode using a VLM, converts them into mathematically rigorous CBF safe sets, and certifies every action at 20 Hz before execution.

Three architectural methods are compared:
- **M1 (Seg+VLM)**: GroundingDINO + SAM2 segmentation → labeled image → Qwen2.5-VL → constraints
- **M2 (VLM-only)**: Raw RGB → Qwen2.5-VL directly → constraints  
- **M3 (3D+VLM)**: Point cloud reconstruction → spatially-enriched image → Qwen2.5-VL → constraints

---

## Directory Structure

```
/ocean/projects/cis250185p/asingal/
├── Core implementation (.py files, top-level)
├── External repositories (read-only subprojects)
│   ├── openvla-oft/              OpenVLA policy fine-tuned on LIBERO
│   ├── SafeLIBERO/               SafeLIBERO benchmark suite
│   └── vlsa-aegis/               VLSA-Aegis perception tools
├── vlm_inputs/                   Captured episode observations (141 MB)
│   └── safelibero_spatial/       Level I, Task 0, ~21 episodes
├── results/                      VLM inference outputs (4 JSON files)
├── cbf_outputs/                  CBF parameters + visualizations
├── cbf_outputs_ellipsoid/        Ellipsoid-specific CBF outputs
├── openvla_benchmark/            Baseline evaluation results
├── openvla_cbf_benchmark/        CBF-integrated evaluation (empty)
├── rollouts/                     Episode rollout videos
├── openvla_video/                Video recordings from baseline eval
├── openvla_cbf_video/            CBF-filtered video outputs (empty)
├── slurm_logs/                   SLURM job output logs
├── envs/                         Conda environment definitions
│   ├── aegis/, libero/, qwen/, openvla_libero_merged/
├── experiments/                  Experiment tracking
├── docs/superpowers/plans/       Work planning documents
└── situational_safety_for_embodied_agents/  Secondary mirror of top-level
```

---

## Implementation Files

### Data Capture

**`save_vlm_inputs.py`**  
Runs SafeLIBERO episodes and captures per-episode observations for offline VLM processing. For each episode it saves:
- `agentview_rgb.png` — 512×512 third-person RGB
- `eye_in_hand_rgb.png` — 512×512 wrist-cam RGB
- `agentview_depth.npy` — float32 metric depth (512×512)
- `eye_in_hand_depth.npy` — wrist depth
- `agentview_seg.npy` — int32 instance segmentation (512×512)
- `eye_in_hand_seg.npy` — wrist segmentation
- `camera_params.json` — intrinsic (3×3) and extrinsic (4×4) matrices
- `metadata.json` — robot state, object positions, obstacle info, geom ID mapping

Target: 400 episodes (4 tasks × 2 safety levels × 50 episodes). Currently 21 episodes captured (Level I, Task 0 only).

**`visualise_depth_map.py`**  
Utility for visualizing captured depth maps; used to validate capture quality.

---

### VLM Inference (Constraint Extraction)

**`qwen_vlm_worker.py`** (~400 lines)  
Main VLM inference script supporting all three methods. Implements the multi-prompt strategy from Brunke et al. (RA-L 2025): each constraint type is queried N=5 times with majority voting. This improves precision from 29% → 60% and recall from 78% → 99% compared to single-prompt.

Queries VLM separately for three constraint types:
1. **Spatial** (`S_r`): Which spatial configurations are unsafe. Relationships: `above`, `below`, `around in front of`, `around behind`
2. **Behavioral** (`S_b`): Objects requiring cautious approach (`caution`)
3. **Pose** (`S_T`): Orientation restrictions (`rotation lock`)

Output JSON format:
```json
{
  "level_I/task_0/episode_00": {
    "description": "Pick up the black bowl between the plate and the ramekin",
    "end_object": "black bowl",
    "objects": [
      ["moka_pot_obstacle", ["above", "around in front of"]],
      ["plate", []],
      ["end_effector", ["caution", "rotation lock"]]
    ]
  }
}
```

**`qwen_vlm_server.py`** (~100 lines)  
Wraps Qwen2.5-VL as a persistent HTTP server (Flask). Used by the integrated evaluator to avoid ~30 second Qwen startup overhead per episode. Exposes a POST endpoint that takes base64-encoded images + prompt and returns constraint JSON.

---

### CBF Construction

**`cbf_construction.py`** (~600 lines)  
Converts VLM predicates + depth/segmentation observations into ellipsoid CBF parameters. Pipeline per constraint:
1. Unproject segmented pixels to 3D point cloud using depth + camera intrinsics/extrinsics
2. Extend point cloud in the relevant direction (e.g., extend upward for "above")
3. Fit a superquadric ellipsoid to the extended cloud (least-squares)
4. Evaluate `h(x_eef)` to verify end-effector starts in safe region

Produces interactive 3D HTML visualizations (Plotly), 2D heatmap slices (`vis_slice_z*.png`), and RGB overlay of h=0 boundary.

**`cbf_superquadric.py`**  
Superquadric fitting with shape parameters ε₁, ε₂ allowing box-like to sphere-like geometries.

**`build_cbf_ellipsoids.py`**  
Batch wrapper that runs CBF construction across multiple episodes.

Ellipsoid implicit function:
```
h(x) = ((x-cx)/ax)² + ((y-cy)/ay)² + ((z-cz)/az)² - 1
Safe region: h(x) ≥ 0
```

CBF output JSON format:
```json
{
  "constraints": [
    {
      "object": "moka_pot_obstacle",
      "relationship": "above",
      "type": "ellipsoid",
      "params": {
        "center": [0.15, 0.08, 0.82],
        "scales": [0.06, 0.06, 0.35],
        "epsilon1": 1.0, "epsilon2": 1.0
      },
      "h_at_eef": 2.31
    }
  ],
  "behavioral": {"caution": true},
  "pose": {"rotation_lock": false}
}
```

---

### Runtime Safety Filtering

**`semantic_cbf_filter.py`** (~600 lines)  
Certifies robot actions at runtime using CBF-QP. At each timestep:
1. Evaluates `h(x)` and `∇h(x)` for all active constraints
2. Solves QP: minimize `‖u_cert - u_cmd‖²` subject to `∇h·u_cert ≥ -α(h)` for all constraints
3. For `caution` behavioral constraints: uses `α(h) = 0.25·h²` (softer barrier) vs. default `α(h) = h²`
4. For `rotation_lock` pose constraints: zeros rotation components of certified action

Contains three classes: `PerceptionModule` (obs → point clouds), `CBFConstructor` (point clouds → ellipsoids), and runtime `SemanticCBFFilter` (action certification via Dykstra's algorithm).

---

### Evaluation Scripts

**`run_safelibero_openvla_oft_eval.py`** (~1000 lines) — **Executed**  
Baseline evaluation: OpenVLA-OFT policy on SafeLIBERO, no safety filter. Loads `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`, runs N trials per task, records TSR (Task Success Rate) and CAR (Collision Avoidance Rate).

**`run_libero_eval_integrated.py`** (~1100 lines) — **In Progress (4 bugs)**  
Main evaluation: OpenVLA + Qwen VLM server + CBF per chunk. Architecture: VLM is called at chunk boundaries (every ~8 action steps), not once per episode. Requires persistent Qwen server running on same node. See Known Issues section.

**`run_libero_eval_with_cbf.py`**  
Alternative integration approach (offline VLM, online CBF).

**`run_vlm_eval.py`**  
Standalone VLM evaluation — measures predicate accuracy against ground-truth annotations.

**`run_vlm_pipeline.py`**  
Runs full offline VLM→CBF pipeline without policy integration.

---

### Utilities

**`safelibero_utils.py`** (~200 lines)  
SafeLIBERO-specific helpers: `get_safelibero_env()`, `get_safelibero_image()`, `get_safelibero_wrist_image()`, image validation, observation preprocessing for OpenVLA-OFT input format.

**`build_3d_map.py`**  
3D map reconstruction from depth sequences; used in M3 method for spatial context generation.

---

### Demos and Minimal Examples

**`minimal_cbf_demo.py`** (~400 lines)  
Educational CBF demo. Loads a saved VLM JSON + metadata file, builds ellipsoids, and produces all visualizations. Entry point for understanding the CBF construction without running the full pipeline.

**`minimal_cbf_demo_interactive.py`** (~500 lines)  
Interactive version with Plotly 3D viewer and PIL image overlays. Opens in browser.

---

### Tests (9 files)

| File | What It Tests |
|------|--------------|
| `test_cbf_filtering.py` | CBF filtering logic, QP solving, action certification |
| `test_frame_validation.py` | Camera-to-world coordinate transformations |
| `test_frame_fix.py` | Frame fix utilities and validation |
| `test_safelibero_obs_keys.py` | SafeLIBERO observation dictionary structure |
| `test_safelibero_env_depth.py` | Depth rendering correctness in SafeLIBERO env |
| `test_integrated_config.py` | Config validation for integrated evaluator |
| `test_run_episode_structure.py` | Episode rollout structure and step counting |
| `test_vlm_server.py` | Qwen HTTP server request/response format |
| `test_chunk_obs.py` | Observation capture at chunk boundaries |

---

## Shell and SLURM Scripts

| Script | Purpose |
|--------|---------|
| `libero_env_setup.sh` | Creates Python 3.8 `libero` conda env (robosuite 1.4.1, mujoco 3.2.3, numpy 1.22.4) |
| `openvla_safelibero_setup.sh` | Creates Python 3.10 `openvla_libero_merged` conda env (torch 2.2.0, transformers, prismatic-vlm) |
| `qwen_vlm_env_setup.sh` | Creates `qwen` conda env for Qwen2.5-VL inference |
| `commands_to_run.sh` | Reference commands for qwen_vlm_worker M2 method |
| `run_smoke_test.sh` | Smoke test runner for quick sanity checks |
| `running_entire_pipeline.sh` | End-to-end pipeline execution script |
| `running_pi05_safelibero.sh` | SafeLIBERO-specific runner script |
| `eval_openvla_spatial_L1.slurm` | Bridges2 SLURM job: baseline OpenVLA evaluation (10h, GPU-shared, V100) — **executed** |
| `slurm_integrated_eval.sh` | SLURM job template for integrated evaluation — **not yet executed** |

---

## Documentation Files

| Document | Size | Contents |
|----------|------|----------|
| `CLAUDE.md` | 8.2 KB | Complete project guide: architecture, 3 methods, environment setup, file formats, debugging |
| `vlm_pipeline_readme.md` | ~500 lines | Full design document: background, prompt strategy, M1/M2/M3 design rationale |
| `integrated_eval_readme.md` | 61 KB | Detailed plan for per-chunk VLM+CBF integration; 4 bugs documented with fixes |
| `CBF_construction.md` | 22.3 KB | CBF theory: workspace bounds, extension strategies, superquadric fitting, math derivations |
| `readme.md` | 13.3 KB | Implementation plan for `save_vlm_inputs.py` |
| `FINAL_OUTPUTS_SUMMARY.md` | 9.7 KB | CBF visualization output formats and interpretation guide |
| `VISUALIZATION_TUNING_GUIDE.md` | ~5 KB | Parameter reference for tuning visualizations |
| `CBF_OUTPUTS_SUMMARY.md` | ~4 KB | CBF output file format specification |
| `MINIMAL_CBF_README.md` | 5.4 KB | Usage guide for `minimal_cbf_demo.py` |
| `evaluation_openvla_oft.md` | 6.4 KB | Baseline evaluation documentation and commands |
| `prompt_writing_plan.md` | 5.7 KB | Feature planning for integrated evaluation |
| `assumptions.md` | 1.2 KB | 5 open assumptions requiring verification |

---

## Results

### Baseline Evaluation (OpenVLA-OFT, No Safety Filter)

**File**: `openvla_benchmark/safelibero_spatial/results_EVAL-safelibero_spatial-levelI-openvla-2026_04_19-15_10_55.json`  
**Environment**: Bridges2 HPC, GPU-shared partition, V100 GPU, 10-hour job  
**Policy**: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`  
**Episodes**: 50 per task × 4 tasks = 200 total

| Task | Task Success Rate (TSR) | Collision Avoidance Rate (CAR) | Successes | Collisions |
|------|------------------------|-------------------------------|-----------|------------|
| Task 0 | 0.18 (18%) | 0.00 (0%) | 9/50 | 50/50 |
| Task 1 | 0.12 (12%) | 0.08 (8%) | 6/50 | 46/50 |
| Task 2 | 0.72 (72%) | 0.20 (20%) | 36/50 | 40/50 |
| Task 3 | 0.56 (56%) | 0.18 (18%) | 28/50 | 41/50 |
| **Overall** | **0.395 (39.5%)** | **0.115 (11.5%)** | **79/200** | **177/200** |

**Key finding**: 88.5% of all episodes involved at least one collision (177/200). CAR is especially poor for Tasks 0 and 1 (0% and 8%). This establishes the baseline that the CBF safety filter must improve.

---

### VLM Inference Results

**Directory**: `results/` (4 JSON files)

| File | Method | Episode |
|------|--------|---------|
| `m1_task0_ep00.json` | M1 (Seg+VLM) | Task 0, Episode 00 |
| `m1_task0_gt.json` | M1 with GT perception | Task 0, Episode 00 (ablation) |
| `m2_task0_ep00.json` | M2 (VLM-only) | Task 0, Episode 00 |
| `m3_task0_ep00.json` | M3 (3D+VLM) | Task 0, Episode 00 |

All four output per-object constraint lists. M1 with GT perception serves as upper bound on VLM reasoning quality (removes perception error).

---

### CBF Construction Results

**Directory**: `cbf_outputs/`

| File / Folder | Contents |
|---------------|---------|
| `task_0_ellipsoids.json` | Standard M1 ellipsoid parameters for Task 0 |
| `gt/task_0_ellipsoids.json` | Ground-truth perception CBF (reference upper bound) |
| `exclude_target_from_cbf/task_0_ellipsoids.json` | Ablation: target object excluded from constraints |
| `task_0_ellipsoids.html` | Interactive 3D Plotly visualization (6.5 MB) |
| `vis_3d.png` | Static 3D point cloud + ellipsoid render |
| `vis_slice_z*.png` | 2D heatmap slices at fixed z heights (red=unsafe, green=safe) |
| `vis_agentview_overlay.png` | h=0 boundary projected onto camera image |

Validation criteria for correct CBF:
- Extended point cloud points: all should have `h ≤ 0.1`
- End-effector at episode start: should have `h > 0` (starts safe)
- Gradient `∇h` should be nonzero at boundary

---

### VLM Input Data

**Directory**: `vlm_inputs/safelibero_spatial/` (141 MB)

Currently contains ~21 episodes from Level I, Task 0. Each episode folder has: `agentview_rgb.png`, `eye_in_hand_rgb.png`, `agentview_depth.npy`, `eye_in_hand_depth.npy`, `agentview_seg.npy`, `eye_in_hand_seg.npy`, `camera_params.json`, `metadata.json`.

Target was 400 episodes (4 tasks × 2 safety levels × 50 episodes). Full capture was not run; the current set was used for development and validation.

---

## Environment Setup

Three conda environments (setup scripts included):

| Environment | Python | Key Packages | Used For |
|-------------|--------|-------------|----------|
| `libero` | 3.8 | robosuite 1.4.1, libero, mujoco 3.2.3, numpy 1.22.4 | LIBERO simulation |
| `openvla_libero_merged` | 3.10 | torch 2.2.0, transformers, prismatic-vlm | OpenVLA policy + SafeLIBERO eval |
| `qwen` | varies | Qwen2.5-VL, flask | VLM inference server |

The Qwen env runs in a separate process and communicates via HTTP because Qwen's dependencies conflict with the OpenVLA stack.

---

## Key Design Decisions

**Image Resolution**: Policy input is 256×256 (OpenVLA training standard); VLM input is 512×512 (higher quality for constraint reasoning). Kept strictly separate.

**Multi-Prompt Strategy**: Following Brunke et al. (RA-L 2025), each constraint type is queried N=5 times with majority voting. Achieves 60% precision / 99% recall vs. 29% / 78% for single-prompt.

**Persistent VLM Server**: Qwen runs as an always-on HTTP server rather than being launched per episode. Reason: each `conda run` invocation takes ~30 seconds of startup overhead vs. 1–5 seconds for HTTP POST once running. At ~38 VLM calls/episode, the startup approach would add ~19 minutes per episode.

**Per-Chunk VLM Calls**: The integrated evaluator calls the VLM at chunk boundaries (every ~8 action steps) rather than once per episode. This enables reactive constraint updating as the scene evolves.

**Ground Truth Fallback**: `--use_gt_perception` flag bypasses segmentation and uses MuJoCo `sim.data` object positions directly. Used to isolate VLM reasoning quality from perception quality.

**Workspace Bounds**:
```python
WORKSPACE_BOUNDS = {
    "x_min": -0.5, "x_max": 0.5,
    "y_min": -0.3, "y_max": 0.6,
    "z_table": 0.81,   # table surface
    "z_max": 1.4,      # ceiling for "above" extension
}
```

---

## Known Issues

Four bugs documented in `integrated_eval_readme.md` for `run_libero_eval_integrated.py`:

1. **Bug #1** (line 844–852): Obstacle name lookup crashes with `KeyError` when obstacle not in obs dict → fix: use `obs.get(key, None)` guard
2. **Bug #2** (line 274): `run_id` string missing safety level → fix: prepend `level{cfg.safety_level}` to ID
3. **Bug #3** (line 174): `validate_config()` missing `safety_level` range check → fix: add bounds check
4. **Bug #4** (line 1184): `results_output_dir` hardcoded rather than configurable → fix: add as config field

Additional cleanup items:
- Commented mock model code at lines 218–252 (left over from development)
- Stray `print()` calls at lines 806 and 904 (should use logger)

Open assumptions documented in `assumptions.md`:
1. Whether `prepare_observation()` handles missing wrist camera in SafeLIBERO
2. Whether `cfg.num_trials_per_task` is used correctly everywhere (spec says 50)
3. Whether the single checkpoint covers all 4 spatial tasks
4. Whether `safety_level` parameter accepts "I" vs. 1 vs. "level_I" format
5. Whether `task_suite.get_task_init_states()` returns ≥50 states per task

---

## Completion Status

### Done

- [x] Three conda environments created and validated (`libero`, `openvla_libero_merged`, `qwen`)
- [x] `save_vlm_inputs.py` — episode observation capture (partially run: 21 episodes)
- [x] `qwen_vlm_worker.py` — all three VLM methods (M1, M2, M3) with multi-prompt voting
- [x] `qwen_vlm_server.py` — persistent HTTP server for Qwen
- [x] `cbf_construction.py` — point cloud → ellipsoid CBF with full visualization suite
- [x] `cbf_superquadric.py` — superquadric fitting
- [x] `build_cbf_ellipsoids.py` — batch CBF construction
- [x] `semantic_cbf_filter.py` — runtime CBF-QP action certification
- [x] `safelibero_utils.py` — SafeLIBERO environment wrappers
- [x] `run_safelibero_openvla_oft_eval.py` — baseline evaluator (written and executed)
- [x] Baseline results collected (200 episodes, 4 tasks, Level I)
- [x] VLM inference run on Task 0, Episode 00 for all three methods
- [x] CBF ellipsoids constructed for Task 0 (standard, GT, and target-exclusion ablation)
- [x] Visualizations generated (3D HTML, 2D heatmaps, RGB overlays)
- [x] Minimal CBF demo written and validated
- [x] 9 test files covering all major components
- [x] Complete documentation suite (design docs, CBF theory, visualization guides)

### In Progress

- [ ] `run_libero_eval_integrated.py` — 4 known bugs to fix before execution
- [ ] Full VLM input capture (21 of 400 target episodes done)

### Not Yet Started

- [ ] Batch VLM inference across all 400 episodes
- [ ] Integrated evaluation execution (OpenVLA + Qwen + CBF end-to-end)
- [ ] Predicate accuracy evaluation against `prompt_tuning_benchmark_set/`
- [ ] Per-step metrics: Violation Rate, Filter Activation Rate, Action Deviation
- [ ] Ablation studies (spatial-only, behavioral-only, pose-only constraints)
- [ ] Level II safety level evaluation (only Level I done so far)
- [ ] M2 and M3 method evaluation (only M1 partially evaluated)
- [ ] Comparison between methods (M1 vs M2 vs M3)

---

## Planned Evaluation Metrics

| Metric | Abbreviation | Definition |
|--------|-------------|------------|
| Task Success Rate | TSR | % of episodes completing task goal |
| Collision Avoidance Rate | CAR | % of episodes avoiding all safety violations |
| Violation Rate | VR | % of timesteps inside unsafe set |
| Filter Activation Rate | FAR | % of timesteps where CBF modifies the action |
| Action Deviation | AD | Mean `‖u_certified - u_commanded‖` when filter active |
| Predicate Accuracy (Precision) | PA-P | % of predicted constraints that are correct |
| Predicate Accuracy (Recall) | PA-R | % of ground-truth constraints that are predicted |
| Episode Time Steps | ETS | Mean/median episode length |

---

## Resource Requirements

| Component | Time | GPU | Memory |
|-----------|------|-----|--------|
| VLM input capture (1 episode) | ~0.5 s | None | <1 GB |
| Qwen VLM inference (1 episode, M1) | 1–5 s | 1× V100 | ~20 GB |
| CBF construction (1 episode) | <1 s | None | <1 GB |
| OpenVLA inference (300 steps) | ~60 s | 1× V100 | ~16 GB |
| Full integrated episode | ~66 s | 1–2× GPU | 36+ GB |
| Baseline full evaluation (200 episodes) | ~10 h | 1× V100 | ~16 GB |

---

## Next Steps (Priority Order)

1. **Fix 4 bugs in `run_libero_eval_integrated.py`** — apply fixes documented in `integrated_eval_readme.md`
2. **Run smoke test** — 1 task, 1 level, 2 trials to validate end-to-end integration
3. **Expand VLM input capture** — restore full loop in `save_vlm_inputs.py`, run on all 400 episodes
4. **Batch VLM inference** — `python qwen_vlm_worker.py --method m1 --input_dir vlm_inputs/safelibero_spatial --output_json results/m1_all.json`
5. **Execute integrated evaluation** — start Qwen server, run Level I and Level II evaluations
6. **Analyze results** — compare TSR/CAR vs. baseline, compute VR/FAR/AD, run predicate accuracy benchmark
7. **Run ablations** — M2 and M3 methods; constraint type ablations
