# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project on situational safety for embodied agents, focused on safe robot manipulation and failure recovery. Two main components:

1. **`mujoco_experimentation/safety_benchmark/`** — A dataset collection framework for evaluating robot manipulation safety across 14 tasks and 4 safety categories, built on LIBERO/robosuite/MuJoCo.
2. **`semantic_cbf/`** — Research prototypes for VLM-derived Control Barrier Functions (CBFs).

## Setup

```bash
# Install LIBERO (required)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . && cd ..

# Install the safety_benchmark package
cd mujoco_experimentation
pip install -e .

# Optional: PyTorch (for latent_cbf.py) and Anthropic API (for VLM features)
pip install torch torchvision transformers pillow anthropic
```

Validate setup:
```bash
python -m safety_benchmark.validate
```

## Common Commands

**Collect a dataset:**
```bash
python -m safety_benchmark.collect_dataset \
    --tasks sem_cup_over_laptop \
    --strategies random scripted perturbed \
    --n_random 20 --n_scripted 5 --n_perturbed 30 \
    --output_dir data/safety_benchmark

# All 14 tasks
python -m safety_benchmark.collect_dataset --tasks all --strategies random scripted --output_dir data/safety_benchmark
```

**Analyze a dataset:**
```bash
python -m safety_benchmark.analysis data/safety_benchmark
```

**Visualize trajectories:**
```bash
python -m safety_benchmark.visualize data/safety_benchmark/sem_cup_over_laptop.hdf5 demo_0 --output video.mp4
python -m safety_benchmark.visualize data/safety_benchmark/sem_cup_over_laptop.hdf5 --category unsafe --mode grid
```

**Evaluate a safety classifier or filter:**
```bash
python -m safety_benchmark.evaluate --mode classifier --data_dir data/safety_benchmark --model_fn my_classifier.predict
python -m safety_benchmark.evaluate --mode filter --data_dir data/safety_benchmark --tasks sem_cup_over_laptop
```

**Run CBF integration demo:**
```bash
python semantic_cbf/vla_cbf_integration.py          # mock mode
python semantic_cbf/vla_cbf_integration.py --use-vlm # with Claude API
```

## Architecture

### Safety Benchmark

The core loop: `collect_dataset.py` orchestrates → `collectors.py` generates trajectories → `safety_monitor.py` annotates per-timestep → `annotation.py` writes to HDF5.

**Key design choice — observation-only monitor:** `SafetyMonitor` (`safety_monitor.py`) only reads from the `obs` dict returned by `env.step()`, never accessing the MuJoCo simulation directly. This decouples it from any specific sim version and makes it portable to real robots.

**Safety categories (10-bit `ViolationType` bitmask in `safety_config.py`):**
- Geometric: `COLLISION`, `SELF_COLLISION`, `WORKSPACE_BOUNDARY`
- Task: `TASK_VIOLATION`, `DROP`, `EXCESSIVE_FORCE`
- Semantic: `SPATIAL_SEMANTIC`, `TILT_SEMANTIC`, `VELOCITY_SEMANTIC`
- Partial observability: `PARTIAL_OBS`

**14 pre-configured tasks** defined in `safety_config.py` as dataclasses covering all 4 categories. Each config includes object definitions with semantic metadata (`semantic_class`, `is_fillable`, `fragility`, `weight_kg`), spatial/tilt/velocity constraints, and hidden-state distributions for partial observability.

**Collection strategies** (`collectors.py`): `random`, `scripted` (handcrafted violation-triggering policies), `perturbed` (Ornstein-Uhlenbeck noise on expert demos), `replay` (annotate existing LIBERO HDF5 demos).

**HDF5 schema** (`annotation.py`): Each demo at `data/demo_N/` stores states, actions, obs images/poses, and a `safety/` group with per-timestep bitmask, severity score, and kinematic metadata. Root-level masks group demos by category (safe/unsafe/collision/semantic/etc.).

### Semantic CBF

Four standalone scripts implementing research directions:

| File | Approach |
|------|----------|
| `vlm_cbf_pipeline.py` | VLMSceneAnalyzer → CBFConstructor → QP-based CBFSafetyFilter |
| `latent_cbf.py` | SafetyMarginNetwork learns h(VLM embedding) > 0 ⟺ safe |
| `multiprompt_pipeline.py` | Separate VLM queries per (object, relationship) with majority voting (from Brunke et al. RA-L 2025) |
| `vla_cbf_integration.py` | Post-hoc CBF filter on VLA actions (AEGIS paradigm) |

### Adding a New Task

1. Add a `SafetyTaskConfig` dataclass instance to the task registry in `safety_config.py`
2. Implement a scripted policy in `collectors.py` (optional, for `scripted` strategy)
3. Add custom MuJoCo object XML under `safety_benchmark/assets/` if needed
4. Run `python -m safety_benchmark.validate` to confirm setup

## Key Dependencies

- **LIBERO**: Robot manipulation benchmark framework (must be installed from source)
- **robosuite / mujoco**: Physics simulation backend
- **h5py**: HDF5 dataset storage
- **scipy**: Quaternion/rotation math in safety monitor
- **anthropic**: Claude API for VLM queries in `semantic_cbf/`
