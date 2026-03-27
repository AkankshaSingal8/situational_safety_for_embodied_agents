# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research prototype for **semantic safety filtering in robot manipulation**. VLMs analyze robot workspace images to extract semantic constraints (avoid water, fire, heat, etc.), which are then converted to Control Barrier Functions (CBFs) that filter robot actions in real time — without requiring explicit 3D perception.

## Running the Code

Each module has a `if __name__ == "__main__"` demo that runs standalone (all VLM calls have mock fallbacks when no API key is available):

```bash
# Core VLM→CBF pipeline
python3 semantic_cbf/vlm_cbf_pipeline.py

# Multi-prompt strategy (main approach from paper)
python3 semantic_cbf/multiprompt_pipeline.py

# VLA + CBF integration demo (mock VLA by default)
python3 semantic_cbf/vla_cbf_integration.py
python3 semantic_cbf/vla_cbf_integration.py --use-vlm   # with Claude API

# Latent-space CBF demo
python3 semantic_cbf/latent_cbf.py
```

Pass a Claude API key via the `api_key` parameter in code, or set `ANTHROPIC_API_KEY` in the environment.

## Dependencies

No `requirements.txt` — install as needed:
- `numpy`, `matplotlib` — always required
- `anthropic` — for live VLM calls (all modules have mock fallbacks)
- `torch`, `torchvision`, `transformers` — only for `latent_cbf.py`

## Architecture

**Data flow:**
```
Image → VLM Analysis → SafetyContext → CBFConstructor → CBFSafetyFilter (QP) → filtered robot action
```

**Three integration paradigms** (all in `vla_cbf_integration.py`):
1. **Post-hoc filter (AEGIS-style)** — VLA generates action → CBF-QP corrects it. No retraining.
2. **Training-time CBF** — CBF constraints embedded in RL reward (`CBFAugmentedReward`).
3. **Latent-space CBF** — Safety filter in VLM embedding space (`latent_cbf.py`).

### Key Classes

| Class | File | Role |
|---|---|---|
| `VLMSceneAnalyzer` | `vlm_cbf_pipeline.py` | Single-prompt VLM scene analysis |
| `MultiPromptVLMAnalyzer` | `multiprompt_pipeline.py` | Per-pair multi-prompt with majority voting (main approach) |
| `CBFConstructor` | `vlm_cbf_pipeline.py` | Converts `SemanticConstraint` objects → differentiable CBF functions (superquadrics) |
| `CBFSafetyFilter` | `vlm_cbf_pipeline.py` | QP safety filter via Dykstra's algorithm |
| `SafetyMarginNetwork` | `latent_cbf.py` | Neural network for learned latent CBF |
| `LatentCBFTrainer` | `latent_cbf.py` | Trains safety margin net with Lipschitz/gradient penalties |
| `VLASafetyFilterLayer` | `vla_cbf_integration.py` | Integration layer connecting VLA output to CBF filter |
| `ManipulationSimulator2D` | `vlm_cbf_pipeline.py` | 2D simulation environment used for all demos |

### Core Data Structures (`vlm_cbf_pipeline.py`)

- `ObjectInfo` — scene object with position, dimensions, semantic label
- `SemanticConstraint` — typed constraint (spatial/behavioral/pose) between objects
- `SafetyContext` — complete VLM analysis output; input to `CBFConstructor`

### Multi-Prompt Strategy

The key research contribution (Brunke et al. RA-L 2025): instead of one monolithic VLM prompt, decompose into per-(object, relationship) queries with majority voting. Claims 99% recall vs 78% for single-prompt baseline.

### CBF Construction

Superquadric barrier functions:
```
h(x) = ((|dx|/ax)^(2/ε) + (|dy|/ay)^(2/ε))^ε - 1
```
`h > 0` = safe, `h < 0` = unsafe. Each semantic relationship type (`above`, `around`, `near`) maps to a different barrier shape.

### Safety Filter (QP)

Solved at each timestep:
```
min ||u - u_cmd||²
s.t. ∇h_i · u ≥ -α_i(h_i)  for all active CBFs
     ||u||∞ ≤ u_max
```
Implemented via Dykstra's iterative projection (no external QP solver needed).
