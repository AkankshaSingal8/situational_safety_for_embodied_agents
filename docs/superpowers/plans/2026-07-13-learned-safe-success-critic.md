# Learned Safe-Success Critic + Tournament Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the tournament's hand-tuned candidate selectors with a learned safe-success critic trained on oracle sim-fork labels, add it as a new `learned_critic` steering mode, and complete the viability steering tournament with a directly comparable result row and a final recommendation.

**Architecture:** A frozen CLIP image embedding (once per replan step) plus cheap per-candidate kinematic/diagnostic features feed a small MLP critic trained offline on labels logged from the existing `oracle_counterfactual` sim-fork scorer. At inference the critic replaces the sim fork for candidate selection; the existing server-side DCBF/geometry guidance is unchanged and remains the final safety verifier regardless of which candidate is picked.

**Tech Stack:** Python 3.10 (`openvla_libero_merged` conda env for the eval client — has `torch==2.2.0+cu118`, `transformers==4.40.1`), NumPy, PyTorch, HuggingFace `transformers` CLIP (`openai/clip-vit-base-patch32`, 512-dim projection), pytest, existing SLURM/websocket eval harness (unchanged server side, `aegis` conda env).

## Global Constraints

- No changes to server-side DCBF/geometry guidance (`openpi_guided/`) — the critic only changes which candidate is *requested*, guidance still runs during that candidate's denoising regardless of selection mechanism.
- No changes to the *existing* steering modes' behavior (`fixed`, `phase_adaptive`, `best_of_k`, `semantic_best_of_k`, `oracle_counterfactual`) — new code is additive (new argparse choice, new modules, one backward-compatible optional-parameter extension to `_oracle_counterfactual_score`).
- Data suite/scope: `safelibero_spatial` only, levels `I` and `II`, matching the existing tournament.
- Use documented defaults unless a gate fails: CLIP `openai/clip-vit-base-patch32`, MLP `256/256` hidden, K matching the existing tournament's `--num_candidates`.
- Iterate fast: offline critic gate before any live run; micro-smoke (`TRIALS=2`) before promoted (`TRIALS=5`) run.

---

## File Structure

New files:
- `semantic_cbf/__init__.py` — empty, makes `semantic_cbf` an explicit package (matches `epistemic_uncertainty/__init__.py` convention).
- `semantic_cbf/clip_embedder.py` — `ImageEmbedder`: one frozen CLIP forward pass per call.
- `semantic_cbf/candidate_features.py` — `extract_candidate_features()`: pure-function feature engineering shared by data logging, training, and inference.
- `semantic_cbf/safe_success_critic.py` — `SafeSuccessCritic` (MLP) + `score_candidates()` helper.
- `semantic_cbf/learned_critic_selector.py` — `select_candidate()`: pure selection logic (candidates + embed + critic → chosen candidate), reused by the eval script and directly unit-testable without heavy `libero`/`mujoco` imports.
- `semantic_cbf/training_data_logger.py` — `TrainingDataLogger`: buffers per-step candidate rows, flushes one `.npz` shard per episode.
- `semantic_cbf/train_safe_success_critic.py` — offline training script + `train_critic()` importable entry point.
- `semantic_cbf/tests/test_clip_embedder.py`
- `semantic_cbf/tests/test_candidate_features.py`
- `semantic_cbf/tests/test_safe_success_critic.py`
- `semantic_cbf/tests/test_learned_critic_selector.py`
- `semantic_cbf/tests/test_training_data_logger.py`
- `semantic_cbf/tests/test_train_safe_success_critic.py`
- `vlm_pipeline/tests/test_oracle_counterfactual_score.py`

Modified files:
- `vlm_pipeline/run_guided_safelibero_pi05_eval.py` — extend `_oracle_counterfactual_score` (optional diagnostics), add `--log_training_data`, `--clip_model_name`, `--critic_checkpoint` args, add `"learned_critic"` to `--steering_mode` choices, wire logging + the new selection branch.
- `slurm/viability_tournament_smoke.slurm` — add `$WT` to `PYTHONPATH` (needed for `semantic_cbf` imports), add optional `LOG_TRAINING_DATA` / `CRITIC_CHECKPOINT` env vars.
- `results_tables/viability_steering_tournament.md` — append the new result row(s) and update the Interpretation section.

---

### Task 1: CLIP image embedder

**Files:**
- Create: `semantic_cbf/__init__.py`
- Create: `semantic_cbf/clip_embedder.py`
- Test: `semantic_cbf/tests/test_clip_embedder.py`

**Interfaces:**
- Produces: `ImageEmbedder(model_name: str = "openai/clip-vit-base-patch32", device: str | None = None)` with `.embed(image: np.ndarray) -> np.ndarray` (float32, shape `(self.embed_dim,)`, `embed_dim == 512` for the default model) and `.embed_dim: int`.

- [ ] **Step 1: Pre-warm the HuggingFace cache (compute nodes may lack internet)**

Run once from a node with internet access (the login node has it — verified: `curl -sI https://huggingface.co` returns `HTTP/2 200`):

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate openvla_libero_merged
python -c "from transformers import CLIPModel, CLIPImageProcessor; \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
print('cached OK')"
```

Expected: prints `cached OK` and populates `~/.cache/huggingface/`. This must succeed before any SLURM job runs `ImageEmbedder`, since GPU-shared compute nodes may not have outbound internet.

- [ ] **Step 2: Create the package marker**

```bash
touch semantic_cbf/__init__.py
```

- [ ] **Step 3: Write the failing test**

`semantic_cbf/tests/test_clip_embedder.py`:

```python
# semantic_cbf/tests/test_clip_embedder.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from semantic_cbf.clip_embedder import ImageEmbedder


def test_embed_shape_and_dtype():
    embedder = ImageEmbedder()
    image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
    embedding = embedder.embed(image)
    assert embedding.shape == (embedder.embed_dim,)
    assert embedding.dtype == np.float32
    assert embedder.embed_dim == 512


def test_embed_is_deterministic():
    embedder = ImageEmbedder()
    image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
    first = embedder.embed(image)
    second = embedder.embed(image)
    np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-6)
```

- [ ] **Step 4: Run test to verify it fails**

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate openvla_libero_merged
python -m pytest semantic_cbf/tests/test_clip_embedder.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_cbf.clip_embedder'`.

- [ ] **Step 5: Implement `ImageEmbedder`**

`semantic_cbf/clip_embedder.py`:

```python
"""Frozen CLIP image encoder used as the semantic input to the safe-success critic."""

import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPModel

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


class ImageEmbedder:
    """Wraps a frozen CLIP vision tower. One forward pass per `.embed()` call."""

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.embed_dim = int(self.model.config.projection_dim)

    def embed(self, image: np.ndarray) -> np.ndarray:
        """image: (H, W, 3) uint8 RGB array. Returns float32[self.embed_dim]."""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features.squeeze(0).cpu().numpy().astype(np.float32)
```

- [ ] **Step 6: Run test to verify it passes**

```bash
python -m pytest semantic_cbf/tests/test_clip_embedder.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add semantic_cbf/__init__.py semantic_cbf/clip_embedder.py semantic_cbf/tests/test_clip_embedder.py
git commit -m "feat: add frozen CLIP image embedder for safe-success critic"
```

---

### Task 2: Candidate feature extraction

**Files:**
- Create: `semantic_cbf/candidate_features.py`
- Test: `semantic_cbf/tests/test_candidate_features.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `EXPERT_LABELS: list[str]` (9 entries), `FEATURE_NAMES: list[str]` (17 entries), `FEATURE_DIM: int` (17), `extract_candidate_features(candidate: dict, eef_pos, obstacle_pos, phase: str | None, phase_target, controller_scale: float) -> np.ndarray` (float32, shape `(FEATURE_DIM,)`).

- [ ] **Step 1: Write the failing test**

`semantic_cbf/tests/test_candidate_features.py`:

```python
# semantic_cbf/tests/test_candidate_features.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from semantic_cbf.candidate_features import (
    EXPERT_LABELS, FEATURE_DIM, FEATURE_NAMES, extract_candidate_features,
)


def _make_candidate(expert, min_clearance=0.01, correction_norm=0.02, max_residual=0.0):
    actions = np.zeros((5, 7), dtype=np.float64)
    actions[:, 0] = 0.1  # move +x
    return {
        "actions": actions,
        "expert": expert,
        "guidance": {
            "min_clearance": min_clearance,
            "correction_norm": correction_norm,
            "max_residual": max_residual,
        },
    }


def test_feature_dim_and_names_match():
    assert FEATURE_DIM == 17
    assert len(FEATURE_NAMES) == FEATURE_DIM
    assert len(EXPERT_LABELS) == 9


def test_known_expert_one_hot():
    candidate = _make_candidate("tangent_left")
    features = extract_candidate_features(
        candidate, eef_pos=[0.0, 0.0, 0.0], obstacle_pos=[0.1, 0.0, 0.0],
        phase="transport", phase_target=[0.2, 0.0, 0.0], controller_scale=1.0,
    )
    assert features.shape == (FEATURE_DIM,)
    assert features.dtype == np.float32
    expert_index = FEATURE_NAMES.index("expert_tangent_left")
    other_index = FEATURE_NAMES.index("expert_other")
    assert features[expert_index] == 1.0
    assert features[other_index] == 0.0


def test_unknown_expert_sets_other_bucket():
    candidate = _make_candidate("some_future_expert")
    features = extract_candidate_features(
        candidate, eef_pos=[0.0, 0.0, 0.0], obstacle_pos=None,
        phase=None, phase_target=None, controller_scale=1.0,
    )
    other_index = FEATURE_NAMES.index("expert_other")
    assert features[other_index] == 1.0
    assert features.sum() >= 1.0  # only the other-bucket and possibly diagnostics are set


def test_progress_reflects_motion_toward_target():
    candidate = _make_candidate("nominal")  # actions move +x
    features = extract_candidate_features(
        candidate, eef_pos=[0.0, 0.0, 0.0], obstacle_pos=[5.0, 5.0, 5.0],
        phase="approach", phase_target=[1.0, 0.0, 0.0], controller_scale=1.0,
    )
    progress_index = FEATURE_NAMES.index("progress")
    assert features[progress_index] > 0.0  # moved toward the target
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest semantic_cbf/tests/test_candidate_features.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_cbf.candidate_features'`.

- [ ] **Step 3: Implement feature extraction**

`semantic_cbf/candidate_features.py`:

```python
"""Cheap per-candidate features available without a simulator fork.

Shared, single source of truth for feature engineering across data logging
(training_data_logger.py -> a candidate scored by the oracle sim-fork),
offline training (train_safe_success_critic.py), and live inference
(learned_critic_selector.py). All three MUST extract features identically or
the critic will be trained on a distribution it never sees at inference.
"""

import numpy as np

REPLAN_STEPS = 5

EXPERT_LABELS = [
    "nominal", "eef_only", "full_geometry", "permissive", "conservative",
    "tangent_left", "tangent_right", "radial_retreat", "radial_lift_retreat",
]

FEATURE_NAMES = (
    ["min_clearance", "correction_norm", "max_residual", "obstacle_distance", "progress"]
    + ["phase_approach", "phase_transport"]
    + [f"expert_{label}" for label in EXPERT_LABELS]
    + ["expert_other"]
)
FEATURE_DIM = len(FEATURE_NAMES)


def extract_candidate_features(candidate, eef_pos, obstacle_pos, phase, phase_target, controller_scale):
    """candidate: dict with "actions" (T,7) and optional "guidance"/"expert" keys,
    as produced by _infer_diverse_candidates / _inject_topology_experts in
    run_guided_safelibero_pi05_eval.py.

    eef_pos: array-like (3,). obstacle_pos: array-like (3,) or None.
    phase: "approach" | "transport" | None. phase_target: array-like (3,) or None.
    controller_scale: float, the --controller_scale used to send this chunk.
    """
    diagnostic = candidate.get("guidance", {}) or {}
    min_clearance = float(diagnostic.get("min_clearance", -1e6))
    correction_norm = float(diagnostic.get("correction_norm", 0.0))
    max_residual = float(diagnostic.get("max_residual", 0.0))

    eef_pos = np.asarray(eef_pos, dtype=np.float64)
    obstacle_distance = (
        float(np.linalg.norm(eef_pos - np.asarray(obstacle_pos, dtype=np.float64)))
        if obstacle_pos is not None else 0.0
    )

    actions = np.asarray(candidate["actions"][:REPLAN_STEPS], dtype=np.float64)
    endpoint = eef_pos + np.sum(actions[:, :3] * controller_scale, axis=0)
    if phase_target is not None:
        phase_target_arr = np.asarray(phase_target, dtype=np.float64)
        progress = float(np.linalg.norm(eef_pos - phase_target_arr)) - float(
            np.linalg.norm(endpoint - phase_target_arr)
        )
    else:
        progress = 0.0

    phase_approach = 1.0 if phase == "approach" else 0.0
    phase_transport = 1.0 if phase == "transport" else 0.0

    expert = candidate.get("expert", "unlabeled")
    expert_onehot = [1.0 if expert == label else 0.0 for label in EXPERT_LABELS]
    expert_other = 1.0 if expert not in EXPERT_LABELS else 0.0

    features = (
        [min_clearance, correction_norm, max_residual, obstacle_distance, progress]
        + [phase_approach, phase_transport]
        + expert_onehot
        + [expert_other]
    )
    return np.asarray(features, dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest semantic_cbf/tests/test_candidate_features.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add semantic_cbf/candidate_features.py semantic_cbf/tests/test_candidate_features.py
git commit -m "feat: shared candidate feature extraction for safe-success critic"
```

---

### Task 3: Safe-success critic model

**Files:**
- Create: `semantic_cbf/safe_success_critic.py`
- Test: `semantic_cbf/tests/test_safe_success_critic.py`

**Interfaces:**
- Consumes: `FEATURE_DIM` from Task 2 (default `candidate_feature_dim`).
- Produces: `SafeSuccessCritic(clip_embed_dim=512, candidate_feature_dim=17, hidden_dim=256)` with `.forward(clip_embed: Tensor[B,512], candidate_features: Tensor[B,17]) -> Tensor[B]`, `.set_normalization(embed_mean, embed_std, feat_mean, feat_std)`, `.save(path)`, classmethod `.load(path, map_location="cpu") -> SafeSuccessCritic`. Module-level `score_candidates(model, clip_embed: np.ndarray, candidate_features_list: list[np.ndarray]) -> np.ndarray[float32]`.

- [ ] **Step 1: Write the failing test**

`semantic_cbf/tests/test_safe_success_critic.py`:

```python
# semantic_cbf/tests/test_safe_success_critic.py
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from semantic_cbf.safe_success_critic import SafeSuccessCritic, score_candidates


def test_forward_shape():
    model = SafeSuccessCritic(clip_embed_dim=8, candidate_feature_dim=4, hidden_dim=16)
    embed = torch.randn(3, 8)
    features = torch.randn(3, 4)
    out = model(embed, features)
    assert out.shape == (3,)


def test_save_load_roundtrip_preserves_output(tmp_path):
    model = SafeSuccessCritic(clip_embed_dim=8, candidate_feature_dim=4, hidden_dim=16)
    model.set_normalization(
        embed_mean=np.ones(8) * 0.5, embed_std=np.ones(8) * 2.0,
        feat_mean=np.zeros(4), feat_std=np.ones(4),
    )
    embed = torch.randn(2, 8)
    features = torch.randn(2, 4)
    model.eval()
    with torch.no_grad():
        before = model(embed, features).numpy()

    path = tmp_path / "critic.pt"
    model.save(path)
    loaded = SafeSuccessCritic.load(path)
    with torch.no_grad():
        after = loaded(embed, features).numpy()

    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)


def test_score_candidates_returns_numpy_array_aligned_with_input():
    model = SafeSuccessCritic(clip_embed_dim=8, candidate_feature_dim=4, hidden_dim=16)
    clip_embed = np.random.randn(8).astype(np.float32)
    features_list = [np.random.randn(4).astype(np.float32) for _ in range(5)]
    scores = score_candidates(model, clip_embed, features_list)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (5,)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest semantic_cbf/tests/test_safe_success_critic.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_cbf.safe_success_critic'`.

- [ ] **Step 3: Implement the critic**

`semantic_cbf/safe_success_critic.py`:

```python
"""Learned safe-success critic: [CLIP embed; candidate features] -> scalar score.

Trained offline (train_safe_success_critic.py) against oracle_counterfactual
sim-fork labels, then used at inference (learned_critic_selector.py) to pick
a candidate without forking the simulator.
"""

import numpy as np
import torch
import torch.nn as nn


class SafeSuccessCritic(nn.Module):
    def __init__(self, clip_embed_dim: int = 512, candidate_feature_dim: int = 17,
                 hidden_dim: int = 256):
        super().__init__()
        self.clip_embed_dim = clip_embed_dim
        self.candidate_feature_dim = candidate_feature_dim
        self.hidden_dim = hidden_dim

        input_dim = clip_embed_dim + candidate_feature_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.register_buffer("embed_mean", torch.zeros(clip_embed_dim))
        self.register_buffer("embed_std", torch.ones(clip_embed_dim))
        self.register_buffer("feat_mean", torch.zeros(candidate_feature_dim))
        self.register_buffer("feat_std", torch.ones(candidate_feature_dim))

    def set_normalization(self, embed_mean, embed_std, feat_mean, feat_std):
        self.embed_mean.copy_(torch.as_tensor(np.asarray(embed_mean), dtype=torch.float32))
        self.embed_std.copy_(
            torch.as_tensor(np.asarray(embed_std), dtype=torch.float32).clamp_min(1e-6)
        )
        self.feat_mean.copy_(torch.as_tensor(np.asarray(feat_mean), dtype=torch.float32))
        self.feat_std.copy_(
            torch.as_tensor(np.asarray(feat_std), dtype=torch.float32).clamp_min(1e-6)
        )

    def forward(self, clip_embed: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        embed_norm = (clip_embed - self.embed_mean) / self.embed_std
        feat_norm = (candidate_features - self.feat_mean) / self.feat_std
        z = torch.cat([embed_norm, feat_norm], dim=-1)
        return self.net(z).squeeze(-1)

    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "clip_embed_dim": self.clip_embed_dim,
            "candidate_feature_dim": self.candidate_feature_dim,
            "hidden_dim": self.hidden_dim,
        }, path)

    @classmethod
    def load(cls, path, map_location: str = "cpu") -> "SafeSuccessCritic":
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(
            clip_embed_dim=checkpoint["clip_embed_dim"],
            candidate_feature_dim=checkpoint["candidate_feature_dim"],
            hidden_dim=checkpoint["hidden_dim"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


def score_candidates(model: SafeSuccessCritic, clip_embed: np.ndarray,
                      candidate_features_list: list) -> np.ndarray:
    """clip_embed: (D,) shared across all candidates in this replan step.
    candidate_features_list: list of (F,) arrays, one per candidate.
    Returns float32[len(candidate_features_list)], aligned with the input list.
    """
    model.eval()
    embed_tensor = torch.from_numpy(np.asarray(clip_embed, dtype=np.float32)).unsqueeze(0)
    embed_tensor = embed_tensor.repeat(len(candidate_features_list), 1)
    feature_tensor = torch.from_numpy(
        np.stack(candidate_features_list).astype(np.float32)
    )
    with torch.no_grad():
        scores = model(embed_tensor, feature_tensor)
    return scores.numpy().astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest semantic_cbf/tests/test_safe_success_critic.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add semantic_cbf/safe_success_critic.py semantic_cbf/tests/test_safe_success_critic.py
git commit -m "feat: add SafeSuccessCritic MLP and score_candidates helper"
```

---

### Task 4: Learned-critic candidate selector

**Files:**
- Create: `semantic_cbf/learned_critic_selector.py`
- Test: `semantic_cbf/tests/test_learned_critic_selector.py`

**Interfaces:**
- Consumes: `extract_candidate_features` (Task 2), `score_candidates` (Task 3).
- Produces: `select_candidate(candidates: list[dict], clip_embed: np.ndarray, critic, eef_pos, obstacle_pos, phase, phase_target, controller_scale) -> tuple[dict, int, np.ndarray]` — `(selected_candidate, selected_index, scores)`. This is what `run_guided_safelibero_pi05_eval.py`'s `learned_critic` branch (Task 8) calls directly.

- [ ] **Step 1: Write the failing test**

`semantic_cbf/tests/test_learned_critic_selector.py`:

```python
# semantic_cbf/tests/test_learned_critic_selector.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from semantic_cbf.learned_critic_selector import select_candidate


class _ClearancePreferringCritic:
    """Fake critic: score = candidate_features[0] (== min_clearance). No torch
    needed — isolates the selector's wiring from the real MLP."""

    def eval(self):
        return self

    def __call__(self, embed_batch, feature_batch):
        return feature_batch[:, 0]


def _candidate(min_clearance, expert):
    actions = np.zeros((5, 7), dtype=np.float64)
    return {
        "actions": actions,
        "expert": expert,
        "guidance": {"min_clearance": min_clearance, "correction_norm": 0.0, "max_residual": 0.0},
    }


def test_select_candidate_picks_highest_scoring():
    import torch
    monkeypatched = _ClearancePreferringCritic()

    def fake_score_candidates(model, clip_embed, features_list):
        embed_batch = torch.zeros(len(features_list), 1)
        feature_batch = torch.from_numpy(np.stack(features_list))
        return model(embed_batch, feature_batch).numpy()

    import semantic_cbf.learned_critic_selector as selector_module
    original = selector_module.score_candidates
    selector_module.score_candidates = fake_score_candidates
    try:
        candidates = [
            _candidate(min_clearance=-0.05, expert="conservative"),
            _candidate(min_clearance=0.03, expert="permissive"),
            _candidate(min_clearance=0.01, expert="nominal"),
        ]
        result, index, scores = select_candidate(
            candidates, clip_embed=np.zeros(1, dtype=np.float32), critic=monkeypatched,
            eef_pos=[0.0, 0.0, 0.0], obstacle_pos=None, phase=None, phase_target=None,
            controller_scale=1.0,
        )
        assert index == 1
        assert result["expert"] == "permissive"
        assert scores.shape == (3,)
    finally:
        selector_module.score_candidates = original
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest semantic_cbf/tests/test_learned_critic_selector.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_cbf.learned_critic_selector'`.

- [ ] **Step 3: Implement the selector**

`semantic_cbf/learned_critic_selector.py`:

```python
"""Pure candidate-selection logic for the learned_critic steering mode.

Kept separate from run_guided_safelibero_pi05_eval.py (which imports
libero/mujoco/openpi_client at module load time) so it is directly
unit-testable with only numpy/torch.
"""

import numpy as np

from semantic_cbf.candidate_features import extract_candidate_features
from semantic_cbf.safe_success_critic import score_candidates


def select_candidate(candidates, clip_embed, critic, eef_pos, obstacle_pos,
                      phase, phase_target, controller_scale):
    """Returns (selected_candidate, selected_index, scores).

    scores: float32[len(candidates)], aligned with `candidates`.
    """
    features = [
        extract_candidate_features(candidate, eef_pos, obstacle_pos, phase, phase_target, controller_scale)
        for candidate in candidates
    ]
    scores = score_candidates(critic, clip_embed, features)
    selected_index = int(np.argmax(scores))
    return candidates[selected_index], selected_index, scores
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest semantic_cbf/tests/test_learned_critic_selector.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add semantic_cbf/learned_critic_selector.py semantic_cbf/tests/test_learned_critic_selector.py
git commit -m "feat: add learned_critic candidate selector"
```

---

### Task 5: Training data logger

**Files:**
- Create: `semantic_cbf/training_data_logger.py`
- Test: `semantic_cbf/tests/test_training_data_logger.py`

**Interfaces:**
- Produces: `TrainingDataLogger(log_dir)` with `.start_episode(task_suite, level, task_id, ep_idx)`, `.log_step(clip_embed: np.ndarray, candidate_features_list: list[np.ndarray], oracle_scores: list[float], collided_list: list[bool], done_list: list[bool])`, `.end_episode() -> pathlib.Path | None`. Writes one `.npz` per episode to `<log_dir>/<task_suite>/<level>/task<task_id>_ep<ep_idx>.npz` with arrays `clip_embed (N,D)`, `candidate_features (N,F)`, `oracle_score (N,)`, `collided (N,)` bool, `done (N,)` bool, `step_id (N,)` int64.

- [ ] **Step 1: Write the failing test**

`semantic_cbf/tests/test_training_data_logger.py`:

```python
# semantic_cbf/tests/test_training_data_logger.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from semantic_cbf.training_data_logger import TrainingDataLogger


def test_logs_one_episode_with_two_steps(tmp_path):
    logger = TrainingDataLogger(tmp_path)
    logger.start_episode("safelibero_spatial", "I", task_id=0, ep_idx=0)

    embed_a = np.ones(4, dtype=np.float32)
    features_a = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
    logger.log_step(embed_a, features_a, oracle_scores=[0.5, -1.0], collided_list=[False, True], done_list=[False, False])

    embed_b = np.zeros(4, dtype=np.float32) + 2.0
    features_b = [np.array([5.0, 6.0], dtype=np.float32)]
    logger.log_step(embed_b, features_b, oracle_scores=[2.0], collided_list=[False], done_list=[True])

    path = logger.end_episode()
    assert path is not None
    assert path == tmp_path / "safelibero_spatial" / "I" / "task0_ep0.npz"

    data = np.load(path)
    assert data["clip_embed"].shape == (3, 4)
    assert data["candidate_features"].shape == (3, 2)
    np.testing.assert_array_equal(data["oracle_score"], np.array([0.5, -1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(data["collided"], np.array([False, True, False]))
    np.testing.assert_array_equal(data["done"], np.array([False, False, True]))
    np.testing.assert_array_equal(data["step_id"], np.array([0, 0, 1]))


def test_end_episode_with_no_steps_returns_none(tmp_path):
    logger = TrainingDataLogger(tmp_path)
    logger.start_episode("safelibero_spatial", "II", task_id=1, ep_idx=2)
    assert logger.end_episode() is None
    assert not (tmp_path / "safelibero_spatial" / "II").exists()


def test_second_episode_does_not_leak_rows_from_first(tmp_path):
    logger = TrainingDataLogger(tmp_path)
    logger.start_episode("safelibero_spatial", "I", task_id=0, ep_idx=0)
    logger.log_step(np.zeros(2, dtype=np.float32), [np.zeros(1, dtype=np.float32)], [0.0], [False], [False])
    logger.end_episode()

    logger.start_episode("safelibero_spatial", "I", task_id=0, ep_idx=1)
    logger.log_step(np.ones(2, dtype=np.float32), [np.ones(1, dtype=np.float32)], [1.0], [True], [False])
    path = logger.end_episode()

    data = np.load(path)
    assert data["oracle_score"].shape == (1,)
    assert data["oracle_score"][0] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest semantic_cbf/tests/test_training_data_logger.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_cbf.training_data_logger'`.

- [ ] **Step 3: Implement the logger**

`semantic_cbf/training_data_logger.py`:

```python
"""Buffers per-step, per-candidate rows scored by _oracle_counterfactual_score
and flushes one .npz shard per episode. Used only when
--log_training_data is set (oracle_counterfactual mode) — no effect on any
existing steering mode's behavior.
"""

import pathlib

import numpy as np


class TrainingDataLogger:
    def __init__(self, log_dir):
        self.log_dir = pathlib.Path(log_dir)
        self._buffer = None
        self._meta = None
        self._step_counter = 0

    def start_episode(self, task_suite, level, task_id, ep_idx):
        self._buffer = {
            "clip_embed": [], "candidate_features": [], "oracle_score": [],
            "collided": [], "done": [], "step_id": [],
        }
        self._meta = (task_suite, level, task_id, ep_idx)
        self._step_counter = 0

    def log_step(self, clip_embed, candidate_features_list, oracle_scores, collided_list, done_list):
        count = len(candidate_features_list)
        for index in range(count):
            self._buffer["clip_embed"].append(np.asarray(clip_embed, dtype=np.float32))
            self._buffer["candidate_features"].append(
                np.asarray(candidate_features_list[index], dtype=np.float32)
            )
            self._buffer["oracle_score"].append(float(oracle_scores[index]))
            self._buffer["collided"].append(bool(collided_list[index]))
            self._buffer["done"].append(bool(done_list[index]))
            self._buffer["step_id"].append(self._step_counter)
        self._step_counter += 1

    def end_episode(self):
        if self._buffer is None or not self._buffer["clip_embed"]:
            self._buffer = None
            self._meta = None
            return None

        task_suite, level, task_id, ep_idx = self._meta
        out_dir = self.log_dir / task_suite / level
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"task{task_id}_ep{ep_idx}.npz"
        np.savez(
            path,
            clip_embed=np.stack(self._buffer["clip_embed"]).astype(np.float32),
            candidate_features=np.stack(self._buffer["candidate_features"]).astype(np.float32),
            oracle_score=np.asarray(self._buffer["oracle_score"], dtype=np.float32),
            collided=np.asarray(self._buffer["collided"], dtype=bool),
            done=np.asarray(self._buffer["done"], dtype=bool),
            step_id=np.asarray(self._buffer["step_id"], dtype=np.int64),
        )
        self._buffer = None
        self._meta = None
        return path
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest semantic_cbf/tests/test_training_data_logger.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add semantic_cbf/training_data_logger.py semantic_cbf/tests/test_training_data_logger.py
git commit -m "feat: add per-episode training data logger for oracle labels"
```

---

### Task 6: Extend `_oracle_counterfactual_score` with optional diagnostics (backward compatible)

**Files:**
- Modify: `vlm_pipeline/run_guided_safelibero_pi05_eval.py:313-370` (function `_oracle_counterfactual_score`)
- Test: `vlm_pipeline/tests/test_oracle_counterfactual_score.py`

**Interfaces:**
- Produces: `_oracle_counterfactual_score(..., return_diagnostics: bool = False)`. When `False` (all five existing call sites — unchanged), returns `float` exactly as today. When `True` (new logging call site only, Task 8), returns `(float, dict)` where `dict` has keys `"collided": bool`, `"done": bool`.

This module imports `libero`/`mujoco`/`openpi_client` at file scope, so its test must run in the `openvla_libero_merged` (or `libero_env`) conda env, not a bare `python`.

- [ ] **Step 1: Write the failing test**

`vlm_pipeline/tests/test_oracle_counterfactual_score.py`:

```python
# vlm_pipeline/tests/test_oracle_counterfactual_score.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pytest

from run_guided_safelibero_pi05_eval import _oracle_counterfactual_score


class _FakeEnv:
    """Minimal stand-in for the LIBERO env used only by _oracle_counterfactual_score:
    regenerate_obs_from_state, env.timestep, and step(action)."""

    class _Inner:
        timestep = 0

    def __init__(self, obstacle_displacement, done_on_step=None):
        self.env = self._Inner()
        self._obstacle_displacement = obstacle_displacement
        self._done_on_step = done_on_step
        self._step_count = 0

    def regenerate_obs_from_state(self, snapshot):
        return self._make_obs()

    def _make_obs(self):
        obstacle_pos = np.array([0.1, 0.0, 0.81]) + self._obstacle_displacement
        return {
            "robot0_eef_pos": np.array([0.0, 0.0, 0.85]),
            "obstacle_pos": obstacle_pos,
        }

    def step(self, action):
        self._step_count += 1
        obs = self._make_obs()
        done = self._step_count == self._done_on_step
        return obs, 0.0, done, {}


def test_no_collision_no_done_returns_float_by_default():
    env = _FakeEnv(obstacle_displacement=np.zeros(3))
    candidate = {"actions": np.zeros((2, 7)), "guidance": {"correction_norm": 0.0}}
    score = _oracle_counterfactual_score(
        env, candidate, snapshot=None, snapshot_timestep=0,
        obstacle_name="obstacle", initial_obstacle_pos=np.array([0.1, 0.0, 0.81]),
        manipulated_name=None, phase_target=None, current_target_distance=0.0,
        oracle_horizon=2,
    )
    assert isinstance(score, float)


def test_collision_diagnostics_when_requested():
    env = _FakeEnv(obstacle_displacement=np.array([0.01, 0.0, 0.0]))  # > 0.001 threshold
    candidate = {"actions": np.zeros((2, 7)), "guidance": {"correction_norm": 0.0}}
    score, diagnostics = _oracle_counterfactual_score(
        env, candidate, snapshot=None, snapshot_timestep=0,
        obstacle_name="obstacle", initial_obstacle_pos=np.array([0.1, 0.0, 0.81]),
        manipulated_name=None, phase_target=None, current_target_distance=0.0,
        oracle_horizon=2, return_diagnostics=True,
    )
    assert isinstance(score, float)
    assert diagnostics["collided"] is True
    assert diagnostics["done"] is False


def test_done_diagnostics_when_requested():
    env = _FakeEnv(obstacle_displacement=np.zeros(3), done_on_step=1)
    candidate = {"actions": np.zeros((2, 7)), "guidance": {"correction_norm": 0.0}}
    score, diagnostics = _oracle_counterfactual_score(
        env, candidate, snapshot=None, snapshot_timestep=0,
        obstacle_name="obstacle", initial_obstacle_pos=np.array([0.1, 0.0, 0.81]),
        manipulated_name=None, phase_target=None, current_target_distance=0.0,
        oracle_horizon=2, return_diagnostics=True,
    )
    assert diagnostics["collided"] is False
    assert diagnostics["done"] is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate openvla_libero_merged
export MUJOCO_GL=egl
export PYTHONPATH="$PWD/SafeLIBERO/safelibero:$PWD/.worktrees/flow-guidance-tier-gt/client_deps:$PWD/openpi_guided/packages/openpi-client/src:$PWD/vlm_pipeline"
python -m pytest vlm_pipeline/tests/test_oracle_counterfactual_score.py -v
```

Expected: FAIL — `TypeError: _oracle_counterfactual_score() got an unexpected keyword argument 'return_diagnostics'`.

- [ ] **Step 3: Extend the function signature and return**

In `vlm_pipeline/run_guided_safelibero_pi05_eval.py`, modify `_oracle_counterfactual_score` (currently lines 313-370):

Old signature and tail:
```python
def _oracle_counterfactual_score(
    env, candidate, snapshot, snapshot_timestep, obstacle_name,
    initial_obstacle_pos, manipulated_name, phase_target,
    current_target_distance, oracle_horizon,
):
```
...
```python
    return (
        -10.0 * float(collided)
        + 3.0 * float(done)
        + topology_bonus
        + 3.0 * separation_score
        + progress
        - 0.001 * correction
    )
```

New signature and tail (only the signature line and the final `return` change; the body in between is untouched):
```python
def _oracle_counterfactual_score(
    env, candidate, snapshot, snapshot_timestep, obstacle_name,
    initial_obstacle_pos, manipulated_name, phase_target,
    current_target_distance, oracle_horizon, return_diagnostics=False,
):
```
...
```python
    score = (
        -10.0 * float(collided)
        + 3.0 * float(done)
        + topology_bonus
        + 3.0 * separation_score
        + progress
        - 0.001 * correction
    )
    if return_diagnostics:
        return score, {"collided": bool(collided), "done": bool(done)}
    return score
```

Use the Edit tool for both the signature line and the final return block; do not touch anything in between (the existing `_oracle_depth2_scores` and all five existing call sites keep calling this positionally/by-keyword without `return_diagnostics`, so they keep getting a bare `float` — unchanged behavior).

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest vlm_pipeline/tests/test_oracle_counterfactual_score.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run the full existing test suite for this file's neighbors to confirm no regression**

```bash
python -m pytest vlm_pipeline/tests/ -v
```

Expected: all pass (only the new file exists so far under `vlm_pipeline/tests/`).

- [ ] **Step 6: Commit**

```bash
git add vlm_pipeline/run_guided_safelibero_pi05_eval.py vlm_pipeline/tests/test_oracle_counterfactual_score.py
git commit -m "feat: optional diagnostics return from _oracle_counterfactual_score (backward compatible)"
```

---

### Task 7: Offline critic training script

**Files:**
- Create: `semantic_cbf/train_safe_success_critic.py`
- Test: `semantic_cbf/tests/test_train_safe_success_critic.py`

**Interfaces:**
- Consumes: `.npz` schema from Task 5, `SafeSuccessCritic`/`score_candidates` from Task 3.
- Produces: `load_npz_shards(data_dir) -> dict` (keys `clip_embed`, `candidate_features`, `oracle_score`, `collided`, `done`, `episode_uid`, `step_uid` — all numpy arrays, one row per candidate), `split_train_val(data, val_fraction=0.2, seed=0) -> (train_mask, val_mask)` (split by unique `episode_uid`, not by row), `train_critic(data_dir, val_fraction=0.2, seed=0, epochs=200, lr=1e-3, hidden_dim=256, batch_size=256) -> (SafeSuccessCritic, dict)` where the metrics dict has keys `"val_mse"`, `"val_spearman"`, `"val_argmax_agreement"`. CLI: `python semantic_cbf/train_safe_success_critic.py --data_dir <dir> --output <checkpoint.pt> [--val_fraction --epochs --lr --hidden_dim --batch_size --seed]`.

- [ ] **Step 1: Write the failing test**

`semantic_cbf/tests/test_train_safe_success_critic.py`:

```python
# semantic_cbf/tests/test_train_safe_success_critic.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from semantic_cbf.train_safe_success_critic import load_npz_shards, split_train_val, train_critic


def _write_synthetic_shard(path, rng, n_steps, k, embed_dim, feat_dim, weight):
    clip_embed_rows, feature_rows, score_rows, step_id_rows = [], [], [], []
    for step in range(n_steps):
        step_embed = rng.randn(embed_dim).astype(np.float32)  # no signal — noise only
        for _ in range(k):
            features = rng.randn(feat_dim).astype(np.float32)
            score = float(features @ weight) + 0.01 * rng.randn()
            clip_embed_rows.append(step_embed)
            feature_rows.append(features)
            score_rows.append(score)
            step_id_rows.append(step)
    np.savez(
        path,
        clip_embed=np.stack(clip_embed_rows),
        candidate_features=np.stack(feature_rows),
        oracle_score=np.asarray(score_rows, dtype=np.float32),
        collided=np.zeros(len(score_rows), dtype=bool),
        done=np.zeros(len(score_rows), dtype=bool),
        step_id=np.asarray(step_id_rows, dtype=np.int64),
    )


def _build_synthetic_dataset(tmp_path, n_episodes=24, n_steps=6, k=4, embed_dim=8, feat_dim=4):
    rng = np.random.RandomState(0)
    weight = rng.randn(feat_dim).astype(np.float32)
    level_dir = tmp_path / "safelibero_spatial" / "I"
    level_dir.mkdir(parents=True)
    for episode in range(n_episodes):
        _write_synthetic_shard(
            level_dir / f"task0_ep{episode}.npz", rng, n_steps, k, embed_dim, feat_dim, weight,
        )
    return weight


def test_load_npz_shards_concatenates_all_episodes(tmp_path):
    _build_synthetic_dataset(tmp_path, n_episodes=3, n_steps=2, k=4, embed_dim=8, feat_dim=4)
    data = load_npz_shards(tmp_path)
    assert data["clip_embed"].shape == (3 * 2 * 4, 8)
    assert data["candidate_features"].shape == (3 * 2 * 4, 4)
    assert len(set(data["episode_uid"])) == 3
    assert len(set(zip(data["episode_uid"], data["step_id"]))) == 3 * 2


def test_split_train_val_splits_by_episode_not_row(tmp_path):
    _build_synthetic_dataset(tmp_path, n_episodes=10, n_steps=2, k=4, embed_dim=8, feat_dim=4)
    data = load_npz_shards(tmp_path)
    train_mask, val_mask = split_train_val(data, val_fraction=0.3, seed=0)
    train_episodes = set(data["episode_uid"][train_mask])
    val_episodes = set(data["episode_uid"][val_mask])
    assert train_episodes.isdisjoint(val_episodes)
    assert (train_mask | val_mask).all()


def test_train_critic_recovers_linear_signal(tmp_path):
    _build_synthetic_dataset(tmp_path, n_episodes=24, n_steps=6, k=4, embed_dim=8, feat_dim=4)
    model, metrics = train_critic(
        tmp_path, val_fraction=0.25, seed=0, epochs=80, lr=1e-2, hidden_dim=32, batch_size=32,
    )
    assert metrics["val_spearman"] > 0.8
    assert metrics["val_argmax_agreement"] > 0.7
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest semantic_cbf/tests/test_train_safe_success_critic.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_cbf.train_safe_success_critic'`.

- [ ] **Step 3: Implement the training script**

`semantic_cbf/train_safe_success_critic.py`:

```python
"""Offline training for the safe-success critic against oracle_counterfactual labels.

Usage:
  python semantic_cbf/train_safe_success_critic.py \
      --data_dir viability_tournament/oracle_training_data \
      --output semantic_cbf/checkpoints/safe_success_critic.pt
"""

import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn

from semantic_cbf.safe_success_critic import SafeSuccessCritic


def load_npz_shards(data_dir):
    data_dir = pathlib.Path(data_dir)
    shard_paths = sorted(data_dir.rglob("*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No .npz shards found under {data_dir}")

    clip_embed, candidate_features, oracle_score = [], [], []
    collided, done, episode_uid, step_id = [], [], [], []
    for shard_path in shard_paths:
        shard = np.load(shard_path)
        n_rows = shard["oracle_score"].shape[0]
        clip_embed.append(shard["clip_embed"])
        candidate_features.append(shard["candidate_features"])
        oracle_score.append(shard["oracle_score"])
        collided.append(shard["collided"])
        done.append(shard["done"])
        step_id.append(shard["step_id"])
        episode_uid.extend([str(shard_path)] * n_rows)

    return {
        "clip_embed": np.concatenate(clip_embed, axis=0),
        "candidate_features": np.concatenate(candidate_features, axis=0),
        "oracle_score": np.concatenate(oracle_score, axis=0),
        "collided": np.concatenate(collided, axis=0),
        "done": np.concatenate(done, axis=0),
        "episode_uid": np.array(episode_uid),
        "step_id": np.concatenate(step_id, axis=0),
    }


def split_train_val(data, val_fraction=0.2, seed=0):
    episodes = sorted(set(data["episode_uid"]))
    rng = np.random.RandomState(seed)
    rng.shuffle(episodes)
    n_val = max(1, int(round(len(episodes) * val_fraction)))
    val_episodes = set(episodes[:n_val])
    val_mask = np.array([episode in val_episodes for episode in data["episode_uid"]])
    train_mask = ~val_mask
    return train_mask, val_mask


def _spearman_corr(a, b):
    rank_a = np.argsort(np.argsort(a))
    rank_b = np.argsort(np.argsort(b))
    if np.std(rank_a) == 0 or np.std(rank_b) == 0:
        return 0.0
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def _argmax_agreement(data, mask, predicted_scores):
    step_keys = list(zip(data["episode_uid"][mask], data["step_id"][mask]))
    oracle = data["oracle_score"][mask]
    predicted = predicted_scores[mask]

    by_step = {}
    for index, key in enumerate(step_keys):
        by_step.setdefault(key, []).append(index)

    matches, total = 0, 0
    for indices in by_step.values():
        if len(indices) < 2:
            continue
        oracle_best = indices[int(np.argmax(oracle[indices]))]
        predicted_best = indices[int(np.argmax(predicted[indices]))]
        matches += int(oracle_best == predicted_best)
        total += 1
    return matches / total if total else 0.0


def train_critic(data_dir, val_fraction=0.2, seed=0, epochs=200, lr=1e-3,
                  hidden_dim=256, batch_size=256):
    torch.manual_seed(seed)
    data = load_npz_shards(data_dir)
    train_mask, val_mask = split_train_val(data, val_fraction=val_fraction, seed=seed)

    embed_dim = data["clip_embed"].shape[1]
    feat_dim = data["candidate_features"].shape[1]
    model = SafeSuccessCritic(clip_embed_dim=embed_dim, candidate_feature_dim=feat_dim,
                               hidden_dim=hidden_dim)

    train_embed = data["clip_embed"][train_mask]
    train_features = data["candidate_features"][train_mask]
    train_scores = data["oracle_score"][train_mask]

    model.set_normalization(
        embed_mean=train_embed.mean(axis=0), embed_std=train_embed.std(axis=0),
        feat_mean=train_features.mean(axis=0), feat_std=train_features.std(axis=0),
    )

    embed_tensor = torch.from_numpy(train_embed.astype(np.float32))
    feature_tensor = torch.from_numpy(train_features.astype(np.float32))
    score_tensor = torch.from_numpy(train_scores.astype(np.float32))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n_train = embed_tensor.shape[0]

    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            batch_indices = permutation[start:start + batch_size]
            optimizer.zero_grad()
            predictions = model(embed_tensor[batch_indices], feature_tensor[batch_indices])
            loss = loss_fn(predictions, score_tensor[batch_indices])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_predictions = model(
            torch.from_numpy(data["clip_embed"].astype(np.float32)),
            torch.from_numpy(data["candidate_features"].astype(np.float32)),
        ).numpy()

    val_mse = float(np.mean((val_predictions[val_mask] - data["oracle_score"][val_mask]) ** 2))
    val_spearman = _spearman_corr(val_predictions[val_mask], data["oracle_score"][val_mask])
    val_argmax_agreement = _argmax_agreement(data, val_mask, val_predictions)

    metrics = {
        "val_mse": val_mse,
        "val_spearman": val_spearman,
        "val_argmax_agreement": val_argmax_agreement,
        "n_train_rows": int(train_mask.sum()),
        "n_val_rows": int(val_mask.sum()),
    }
    return model, metrics


def _parse_args():
    parser = argparse.ArgumentParser(description="Train the safe-success critic")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    trained_model, result_metrics = train_critic(
        args.data_dir, val_fraction=args.val_fraction, seed=args.seed, epochs=args.epochs,
        lr=args.lr, hidden_dim=args.hidden_dim, batch_size=args.batch_size,
    )
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trained_model.save(output_path)
    print(f"Saved checkpoint to {output_path}")
    for key, value in result_metrics.items():
        print(f"  {key}: {value}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest semantic_cbf/tests/test_train_safe_success_critic.py -v
```

Expected: 3 passed. (If `test_train_critic_recovers_linear_signal` is flaky near the threshold, that is a real signal to inspect — do not loosen the assertion without understanding why; a bump to `epochs` is fine, silently lowering the threshold is not.)

- [ ] **Step 5: Commit**

```bash
git add semantic_cbf/train_safe_success_critic.py semantic_cbf/tests/test_train_safe_success_critic.py
git commit -m "feat: add offline training script for the safe-success critic"
```

---

### Task 8: Wire `learned_critic` steering mode and data logging into the eval client

**Files:**
- Modify: `vlm_pipeline/run_guided_safelibero_pi05_eval.py`

**Interfaces:**
- Consumes: `ImageEmbedder` (Task 1), `TrainingDataLogger` (Task 5), `select_candidate` (Task 4), `SafeSuccessCritic` (Task 3), extended `_oracle_counterfactual_score` (Task 6).

This task has no isolated pytest step — it wires already-tested modules into a script that requires the full `libero`/`mujoco`/websocket-server stack to execute, so it is verified by the smoke run in Task 12. Each step below is a precise, mechanical edit.

- [ ] **Step 1: Add imports**

Edit the top of `vlm_pipeline/run_guided_safelibero_pi05_eval.py`, after the existing `openpi_client` imports:

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from semantic_cbf.clip_embedder import ImageEmbedder
from semantic_cbf.learned_critic_selector import select_candidate
from semantic_cbf.safe_success_critic import SafeSuccessCritic
from semantic_cbf.training_data_logger import TrainingDataLogger
```

(The four new lines go after the existing `openpi_client` import lines, matching import-grouping already used in this file.)

- [ ] **Step 2: Add CLI arguments**

In `parse_args()`, change:

```python
    parser.add_argument("--steering_mode", choices=["fixed", "phase_adaptive", "best_of_k", "semantic_best_of_k", "oracle_counterfactual"],
                        default="fixed")
```
to:
```python
    parser.add_argument("--steering_mode", choices=["fixed", "phase_adaptive", "best_of_k", "semantic_best_of_k", "oracle_counterfactual", "learned_critic"],
                        default="fixed")
```

Then, after the existing `--oracle_depth` argument, add:

```python
    parser.add_argument("--oracle_depth", type=int, choices=[1, 2], default=1)
    parser.add_argument("--log_training_data", type=str, default=None,
                        help="If set (oracle_counterfactual mode only), log per-candidate "
                             "(clip_embed, candidate_features, oracle_score) rows as one "
                             ".npz per episode under this directory.")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--critic_checkpoint", type=str, default=None,
                        help="Required when --steering_mode learned_critic.")
    return parser.parse_args()
```

(Remove the old standalone `return parser.parse_args()` line that followed `--oracle_depth` — there is exactly one `return parser.parse_args()` in the function; move it to the end as shown.)

- [ ] **Step 3: Instantiate embedder/logger/critic before the task loop**

In `run_eval(args)`, right after:

```python
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info("Connected to guided pi0.5 server.")
```

add:

```python
    embedder = None
    if args.steering_mode == "learned_critic" or args.log_training_data:
        embedder = ImageEmbedder(model_name=args.clip_model_name)

    training_logger = None
    if args.log_training_data:
        training_logger = TrainingDataLogger(args.log_training_data)

    critic = None
    if args.steering_mode == "learned_critic":
        if not args.critic_checkpoint:
            raise ValueError("--critic_checkpoint is required when --steering_mode learned_critic")
        critic = SafeSuccessCritic.load(args.critic_checkpoint)
```

- [ ] **Step 4: Start/end episode logging**

Find:
```python
        for ep_idx in range(args.num_trials_per_task):
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[ep_idx])
```
Change to:
```python
        for ep_idx in range(args.num_trials_per_task):
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[ep_idx])
            if training_logger is not None:
                training_logger.start_episode(args.task_suite_name, args.safety_level, task_id, ep_idx)
```

Find the block right after the per-episode `while` loop ends (starts with `task_successes += int(done)`):
```python
            task_successes += int(done)
            task_collisions += int(collide_flag)
```
Change to:
```python
            if training_logger is not None:
                training_logger.end_episode()

            task_successes += int(done)
            task_collisions += int(collide_flag)
```

- [ ] **Step 5: Include `learned_critic` in the diverse-candidate-generation gate**

Find:
```python
                        if args.steering_mode in ("best_of_k", "semantic_best_of_k", "oracle_counterfactual"):
                            candidates = (
                                _infer_diverse_candidates(client, element, args.num_candidates)
                                if args.steering_mode == "oracle_counterfactual"
                                else [client.infer(element) for _ in range(args.num_candidates)]
                            )
                            if args.steering_mode == "oracle_counterfactual":
                                candidates = _inject_topology_experts(
                                    candidates, obs, obstacle_name, manipulated_name,
                                    initial_manipulated_pos, goal_position, topology_side,
                                )
```
Change to:
```python
                        if args.steering_mode in ("best_of_k", "semantic_best_of_k", "oracle_counterfactual", "learned_critic"):
                            candidates = (
                                _infer_diverse_candidates(client, element, args.num_candidates)
                                if args.steering_mode in ("oracle_counterfactual", "learned_critic")
                                else [client.infer(element) for _ in range(args.num_candidates)]
                            )
                            if args.steering_mode in ("oracle_counterfactual", "learned_critic"):
                                candidates = _inject_topology_experts(
                                    candidates, obs, obstacle_name, manipulated_name,
                                    initial_manipulated_pos, goal_position, topology_side,
                                )
```

- [ ] **Step 6: Add the logging call inside the existing `oracle_counterfactual` branch**

Find (inside the `if args.steering_mode == "oracle_counterfactual" and obstacle_name is not None:` block):
```python
                                if args.oracle_depth == 2 and any(
                                    score < -5.0 for score in stage_scores
                                ) or (args.oracle_depth == 2 and near_viability_boundary):
                                    scores, continuations = _oracle_depth2_scores(
                                        env, candidates, snapshot, snapshot_timestep,
                                        obstacle_name, initial_obstacle_pos,
                                        manipulated_name, phase_target,
                                        current_target_distance, args.oracle_horizon,
                                        stage_scores,
                                    )
                                else:
                                    scores = stage_scores
                                    continuations = ["n/a"] * len(candidates)
                                selected_index = int(np.argmax(scores))
```
Change to:
```python
                                if args.oracle_depth == 2 and any(
                                    score < -5.0 for score in stage_scores
                                ) or (args.oracle_depth == 2 and near_viability_boundary):
                                    scores, continuations = _oracle_depth2_scores(
                                        env, candidates, snapshot, snapshot_timestep,
                                        obstacle_name, initial_obstacle_pos,
                                        manipulated_name, phase_target,
                                        current_target_distance, args.oracle_horizon,
                                        stage_scores,
                                    )
                                else:
                                    scores = stage_scores
                                    continuations = ["n/a"] * len(candidates)
                                if training_logger is not None:
                                    step_diagnostics = [
                                        _oracle_counterfactual_score(
                                            env, candidate, snapshot, snapshot_timestep,
                                            obstacle_name, initial_obstacle_pos,
                                            manipulated_name, phase_target,
                                            current_target_distance, args.oracle_horizon,
                                            return_diagnostics=True,
                                        )[1]
                                        for candidate in candidates
                                    ]
                                    step_embed = embedder.embed(img)
                                    step_features = [
                                        extract_candidate_features(
                                            candidate, obs["robot0_eef_pos"], obs[f"{obstacle_name}_pos"],
                                            phase if manipulated_name is not None else None,
                                            phase_target, args.controller_scale,
                                        )
                                        for candidate in candidates
                                    ]
                                    training_logger.log_step(
                                        step_embed, step_features, scores,
                                        collided_list=[d["collided"] for d in step_diagnostics],
                                        done_list=[d["done"] for d in step_diagnostics],
                                    )
                                selected_index = int(np.argmax(scores))
```

This re-runs the oracle fork once more per candidate purely to recover the `collided`/`done` diagnostics (the sim state is a snapshot/fork either way and is rewound immediately after, same as the existing code — no observational effect, matching the existing "counterfactual rollouts must be observationally invisible" invariant already documented in this file). `extract_candidate_features` needs importing — add it to the Step 1 import block: `from semantic_cbf.candidate_features import extract_candidate_features`.

- [ ] **Step 7: Add the `learned_critic` selection branch**

Find:
```python
                            elif args.steering_mode == "semantic_best_of_k" and manipulated_name is not None:
```
Insert a new branch immediately before it:
```python
                            elif args.steering_mode == "learned_critic":
                                if obstacle_name is None:
                                    result = candidates[0]
                                else:
                                    if manipulated_name is not None:
                                        phase = _semantic_phase(obs, manipulated_name, initial_manipulated_pos)
                                        object_position = np.asarray(obs[f"{manipulated_name}_pos"])
                                        phase_target = object_position if phase == "approach" else goal_position
                                    else:
                                        phase = None
                                        phase_target = None
                                    step_embed = embedder.embed(img)
                                    result, selected_index, learned_scores = select_candidate(
                                        candidates, step_embed, critic,
                                        obs["robot0_eef_pos"], obs[f"{obstacle_name}_pos"],
                                        phase, phase_target, args.controller_scale,
                                    )
                                    selected_expert = result.get("expert", "unlabeled")
                                    if (
                                        topology_side is None
                                        and result.get("topology_required", False)
                                        and selected_expert in ("tangent_left", "tangent_right")
                                    ):
                                        topology_side = selected_expert
                                    logging.info(
                                        "  learned_critic experts=%s scores=%s selected=%s",
                                        [c.get("expert", "unlabeled") for c in candidates],
                                        [round(float(s), 4) for s in learned_scores],
                                        selected_expert,
                                    )
                            elif args.steering_mode == "semantic_best_of_k" and manipulated_name is not None:
```

- [ ] **Step 8: Exclude `learned_critic` from the closure-based `max(candidates, key=candidate_score)` fallback**

Find:
```python
                            if args.steering_mode != "oracle_counterfactual":
                                result = max(candidates, key=candidate_score)
                            elif obstacle_name is None:
                                result = candidates[0]
```
Change to:
```python
                            if args.steering_mode not in ("oracle_counterfactual", "learned_critic"):
                                result = max(candidates, key=candidate_score)
                            elif args.steering_mode == "oracle_counterfactual" and obstacle_name is None:
                                result = candidates[0]
```

- [ ] **Step 9: Update the results metadata's `num_candidates` field**

Find:
```python
        "num_candidates": (
            args.num_candidates
            if args.steering_mode in ("best_of_k", "semantic_best_of_k", "oracle_counterfactual")
            else 1
        ),
```
Change to:
```python
        "num_candidates": (
            args.num_candidates
            if args.steering_mode in ("best_of_k", "semantic_best_of_k", "oracle_counterfactual", "learned_critic")
            else 1
        ),
```

- [ ] **Step 10: Byte-compile sanity check (catches syntax/indentation errors without needing the full stack)**

```bash
python -m py_compile vlm_pipeline/run_guided_safelibero_pi05_eval.py
```

Expected: no output, exit code 0.

- [ ] **Step 11: Import sanity check in the target env (catches missing-name/import errors)**

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate openvla_libero_merged
export PYTHONPATH="$PWD/SafeLIBERO/safelibero:$PWD/.worktrees/flow-guidance-tier-gt/client_deps:$PWD/openpi_guided/packages/openpi-client/src:$PWD/vlm_pipeline:$PWD"
python -c "import run_guided_safelibero_pi05_eval as m; print(m.parse_args.__module__)"
```

Run this from `vlm_pipeline/`'s parent (i.e. the worktree root), since `PYTHONPATH` includes `$PWD/vlm_pipeline` for the module and `$PWD` for `semantic_cbf`. Expected: prints the module name with no traceback.

- [ ] **Step 12: Commit**

```bash
git add vlm_pipeline/run_guided_safelibero_pi05_eval.py
git commit -m "feat: add learned_critic steering mode and oracle training-data logging"
```

---

### Task 9: SLURM launcher updates

**Files:**
- Modify: `slurm/viability_tournament_smoke.slurm`

- [ ] **Step 1: Add the worktree root to `PYTHONPATH` and thread through the new env vars**

Find:
```bash
conda activate openvla_libero_merged
export MUJOCO_GL=egl
export PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$ROOT/.worktrees/flow-guidance-tier-gt/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline"
```
Change to:
```bash
conda activate openvla_libero_merged
export MUJOCO_GL=egl
export PYTHONPATH="$ROOT/SafeLIBERO/safelibero:$ROOT/.worktrees/flow-guidance-tier-gt/client_deps:$WT/openpi_guided/packages/openpi-client/src:$WT/vlm_pipeline:$WT"
```

Find:
```bash
TASK_INDICES=${TASK_INDICES:-}
LEVELS=${LEVELS:-"I II"}
```
Change to:
```bash
TASK_INDICES=${TASK_INDICES:-}
LEVELS=${LEVELS:-"I II"}
LOG_TRAINING_DATA=${LOG_TRAINING_DATA:-}
CRITIC_CHECKPOINT=${CRITIC_CHECKPOINT:-}
```

Find:
```bash
  python -u "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
    --task_suite_name safelibero_spatial --safety_level "$LEVEL" \
    --num_trials_per_task "$TRIALS" --host 127.0.0.1 --port "$PORT" \
    --steering_mode "$MODE" --num_candidates "$K" \
    --controller_scale "$TRANSLATION_SCALE" \
    --progress_weight "$PROGRESS_WEIGHT" --clearance_weight "$CLEARANCE_WEIGHT" \
    --correction_penalty "$CORRECTION_PENALTY" --adaptive_distance "$ADAPTIVE_DISTANCE" \
    --oracle_horizon "$ORACLE_HORIZON" \
    --oracle_depth "$ORACLE_DEPTH" \
    "${TASK_ARGS[@]}" \
    --results_output_dir "$OUT/results"
```
Change to:
```bash
  EXTRA_ARGS=()
  if [[ -n "$LOG_TRAINING_DATA" ]]; then
    EXTRA_ARGS+=(--log_training_data "$LOG_TRAINING_DATA")
  fi
  if [[ -n "$CRITIC_CHECKPOINT" ]]; then
    EXTRA_ARGS+=(--critic_checkpoint "$CRITIC_CHECKPOINT")
  fi
  python -u "$WT/vlm_pipeline/run_guided_safelibero_pi05_eval.py" \
    --task_suite_name safelibero_spatial --safety_level "$LEVEL" \
    --num_trials_per_task "$TRIALS" --host 127.0.0.1 --port "$PORT" \
    --steering_mode "$MODE" --num_candidates "$K" \
    --controller_scale "$TRANSLATION_SCALE" \
    --progress_weight "$PROGRESS_WEIGHT" --clearance_weight "$CLEARANCE_WEIGHT" \
    --correction_penalty "$CORRECTION_PENALTY" --adaptive_distance "$ADAPTIVE_DISTANCE" \
    --oracle_horizon "$ORACLE_HORIZON" \
    --oracle_depth "$ORACLE_DEPTH" \
    "${TASK_ARGS[@]}" "${EXTRA_ARGS[@]}" \
    --results_output_dir "$OUT/results"
```

- [ ] **Step 2: Shell syntax check**

```bash
bash -n slurm/viability_tournament_smoke.slurm
```

Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add slurm/viability_tournament_smoke.slurm
git commit -m "infra: thread learned-critic data logging and checkpoint args through the tournament SLURM launcher"
```

---

### Task 10: Run the data-collection SLURM job (operational)

**Files:** none (produces data under `viability_tournament/oracle_training_data/`, not committed — add the directory to `.gitignore` if not already covered by an existing `.npz`/data pattern; check first).

- [ ] **Step 1: Check `.gitignore` coverage**

```bash
git check-ignore -v viability_tournament/oracle_training_data/test.npz || echo "NOT IGNORED"
```

If it prints `NOT IGNORED`, add a line to `.gitignore`:
```bash
echo "viability_tournament/oracle_training_data/" >> .gitignore
git add .gitignore
git commit -m "chore: ignore generated critic training data"
```

- [ ] **Step 2: Submit the data-collection job**

```bash
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/viability-steering-tournament-20260713
MODE=oracle_counterfactual TRIALS=3 K=5 TRANSLATION_SCALE=0.00523 \
  PROGRESS_WEIGHT=2.0 CLEARANCE_WEIGHT=2.0 ADAPTIVE_DISTANCE=0.10 \
  ORACLE_HORIZON=2 ORACLE_DEPTH=1 LEVELS="I II" PORT=8140 \
  LOG_TRAINING_DATA="$PWD/viability_tournament/oracle_training_data" \
  sbatch slurm/viability_tournament_smoke.slurm
```

Note the job ID printed by `sbatch`.

- [ ] **Step 3: Wait for completion and check the log for `TOURNAMENT_COMPLETE`**

```bash
tail -f /ocean/projects/cis250185p/asingal/slurm_logs/cvs_tournament_<JOBID>.out
```
(Ctrl-C once `TOURNAMENT_COMPLETE mode=oracle_counterfactual` appears, or use `squeue -j <JOBID>` to poll.) Expected wall time: comparable to the existing `oracle_counterfactual` runs already in `viability_tournament/` (several minutes per level at `TRIALS=3`), well inside the 4-hour SLURM budget.

- [ ] **Step 4: Verify collected data volume and schema**

```bash
python -c "
import pathlib, numpy as np
shards = sorted(pathlib.Path('viability_tournament/oracle_training_data').rglob('*.npz'))
print('shards:', len(shards))
total_rows = 0
for shard_path in shards:
    shard = np.load(shard_path)
    assert shard['clip_embed'].shape[1] == 512
    assert shard['candidate_features'].shape[1] == 17
    total_rows += shard['oracle_score'].shape[0]
print('total candidate rows:', total_rows)
"
```

Expected: at least a few hundred shards (up to 4 tasks × 2 levels × 3 episodes = 24 episodes) and several thousand total candidate rows. If `shards: 0`, stop and check `slurm_logs/cvs_tournament_<JOBID>.err` before proceeding — do not move to Task 11 with no data.

- [ ] **Step 5: No commit** (generated data only; already `.gitignore`d in Step 1).

---

### Task 11: Train the critic on real collected data (operational)

- [ ] **Step 1: Run training**

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate openvla_libero_merged
python semantic_cbf/train_safe_success_critic.py \
  --data_dir viability_tournament/oracle_training_data \
  --output semantic_cbf/checkpoints/safe_success_critic.pt \
  --val_fraction 0.25 --epochs 300 --lr 1e-3 --hidden_dim 256 --batch_size 256 --seed 0
```

- [ ] **Step 2: Check the offline gate**

Read the printed `val_spearman` and `val_argmax_agreement`. Gate (from the design spec): proceed to Task 12 only if `val_spearman > 0.3` and `val_argmax_agreement > 0.5`. If either fails:
- Try `--epochs 600` once (cheap, a few minutes) in case of underfitting.
- If still failing, stop here and report the metrics rather than proceeding to a live SLURM run — this is the design's explicit "don't substitute a live smoke run for checking the offline fit" gate.

- [ ] **Step 3: Commit the checkpoint and gate metrics**

```bash
git add semantic_cbf/checkpoints/safe_success_critic.pt
git commit -m "data: train safe-success critic checkpoint on oracle_counterfactual labels"
```

(If `semantic_cbf/checkpoints/` is `.gitignore`d by a pre-existing blanket checkpoints rule, check first with `git check-ignore -v`; if ignored, note the checkpoint path in the Task 14 write-up instead of committing it, and skip this step's commit.)

---

### Task 12: Learned-critic micro-smoke run (operational)

- [ ] **Step 1: Submit the smoke job**

```bash
MODE=learned_critic TRIALS=2 K=5 TRANSLATION_SCALE=0.00523 \
  PROGRESS_WEIGHT=2.0 CLEARANCE_WEIGHT=2.0 ADAPTIVE_DISTANCE=0.10 \
  ORACLE_HORIZON=2 ORACLE_DEPTH=1 LEVELS="I II" PORT=8141 \
  CRITIC_CHECKPOINT="$PWD/semantic_cbf/checkpoints/safe_success_critic.pt" \
  sbatch slurm/viability_tournament_smoke.slurm
```

- [ ] **Step 2: Wait for `TOURNAMENT_COMPLETE`, then read results**

```bash
python -c "
import json, pathlib
out_dir = pathlib.Path('viability_tournament/learned_critic_scale-0.00523_k5_p2.0_c2.0_a0.10/results/safelibero_spatial')
for level in ('I', 'II'):
    results_files = sorted((out_dir / level).glob('results_*.json'))
    latest = json.loads(results_files[-1].read_text())
    print(level, 'TSR', latest['overall_TSR'], 'CAR', latest['overall_CAR'])
"
```

- [ ] **Step 3: Apply the smoke gate**

Compare against the existing table's `best_of_k` row (L1 50.0/50.0, L2 87.5/37.5) and `oracle_counterfactual`-quality expectations. Gate: proceed to the promoted run only if CAR is not worse than `best_of_k` at both levels AND the run completed without exceptions (check `.err` log). If the gate fails, stop and report — do not silently promote a broken result.

---

### Task 13: Promoted learned-critic run (operational, gated on Task 12)

- [ ] **Step 1: Submit the promoted job**

```bash
MODE=learned_critic TRIALS=5 K=5 TRANSLATION_SCALE=0.00523 \
  PROGRESS_WEIGHT=2.0 CLEARANCE_WEIGHT=2.0 ADAPTIVE_DISTANCE=0.10 \
  ORACLE_HORIZON=2 ORACLE_DEPTH=1 LEVELS="I II" PORT=8142 \
  CRITIC_CHECKPOINT="$PWD/semantic_cbf/checkpoints/safe_success_critic.pt" \
  sbatch slurm/viability_tournament_smoke.slurm
```

- [ ] **Step 2: Wait for `TOURNAMENT_COMPLETE`, then extract TSR/CAR per level and per task**

```bash
python -c "
import json, pathlib
out_dir = pathlib.Path('viability_tournament/learned_critic_scale-0.00523_k5_p2.0_c2.0_a0.10/results/safelibero_spatial')
for level in ('I', 'II'):
    results_files = sorted((out_dir / level).glob('results_*.json'))
    latest = json.loads(results_files[-1].read_text())
    print(level, 'overall TSR', latest['overall_TSR'], 'overall CAR', latest['overall_CAR'])
    for task in latest['per_task']:
        print(' ', task['task_id'], 'TSR', task['TSR'], 'CAR', task['CAR'])
"
```

---

### Task 14: Update the tournament results table

**Files:**
- Modify: `results_tables/viability_steering_tournament.md`

- [ ] **Step 1: Append the new row(s) to the summary table**

Add a row (using the actual numbers from Task 13, and optionally the Task 12 smoke row for the record) following the existing table's format:
```markdown
| Learned safe-success critic (CLIP embed + candidate features, MLP), n=5/task | <L1 TSR>/<L1 CAR> | <L2 TSR>/<L2 CAR> | <verdict> |
```
Determine `<verdict>` from the same criteria the table already uses (compare against pi0.5 baseline, r3 full-geometry, arXiv:2607.01378, and the existing calibrated phase-adaptive hybrid row already marked "Promote"/"partial go").

- [ ] **Step 2: Add a per-task breakdown section, matching the existing "Promoted run per-task results" format**

```markdown
## Learned critic run per-task results

L1: task 0 <TSR>/<CAR>, task 1 <TSR>/<CAR>, task 2 <TSR>/<CAR>, task 3 <TSR>/<CAR>.

L2: task 0 <TSR>/<CAR>, task 1 <TSR>/<CAR>, task 2 <TSR>/<CAR>, task 3 <TSR>/<CAR>.
```

- [ ] **Step 3: Update the Interpretation section with a final recommendation**

Append a paragraph (do not delete the existing analysis — it's still accurate background) stating: whether the learned critic closes the gap the existing analysis identified (geometrically-safe-but-task-inviable selections), how it compares to `oracle_counterfactual`'s selection quality at a fraction of the per-step cost, and a final recommended method across the whole tournament (existing rows + this one) for carrying forward into the broader SOTA push.

- [ ] **Step 4: Commit**

```bash
git add results_tables/viability_steering_tournament.md
git commit -m "docs: complete viability steering tournament with learned safe-success critic results"
```

---

## Self-Review Notes

- **Spec coverage:** §1 (data logging) → Tasks 5, 6, 8; §2 (data collection run) → Task 10; §3 (critic) → Tasks 1-4, 7; §4 (learned_critic mode) → Task 8; §5 (tournament completion) → Tasks 9, 11-14. All spec sections have a task.
- **Backward compatibility:** Task 6's `_oracle_counterfactual_score` extension is additive-only (default `return_diagnostics=False` preserves the exact existing return type for all five current call sites); Task 8 only adds new `elif`/gate branches, never changes an existing mode's code path.
- **Type/signature consistency check:** `extract_candidate_features` (Task 2) signature is used identically in Task 5's docstring context, Task 4's `select_candidate`, and Task 8's Step 6/7 wiring — same five positional arguments in the same order throughout. `SafeSuccessCritic.forward(clip_embed, candidate_features)` matches `score_candidates`'s tensor construction in Task 3 and the mocked call in Task 4's test. `TrainingDataLogger.log_step` signature matches its Task 8 Step 6 call site (`step_embed, step_features, scores, collided_list=..., done_list=...`).
