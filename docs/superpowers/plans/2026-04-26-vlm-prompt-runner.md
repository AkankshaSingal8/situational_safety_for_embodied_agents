# VLM Prompt Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone `vlm_prompt_runner/` package that runs any VLM (local Qwen or remote API) on SafeLIBERO episode images using a prompt `.md` file, writing per-episode JSON output to a mirrored folder structure.

**Architecture:** A thin main script dispatches to a swappable `VLMBackend` ABC, so adding a new model is a single new file. Episode discovery, prompt loading, and output path logic are pure functions tested in isolation. The bash launcher wraps the main script with sensible defaults for the HPC environment.

**Tech Stack:** Python 3.10+, `transformers` (Qwen local), `anthropic` SDK (Claude API), `Pillow`, `pytest`

---

## File Structure

```
vlm_prompt_runner/
├── backends/
│   ├── __init__.py          # exports get_backend()
│   ├── base.py              # VLMBackend ABC
│   ├── dry_run.py           # no-op backend for testing
│   ├── qwen_local.py        # Qwen2.5/Qwen2 via transformers
│   └── anthropic_api.py     # Claude API via anthropic SDK
├── episode.py               # load_episode(), resolve_episodes(), output_path()
├── prompt_loader.py         # load_prompt() reads .md file
├── runner.py                # run_episode() — wires backend + episode + prompt
├── main.py                  # argparse entry point
├── tests/
│   ├── __init__.py
│   ├── test_episode.py
│   ├── test_prompt_loader.py
│   └── test_runner.py
└── run_vlm.sh               # bash/SLURM launcher
```

**Output root** (sibling of `vlm_inputs/`):
`<project_root>/<prompt_stem>/<suite>/level_<level>/task_<task_id>/episode_<ep:02d>/output.json`

Example: `safety_predicates_prompt/safelibero_spatial/level_I/task_0/episode_00/output.json`

---

## Task 1: Create folder skeleton + VLMBackend ABC

**Files:**
- Create: `vlm_prompt_runner/__init__.py`
- Create: `vlm_prompt_runner/backends/__init__.py`
- Create: `vlm_prompt_runner/backends/base.py`
- Create: `vlm_prompt_runner/backends/dry_run.py`

- [ ] **Step 1: Create directory skeleton**

```bash
mkdir -p vlm_prompt_runner/backends vlm_prompt_runner/tests
touch vlm_prompt_runner/__init__.py
touch vlm_prompt_runner/tests/__init__.py
```

- [ ] **Step 2: Write `backends/base.py`**

```python
from abc import ABC, abstractmethod


class VLMBackend(ABC):
    """Minimal interface all VLM backends must implement."""

    @abstractmethod
    def generate(self, prompt: str, image_paths: list[str],
                 max_new_tokens: int = 1024) -> str:
        """Send prompt + images to the model, return raw text response."""
        ...
```

- [ ] **Step 3: Write `backends/dry_run.py`**

```python
from vlm_prompt_runner.backends.base import VLMBackend


class DryRunBackend(VLMBackend):
    """Returns a fixed JSON string. No GPU or API key needed."""

    def generate(self, prompt: str, image_paths: list[str],
                 max_new_tokens: int = 1024) -> str:
        return '{"dry_run": true, "note": "placeholder response"}'
```

- [ ] **Step 4: Write `backends/__init__.py` with `get_backend()`**

```python
from vlm_prompt_runner.backends.base import VLMBackend


def get_backend(model_key: str, **kwargs) -> VLMBackend:
    """Factory: return the right backend for a model key.

    Qwen keys  : "qwen2.5-vl-7b", "qwen2.5-vl-3b", "qwen2-vl-7b", "qwen3-vl-8b"
    Claude keys: anything starting with "claude-"
    dry-run    : "dry-run"
    """
    if model_key == "dry-run":
        from vlm_prompt_runner.backends.dry_run import DryRunBackend
        return DryRunBackend()
    if model_key.startswith("qwen"):
        from vlm_prompt_runner.backends.qwen_local import QwenLocalBackend
        return QwenLocalBackend(model_key=model_key, **kwargs)
    if model_key.startswith("claude"):
        from vlm_prompt_runner.backends.anthropic_api import AnthropicBackend
        return AnthropicBackend(model_id=model_key, **kwargs)
    raise ValueError(
        f"Unknown model key: {model_key!r}. "
        "Supported prefixes: qwen*, claude*, dry-run"
    )
```

- [ ] **Step 5: Commit**

```bash
git add vlm_prompt_runner/
git commit -m "feat(vlm-runner): add VLMBackend ABC, factory, dry-run backend"
```

---

## Task 2: Episode loader and output path resolver

**Files:**
- Create: `vlm_prompt_runner/episode.py`
- Create: `vlm_prompt_runner/tests/test_episode.py`

- [ ] **Step 1: Write the failing tests**

```python
# vlm_prompt_runner/tests/test_episode.py
import json
import tempfile
from pathlib import Path

import pytest

from vlm_prompt_runner.episode import load_episode, output_path, resolve_episodes


def _make_episode(root: Path, suite: str, level: str,
                  task_id: int, ep_idx: int) -> Path:
    ep = root / suite / f"level_{level}" / f"task_{task_id}" / f"episode_{ep_idx:02d}"
    ep.mkdir(parents=True)
    meta = {
        "task_suite": suite, "safety_level": level,
        "task_id": task_id, "episode_idx": ep_idx,
        "task_description": "pick up the red cube",
    }
    (ep / "metadata.json").write_text(json.dumps(meta))
    for name in ("agentview_rgb.png", "eye_in_hand_rgb.png", "backview_rgb.png"):
        (ep / name).touch()
    return ep


def test_load_episode():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "vlm_inputs"
        ep_dir = _make_episode(root, "safelibero_spatial", "I", 0, 0)
        ep = load_episode(ep_dir)
        assert ep["task_description"] == "pick up the red cube"
        assert Path(ep["agentview"]).name == "agentview_rgb.png"
        assert Path(ep["eye_in_hand"]).name == "eye_in_hand_rgb.png"
        assert Path(ep["backview"]).name == "backview_rgb.png"


def test_load_episode_missing_metadata_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            load_episode(Path(tmp) / "nonexistent")


def test_resolve_episodes_specific():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "vlm_inputs"
        for i in range(5):
            _make_episode(root, "safelibero_spatial", "I", 0, i)
        eps = resolve_episodes(root, "safelibero_spatial", "I", 0, episodes=[0, 2, 4])
        assert len(eps) == 3
        assert all(p.exists() for p in eps)


def test_resolve_episodes_all():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "vlm_inputs"
        for i in range(3):
            _make_episode(root, "safelibero_spatial", "I", 0, i)
        eps = resolve_episodes(root, "safelibero_spatial", "I", 0, episodes=None)
        assert len(eps) == 3


def test_output_path():
    p = output_path(
        output_base=Path("/results"),
        prompt_stem="safety_predicates_prompt",
        suite="safelibero_spatial",
        level="I",
        task_id=0,
        ep_idx=3,
    )
    assert p == Path(
        "/results/safety_predicates_prompt/safelibero_spatial"
        "/level_I/task_0/episode_03/output.json"
    )
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest vlm_prompt_runner/tests/test_episode.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'vlm_prompt_runner.episode'`

- [ ] **Step 3: Implement `episode.py`**

```python
from __future__ import annotations
import json
from pathlib import Path


def load_episode(ep_dir: Path | str) -> dict:
    """Load metadata and image paths for one episode directory."""
    ep_dir = Path(ep_dir)
    meta_path = ep_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {ep_dir}")
    with open(meta_path) as f:
        meta = json.load(f)
    return {
        "task_description": meta.get("task_description", ""),
        "metadata": meta,
        "agentview": str(ep_dir / "agentview_rgb.png"),
        "eye_in_hand": str(ep_dir / "eye_in_hand_rgb.png"),
        "backview": str(ep_dir / "backview_rgb.png"),
        "ep_dir": str(ep_dir),
    }


def resolve_episodes(input_base: Path | str, suite: str, level: str,
                     task_id: int, episodes: list[int] | None) -> list[Path]:
    """Return sorted list of episode directories.

    If episodes is None, returns all episode_* dirs for the task.
    Otherwise returns only the specified episode indices.
    """
    input_base = Path(input_base)
    task_dir = input_base / suite / f"level_{level}" / f"task_{task_id}"
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    if episodes is None:
        return sorted(task_dir.glob("episode_*"))

    paths = []
    for ep_idx in episodes:
        ep_dir = task_dir / f"episode_{ep_idx:02d}"
        if not ep_dir.exists():
            raise FileNotFoundError(f"Episode directory not found: {ep_dir}")
        paths.append(ep_dir)
    return paths


def output_path(output_base: Path | str, prompt_stem: str, suite: str,
                level: str, task_id: int, ep_idx: int) -> Path:
    """Compute the output JSON path, mirroring the input folder structure."""
    return (
        Path(output_base) / prompt_stem / suite
        / f"level_{level}" / f"task_{task_id}"
        / f"episode_{ep_idx:02d}" / "output.json"
    )
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest vlm_prompt_runner/tests/test_episode.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add vlm_prompt_runner/episode.py vlm_prompt_runner/tests/test_episode.py
git commit -m "feat(vlm-runner): add episode loader and output path resolver"
```

---

## Task 3: Prompt loader

**Files:**
- Create: `vlm_prompt_runner/prompt_loader.py`
- Create: `vlm_prompt_runner/tests/test_prompt_loader.py`

- [ ] **Step 1: Write failing tests**

```python
# vlm_prompt_runner/tests/test_prompt_loader.py
import tempfile
from pathlib import Path

import pytest

from vlm_prompt_runner.prompt_loader import load_prompt


def test_load_prompt_returns_string():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "my_prompt.md"
        p.write_text("You are a robot safety expert.\n\nDo something.")
        result = load_prompt(p)
        assert "robot safety expert" in result
        assert isinstance(result, str)


def test_load_prompt_stem():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "safety_predicates_prompt.md"
        p.write_text("content")
        content, stem = load_prompt(p, return_stem=True)
        assert stem == "safety_predicates_prompt"
        assert content == "content"


def test_load_prompt_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt(Path("/nonexistent/prompt.md"))
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest vlm_prompt_runner/tests/test_prompt_loader.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'vlm_prompt_runner.prompt_loader'`

- [ ] **Step 3: Implement `prompt_loader.py`**

```python
from pathlib import Path


def load_prompt(prompt_path: Path | str,
                return_stem: bool = False) -> str | tuple[str, str]:
    """Read a .md prompt file and return its contents.

    If return_stem=True, returns (content, stem) where stem is the
    filename without extension — used to name the output directory.
    """
    prompt_path = Path(prompt_path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    content = prompt_path.read_text(encoding="utf-8")
    if return_stem:
        return content, prompt_path.stem
    return content
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest vlm_prompt_runner/tests/test_prompt_loader.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add vlm_prompt_runner/prompt_loader.py vlm_prompt_runner/tests/test_prompt_loader.py
git commit -m "feat(vlm-runner): add prompt loader"
```

---

## Task 4: QwenLocalBackend

**Files:**
- Create: `vlm_prompt_runner/backends/qwen_local.py`

Reuses inference logic from `vlm_pipeline/qwen_vlm_worker.py` as a standalone copy — the two scripts stay independent.

- [ ] **Step 1: Write `backends/qwen_local.py`**

```python
import logging
import os
import time

from vlm_prompt_runner.backends.base import VLMBackend

logger = logging.getLogger(__name__)

QWEN_MODELS = {
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2-vl-7b":   "Qwen/Qwen2-VL-7B-Instruct",
    "qwen3-vl-8b":   "Qwen/Qwen3-VL-8B",
}


class QwenLocalBackend(VLMBackend):
    """Loads a Qwen VLM checkpoint locally via transformers and runs inference."""

    def __init__(self, model_key: str = "qwen2.5-vl-7b",
                 device: str = "auto", load_in_4bit: bool = False):
        import torch
        from transformers import AutoProcessor

        if model_key not in QWEN_MODELS:
            raise ValueError(
                f"Unknown Qwen model key: {model_key!r}. "
                f"Choose from {list(QWEN_MODELS)}"
            )

        hf_id = QWEN_MODELS[model_key]
        logger.info(f"Loading {hf_id} ...")

        if model_key.startswith("qwen2.5"):
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
        elif model_key.startswith("qwen2"):
            from transformers import Qwen2VLForConditionalGeneration as ModelCls
        else:
            from transformers import AutoModelForImageTextToText as ModelCls

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        load_kwargs: dict = dict(torch_dtype=dtype, device_map=device)

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            )

        t0 = time.time()
        self.model = ModelCls.from_pretrained(hf_id, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(hf_id)
        logger.info(f"Loaded in {time.time()-t0:.1f}s  (class={ModelCls.__name__})")

    def generate(self, prompt: str, image_paths: list[str],
                 max_new_tokens: int = 1024) -> str:
        import torch
        from PIL import Image as PILImage

        content = []
        pil_images = []
        for p in image_paths:
            if p and os.path.exists(p):
                img = PILImage.open(p).convert("RGB")
                pil_images.append(img)
                content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        if pil_images:
            inputs = self.processor(
                text=[prompt_text], images=pil_images,
                return_tensors="pt", padding=True,
            )
        else:
            inputs = self.processor(
                text=[prompt_text], return_tensors="pt", padding=True,
            )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
```

- [ ] **Step 2: Verify the import compiles (no GPU needed)**

```bash
python -c "
from vlm_prompt_runner.backends.qwen_local import QWEN_MODELS
print(list(QWEN_MODELS))
"
```

Expected: `['qwen2.5-vl-3b', 'qwen2.5-vl-7b', 'qwen2-vl-7b', 'qwen3-vl-8b']`

- [ ] **Step 3: Commit**

```bash
git add vlm_prompt_runner/backends/qwen_local.py
git commit -m "feat(vlm-runner): add QwenLocalBackend"
```

---

## Task 5: AnthropicBackend

**Files:**
- Create: `vlm_prompt_runner/backends/anthropic_api.py`

- [ ] **Step 1: Write `backends/anthropic_api.py`**

```python
import base64
import logging
import os
from pathlib import Path

from vlm_prompt_runner.backends.base import VLMBackend

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicBackend(VLMBackend):
    """Calls the Anthropic Messages API with base64-encoded image support.

    Reads ANTHROPIC_API_KEY from the environment unless api_key is passed.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL,
                 api_key: str | None = None, max_new_tokens: int = 1024):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError("Install the Anthropic SDK: pip install anthropic") from e

        self.model_id = model_id
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Set ANTHROPIC_API_KEY environment variable or pass api_key="
            )
        self._default_max_tokens = max_new_tokens

    def generate(self, prompt: str, image_paths: list[str],
                 max_new_tokens: int | None = None) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        max_tokens = max_new_tokens or self._default_max_tokens

        content: list[dict] = []
        for p in image_paths:
            if p and Path(p).exists():
                media_type = _infer_media_type(p)
                with open(p, "rb") as f:
                    b64 = base64.standard_b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    },
                })
            else:
                logger.warning(f"Image not found, skipping: {p}")

        content.append({"type": "text", "text": prompt})

        response = client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text.strip()


def _infer_media_type(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(ext, "image/png")
```

- [ ] **Step 2: Verify import compiles**

```bash
python -c "from vlm_prompt_runner.backends.anthropic_api import AnthropicBackend; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add vlm_prompt_runner/backends/anthropic_api.py
git commit -m "feat(vlm-runner): add AnthropicBackend with base64 image encoding"
```

---

## Task 6: Core runner + tests

**Files:**
- Create: `vlm_prompt_runner/runner.py`
- Create: `vlm_prompt_runner/tests/test_runner.py`

- [ ] **Step 1: Write failing tests**

```python
# vlm_prompt_runner/tests/test_runner.py
import json
import tempfile
from pathlib import Path

from vlm_prompt_runner.backends.dry_run import DryRunBackend
from vlm_prompt_runner.runner import build_prompt, extract_json, run_episode


def test_build_prompt_includes_task_and_system():
    result = build_prompt(
        system_prompt="You are a safety expert.",
        task_description="pick up the red cube",
    )
    assert "pick up the red cube" in result
    assert "You are a safety expert" in result


def test_extract_json_valid_object():
    raw = 'Some preamble {"key": "value"} trailing text'
    result = extract_json(raw)
    assert result == {"key": "value"}


def test_extract_json_valid_array():
    raw = '[{"a": 1}, {"b": 2}]'
    result = extract_json(raw)
    assert isinstance(result, list)
    assert len(result) == 2


def test_extract_json_invalid_returns_raw_wrapped():
    raw = "This is not JSON at all."
    result = extract_json(raw)
    assert result == {"raw_response": raw}


def test_run_episode_writes_output(tmp_path):
    ep_dir = tmp_path / "episode_00"
    ep_dir.mkdir()
    (ep_dir / "metadata.json").write_text(json.dumps({
        "task_description": "pick up the cube",
        "task_id": 0, "safety_level": "I",
        "task_suite": "test_suite", "episode_idx": 0,
    }))
    for name in ("agentview_rgb.png", "eye_in_hand_rgb.png", "backview_rgb.png"):
        (ep_dir / name).touch()

    out_path = tmp_path / "output.json"
    run_episode(
        ep_dir=ep_dir,
        system_prompt="You are a safety expert.",
        backend=DryRunBackend(),
        out_path=out_path,
    )
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert isinstance(data, dict)
    assert "_meta" in data
    assert data["_meta"]["task_description"] == "pick up the cube"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest vlm_prompt_runner/tests/test_runner.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'vlm_prompt_runner.runner'`

- [ ] **Step 3: Implement `runner.py`**

```python
from __future__ import annotations
import json
import logging
import re
from pathlib import Path

from vlm_prompt_runner.backends.base import VLMBackend
from vlm_prompt_runner.episode import load_episode

logger = logging.getLogger(__name__)


def build_prompt(system_prompt: str, task_description: str) -> str:
    """Combine the .md system prompt with the task description from metadata."""
    return (
        f"{system_prompt.strip()}\n\n"
        f"---\n\n"
        f"**Robot task instruction:** {task_description}"
    )


def extract_json(raw: str) -> dict | list:
    """Extract a JSON object or array from the raw VLM response.

    Falls back to wrapping the raw string if no valid JSON is found.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for pattern in (r'\{.*\}', r'\[.*\]'):
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    return {"raw_response": raw}


def run_episode(ep_dir: Path | str, system_prompt: str,
                backend: VLMBackend, out_path: Path | str,
                max_new_tokens: int = 1024) -> dict:
    """Run VLM inference for one episode and write the result to out_path."""
    ep_dir = Path(ep_dir)
    out_path = Path(out_path)

    episode = load_episode(ep_dir)
    prompt = build_prompt(system_prompt, episode["task_description"])
    image_paths = [episode["agentview"], episode["eye_in_hand"], episode["backview"]]

    logger.info(f"Running inference: {ep_dir.name}")
    raw = backend.generate(prompt, image_paths, max_new_tokens=max_new_tokens)
    result = extract_json(raw)

    if isinstance(result, dict):
        result["_meta"] = {
            "task_description": episode["task_description"],
            "ep_dir": str(ep_dir),
        }
    else:
        result = {"data": result, "_meta": {
            "task_description": episode["task_description"],
            "ep_dir": str(ep_dir),
        }}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"  → {out_path}")
    return result
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest vlm_prompt_runner/tests/test_runner.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add vlm_prompt_runner/runner.py vlm_prompt_runner/tests/test_runner.py
git commit -m "feat(vlm-runner): add runner with prompt builder and JSON extractor"
```

---

## Task 7: Main script (`main.py`)

**Files:**
- Create: `vlm_prompt_runner/main.py`

- [ ] **Step 1: Write `main.py`**

```python
#!/usr/bin/env python3
"""
vlm_prompt_runner/main.py

Run VLM inference on SafeLIBERO episode observations.

Output path:
  <output_base>/<prompt_stem>/<suite>/level_<level>/task_<task_id>/episode_<ep:02d>/output.json

Examples:

  # Dry-run single episode
  python -m vlm_prompt_runner.main \\
      --suite safelibero_spatial --level I --task 0 --episodes 0 \\
      --prompt prompts/safety_predicates_prompt.md \\
      --vlm dry-run

  # Qwen local — all episodes for task 0
  python -m vlm_prompt_runner.main \\
      --suite safelibero_spatial --level I --task 0 \\
      --prompt prompts/safelibero_prompt.md \\
      --vlm qwen2.5-vl-7b

  # Claude API — episodes 0-4
  python -m vlm_prompt_runner.main \\
      --suite safelibero_spatial --level I --task 0 --episodes 0 1 2 3 4 \\
      --prompt prompts/safelibero_prompt.md \\
      --vlm claude-sonnet-4-6
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_BASE = _PROJECT_ROOT / "vlm_inputs"
DEFAULT_OUTPUT_BASE = _PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run VLM inference on SafeLIBERO episodes with a .md prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--suite", required=True,
                   help="Task suite name, e.g. safelibero_spatial")
    p.add_argument("--level", required=True,
                   help="Safety level, e.g. I or II")
    p.add_argument("--task", type=int, required=True,
                   help="Task ID (integer), e.g. 0")
    p.add_argument("--episodes", type=int, nargs="*", default=None,
                   help="Episode indices to process. Omit to process all.")
    p.add_argument("--prompt", required=True,
                   help="Path to .md prompt file, e.g. prompts/safelibero_prompt.md")
    p.add_argument("--vlm", required=True,
                   help=("VLM key: qwen2.5-vl-7b | qwen2.5-vl-3b | qwen2-vl-7b | "
                         "qwen3-vl-8b | claude-sonnet-4-6 | claude-opus-4-7 | dry-run"))
    p.add_argument("--load_in_4bit", action="store_true",
                   help="4-bit quantization for Qwen local models")
    p.add_argument("--device", default="auto",
                   help="Device map for Qwen local (default: auto)")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--input_base", default=str(DEFAULT_INPUT_BASE),
                   help="Root directory of vlm_inputs/ "
                        f"(default: {DEFAULT_INPUT_BASE})")
    p.add_argument("--output_base", default=str(DEFAULT_OUTPUT_BASE),
                   help="Root directory for outputs "
                        f"(default: {DEFAULT_OUTPUT_BASE})")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from vlm_prompt_runner.backends import get_backend
    from vlm_prompt_runner.episode import output_path, resolve_episodes
    from vlm_prompt_runner.prompt_loader import load_prompt
    from vlm_prompt_runner.runner import run_episode

    prompt_content, prompt_stem = load_prompt(args.prompt, return_stem=True)
    logger.info(f"Prompt: {args.prompt}  (stem={prompt_stem!r})")

    episode_dirs = resolve_episodes(
        input_base=Path(args.input_base),
        suite=args.suite,
        level=args.level,
        task_id=args.task,
        episodes=args.episodes,
    )
    logger.info(
        f"Processing {len(episode_dirs)} episode(s) "
        f"[suite={args.suite}  level={args.level}  task={args.task}]"
    )

    backend_kwargs: dict = {}
    if args.vlm.startswith("qwen"):
        backend_kwargs = dict(device=args.device, load_in_4bit=args.load_in_4bit)
    backend = get_backend(args.vlm, **backend_kwargs)

    for ep_dir in episode_dirs:
        ep_idx = int(ep_dir.name.split("_")[-1])
        out = output_path(
            output_base=Path(args.output_base),
            prompt_stem=prompt_stem,
            suite=args.suite,
            level=args.level,
            task_id=args.task,
            ep_idx=ep_idx,
        )
        try:
            run_episode(
                ep_dir=ep_dir,
                system_prompt=prompt_content,
                backend=backend,
                out_path=out,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            logger.error(f"Failed on {ep_dir.name}: {e}", exc_info=True)

    logger.info("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test with dry-run**

```bash
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
python -m vlm_prompt_runner.main \
    --suite safelibero_spatial --level I --task 0 --episodes 0 1 \
    --prompt prompts/safety_predicates_prompt.md \
    --vlm dry-run \
    --output_base /tmp/vlm_runner_test
```

Expected output files:
```
/tmp/vlm_runner_test/safety_predicates_prompt/safelibero_spatial/level_I/task_0/episode_00/output.json
/tmp/vlm_runner_test/safety_predicates_prompt/safelibero_spatial/level_I/task_0/episode_01/output.json
```

Verify: `cat /tmp/vlm_runner_test/safety_predicates_prompt/safelibero_spatial/level_I/task_0/episode_00/output.json`

- [ ] **Step 3: Commit**

```bash
git add vlm_prompt_runner/main.py
git commit -m "feat(vlm-runner): add main.py argparse entry point"
```

---

## Task 8: Bash launcher script

**Files:**
- Create: `vlm_prompt_runner/run_vlm.sh`

- [ ] **Step 1: Write `run_vlm.sh`**

```bash
#!/usr/bin/env bash
# vlm_prompt_runner/run_vlm.sh
#
# Convenience launcher for VLM inference on SafeLIBERO episodes.
# Activates the correct conda env and sets required env vars.
#
# Usage (interactive node or inside a SLURM job):
#   bash vlm_prompt_runner/run_vlm.sh \
#       --suite safelibero_spatial --level I --task 0 \
#       --prompt prompts/safety_predicates_prompt.md \
#       --vlm qwen2.5-vl-7b
#
# Environment variables:
#   CONDA_ENV     conda env to activate (default: qwen_vlm)
#   PROJECT_ROOT  project root path (default: parent of this script)

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${PROJECT_ROOT:-$( dirname "$SCRIPT_DIR" )}"
CONDA_ENV="${CONDA_ENV:-qwen_vlm}"

# Activate conda if needed
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    echo "[run_vlm.sh] Activating conda env: $CONDA_ENV"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# Headless GPU rendering required on Bridges2 HPC
export MUJOCO_GL="${MUJOCO_GL:-egl}"

echo "[run_vlm.sh] PROJECT_ROOT=$PROJECT_ROOT"
echo "[run_vlm.sh] Args: $*"

cd "$PROJECT_ROOT"
python -m vlm_prompt_runner.main "$@"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x vlm_prompt_runner/run_vlm.sh
```

- [ ] **Step 3: Verify help works**

```bash
bash vlm_prompt_runner/run_vlm.sh --help 2>&1 | grep -E "(suite|level|task|prompt|vlm)"
```

Expected: lines showing `--suite`, `--level`, `--task`, `--prompt`, `--vlm` options.

- [ ] **Step 4: Commit**

```bash
git add vlm_prompt_runner/run_vlm.sh
git commit -m "feat(vlm-runner): add bash launcher with conda activation"
```

---

## Task 9: Full test suite + integration check

- [ ] **Step 1: Run all unit tests**

```bash
python -m pytest vlm_prompt_runner/tests/ -v
```

Expected: ≥13 tests, all PASS

- [ ] **Step 2: Dry-run across all prompts**

```bash
for PROMPT in prompts/*.md; do
    STEM=$(basename "$PROMPT" .md)
    echo "=== $STEM ==="
    python -m vlm_prompt_runner.main \
        --suite safelibero_spatial --level I --task 0 --episodes 0 \
        --prompt "$PROMPT" \
        --vlm dry-run \
        --output_base /tmp/vlm_runner_allprompts
    cat "/tmp/vlm_runner_allprompts/${STEM}/safelibero_spatial/level_I/task_0/episode_00/output.json"
    echo
done
```

Expected: one `output.json` per prompt, each containing `{"dry_run": true, ..., "_meta": {...}}`.

- [ ] **Step 3: Final commit**

```bash
git add vlm_prompt_runner/
git commit -m "feat(vlm-runner): complete vlm_prompt_runner package with all tests passing"
```

---

## Self-Review: Spec Coverage

| Requirement | Task |
|---|---|
| New script takes vlm_inputs as input | Task 2 `episode.py`, Task 7 `--input_base` |
| Output mirrors input folder structure as `qwen_{prompt_stem}/...` | Task 2 `output_path()` |
| Input args: suite, level, task, episodes | Task 7 `main.py` argparse |
| Prompt from `.md` file in `prompts/` | Task 3 `prompt_loader.py`, Task 7 `--prompt` |
| 3 views (agentview, eye_in_hand, backview) + task description as VLM input | Task 6 `runner.py` |
| Easy to swap VLM/API backends | Task 1 `VLMBackend` ABC + factory |
| VLM model as CLI argument | Task 7 `--vlm` |
| New folder with python + bash | `vlm_prompt_runner/` directory |
| Do not modify existing scripts | All code in new `vlm_prompt_runner/` |
