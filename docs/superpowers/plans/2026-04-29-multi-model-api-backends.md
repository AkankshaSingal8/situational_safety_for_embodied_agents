# Multi-Model API Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `vlm_prompt_runner/` to load API keys from `.env`, add OpenAI (GPT) and Google Gemini backends, and add a multi-model experiment runner that compares accuracy across all specified models and prompts in a single command.

**Architecture:** Add a `config.py` module that loads `.env` on import (using `python-dotenv`), create `backends/openai_api.py` and `backends/gemini_api.py` following the existing `VLMBackend` interface, register them in the backend factory with `gpt-*` and `gemini-*` prefixes, then add `run_multi_model_experiment.py` that loops over `(model, prompt)` pairs and outputs a comparison table.

**Tech Stack:** `python-dotenv`, `openai` SDK, `google-generativeai` SDK, existing `anthropic` SDK, all following the established `VLMBackend` ABC.

---

## Model Key Reference

The backend factory dispatches by prefix. The key you pass on the CLI **is** the model ID sent to the API (except where noted below):

| CLI key | Provider | Actual API model string |
|---|---|---|
| `gpt-4.5-preview` | OpenAI | `gpt-4.5-preview` |
| `gpt-4.1` | OpenAI | `gpt-4.1` |
| `gpt-4.1-mini` | OpenAI | `gpt-4.1-mini` |
| `claude-sonnet-4-6` | Anthropic | `claude-sonnet-4-6` (already works) |
| `claude-opus-4-7` | Anthropic | `claude-opus-4-7` (already works) |
| `gemini-2.5-pro` | Google | `gemini-2.5-pro` |
| `gemini-2.0-flash` | Google | `gemini-2.0-flash` |

> **Note:** OpenAI and Google model names change frequently. If the user specified "gpt 5.5 / gpt 5.4 / gpt-5-mini / gemini 3.1 / gemini 3 flash" use the actual model IDs that the respective API accepts — set them in `.env` as `OPENAI_DEFAULT_MODEL` / `GEMINI_DEFAULT_MODEL` or just pass the right key on the CLI. The key is passed verbatim to the API.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| **Create** | `vlm_prompt_runner/config.py` | Load `.env` / `.envfile` at import time, expose `load_dotenv()` helper |
| **Create** | `.env.example` | Template showing required keys |
| **Create** | `vlm_prompt_runner/backends/openai_api.py` | OpenAI vision backend (GPT-4o, 4.5, 4.1, 4.1-mini, etc.) |
| **Create** | `vlm_prompt_runner/backends/gemini_api.py` | Google Gemini vision backend |
| **Modify** | `vlm_prompt_runner/backends/__init__.py` | Add `gpt-*` and `gemini-*` dispatch |
| **Modify** | `vlm_prompt_runner/main.py` | Import `config` at startup; update help text |
| **Modify** | `vlm_prompt_runner/run_prompt_experiment.py` | Import `config` at startup; update help text |
| **Create** | `vlm_prompt_runner/run_multi_model_experiment.py` | Outer loop over models × prompts; writes per-model JSON + comparison table |
| **Create** | `vlm_prompt_runner/tests/test_openai_backend.py` | Unit tests (mocked) for OpenAI backend |
| **Create** | `vlm_prompt_runner/tests/test_gemini_backend.py` | Unit tests (mocked) for Gemini backend |
| **Create** | `vlm_prompt_runner/tests/test_config.py` | Unit tests for .env loading |

---

## Task 1: .env Loading Module

**Files:**
- Create: `vlm_prompt_runner/config.py`
- Create: `.env.example`
- Create: `vlm_prompt_runner/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# vlm_prompt_runner/tests/test_config.py
import os
import tempfile
from pathlib import Path
import pytest


def test_load_dotenv_sets_vars(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=test-openai-key\nGOOGLE_API_KEY=test-google-key\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    from vlm_prompt_runner.config import load_env
    load_env(env_path=str(env_file))

    assert os.environ["OPENAI_API_KEY"] == "test-openai-key"
    assert os.environ["GOOGLE_API_KEY"] == "test-google-key"


def test_load_dotenv_does_not_override_existing(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-file\n")
    monkeypatch.setenv("OPENAI_API_KEY", "already-set")

    from vlm_prompt_runner.config import load_env
    load_env(env_path=str(env_file))

    assert os.environ["OPENAI_API_KEY"] == "already-set"


def test_load_dotenv_missing_file_is_silent(tmp_path):
    from vlm_prompt_runner.config import load_env
    load_env(env_path=str(tmp_path / "nonexistent.env"))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
python -m pytest vlm_prompt_runner/tests/test_config.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `config` module doesn't exist yet.

- [ ] **Step 3: Install python-dotenv**

```bash
pip install python-dotenv
```

- [ ] **Step 4: Create `vlm_prompt_runner/config.py`**

```python
"""
vlm_prompt_runner/config.py

Load API keys from .env or .envfile at import time.
Searches for .env / .envfile in the project root (two levels up from this file).
Does NOT override variables already set in the environment.
"""
from pathlib import Path


def load_env(env_path: str | None = None) -> None:
    """Load key=value pairs from env_path into os.environ (no-override)."""
    try:
        from dotenv import load_dotenv
        _use_dotenv = True
    except ImportError:
        _use_dotenv = False

    candidates = (
        [Path(env_path)] if env_path
        else [
            Path(__file__).resolve().parents[1] / ".env",
            Path(__file__).resolve().parents[1] / ".envfile",
        ]
    )

    for path in candidates:
        if path.exists():
            if _use_dotenv:
                load_dotenv(dotenv_path=path, override=False)
            else:
                _manual_load(path)
            break


def _manual_load(path: Path) -> None:
    import os
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


# Load automatically on import so backends pick up keys without extra wiring.
load_env()
```

- [ ] **Step 5: Create `.env.example`**

```bash
cat > /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.env.example << 'EOF'
# Copy this file to .env and fill in your API keys.
# Keys already set in the environment take precedence over values here.

# Anthropic (Claude models)
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI (GPT models)
OPENAI_API_KEY=sk-...

# Google (Gemini models)
GOOGLE_API_KEY=AIza...
EOF
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest vlm_prompt_runner/tests/test_config.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add vlm_prompt_runner/config.py vlm_prompt_runner/tests/test_config.py .env.example
git commit -m "feat: add .env loading via config.py with manual fallback"
```

---

## Task 2: OpenAI Vision Backend

**Files:**
- Create: `vlm_prompt_runner/backends/openai_api.py`
- Create: `vlm_prompt_runner/tests/test_openai_backend.py`

- [ ] **Step 1: Write the failing test**

```python
# vlm_prompt_runner/tests/test_openai_backend.py
import base64
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def fake_image(tmp_path):
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG fake")
    return str(img)


def _make_backend(model_id="gpt-4o", api_key="test-key"):
    with patch.dict("os.environ", {"OPENAI_API_KEY": api_key}):
        from vlm_prompt_runner.backends.openai_api import OpenAIBackend
        return OpenAIBackend(model_id=model_id, api_key=api_key)


def test_generate_sends_images_and_prompt(fake_image):
    backend = _make_backend()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"obstacle": "moka_pot"}'

    with patch.object(backend._client.chat.completions, "create",
                      return_value=mock_response) as mock_create:
        result = backend.generate("Identify the obstacle.", [fake_image], max_new_tokens=256)

    assert result == '{"obstacle": "moka_pot"}'
    call_kwargs = mock_create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    # one image block + one text block
    assert any(b.get("type") == "image_url" for b in content)
    assert any(b.get("type") == "text" for b in content)
    assert call_kwargs["max_tokens"] == 256


def test_generate_skips_missing_images(tmp_path):
    backend = _make_backend()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"

    with patch.object(backend._client.chat.completions, "create",
                      return_value=mock_response) as mock_create:
        result = backend.generate("prompt", [str(tmp_path / "missing.png")])

    content = mock_create.call_args.kwargs["messages"][0]["content"]
    image_blocks = [b for b in content if b.get("type") == "image_url"]
    assert len(image_blocks) == 0


def test_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from vlm_prompt_runner.backends.openai_api import OpenAIBackend
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIBackend(model_id="gpt-4o")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest vlm_prompt_runner/tests/test_openai_backend.py -v
```

Expected: `ImportError` — `openai_api.py` doesn't exist yet.

- [ ] **Step 3: Install openai SDK**

```bash
pip install openai
```

- [ ] **Step 4: Create `vlm_prompt_runner/backends/openai_api.py`**

```python
import base64
import logging
import os
from pathlib import Path

from vlm_prompt_runner.backends.base import VLMBackend

logger = logging.getLogger(__name__)


class OpenAIBackend(VLMBackend):
    """Calls the OpenAI Chat Completions API with base64-encoded vision support.

    The model_id is passed verbatim to the API (e.g. 'gpt-4o', 'gpt-4.5-preview',
    'gpt-4.1', 'gpt-4.1-mini').  Reads OPENAI_API_KEY from the environment unless
    api_key is passed explicitly.
    """

    def __init__(self, model_id: str = "gpt-4o",
                 api_key: str | None = None,
                 max_new_tokens: int = 1024):
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise ImportError("Install the OpenAI SDK: pip install openai") from e

        self.model_id = model_id
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Set OPENAI_API_KEY environment variable or pass api_key="
            )
        self._default_max_tokens = max_new_tokens

        import openai as _openai
        self._client = _openai.OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, image_paths: list[str],
                 max_new_tokens: int | None = None) -> str:
        max_tokens = max_new_tokens if max_new_tokens is not None else self._default_max_tokens

        content: list[dict] = []
        for p in image_paths:
            if p and Path(p).exists():
                media_type = _infer_media_type(p)
                with open(p, "rb") as f:
                    b64 = base64.standard_b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{b64}",
                        "detail": "high",
                    },
                })
            else:
                logger.warning("Image not found, skipping: %s", p)

        content.append({"type": "text", "text": prompt})

        import time as _time

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "OpenAI API error (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1, max_retries, wait, exc,
                    )
                    _time.sleep(wait)
                else:
                    raise


def _infer_media_type(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(ext, "image/png")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest vlm_prompt_runner/tests/test_openai_backend.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add vlm_prompt_runner/backends/openai_api.py vlm_prompt_runner/tests/test_openai_backend.py
git commit -m "feat: add OpenAI vision backend for GPT models"
```

---

## Task 3: Google Gemini Vision Backend

**Files:**
- Create: `vlm_prompt_runner/backends/gemini_api.py`
- Create: `vlm_prompt_runner/tests/test_gemini_backend.py`

- [ ] **Step 1: Write the failing test**

```python
# vlm_prompt_runner/tests/test_gemini_backend.py
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def fake_image(tmp_path):
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG fake")
    return str(img)


def _make_backend(model_id="gemini-2.0-flash", api_key="test-key"):
    with patch.dict("os.environ", {"GOOGLE_API_KEY": api_key}):
        from vlm_prompt_runner.backends.gemini_api import GeminiBackend
        return GeminiBackend(model_id=model_id, api_key=api_key)


def test_generate_sends_images_and_prompt(fake_image):
    backend = _make_backend()

    mock_response = MagicMock()
    mock_response.text = '{"obstacle": "bowl"}'

    with patch.object(backend._model, "generate_content",
                      return_value=mock_response) as mock_gen:
        result = backend.generate("Identify the obstacle.", [fake_image], max_new_tokens=512)

    assert result == '{"obstacle": "bowl"}'
    args = mock_gen.call_args.args[0]  # positional: parts list
    # should contain at least one non-string (image Part) and the prompt string
    assert any(isinstance(a, str) or hasattr(a, "data") for a in args)


def test_generate_skips_missing_images(tmp_path):
    backend = _make_backend()
    mock_response = MagicMock()
    mock_response.text = "ok"

    with patch.object(backend._model, "generate_content",
                      return_value=mock_response):
        result = backend.generate("prompt", [str(tmp_path / "missing.png")])

    assert result == "ok"


def test_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from vlm_prompt_runner.backends.gemini_api import GeminiBackend
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        GeminiBackend(model_id="gemini-2.0-flash")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest vlm_prompt_runner/tests/test_gemini_backend.py -v
```

Expected: `ImportError` — `gemini_api.py` doesn't exist yet.

- [ ] **Step 3: Install google-generativeai SDK**

```bash
pip install google-generativeai
```

- [ ] **Step 4: Create `vlm_prompt_runner/backends/gemini_api.py`**

```python
import logging
import os
from pathlib import Path

from vlm_prompt_runner.backends.base import VLMBackend

logger = logging.getLogger(__name__)


class GeminiBackend(VLMBackend):
    """Calls the Google Generative AI (Gemini) API with image support.

    The model_id is passed verbatim to the API (e.g. 'gemini-2.5-pro',
    'gemini-2.0-flash').  Reads GOOGLE_API_KEY from the environment unless
    api_key is passed explicitly.
    """

    def __init__(self, model_id: str = "gemini-2.0-flash",
                 api_key: str | None = None,
                 max_new_tokens: int = 1024):
        try:
            import google.generativeai as genai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Install the Google AI SDK: pip install google-generativeai"
            ) from e

        self.model_id = model_id
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Set GOOGLE_API_KEY environment variable or pass api_key="
            )
        self._default_max_tokens = max_new_tokens

        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(model_id)

    def generate(self, prompt: str, image_paths: list[str],
                 max_new_tokens: int | None = None) -> str:
        max_tokens = max_new_tokens if max_new_tokens is not None else self._default_max_tokens

        parts: list = []
        for p in image_paths:
            if p and Path(p).exists():
                media_type = _infer_media_type(p)
                with open(p, "rb") as f:
                    data = f.read()
                parts.append({"mime_type": media_type, "data": data})
            else:
                logger.warning("Image not found, skipping: %s", p)

        parts.append(prompt)

        generation_config = self._genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
        )

        import time as _time

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._model.generate_content(
                    parts,
                    generation_config=generation_config,
                )
                return response.text.strip()
            except Exception as exc:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Gemini API error (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1, max_retries, wait, exc,
                    )
                    _time.sleep(wait)
                else:
                    raise


def _infer_media_type(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(ext, "image/png")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest vlm_prompt_runner/tests/test_gemini_backend.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add vlm_prompt_runner/backends/gemini_api.py vlm_prompt_runner/tests/test_gemini_backend.py
git commit -m "feat: add Google Gemini vision backend"
```

---

## Task 4: Register New Backends in Factory + Load .env in Entry Points

**Files:**
- Modify: `vlm_prompt_runner/backends/__init__.py`
- Modify: `vlm_prompt_runner/main.py`
- Modify: `vlm_prompt_runner/run_prompt_experiment.py`

- [ ] **Step 1: Update `backends/__init__.py`**

Replace the entire file with:

```python
from vlm_prompt_runner.backends.base import VLMBackend


def get_backend(model_key: str, **kwargs) -> VLMBackend:
    """Factory: return the right backend for a model key.

    Dispatch rules (by prefix):
      qwen*      -> QwenLocalBackend   (local inference)
      claude-*   -> AnthropicBackend   (Anthropic API)
      gpt-*      -> OpenAIBackend      (OpenAI API)
      gemini-*   -> GeminiBackend      (Google Generative AI API)
      dry-run    -> DryRunBackend

    The model_key is passed verbatim to the API as the model ID, so use the
    exact model string the provider accepts (e.g. 'gpt-4.5-preview',
    'gemini-2.5-pro', 'claude-sonnet-4-6').
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
    if model_key.startswith("gpt"):
        from vlm_prompt_runner.backends.openai_api import OpenAIBackend
        return OpenAIBackend(model_id=model_key, **kwargs)
    if model_key.startswith("gemini"):
        from vlm_prompt_runner.backends.gemini_api import GeminiBackend
        return GeminiBackend(model_id=model_key, **kwargs)
    raise ValueError(
        f"Unknown model key: {model_key!r}. "
        "Supported prefixes: qwen*, claude-*, gpt-*, gemini-*, dry-run"
    )
```

- [ ] **Step 2: Add `import vlm_prompt_runner.config` to `main.py`**

Add this line immediately after the standard library imports (before `from vlm_prompt_runner` imports), at the top of the `main()` function body — or as a top-level import. The simplest change: add one import line at line 10 (after `from pathlib import Path`):

In `vlm_prompt_runner/main.py`, insert after `from pathlib import Path`:

```python
import vlm_prompt_runner.config  # loads .env / .envfile on import
```

Also update the `--vlm` help string on line 63 to include the new keys:

```python
    p.add_argument("--vlm", required=True,
                   help=("VLM key: qwen2.5-vl-7b | qwen2.5-vl-3b | qwen2-vl-7b | "
                         "qwen3-vl-8b | claude-sonnet-4-6 | claude-opus-4-7 | "
                         "gpt-4.5-preview | gpt-4.1 | gpt-4.1-mini | "
                         "gemini-2.5-pro | gemini-2.0-flash | dry-run"))
```

- [ ] **Step 3: Add `import vlm_prompt_runner.config` to `run_prompt_experiment.py`**

In `vlm_prompt_runner/run_prompt_experiment.py`, insert after `from pathlib import Path` (line 25):

```python
import vlm_prompt_runner.config  # loads .env / .envfile on import
```

Also update the `--model` help string on line 47:

```python
    p.add_argument("--model", required=True,
                   help=("VLM model key: qwen3-vl-8b | qwen2.5-vl-7b | qwen2.5-vl-3b | "
                         "claude-sonnet-4-6 | claude-opus-4-7 | "
                         "gpt-4.5-preview | gpt-4.1 | gpt-4.1-mini | "
                         "gemini-2.5-pro | gemini-2.0-flash | dry-run"))
```

- [ ] **Step 4: Verify backend factory test**

```bash
python -m pytest vlm_prompt_runner/tests/ -v -k "not test_openai and not test_gemini"
```

Expected: all existing tests PASS (no regressions).

- [ ] **Step 5: Smoke-test backend factory with dry-run**

```bash
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
python -c "
from vlm_prompt_runner.backends import get_backend
b = get_backend('dry-run')
print(b.generate('test', []))
"
```

Expected: `{"dry_run": true, "note": "placeholder response"}`

- [ ] **Step 6: Commit**

```bash
git add vlm_prompt_runner/backends/__init__.py vlm_prompt_runner/main.py vlm_prompt_runner/run_prompt_experiment.py
git commit -m "feat: register gpt-* and gemini-* backends; auto-load .env in entry points"
```

---

## Task 5: Multi-Model Experiment Runner

**Files:**
- Create: `vlm_prompt_runner/run_multi_model_experiment.py`

This script is the main deliverable: one command to compare N models × M prompts and get a cross-model accuracy table.

- [ ] **Step 1: Create `vlm_prompt_runner/run_multi_model_experiment.py`**

```python
#!/usr/bin/env python3
"""
Run all combinations of models × prompts, compute per-combo accuracy,
and output a comparison table.

Example — compare 3 API models on all prompts:
  python -m vlm_prompt_runner.run_multi_model_experiment \\
      --prompts-dir prompts/obstacle_id \\
      --models claude-sonnet-4-6 gpt-4.1 gemini-2.0-flash \\
      --suite safelibero_spatial --level I --task 0 \\
      --results-out vlm_prompt_runner/results/multimodel_comparison.json

Example — specific prompts only:
  python -m vlm_prompt_runner.run_multi_model_experiment \\
      --prompts-dir prompts/obstacle_id \\
      --prompts p3_candidate_list p5_cot_structured \\
      --models claude-sonnet-4-6 gpt-4.5-preview gemini-2.5-pro \\
      --suite safelibero_spatial --level I --task 0 \\
      --results-out vlm_prompt_runner/results/multimodel_phase2.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import vlm_prompt_runner.config  # loads .env / .envfile on import

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_BASE = _PROJECT_ROOT / "vlm_inputs"


def _model_slug(model_key: str) -> str:
    """Convert model key to filesystem-safe directory name."""
    return model_key.replace(".", "_").replace("-", "_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare multiple VLM models across prompts; report accuracy table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--prompts-dir", required=True,
                   help="Directory containing .md prompt files")
    p.add_argument("--models", nargs="+", required=True,
                   help=(
                       "One or more model keys: "
                       "claude-sonnet-4-6 | claude-opus-4-7 | "
                       "gpt-4.5-preview | gpt-4.1 | gpt-4.1-mini | "
                       "gemini-2.5-pro | gemini-2.0-flash | "
                       "qwen3-vl-8b | dry-run"
                   ))
    p.add_argument("--suite", required=True)
    p.add_argument("--level", required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--prompts", nargs="*", default=None,
                   help="Specific prompt stems to run (default: all *.md in --prompts-dir)")
    p.add_argument("--results-out", required=True,
                   help="Path to write the JSON results summary")
    p.add_argument("--input-base", default=str(DEFAULT_INPUT_BASE))
    p.add_argument("--output-base", default=str(_PROJECT_ROOT / "vlm_prompt_runner" / "outputs"),
                   help="Root for per-model output dirs (model slug appended automatically)")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="4-bit quantization for Qwen local models")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip episodes whose output.json already exists (default: True)")
    return p.parse_args()


def run_model(
    model_key: str,
    prompt_files: list[Path],
    episode_dirs: list[Path],
    args: argparse.Namespace,
    vlm_inputs_dir: Path,
) -> dict:
    """Run all prompts for one model. Returns {prompt_stem: {correct, total, accuracy}}."""
    from vlm_prompt_runner.accuracy import compute_accuracy
    from vlm_prompt_runner.backends import get_backend
    from vlm_prompt_runner.episode import output_path
    from vlm_prompt_runner.prompt_loader import load_prompt
    from vlm_prompt_runner.runner import run_episode

    slug = _model_slug(model_key)
    model_output_base = Path(args.output_base) / slug

    backend_kwargs: dict = {}
    if model_key.startswith("qwen"):
        backend_kwargs = {"load_in_4bit": args.load_in_4bit}

    logger.info("=== Model: %s ===", model_key)
    backend = get_backend(model_key, **backend_kwargs)

    model_results: dict = {}

    for prompt_file in prompt_files:
        stem = prompt_file.stem
        logger.info("  Prompt: %s", stem)
        prompt_content = load_prompt(prompt_file)

        for ep_dir in episode_dirs:
            ep_idx = int(ep_dir.name.removeprefix("episode_"))
            out = output_path(
                output_base=model_output_base,
                prompt_stem=stem,
                suite=args.suite,
                level=args.level,
                task_id=args.task,
                ep_idx=ep_idx,
            )
            if args.skip_existing and out.exists():
                logger.debug("    Skipping %s (output exists)", ep_dir.name)
                continue
            try:
                run_episode(
                    ep_dir=ep_dir,
                    system_prompt=prompt_content,
                    backend=backend,
                    out_path=out,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as exc:
                logger.error("    Failed %s: %s", ep_dir.name, exc, exc_info=True)

        stl_dir = model_output_base / stem
        acc = compute_accuracy(stl_dir, vlm_inputs_dir)
        overall = acc.get("_totals", {}).get("overall", {})
        model_results[stem] = {
            "correct": overall.get("correct", 0),
            "total": overall.get("total", 0),
            "accuracy": overall.get("accuracy", 0.0),
        }
        logger.info("    %s  →  %d/%d  (%.1f%%)",
                    stem,
                    model_results[stem]["correct"],
                    model_results[stem]["total"],
                    model_results[stem]["accuracy"] * 100)

    return model_results


def print_comparison_table(summary: dict) -> None:
    models = list(summary["models"].keys())
    all_prompts = sorted({
        p for m in models for p in summary["models"][m]
    })

    col_w = max(12, *(len(m) for m in models))
    prompt_w = max(30, *(len(p) for p in all_prompts))

    header = f"{'Prompt':<{prompt_w}}" + "".join(f"  {m:>{col_w}}" for m in models)
    print()
    print("=== Multi-Model Accuracy Comparison ===")
    print(f"Suite: {summary['suite']}  |  Level: {summary['level']}  |  Task: {summary['task']}")
    print(header)
    print("-" * len(header))

    for prompt in all_prompts:
        row = f"{prompt:<{prompt_w}}"
        for model in models:
            stats = summary["models"][model].get(prompt)
            if stats:
                row += f"  {stats['accuracy']:>{col_w}.1%}"
            else:
                row += f"  {'N/A':>{col_w}}"
        print(row)

    # Best model per prompt
    print()
    print("Best model per prompt:")
    for prompt in all_prompts:
        best_model = max(
            models,
            key=lambda m: summary["models"][m].get(prompt, {}).get("accuracy", -1),
        )
        best_acc = summary["models"][best_model].get(prompt, {}).get("accuracy", 0)
        print(f"  {prompt:<{prompt_w}}  {best_model}  ({best_acc:.1%})")
    print()


def main() -> None:
    args = parse_args()

    from vlm_prompt_runner.episode import resolve_episodes
    from vlm_prompt_runner.prompt_loader import load_prompt  # noqa: F401 — verify import

    prompts_dir = Path(args.prompts_dir)
    all_prompt_files = sorted(
        f for f in prompts_dir.glob("*.md")
        if f.stem.lower() != "readme"
    )
    if args.prompts:
        all_prompt_files = [f for f in all_prompt_files if f.stem in args.prompts]
    if not all_prompt_files:
        logger.error("No prompt files found in %s (filter=%s)", prompts_dir, args.prompts)
        sys.exit(1)

    episode_dirs = resolve_episodes(
        input_base=Path(args.input_base),
        suite=args.suite,
        level=args.level,
        task_id=args.task,
        episodes=None,
    )
    logger.info("Episodes: %d  |  Prompts: %d  |  Models: %s",
                len(episode_dirs), len(all_prompt_files), args.models)

    summary: dict = {
        "suite": args.suite,
        "level": args.level,
        "task": args.task,
        "models": {},
    }

    vlm_inputs_dir = Path(args.input_base)

    for model_key in args.models:
        summary["models"][model_key] = run_model(
            model_key=model_key,
            prompt_files=all_prompt_files,
            episode_dirs=episode_dirs,
            args=args,
            vlm_inputs_dir=vlm_inputs_dir,
        )

    Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results_out).write_text(json.dumps(summary, indent=2))
    logger.info("Results saved to %s", args.results_out)

    print_comparison_table(summary)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test with dry-run**

```bash
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
python -m vlm_prompt_runner.run_multi_model_experiment \
    --prompts-dir prompts/obstacle_id \
    --models dry-run \
    --suite safelibero_spatial --level I --task 0 \
    --results-out /tmp/test_multimodel.json
```

Expected: runs without error, prints a comparison table, writes JSON to `/tmp/test_multimodel.json`.

- [ ] **Step 3: Verify JSON output structure**

```bash
python -c "
import json
d = json.loads(open('/tmp/test_multimodel.json').read())
print('keys:', list(d.keys()))
print('models:', list(d['models'].keys()))
print('dry-run prompts:', list(d['models']['dry-run'].keys()))
"
```

Expected output structure:
```
keys: ['suite', 'level', 'task', 'models']
models: ['dry-run']
dry-run prompts: ['p1_vision_only', 'p2_task_roles', ...]
```

- [ ] **Step 4: Commit**

```bash
git add vlm_prompt_runner/run_multi_model_experiment.py
git commit -m "feat: add run_multi_model_experiment.py for cross-model accuracy comparison"
```

---

## Task 6: Create .env from .env.example + Full Test Suite

**Files:**
- Create: `.env` (user action — not committed)
- Run: full test suite

- [ ] **Step 1: Copy .env.example to .env and fill in keys**

```bash
cp /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.env.example \
   /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.env
```

Then edit `.env` to add actual keys:
```
ANTHROPIC_API_KEY=<your-anthropic-key>
OPENAI_API_KEY=<your-openai-key>
GOOGLE_API_KEY=<your-google-key>
```

- [ ] **Step 2: Add .env to .gitignore**

```bash
echo ".env" >> /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.gitignore
echo ".envfile" >> /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.gitignore
```

- [ ] **Step 3: Run full test suite**

```bash
cd /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents
python -m pytest vlm_prompt_runner/tests/ -v
```

Expected: all tests PASS (existing + new config, openai, gemini tests).

- [ ] **Step 4: Commit .gitignore update**

```bash
git add .gitignore
git commit -m "chore: ignore .env and .envfile"
```

---

## Example Commands After Implementation

### Single API model, all prompts:
```bash
python -m vlm_prompt_runner.run_prompt_experiment \
    --prompts-dir prompts/obstacle_id \
    --model gpt-4.1 \
    --suite safelibero_spatial --level I --task 0 \
    --results-out vlm_prompt_runner/results/phase_gpt41.json
```

### All API models comparison:
```bash
python -m vlm_prompt_runner.run_multi_model_experiment \
    --prompts-dir prompts/obstacle_id \
    --models claude-sonnet-4-6 claude-opus-4-7 gpt-4.5-preview gpt-4.1 gpt-4.1-mini gemini-2.5-pro gemini-2.0-flash \
    --suite safelibero_spatial --level I --task 0 \
    --results-out vlm_prompt_runner/results/multimodel_all.json
```

### Specific prompts, specific models:
```bash
python -m vlm_prompt_runner.run_multi_model_experiment \
    --prompts-dir prompts/obstacle_id \
    --prompts p3_candidate_list p5_cot_structured \
    --models claude-opus-4-7 gpt-4.1 gemini-2.5-pro \
    --suite safelibero_spatial --level I --task 0 \
    --results-out vlm_prompt_runner/results/multimodel_top_prompts.json
```

---

## Self-Review

**Spec coverage check:**
- ✅ API keys from .env / .envfile → Task 1 (`config.py`)
- ✅ GPT models (gpt-4.5-preview, gpt-4.1, gpt-4.1-mini) → Task 2 + 4
- ✅ Claude models (sonnet-4-6, opus-4-7) → already existed; Task 4 wires in .env load
- ✅ Gemini models (gemini-2.5-pro, gemini-2.0-flash) → Task 3 + 4
- ✅ Run all or specified prompts → `--prompts` filter in Task 5
- ✅ Get accuracy → `compute_accuracy` called after each (model, prompt) run in Task 5

**Placeholder check:** No TBD or TODO markers — all code blocks are complete.

**Type consistency:** `output_path`, `resolve_episodes`, `compute_accuracy`, `run_episode` are called identically across Tasks 5 and existing code. `VLMBackend.generate(prompt, image_paths, max_new_tokens)` signature used consistently in both new backends.
