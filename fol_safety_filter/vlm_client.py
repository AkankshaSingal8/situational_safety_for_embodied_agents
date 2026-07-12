"""
FOL Safety Filter v6 — Backend-Agnostic VLM Client

Routes to:
  1. Local Qwen server (QWEN_SERVER_URL set) — /infer_v6 endpoint
  2. Anthropic Claude API (ANTHROPIC_API_KEY set, no Qwen server)
  3. OpenAI API (OPENAI_API_KEY set, no Qwen or Anthropic)

Priority: Qwen server > Anthropic > OpenAI
Falls back to mock response if no backend is available.

Each call: infer(prompt, images, max_tokens) -> str
"""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Env-var controlled model overrides
_DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # cheapest claude vision
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class VLMClient:
    """
    Single interface for VLM inference across backends.

    Usage:
        client = VLMClient()
        text = client.infer("Describe this scene.", [rgb_array1, rgb_array2])
    """

    def __init__(self):
        self._backend = self._detect_backend()
        logger.info(f"[VLMClient] Backend: {self._backend}")

    def _detect_backend(self) -> str:
        override = os.environ.get("FOL_VLM_BACKEND", "").strip().lower()
        if override in ("anthropic", "openai"):
            return override
        if override:
            logger.warning(
                f"[VLMClient] Unrecognized FOL_VLM_BACKEND={override!r}; "
                "falling back to auto-detection"
            )
        qwen_url = os.environ.get("QWEN_SERVER_URL", "").strip()
        if qwen_url:
            return "qwen"
        if os.environ.get("ANTHROPIC_API_KEY", "").strip():
            return "anthropic"
        if os.environ.get("OPENAI_API_KEY", "").strip():
            return "openai"
        return "mock"

    @property
    def backend_name(self) -> str:
        return self._backend

    def infer(
        self,
        prompt: str,
        images: Optional[List[np.ndarray]] = None,
        max_tokens: int = 512,
    ) -> str:
        """
        Run VLM inference.

        Args:
            prompt: Text prompt (includes any accumulated CoT context)
            images: List of RGB uint8 arrays. None or [] means text-only.
            max_tokens: Max response tokens.

        Returns:
            Model response as a string.
        """
        images = [img for img in (images or []) if img is not None]

        try:
            if self._backend == "qwen":
                return self._infer_qwen(prompt, images, max_tokens)
            elif self._backend == "anthropic":
                return self._infer_anthropic(prompt, images, max_tokens)
            elif self._backend == "openai":
                return self._infer_openai(prompt, images, max_tokens)
            else:
                logger.warning("[VLMClient] No backend available — returning empty string")
                return ""
        except Exception as e:
            logger.warning(f"[VLMClient] {self._backend} inference failed: {e}")
            return ""

    # ------------------------------------------------------------------ #
    # Qwen server backend
    # ------------------------------------------------------------------ #

    def _infer_qwen(self, prompt: str, images: List[np.ndarray], max_tokens: int) -> str:
        import requests
        from PIL import Image

        server_url = os.environ["QWEN_SERVER_URL"].rstrip("/")
        images_b64 = []
        for img in images:
            pil = Image.fromarray(img.astype(np.uint8))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        resp = requests.post(
            f"{server_url}/infer_v6",
            json={"prompt": prompt, "images_b64": images_b64, "max_new_tokens": max_tokens},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("text", "")

    # ------------------------------------------------------------------ #
    # Anthropic backend (uses vlm_prompt_runner if available, else direct)
    # ------------------------------------------------------------------ #

    def _infer_anthropic(self, prompt: str, images: List[np.ndarray], max_tokens: int) -> str:
        # Try vlm_prompt_runner AnthropicBackend first (saves temp files for us)
        try:
            return self._infer_anthropic_via_runner(prompt, images, max_tokens)
        except Exception:
            return self._infer_anthropic_direct(prompt, images, max_tokens)

    def _infer_anthropic_via_runner(
        self, prompt: str, images: List[np.ndarray], max_tokens: int
    ) -> str:
        import sys, os as _os
        _runner_dir = _os.path.join(_os.path.dirname(__file__), "..", "..", "vlm_prompt_runner")
        if _runner_dir not in sys.path:
            sys.path.insert(0, _runner_dir)
        from vlm_prompt_runner.backends.anthropic_api import AnthropicBackend

        model = os.environ.get("ANTHROPIC_MODEL", _DEFAULT_CLAUDE_MODEL)
        backend = AnthropicBackend(model_id=model, max_new_tokens=max_tokens)

        tmp_paths = _save_images_tmp(images)
        try:
            return backend.generate(prompt, tmp_paths, max_new_tokens=max_tokens)
        finally:
            _cleanup_tmp(tmp_paths)

    def _infer_anthropic_direct(
        self, prompt: str, images: List[np.ndarray], max_tokens: int
    ) -> str:
        import anthropic
        from PIL import Image

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        model = os.environ.get("ANTHROPIC_MODEL", _DEFAULT_CLAUDE_MODEL)

        content = []
        for img in images:
            pil = Image.fromarray(img.astype(np.uint8))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            })
        content.append({"type": "text", "text": prompt})

        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return msg.content[0].text

    # ------------------------------------------------------------------ #
    # OpenAI backend
    # ------------------------------------------------------------------ #

    def _infer_openai(self, prompt: str, images: List[np.ndarray], max_tokens: int) -> str:
        try:
            return self._infer_openai_via_runner(prompt, images, max_tokens)
        except Exception:
            return self._infer_openai_direct(prompt, images, max_tokens)

    def _infer_openai_via_runner(
        self, prompt: str, images: List[np.ndarray], max_tokens: int
    ) -> str:
        import sys, os as _os
        _runner_dir = _os.path.join(_os.path.dirname(__file__), "..", "..", "vlm_prompt_runner")
        if _runner_dir not in sys.path:
            sys.path.insert(0, _runner_dir)
        from vlm_prompt_runner.backends.openai_api import OpenAIBackend

        model = os.environ.get("OPENAI_MODEL", _DEFAULT_OPENAI_MODEL)
        backend = OpenAIBackend(model_id=model)
        tmp_paths = _save_images_tmp(images)
        try:
            return backend.generate(prompt, tmp_paths, max_new_tokens=max_tokens)
        finally:
            _cleanup_tmp(tmp_paths)

    def _infer_openai_direct(
        self, prompt: str, images: List[np.ndarray], max_tokens: int
    ) -> str:
        import openai
        from PIL import Image

        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = os.environ.get("OPENAI_MODEL", _DEFAULT_OPENAI_MODEL)

        content = []
        for img in images:
            pil = Image.fromarray(img.astype(np.uint8))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt})

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_images_tmp(images: List[np.ndarray]) -> List[str]:
    from PIL import Image
    paths = []
    for img in images:
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(img.astype(np.uint8)).save(f.name)
        f.close()
        paths.append(f.name)
    return paths


def _cleanup_tmp(paths: List[str]) -> None:
    for p in paths:
        try:
            os.unlink(p)
        except Exception:
            pass
