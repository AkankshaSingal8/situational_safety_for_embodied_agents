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
