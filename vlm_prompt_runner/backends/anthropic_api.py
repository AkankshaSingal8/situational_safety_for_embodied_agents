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
        import anthropic as _anthropic
        self._client = _anthropic.Anthropic(api_key=self.api_key)

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

        import time as _time

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._client.messages.create(
                    model=self.model_id,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": content}],
                )
                return response.content[0].text.strip()
            except Exception as exc:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Anthropic API error (attempt %d/%d), retrying in %ds: %s",
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
