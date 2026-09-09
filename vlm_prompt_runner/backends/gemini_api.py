import base64
import logging
import os
from pathlib import Path

from vlm_prompt_runner.backends.base import VLMBackend

logger = logging.getLogger(__name__)


class GeminiBackend(VLMBackend):
    """Calls the Google Gemini API with image support via google-genai SDK.

    The model_id is passed verbatim to the API (e.g. 'gemini-2.5-pro',
    'gemini-2.5-flash').  Reads GOOGLE_API_KEY from the environment unless
    api_key is passed explicitly.
    """

    def __init__(self, model_id: str = "gemini-2.5-flash",
                 api_key: str | None = None,
                 max_new_tokens: int = 8192):
        try:
            from google import genai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Install the Google AI SDK: pip install google-genai"
            ) from e

        self.model_id = model_id
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Set GOOGLE_API_KEY environment variable or pass api_key="
            )
        self._default_max_tokens = max_new_tokens

        from google import genai
        self._client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str, image_paths: list[str],
                 max_new_tokens: int | None = None) -> str:
        from google.genai import types

        max_tokens = max_new_tokens if max_new_tokens is not None else self._default_max_tokens

        parts: list = []
        for p in image_paths:
            if p and Path(p).exists():
                media_type = _infer_media_type(p)
                with open(p, "rb") as f:
                    data = f.read()
                parts.append(types.Part.from_bytes(data=data, mime_type=media_type))
            else:
                logger.warning("Image not found, skipping: %s", p)

        parts.append(prompt)

        config = types.GenerateContentConfig(max_output_tokens=max_tokens)

        import time as _time

        max_retries = 6
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model_id,
                    contents=parts,
                    config=config,
                )
                return response.text.strip()
            except Exception as exc:
                # Retry on transient server/rate-limit errors (503, 429)
                err_str = str(exc)
                is_retryable = any(code in err_str for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
                if attempt < max_retries - 1 and is_retryable:
                    wait = min(5 * (2 ** attempt), 120)  # 5, 10, 20, 40, 80, 120s
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
