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
