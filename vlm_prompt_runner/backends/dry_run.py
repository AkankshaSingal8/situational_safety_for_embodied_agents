from vlm_prompt_runner.backends.base import VLMBackend


class DryRunBackend(VLMBackend):
    """Returns a fixed JSON string. No GPU or API key needed."""

    def generate(self, prompt: str, image_paths: list[str],
                 max_new_tokens: int = 1024) -> str:
        return '{"dry_run": true, "note": "placeholder response"}'
