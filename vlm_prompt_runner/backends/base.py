from abc import ABC, abstractmethod


class VLMBackend(ABC):
    """Minimal interface all VLM backends must implement."""

    @abstractmethod
    def generate(self, prompt: str, image_paths: list[str],
                 max_new_tokens: int = 1024) -> str:
        """Send prompt + images to the model, return raw text response."""
        ...
