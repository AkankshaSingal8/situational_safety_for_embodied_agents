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
