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
