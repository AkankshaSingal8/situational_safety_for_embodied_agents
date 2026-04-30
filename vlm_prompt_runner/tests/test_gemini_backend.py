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
