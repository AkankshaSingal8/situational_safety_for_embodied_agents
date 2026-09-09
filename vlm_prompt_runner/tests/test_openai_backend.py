# vlm_prompt_runner/tests/test_openai_backend.py
import base64
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def fake_image(tmp_path):
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG fake")
    return str(img)


def _make_backend(model_id="gpt-4o", api_key="test-key"):
    with patch.dict("os.environ", {"OPENAI_API_KEY": api_key}):
        from vlm_prompt_runner.backends.openai_api import OpenAIBackend
        return OpenAIBackend(model_id=model_id, api_key=api_key)


def test_generate_sends_images_and_prompt(fake_image):
    backend = _make_backend()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"obstacle": "moka_pot"}'

    with patch.object(backend._client.chat.completions, "create",
                      return_value=mock_response) as mock_create:
        result = backend.generate("Identify the obstacle.", [fake_image], max_new_tokens=256)

    assert result == '{"obstacle": "moka_pot"}'
    call_kwargs = mock_create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    # one image block + one text block
    assert any(b.get("type") == "image_url" for b in content)
    assert any(b.get("type") == "text" for b in content)
    assert call_kwargs["max_tokens"] == 256


def test_generate_skips_missing_images(tmp_path):
    backend = _make_backend()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"

    with patch.object(backend._client.chat.completions, "create",
                      return_value=mock_response) as mock_create:
        result = backend.generate("prompt", [str(tmp_path / "missing.png")])

    content = mock_create.call_args.kwargs["messages"][0]["content"]
    image_blocks = [b for b in content if b.get("type") == "image_url"]
    assert len(image_blocks) == 0


def test_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from vlm_prompt_runner.backends.openai_api import OpenAIBackend
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIBackend(model_id="gpt-4o")
