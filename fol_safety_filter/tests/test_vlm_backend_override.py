"""Tests for the FOL_VLM_BACKEND override in VLMClient._detect_backend()."""

from fol_safety_filter.vlm_client import VLMClient


def test_override_anthropic_wins_over_qwen(monkeypatch):
    monkeypatch.setenv("FOL_VLM_BACKEND", "anthropic")
    monkeypatch.setenv("QWEN_SERVER_URL", "http://localhost:9999")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    client = VLMClient()
    assert client.backend_name == "anthropic"


def test_override_openai_wins_over_anthropic_key(monkeypatch):
    monkeypatch.delenv("QWEN_SERVER_URL", raising=False)
    monkeypatch.setenv("FOL_VLM_BACKEND", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test2")
    client = VLMClient()
    assert client.backend_name == "openai"


def test_unset_preserves_existing_priority(monkeypatch):
    monkeypatch.delenv("FOL_VLM_BACKEND", raising=False)
    monkeypatch.setenv("QWEN_SERVER_URL", "http://localhost:9999")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    client = VLMClient()
    assert client.backend_name == "qwen"


def test_garbage_value_falls_back_to_autodetect(monkeypatch):
    monkeypatch.setenv("FOL_VLM_BACKEND", "not-a-real-backend")
    monkeypatch.delenv("QWEN_SERVER_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    client = VLMClient()
    assert client.backend_name == "anthropic"
