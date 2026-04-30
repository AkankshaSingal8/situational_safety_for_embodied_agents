# vlm_prompt_runner/tests/test_config.py
import os
import tempfile
from pathlib import Path
import pytest


def test_load_dotenv_sets_vars(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=test-openai-key\nGOOGLE_API_KEY=test-google-key\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    from vlm_prompt_runner.config import load_env
    load_env(env_path=str(env_file))

    assert os.environ["OPENAI_API_KEY"] == "test-openai-key"
    assert os.environ["GOOGLE_API_KEY"] == "test-google-key"


def test_load_dotenv_does_not_override_existing(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-file\n")
    monkeypatch.setenv("OPENAI_API_KEY", "already-set")

    from vlm_prompt_runner.config import load_env
    load_env(env_path=str(env_file))

    assert os.environ["OPENAI_API_KEY"] == "already-set"


def test_load_dotenv_missing_file_is_silent(tmp_path):
    from vlm_prompt_runner.config import load_env
    load_env(env_path=str(tmp_path / "nonexistent.env"))
