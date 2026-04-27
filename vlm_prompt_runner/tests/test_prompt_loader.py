import tempfile
from pathlib import Path

import pytest

from vlm_prompt_runner.prompt_loader import load_prompt


def test_load_prompt_returns_string():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "my_prompt.md"
        p.write_text("You are a robot safety expert.\n\nDo something.")
        result = load_prompt(p)
        assert "robot safety expert" in result
        assert isinstance(result, str)


def test_load_prompt_stem():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "safety_predicates_prompt.md"
        p.write_text("content")
        content, stem = load_prompt(p, return_stem=True)
        assert stem == "safety_predicates_prompt"
        assert content == "content"


def test_load_prompt_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt(Path("/nonexistent/prompt.md"))


def test_load_prompt_real_file():
    """Verify load_prompt works on the actual prompts in the repo."""
    prompt_dir = Path("/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/prompts")
    for md_file in prompt_dir.glob("*.md"):
        content, stem = load_prompt(md_file, return_stem=True)
        assert isinstance(content, str)
        assert len(content) > 0
        assert stem == md_file.stem
