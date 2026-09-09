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


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_prompt_real_file():
    """Verify load_prompt works on the actual prompts in the repo."""
    prompt_dir = REPO_ROOT / "prompts"
    md_files = sorted(prompt_dir.glob("*.md"))
    # Without this the loop body never runs and the test passes vacuously.
    assert md_files, f"no prompt files found under {prompt_dir}"
    for md_file in md_files:
        content, stem = load_prompt(md_file, return_stem=True)
        assert isinstance(content, str)
        assert len(content) > 0
        assert stem == md_file.stem
