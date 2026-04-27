import json
import tempfile
from pathlib import Path

from vlm_prompt_runner.backends.dry_run import DryRunBackend
from vlm_prompt_runner.runner import build_prompt, extract_json, run_episode


def test_build_prompt_includes_task_and_system():
    result = build_prompt(
        system_prompt="You are a safety expert.",
        task_description="pick up the red cube",
    )
    assert "pick up the red cube" in result
    assert "You are a safety expert" in result


def test_extract_json_valid_object():
    raw = 'Some preamble {"key": "value"} trailing text'
    result = extract_json(raw)
    assert result == {"key": "value"}


def test_extract_json_valid_array():
    raw = '[{"a": 1}, {"b": 2}]'
    result = extract_json(raw)
    assert isinstance(result, list)
    assert len(result) == 2


def test_extract_json_invalid_returns_raw_wrapped():
    raw = "This is not JSON at all."
    result = extract_json(raw)
    assert result == {"raw_response": raw}


def test_run_episode_writes_output(tmp_path):
    ep_dir = tmp_path / "episode_00"
    ep_dir.mkdir()
    (ep_dir / "metadata.json").write_text(json.dumps({
        "task_description": "pick up the cube",
        "task_id": 0, "safety_level": "I",
        "task_suite": "test_suite", "episode_idx": 0,
    }))
    for name in ("agentview_rgb.png", "eye_in_hand_rgb.png", "backview_rgb.png"):
        (ep_dir / name).touch()

    out_path = tmp_path / "output.json"
    run_episode(
        ep_dir=ep_dir,
        system_prompt="You are a safety expert.",
        backend=DryRunBackend(),
        out_path=out_path,
    )
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert isinstance(data, dict)
    assert "_meta" in data
    assert data["_meta"]["task_description"] == "pick up the cube"


def test_run_episode_creates_parent_dirs(tmp_path):
    """output directory should be created automatically if it doesn't exist."""
    ep_dir = tmp_path / "episode_00"
    ep_dir.mkdir()
    (ep_dir / "metadata.json").write_text(json.dumps({
        "task_description": "pick up the cube",
        "task_id": 0, "safety_level": "I",
        "task_suite": "test_suite", "episode_idx": 0,
    }))
    for name in ("agentview_rgb.png", "eye_in_hand_rgb.png", "backview_rgb.png"):
        (ep_dir / name).touch()

    # nested path that does NOT exist yet
    out_path = tmp_path / "safety_predicates_prompt" / "safelibero_spatial" / "level_I" / "task_0" / "episode_00" / "output.json"
    run_episode(
        ep_dir=ep_dir,
        system_prompt="You are a safety expert.",
        backend=DryRunBackend(),
        out_path=out_path,
    )
    assert out_path.exists()
    assert out_path.parent.is_dir()


def test_run_episode_array_response_wrapped(tmp_path):
    """If VLM returns a JSON array, it should be wrapped under 'data' key."""
    from vlm_prompt_runner.backends.base import VLMBackend

    class ArrayBackend(VLMBackend):
        def generate(self, prompt, image_paths, max_new_tokens=1024):
            return '[{"obj": "plate", "unsafe": ["above"]}]'

    ep_dir = tmp_path / "episode_00"
    ep_dir.mkdir()
    (ep_dir / "metadata.json").write_text(json.dumps({
        "task_description": "move the bowl",
        "task_id": 0, "safety_level": "I",
        "task_suite": "test_suite", "episode_idx": 0,
    }))
    for name in ("agentview_rgb.png", "eye_in_hand_rgb.png", "backview_rgb.png"):
        (ep_dir / name).touch()

    out_path = tmp_path / "output.json"
    run_episode(
        ep_dir=ep_dir,
        system_prompt="You are a safety expert.",
        backend=ArrayBackend(),
        out_path=out_path,
    )
    data = json.loads(out_path.read_text())
    assert "data" in data
    assert "_meta" in data
