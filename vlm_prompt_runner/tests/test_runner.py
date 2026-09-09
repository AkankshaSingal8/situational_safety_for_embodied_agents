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


def test_build_prompt_template_substitution():
    result = build_prompt(
        system_prompt="Objects:\n{object_list}",
        task_description="pick up the cube",
        template_vars={"object_list": "- plate_1\n- moka_pot_obstacle_1"},
    )
    assert "- plate_1" in result
    assert "- moka_pot_obstacle_1" in result
    assert "pick up the cube" in result


def test_build_prompt_missing_template_var_passes_through():
    """Unknown placeholders should not raise — leave them as-is."""
    result = build_prompt(
        system_prompt="Objects: {object_list} extra: {unknown_var}",
        task_description="pick up the cube",
        template_vars={"object_list": "- plate_1"},
    )
    assert "- plate_1" in result
    assert "{unknown_var}" in result


def test_build_prompt_no_template_vars_unchanged():
    """Calling without template_vars must not alter existing behaviour."""
    result = build_prompt(
        system_prompt="You are a safety expert.",
        task_description="pick up the red cube",
    )
    assert "pick up the red cube" in result
    assert "You are a safety expert" in result


def test_run_episode_passes_object_list_to_prompt(tmp_path):
    """run_episode should substitute {object_list} from episode metadata."""
    from vlm_prompt_runner.backends.base import VLMBackend

    received_prompts = []

    class CapturingBackend(VLMBackend):
        def generate(self, prompt, image_paths, max_new_tokens=1024):
            received_prompts.append(prompt)
            return '{"reasoning": "test", "object": "moka_pot_obstacle_1"}'

    ep_dir = tmp_path / "episode_00"
    ep_dir.mkdir()
    (ep_dir / "metadata.json").write_text(json.dumps({
        "task_description": "pick up the bowl",
        "task_id": 0, "safety_level": "I",
        "task_suite": "test", "episode_idx": 0,
        "objects": {
            "moka_pot_obstacle_1": {"position": [0.1, 0.1, 0.9], "quaternion": [0, 0, 0, 1]},
        },
    }))
    for name in ("agentview_rgb.png", "eye_in_hand_rgb.png", "backview_rgb.png"):
        (ep_dir / name).touch()

    out_path = tmp_path / "output.json"
    run_episode(
        ep_dir=ep_dir,
        system_prompt="Objects:\n{object_list}",
        backend=CapturingBackend(),
        out_path=out_path,
    )
    assert len(received_prompts) == 1
    assert "moka_pot_obstacle_1" in received_prompts[0]


def test_run_episode_preserves_reasoning_field(tmp_path):
    """If VLM returns reasoning field it should be saved in output.json."""
    from vlm_prompt_runner.backends.base import VLMBackend

    class ReasoningBackend(VLMBackend):
        def generate(self, prompt, image_paths, max_new_tokens=1024):
            return '{"reasoning": "I see a moka pot.", "object": "moka_pot_obstacle_1"}'

    ep_dir = tmp_path / "episode_00"
    ep_dir.mkdir()
    (ep_dir / "metadata.json").write_text(json.dumps({
        "task_description": "pick up the bowl",
        "task_id": 0, "safety_level": "I",
        "task_suite": "test", "episode_idx": 0,
    }))
    for name in ("agentview_rgb.png", "eye_in_hand_rgb.png", "backview_rgb.png"):
        (ep_dir / name).touch()

    out_path = tmp_path / "output.json"
    run_episode(
        ep_dir=ep_dir,
        system_prompt="Identify the obstacle.",
        backend=ReasoningBackend(),
        out_path=out_path,
    )
    data = json.loads(out_path.read_text())
    assert data["reasoning"] == "I see a moka pot."
    assert data["object"] == "moka_pot_obstacle_1"
