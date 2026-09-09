import json
import tempfile
from pathlib import Path

import pytest

from vlm_prompt_runner.episode import load_episode, output_path, resolve_episodes


def _make_episode(root: Path, suite: str, level: str,
                  task_id: int, ep_idx: int) -> Path:
    ep = root / suite / f"level_{level}" / f"task_{task_id}" / f"episode_{ep_idx:02d}"
    ep.mkdir(parents=True)
    meta = {
        "task_suite": suite, "safety_level": level,
        "task_id": task_id, "episode_idx": ep_idx,
        "task_description": "pick up the red cube",
    }
    (ep / "metadata.json").write_text(json.dumps(meta))
    for name in ("agentview_rgb.png", "eye_in_hand_rgb.png", "backview_rgb.png"):
        (ep / name).touch()
    return ep


def test_load_episode():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "vlm_inputs"
        ep_dir = _make_episode(root, "safelibero_spatial", "I", 0, 0)
        ep = load_episode(ep_dir)
        assert ep["task_description"] == "pick up the red cube"
        assert Path(ep["agentview"]).name == "agentview_rgb.png"
        assert Path(ep["eye_in_hand"]).name == "eye_in_hand_rgb.png"
        assert Path(ep["backview"]).name == "backview_rgb.png"


def test_load_episode_missing_metadata_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            load_episode(Path(tmp) / "nonexistent")


def test_resolve_episodes_specific():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "vlm_inputs"
        for i in range(5):
            _make_episode(root, "safelibero_spatial", "I", 0, i)
        eps = resolve_episodes(root, "safelibero_spatial", "I", 0, episodes=[0, 2, 4])
        assert len(eps) == 3
        assert all(p.exists() for p in eps)


def test_resolve_episodes_all():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "vlm_inputs"
        for i in range(3):
            _make_episode(root, "safelibero_spatial", "I", 0, i)
        eps = resolve_episodes(root, "safelibero_spatial", "I", 0, episodes=None)
        assert len(eps) == 3


def test_resolve_episodes_missing_task_dir_raises():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "vlm_inputs"
        with pytest.raises(FileNotFoundError, match="Task directory not found"):
            resolve_episodes(root, "safelibero_spatial", "I", 99, episodes=None)


def test_resolve_episodes_missing_episode_raises():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "vlm_inputs"
        _make_episode(root, "safelibero_spatial", "I", 0, 0)
        with pytest.raises(FileNotFoundError, match="Episode directory not found"):
            resolve_episodes(root, "safelibero_spatial", "I", 0, episodes=[0, 99])


def test_output_path():
    p = output_path(
        output_base=Path("/results"),
        prompt_stem="safety_predicates_prompt",
        suite="safelibero_spatial",
        level="I",
        task_id=0,
        ep_idx=3,
    )
    assert p == Path(
        "/results/safety_predicates_prompt/safelibero_spatial"
        "/level_I/task_0/episode_03/output.json"
    )


def test_load_episode_includes_object_list(tmp_path):
    ep_dir = tmp_path / "episode_00"
    ep_dir.mkdir()
    meta = {
        "task_description": "pick up the cube",
        "task_id": 0, "safety_level": "I",
        "task_suite": "test", "episode_idx": 0,
        "objects": {
            "moka_pot_obstacle_1": {"position": [0.1, 0.1, 0.9], "quaternion": [0, 0, 0, 1]},
            "plate_1": {"position": [0.2, 0.2, 0.9], "quaternion": [0, 0, 0, 1]},
            "table_collision": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]},
        },
    }
    (ep_dir / "metadata.json").write_text(json.dumps(meta))
    for name in ("agentview_rgb.png", "eye_in_hand_rgb.png", "backview_rgb.png"):
        (ep_dir / name).touch()

    ep = load_episode(ep_dir)
    assert "object_list" in ep
    assert "moka_pot_obstacle_1" in ep["object_list"]
    assert "plate_1" in ep["object_list"]
    assert "table_collision" not in ep["object_list"]   # structural — excluded


def test_load_episode_object_list_with_positions(tmp_path):
    ep_dir = tmp_path / "episode_00"
    ep_dir.mkdir()
    meta = {
        "task_description": "pick up the cube",
        "task_id": 0, "safety_level": "I",
        "task_suite": "test", "episode_idx": 0,
        "objects": {
            "moka_pot_obstacle_1": {"position": [0.1, 0.2, 0.9], "quaternion": [0, 0, 0, 1]},
        },
    }
    (ep_dir / "metadata.json").write_text(json.dumps(meta))
    for name in ("agentview_rgb.png", "eye_in_hand_rgb.png", "backview_rgb.png"):
        (ep_dir / name).touch()

    ep = load_episode(ep_dir)
    assert "object_list_with_positions" in ep
    assert "x=0.100" in ep["object_list_with_positions"]
    assert "y=0.200" in ep["object_list_with_positions"]
    assert "z=0.900" in ep["object_list_with_positions"]


def test_load_episode_no_objects_key(tmp_path):
    ep_dir = tmp_path / "episode_00"
    ep_dir.mkdir()
    meta = {
        "task_description": "pick up the cube",
        "task_id": 0, "safety_level": "I",
        "task_suite": "test", "episode_idx": 0,
    }
    (ep_dir / "metadata.json").write_text(json.dumps(meta))
    for name in ("agentview_rgb.png", "eye_in_hand_rgb.png", "backview_rgb.png"):
        (ep_dir / name).touch()

    ep = load_episode(ep_dir)
    assert ep["object_list"] == ""
    assert ep["object_list_with_positions"] == ""


def test_load_episode_structural_prefixes_excluded(tmp_path):
    ep_dir = tmp_path / "episode_00"
    ep_dir.mkdir()
    meta = {
        "task_description": "pick up the cube",
        "task_id": 0, "safety_level": "I",
        "task_suite": "test", "episode_idx": 0,
        "objects": {
            "robot0_link1": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]},
            "mount0_pedestal": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]},
            "gripper0_finger1": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]},
            "wall_left_visual": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]},
            "floor": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]},
            "table_collision": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]},
            "box_base_1": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]},
            "wooden_cabinet_1": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]},
            "flat_stove_1": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]},
            "milk_obstacle_1": {"position": [0.1, 0.1, 0.9], "quaternion": [0, 0, 0, 1]},
        },
    }
    (ep_dir / "metadata.json").write_text(json.dumps(meta))
    for name in ("agentview_rgb.png", "eye_in_hand_rgb.png", "backview_rgb.png"):
        (ep_dir / name).touch()

    ep = load_episode(ep_dir)
    assert "milk_obstacle_1" in ep["object_list"]
    for structural in ["robot0_link1", "mount0_pedestal", "gripper0_finger1",
                       "wall_left_visual", "floor", "table_collision",
                       "box_base_1", "wooden_cabinet_1", "flat_stove_1"]:
        assert structural not in ep["object_list"]
