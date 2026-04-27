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
