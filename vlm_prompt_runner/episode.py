from __future__ import annotations
import json
from pathlib import Path


def load_episode(ep_dir: Path | str) -> dict:
    """Load metadata and image paths for one episode directory."""
    ep_dir = Path(ep_dir)
    meta_path = ep_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {ep_dir}")
    with open(meta_path) as f:
        meta = json.load(f)
    return {
        "task_description": meta.get("task_description", ""),
        "metadata": meta,
        "agentview": str(ep_dir / "agentview_rgb.png"),
        "eye_in_hand": str(ep_dir / "eye_in_hand_rgb.png"),
        "backview": str(ep_dir / "backview_rgb.png"),
        "ep_dir": str(ep_dir),
    }


def resolve_episodes(input_base: Path | str, suite: str, level: str,
                     task_id: int, episodes: list[int] | None) -> list[Path]:
    """Return sorted list of episode directories.

    If episodes is None, returns all episode_* dirs for the task.
    Otherwise returns only the specified episode indices.
    """
    input_base = Path(input_base)
    task_dir = input_base / suite / f"level_{level}" / f"task_{task_id}"
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    if episodes is None:
        return sorted(task_dir.glob("episode_*"))

    paths = []
    for ep_idx in episodes:
        ep_dir = task_dir / f"episode_{ep_idx:02d}"
        if not ep_dir.exists():
            raise FileNotFoundError(f"Episode directory not found: {ep_dir}")
        paths.append(ep_dir)
    return paths


def output_path(output_base: Path | str, prompt_stem: str, suite: str,
                level: str, task_id: int, ep_idx: int) -> Path:
    """Compute the output JSON path, mirroring the input folder structure."""
    return (
        Path(output_base) / prompt_stem / suite
        / f"level_{level}" / f"task_{task_id}"
        / f"episode_{ep_idx:02d}" / "output.json"
    )
