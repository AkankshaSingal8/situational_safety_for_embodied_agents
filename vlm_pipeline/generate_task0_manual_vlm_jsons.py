#!/usr/bin/env python3
"""Generate one-obstacle manual VLM JSONs for SafeLIBERO task 0 episodes.

The generated files match the existing ``vlm_manual_scene_safety_epXX_single``
schema, but the obstacle name is read from each episode's metadata instead of
being hand-entered.
"""

import argparse
import json
from pathlib import Path


DEFAULT_INPUT_ROOT = Path("vlm_pipeline/vlm_inputs/safelibero_spatial")
DEFAULT_OUTPUT_ROOT = Path("vlm_pipeline/vlm_outputs/task_0")
DEFAULT_LEVELS = ("level_I", "level_II")
DEFAULT_TASK_ID = 0
DEFAULT_EPISODES = range(50)


def build_vlm_json(obstacle_name):
    return {
        "single": {
            "description": f"Avoid {obstacle_name} while moving the bowl.",
            "end_object": "akita_black_bowl",
            "objects": [
                [
                    obstacle_name,
                    [
                        "around",
                    ],
                ],
                [
                    "end_effector",
                    [
                        "caution",
                    ],
                ],
            ],
        },
    }


def load_active_obstacle(metadata_path):
    with metadata_path.open() as f:
        metadata = json.load(f)

    obstacle = metadata.get("obstacle") or {}
    name = obstacle.get("name")
    if not name:
        raise ValueError(f"No obstacle.name found in {metadata_path}")
    return name


def generate_for_level(input_root, output_root, level, task_id, episodes):
    task_input = input_root / level / f"task_{task_id}"
    task_output = output_root / level
    task_output.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for episode_idx in episodes:
        metadata_path = task_input / f"episode_{episode_idx:02d}" / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing metadata: {metadata_path}")

        obstacle_name = load_active_obstacle(metadata_path)
        output_path = task_output / f"episode_{episode_idx:02d}.json"
        with output_path.open("w") as f:
            json.dump(build_vlm_json(obstacle_name), f, indent=2)
            f.write("\n")

        manifest[f"episode_{episode_idx:02d}"] = {
            "obstacle": obstacle_name,
            "json": str(output_path),
        }

    manifest_path = task_output / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest_path


def parse_episode_indices(raw):
    if raw.strip().lower() == "all":
        return list(DEFAULT_EPISODES)
    return [int(s) for s in raw.split(",") if s.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Generate task_0 manual VLM JSONs from saved metadata.")
    parser.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--levels", type=str, default=",".join(DEFAULT_LEVELS))
    parser.add_argument("--task_id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--episode_indices", type=str, default="all")
    args = parser.parse_args()

    levels = [s.strip() for s in args.levels.split(",") if s.strip()]
    episodes = parse_episode_indices(args.episode_indices)

    for level in levels:
        manifest_path = generate_for_level(
            input_root=args.input_root,
            output_root=args.output_root,
            level=level,
            task_id=args.task_id,
            episodes=episodes,
        )
        print(f"Wrote {len(episodes)} JSONs for {level}: {manifest_path}")


if __name__ == "__main__":
    main()
