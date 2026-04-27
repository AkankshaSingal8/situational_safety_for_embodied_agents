from __future__ import annotations
import json
import logging
import re
from pathlib import Path

from vlm_prompt_runner.backends.base import VLMBackend
from vlm_prompt_runner.episode import load_episode

logger = logging.getLogger(__name__)


def build_prompt(system_prompt: str, task_description: str) -> str:
    """Combine the .md system prompt with the task description from metadata."""
    return (
        f"{system_prompt.strip()}\n\n"
        f"---\n\n"
        f"**Robot task instruction:** {task_description}"
    )


def extract_json(raw: str) -> dict | list:
    """Extract a JSON object or array from the raw VLM response.

    Falls back to wrapping the raw string if no valid JSON is found.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for pattern in (r'\{.*\}', r'\[.*\]'):
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    return {"raw_response": raw}


def run_episode(ep_dir: Path | str, system_prompt: str,
                backend: VLMBackend, out_path: Path | str,
                max_new_tokens: int = 1024) -> dict:
    """Run VLM inference for one episode and write the result to out_path."""
    ep_dir = Path(ep_dir)
    out_path = Path(out_path)

    episode = load_episode(ep_dir)
    prompt = build_prompt(system_prompt, episode["task_description"])
    image_paths = [episode["agentview"], episode["eye_in_hand"], episode["backview"]]

    logger.info(f"Running inference: {ep_dir.name}")
    raw = backend.generate(prompt, image_paths, max_new_tokens=max_new_tokens)
    result = extract_json(raw)

    if isinstance(result, dict):
        result["_meta"] = {
            "task_description": episode["task_description"],
            "ep_dir": str(ep_dir),
        }
    else:
        result = {"data": result, "_meta": {
            "task_description": episode["task_description"],
            "ep_dir": str(ep_dir),
        }}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"  → {out_path}")
    return result
