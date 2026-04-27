from __future__ import annotations
import json
import logging
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

    Tries direct parse first, then a brace-counting extractor to find the
    first complete JSON block, then falls back to wrapping the raw string.
    """
    # Direct parse first (handles clean JSON responses)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Brace-counting extraction — handles JSON embedded in prose / markdown
    block = _extract_first_json_block(raw)
    if block is not None:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass

    logger.warning(
        "Could not parse JSON from VLM response; storing raw text. "
        "Preview: %.120s", raw
    )
    return {"raw_response": raw}


def _extract_first_json_block(raw: str) -> str | None:
    """Return the first complete JSON object or array in raw, or None.

    Picks whichever opener (``{`` or ``[``) appears first in the string so
    that an array like ``[{...}]`` is returned as an array rather than its
    first inner object.
    """
    pairs = [('{', '}'), ('[', ']')]
    # Sort pairs so we try the opener that appears earliest in the string first
    pairs.sort(key=lambda p: raw.find(p[0]) if raw.find(p[0]) != -1 else len(raw))

    for start_char, end_char in pairs:
        start = raw.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(raw[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return raw[start:i + 1]
    return None


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
    try:
        raw = backend.generate(prompt, image_paths, max_new_tokens=max_new_tokens)
    except Exception as exc:
        logger.error("backend.generate failed for %s: %s", ep_dir.name, exc)
        raise
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
