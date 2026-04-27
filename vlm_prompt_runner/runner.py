from __future__ import annotations
import json
import logging
import re
from pathlib import Path

from vlm_prompt_runner.backends.base import VLMBackend
from vlm_prompt_runner.episode import load_episode

logger = logging.getLogger(__name__)


class _PassThrough(dict):
    """format_map helper: unknown keys are returned as-is as {key}."""
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def build_prompt(system_prompt: str, task_description: str,
                 template_vars: dict | None = None) -> str:
    """Combine the .md system prompt with the task description from metadata.

    If template_vars is provided, {placeholder} patterns in system_prompt are
    substituted before the prompt is assembled. Unknown placeholders pass through
    unchanged so prompts that don't use all variables still work.
    """
    if template_vars:
        system_prompt = system_prompt.format_map(_PassThrough(template_vars))
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
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

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
    """Return the first complete JSON object or array in raw, or None."""
    pairs = [('{', '}'), ('[', ']')]
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


def _split_thinking(raw: str) -> tuple[str, str | None]:
    """Split a Qwen3 <think>...</think> block from the answer.

    Returns (answer, thinking) where thinking is None if no block was found.
    """
    match = re.search(r"<think>(.*?)</think>(.*)", raw, re.DOTALL)
    if match:
        return match.group(2).strip(), match.group(1).strip()
    return raw, None


def run_episode(ep_dir: Path | str, system_prompt: str,
                backend: VLMBackend, out_path: Path | str,
                max_new_tokens: int = 1024) -> dict:
    """Run VLM inference for one episode and write the result to out_path."""
    ep_dir = Path(ep_dir)
    out_path = Path(out_path)

    episode = load_episode(ep_dir)

    template_vars = {
        "task_description": episode["task_description"],
        "object_list": episode.get("object_list", ""),
        "object_list_with_positions": episode.get("object_list_with_positions", ""),
    }
    prompt = build_prompt(system_prompt, episode["task_description"], template_vars)
    image_paths = [episode["agentview"], episode["eye_in_hand"], episode["backview"]]

    logger.info(f"Running inference: {ep_dir.name}")
    try:
        raw = backend.generate(prompt, image_paths, max_new_tokens=max_new_tokens)
    except Exception as exc:
        logger.error("backend.generate failed for %s: %s", ep_dir.name, exc)
        raise

    answer, thinking = _split_thinking(raw)
    result = extract_json(answer)

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

    if thinking is not None:
        result["_thinking"] = thinking

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"  → {out_path}")
    return result
