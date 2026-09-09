"""
Run a prompt N times per episode, majority-vote the 'object' field,
write a single output.json compatible with compute_accuracy().

Usage:
    python -m vlm_prompt_runner.run_majority_vote_experiment \
        --prompts-dir prompts/obstacle_id \
        --prompts p10_most_dangerous_prompt \
        --models gpt-5.5 \
        --suite safelibero_spatial \
        --level I --task 0 \
        --n-votes 5 \
        --results-out vlm_prompt_runner/results/majority_vote_gpt55_p10.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import vlm_prompt_runner.config  # loads .env on import
from vlm_prompt_runner.accuracy import compute_accuracy
from vlm_prompt_runner.backends import get_backend
from vlm_prompt_runner.episode import load_episode, output_path, resolve_episodes
from vlm_prompt_runner.prompt_loader import load_prompt
from vlm_prompt_runner.runner import build_prompt, extract_json


def majority_vote(votes: list[str]) -> str:
    """Return the most common element. Raises ValueError if votes is empty."""
    if not votes:
        raise ValueError("votes list is empty")
    counter = Counter(votes)
    return counter.most_common(1)[0][0]


def merge_votes(
    votes: list[str | None],
    raw_responses: list[str],
    meta: dict,
) -> dict:
    """Combine N VLM votes into a single output.json-compatible dict.

    None entries (parse failures) are excluded from voting.
    """
    valid_votes = [v for v in votes if v is not None]
    if not valid_votes:
        winning_object = None
    else:
        winning_object = majority_vote(valid_votes)

    return {
        "object": winning_object,
        "vote_counts": dict(Counter(valid_votes)),
        "all_votes": votes,
        "_raw_responses": raw_responses,
        "_meta": meta,
    }


def run_majority_vote_episode(
    ep_dir: Path,
    system_prompt: str,
    backend,
    out_path: Path,
    n_votes: int = 5,
    max_new_tokens: int | None = None,
) -> dict:
    """Call VLM n_votes times for one episode, majority-vote 'object', write output."""
    if out_path.exists():
        print(f"  [skip] {out_path} already exists")
        return json.loads(out_path.read_text())

    episode = load_episode(ep_dir)
    prompt = build_prompt(system_prompt, episode["task_description"], episode)
    image_paths = [episode["agentview"], episode["eye_in_hand"], episode["backview"]]
    meta = {"task_description": episode["task_description"], "ep_dir": str(ep_dir)}

    votes: list[str | None] = []
    raw_responses: list[str] = []

    for i in range(n_votes):
        raw = backend.generate(prompt, image_paths, max_new_tokens=max_new_tokens)
        raw_responses.append(raw)
        parsed = extract_json(raw)
        obj = parsed.get("object") if isinstance(parsed, dict) else None
        votes.append(obj)
        print(f"    vote {i+1}/{n_votes}: {obj!r}")

    result = merge_votes(votes, raw_responses, meta)
    print(f"  -> majority: {result['object']!r}  counts: {result['vote_counts']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


def _model_slug(model_key: str) -> str:
    return model_key.replace(".", "_").replace("-", "_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Majority-vote VLM experiment runner")
    p.add_argument("--prompts-dir", required=True, type=Path)
    p.add_argument("--prompts", nargs="*", default=None,
                   help="Prompt stems to run; omit for all in prompts-dir")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--level", required=True)
    p.add_argument("--task", required=True, type=int)
    p.add_argument("--episodes", nargs="*", type=int, default=None)
    p.add_argument("--n-votes", type=int, default=5)
    p.add_argument("--results-out", required=True, type=Path)
    p.add_argument("--output-base", type=Path,
                   default=Path("vlm_prompt_runner/outputs"))
    p.add_argument("--input-base", type=Path, default=Path("vlm_inputs"))
    p.add_argument("--max-new-tokens", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    all_prompts = sorted(args.prompts_dir.glob("*.md"))
    if args.prompts:
        stems = set(args.prompts)
        prompt_files = [f for f in all_prompts if f.stem in stems]
    else:
        prompt_files = all_prompts

    episode_dirs = resolve_episodes(
        args.input_base, args.suite, args.level, args.task, args.episodes
    )

    summary: dict = {
        "suite": args.suite,
        "level": args.level,
        "task": args.task,
        "n_votes": args.n_votes,
        "models": {},
    }

    for model_key in args.models:
        print(f"\n=== Model: {model_key} ===")
        slug = _model_slug(model_key)
        backend = get_backend(model_key)
        model_results: dict = {}

        for prompt_file in prompt_files:
            system_prompt, prompt_stem = load_prompt(prompt_file, return_stem=True)
            print(f"\n  Prompt: {prompt_stem}")
            output_base = args.output_base / slug

            for ep_dir in episode_dirs:
                ep_idx = int(ep_dir.name.replace("episode_", ""))
                out = output_path(
                    output_base, prompt_stem, args.suite, args.level, args.task, ep_idx
                )
                run_majority_vote_episode(
                    ep_dir, system_prompt, backend, out,
                    n_votes=args.n_votes,
                    max_new_tokens=args.max_new_tokens,
                )

            stl_dir = output_base / prompt_stem
            acc = compute_accuracy(stl_dir, args.input_base)
            totals = acc.get("_totals", {}).get("overall", {})
            model_results[prompt_stem] = {
                "correct": totals.get("correct", 0),
                "total": totals.get("total", 0),
                "accuracy": totals.get("accuracy", 0.0),
            }
            print(f"  {prompt_stem}: {model_results[prompt_stem]}")

        summary["models"][model_key] = model_results

    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_out.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {args.results_out}")


if __name__ == "__main__":
    main()
