#!/usr/bin/env python3
"""
vlm_prompt_runner/main.py

Run VLM inference on SafeLIBERO episode observations.

Output path:
  <output_base>/<prompt_stem>/<suite>/level_<level>/task_<task_id>/episode_<ep:02d>/output.json

Examples:

  # Dry-run single episode
  python -m vlm_prompt_runner.main \\
      --suite safelibero_spatial --level I --task 0 --episodes 0 \\
      --prompt prompts/safety_predicates_prompt.md \\
      --vlm dry-run

  # Qwen local — all episodes for task 0
  python -m vlm_prompt_runner.main \\
      --suite safelibero_spatial --level I --task 0 \\
      --prompt prompts/safelibero_prompt.md \\
      --vlm qwen2.5-vl-7b

  # Claude API — episodes 0-4
  python -m vlm_prompt_runner.main \\
      --suite safelibero_spatial --level I --task 0 --episodes 0 1 2 3 4 \\
      --prompt prompts/safelibero_prompt.md \\
      --vlm claude-sonnet-4-6
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_BASE = _PROJECT_ROOT / "vlm_inputs"
DEFAULT_OUTPUT_BASE = _PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run VLM inference on SafeLIBERO episodes with a .md prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--suite", required=True,
                   help="Task suite name, e.g. safelibero_spatial")
    p.add_argument("--level", required=True,
                   help="Safety level, e.g. I or II")
    p.add_argument("--task", type=int, required=True,
                   help="Task ID (integer), e.g. 0")
    p.add_argument("--episodes", type=int, nargs="*", default=None,
                   help="Episode indices to process. Omit to process all.")
    p.add_argument("--prompt", required=True,
                   help="Path to .md prompt file, e.g. prompts/safelibero_prompt.md")
    p.add_argument("--vlm", required=True,
                   help=("VLM key: qwen2.5-vl-7b | qwen2.5-vl-3b | qwen2-vl-7b | "
                         "qwen3-vl-8b | claude-sonnet-4-6 | claude-opus-4-7 | dry-run"))
    p.add_argument("--load_in_4bit", action="store_true",
                   help="4-bit quantization for Qwen local models")
    p.add_argument("--device", default="auto",
                   help="Device map for Qwen local (default: auto)")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--input_base", default=str(DEFAULT_INPUT_BASE),
                   help="Root directory of vlm_inputs/")
    p.add_argument("--output_base", default=str(DEFAULT_OUTPUT_BASE),
                   help="Root directory for outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from vlm_prompt_runner.backends import get_backend
    from vlm_prompt_runner.episode import output_path, resolve_episodes
    from vlm_prompt_runner.prompt_loader import load_prompt
    from vlm_prompt_runner.runner import run_episode

    prompt_content, prompt_stem = load_prompt(args.prompt, return_stem=True)
    logger.info(f"Prompt: {args.prompt}  (stem={prompt_stem!r})")

    episode_dirs = resolve_episodes(
        input_base=Path(args.input_base),
        suite=args.suite,
        level=args.level,
        task_id=args.task,
        episodes=args.episodes,
    )
    logger.info(
        f"Processing {len(episode_dirs)} episode(s) "
        f"[suite={args.suite}  level={args.level}  task={args.task}]"
    )
    if not episode_dirs:
        logger.warning("No episodes to process. Check --episodes and --input_base.")
        return

    backend_kwargs: dict = {}
    if args.vlm.startswith("qwen"):
        backend_kwargs = dict(device=args.device, load_in_4bit=args.load_in_4bit)
    backend = get_backend(args.vlm, **backend_kwargs)

    for ep_dir in episode_dirs:
        ep_idx = int(ep_dir.name.removeprefix("episode_"))
        out = output_path(
            output_base=Path(args.output_base),
            prompt_stem=prompt_stem,
            suite=args.suite,
            level=args.level,
            task_id=args.task,
            ep_idx=ep_idx,
        )
        try:
            run_episode(
                ep_dir=ep_dir,
                system_prompt=prompt_content,
                backend=backend,
                out_path=out,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            logger.error(f"Failed on {ep_dir.name}: {e}", exc_info=True)

    logger.info("Done.")


if __name__ == "__main__":
    main()
