#!/usr/bin/env python3
"""
Run all prompts in a directory against a task/level with a given model,
compute accuracy per prompt, and save a results JSON.

Example — all 5 prompts with qwen3:
  python -m vlm_prompt_runner.run_prompt_experiment \\
      --prompts-dir prompts/obstacle_id \\
      --model qwen3-vl-8b \\
      --suite safelibero_spatial --level I --task 0 \\
      --results-out vlm_prompt_runner/results/phase1_qwen3_vl_8b.json

Example — one specific prompt with qwen2.5-vl-7b:
  python -m vlm_prompt_runner.run_prompt_experiment \\
      --prompts-dir prompts/obstacle_id \\
      --prompts p3_candidate_list \\
      --model qwen2.5-vl-7b \\
      --suite safelibero_spatial --level I --task 0 \\
      --results-out vlm_prompt_runner/results/phase2_qwen25_vl_7b.json
"""
import argparse
import json
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
        description="Run obstacle-id prompt experiment and report per-prompt accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--prompts-dir", required=True,
                   help="Directory containing .md prompt files")
    p.add_argument("--model", required=True,
                   help="VLM model key: qwen3-vl-8b | qwen2.5-vl-7b | qwen2.5-vl-3b | dry-run")
    p.add_argument("--suite", required=True)
    p.add_argument("--level", required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--prompts", nargs="*", default=None,
                   help="Specific prompt stems to run (default: all *.md in --prompts-dir)")
    p.add_argument("--results-out", required=True,
                   help="Path to write the JSON results summary")
    p.add_argument("--input-base", default=str(DEFAULT_INPUT_BASE))
    p.add_argument("--output-base", default=str(DEFAULT_OUTPUT_BASE))
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from vlm_prompt_runner.accuracy import compute_accuracy
    from vlm_prompt_runner.backends import get_backend
    from vlm_prompt_runner.episode import output_path, resolve_episodes
    from vlm_prompt_runner.prompt_loader import load_prompt
    from vlm_prompt_runner.runner import run_episode

    prompts_dir = Path(args.prompts_dir)
    all_prompt_files = sorted(
        f for f in prompts_dir.glob("*.md")
        if f.stem.lower() != "readme"
    )
    if args.prompts:
        all_prompt_files = [f for f in all_prompt_files if f.stem in args.prompts]
    if not all_prompt_files:
        logger.error("No prompt files found in %s (filter=%s)", prompts_dir, args.prompts)
        sys.exit(1)

    episode_dirs = resolve_episodes(
        input_base=Path(args.input_base),
        suite=args.suite,
        level=args.level,
        task_id=args.task,
        episodes=None,
    )
    logger.info("Episodes: %d  |  Prompts: %d  |  Model: %s",
                len(episode_dirs), len(all_prompt_files), args.model)

    backend_kwargs: dict = {}
    if args.model.startswith("qwen"):
        backend_kwargs = {"load_in_4bit": args.load_in_4bit}
    backend = get_backend(args.model, **backend_kwargs)

    vlm_inputs_dir = Path(args.input_base)
    summary: dict = {
        "model": args.model,
        "suite": args.suite,
        "level": args.level,
        "task": args.task,
        "prompts": {},
    }

    for prompt_file in all_prompt_files:
        stem = prompt_file.stem
        logger.info("--- Prompt: %s ---", stem)
        prompt_content = load_prompt(prompt_file)

        for ep_dir in episode_dirs:
            ep_idx = int(ep_dir.name.removeprefix("episode_"))
            out = output_path(
                output_base=Path(args.output_base),
                prompt_stem=stem,
                suite=args.suite,
                level=args.level,
                task_id=args.task,
                ep_idx=ep_idx,
            )
            if out.exists():
                logger.info("  Skipping %s (output exists)", ep_dir.name)
                continue
            try:
                run_episode(
                    ep_dir=ep_dir,
                    system_prompt=prompt_content,
                    backend=backend,
                    out_path=out,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as exc:
                logger.error("  Failed %s: %s", ep_dir.name, exc, exc_info=True)

        stl_dir = Path(args.output_base) / stem
        acc = compute_accuracy(stl_dir, vlm_inputs_dir)
        overall = acc.get("_totals", {}).get("overall", {})
        summary["prompts"][stem] = {
            "correct": overall.get("correct", 0),
            "total": overall.get("total", 0),
            "accuracy": overall.get("accuracy", 0.0),
        }
        logger.info("  Accuracy: %d/%d  (%.1f%%)",
                    summary["prompts"][stem]["correct"],
                    summary["prompts"][stem]["total"],
                    summary["prompts"][stem]["accuracy"] * 100)

    if summary["prompts"]:
        best = max(summary["prompts"], key=lambda k: summary["prompts"][k]["accuracy"])
        summary["best_prompt"] = best
    else:
        best = None
        summary["best_prompt"] = None

    Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results_out).write_text(json.dumps(summary, indent=2))
    logger.info("Results saved to %s", args.results_out)

    # Human-readable table
    print("\n=== Prompt Accuracy Summary ===")
    print(f"Model: {args.model}  |  Suite: {args.suite}  |  Level: {args.level}  |  Task: {args.task}")
    print(f"{'Prompt':<35} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print("-" * 65)
    for stem, stats in sorted(summary["prompts"].items(),
                               key=lambda x: -x[1]["accuracy"]):
        marker = "  ← best" if stem == best else ""
        print(f"{stem:<35} {stats['correct']:>8} {stats['total']:>6} "
              f"{stats['accuracy']:>9.1%}{marker}")
    print()


if __name__ == "__main__":
    main()
