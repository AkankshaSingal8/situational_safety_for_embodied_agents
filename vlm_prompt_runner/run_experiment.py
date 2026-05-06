#!/usr/bin/env python3
"""
Run one or more VLM models across prompts, compute accuracy, and print a summary table.
Works with local Qwen models and remote API models (Claude, GPT, Gemini) in a single command.

Outputs are stored under {output-base}/{model-slug}/{prompt-stem}/...

Examples
--------
Single model (Qwen local):
  HF_HOME=/ocean/projects/cis250185p/asingal/hf_cache \\
  python -m vlm_prompt_runner.run_experiment \\
      --prompts-dir prompts/obstacle_id \\
      --models qwen3-vl-8b \\
      --suite safelibero_spatial --level I --task 0 \\
      --results-out vlm_prompt_runner/results/qwen3_vl_8b.json

Single model (API):
  python -m vlm_prompt_runner.run_experiment \\
      --prompts-dir prompts/obstacle_id --prompts p8_ranked_collision \\
      --models gpt-5.5 \\
      --suite safelibero_spatial --level I --task 0 \\
      --results-out vlm_prompt_runner/results/test_gpt55_p8.json

Multiple models, specific prompts:
  python -m vlm_prompt_runner.run_experiment \\
      --prompts-dir prompts/obstacle_id \\
      --prompts p5_cot_structured p8_ranked_collision \\
      --models claude-sonnet-4-6 gpt-5.5 gemini-2.5-pro qwen3-vl-8b \\
      --suite safelibero_spatial --level I --task 0 \\
      --results-out vlm_prompt_runner/results/multimodel.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import vlm_prompt_runner.config  # loads .env / .envfile on import

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_BASE = _PROJECT_ROOT / "vlm_inputs"
DEFAULT_OUTPUT_BASE = _PROJECT_ROOT / "vlm_prompt_runner" / "outputs"


def _model_slug(model_key: str) -> str:
    return model_key.replace(".", "_").replace("-", "_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run VLM prompt experiments (local or API) and report accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--prompts-dir", required=True,
                   help="Directory containing .md prompt files")

    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--models", nargs="+",
                             metavar="MODEL",
                             help="One or more model keys")
    model_group.add_argument("--model",
                             metavar="MODEL",
                             help="Single model key (alias for --models with one value)")

    p.add_argument("--suite", required=True)
    p.add_argument("--level", required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--prompts", nargs="*", default=None,
                   help="Specific prompt stems to run (default: all *.md in --prompts-dir)")
    p.add_argument("--results-out", required=True,
                   help="Path to write the JSON results summary")
    p.add_argument("--input-base", default=str(DEFAULT_INPUT_BASE))
    p.add_argument("--output-base", default=str(DEFAULT_OUTPUT_BASE),
                   help="Root output directory; model slug is appended automatically")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="4-bit quantisation for local Qwen models")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Token limit passed to the model (default: let the API decide)")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-run episodes even if output.json already exists")
    return p.parse_args()


def run_model(
    model_key: str,
    prompt_files: list[Path],
    episode_dirs: list[Path],
    args: argparse.Namespace,
    vlm_inputs_dir: Path,
) -> dict:
    """Run all prompt × episode combos for one model. Returns {prompt_stem: stats}."""
    from vlm_prompt_runner.accuracy import compute_accuracy
    from vlm_prompt_runner.backends import get_backend
    from vlm_prompt_runner.episode import output_path
    from vlm_prompt_runner.prompt_loader import load_prompt
    from vlm_prompt_runner.runner import run_episode

    slug = _model_slug(model_key)
    model_output_base = Path(args.output_base) / slug

    backend_kwargs: dict = {}
    if model_key.startswith("qwen"):
        backend_kwargs["load_in_4bit"] = args.load_in_4bit

    logger.info("=== Model: %s ===", model_key)
    backend = get_backend(model_key, **backend_kwargs)
    skip_existing = not args.no_skip_existing

    model_results: dict = {}

    for prompt_file in prompt_files:
        stem = prompt_file.stem
        logger.info("  Prompt: %s", stem)
        prompt_content = load_prompt(prompt_file)

        for ep_dir in episode_dirs:
            ep_idx = int(ep_dir.name.removeprefix("episode_"))
            out = output_path(
                output_base=model_output_base,
                prompt_stem=stem,
                suite=args.suite,
                level=args.level,
                task_id=args.task,
                ep_idx=ep_idx,
            )
            if skip_existing and out.exists():
                logger.debug("    Skipping %s (output exists)", ep_dir.name)
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
                logger.error("    Failed %s: %s", ep_dir.name, exc, exc_info=True)

        acc = compute_accuracy(model_output_base / stem, vlm_inputs_dir)
        overall = acc.get("_totals", {}).get("overall", {})
        model_results[stem] = {
            "correct": overall.get("correct", 0),
            "total": overall.get("total", 0),
            "accuracy": overall.get("accuracy", 0.0),
        }
        logger.info("    %s  →  %d/%d  (%.1f%%)",
                    stem,
                    model_results[stem]["correct"],
                    model_results[stem]["total"],
                    model_results[stem]["accuracy"] * 100)

    return model_results


def _print_single_model_table(model_key: str, results: dict) -> None:
    best = max(results, key=lambda k: results[k]["accuracy"]) if results else None
    print(f"\n=== Prompt Accuracy — {model_key} ===")
    print(f"{'Prompt':<35} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print("-" * 65)
    for stem, stats in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
        marker = "  ← best" if stem == best else ""
        print(f"{stem:<35} {stats['correct']:>8} {stats['total']:>6} "
              f"{stats['accuracy']:>9.1%}{marker}")
    print()


def _print_multi_model_table(summary: dict) -> None:
    models = list(summary["models"].keys())
    all_prompts = sorted({p for m in models for p in summary["models"][m]})
    if not models or not all_prompts:
        return

    col_w = max(12, *(len(m) for m in models))
    prompt_w = max(30, *(len(p) for p in all_prompts))

    header = f"{'Prompt':<{prompt_w}}" + "".join(f"  {m:>{col_w}}" for m in models)
    print(f"\n=== Multi-Model Accuracy Comparison ===")
    print(f"Suite: {summary['suite']}  |  Level: {summary['level']}  |  Task: {summary['task']}")
    print(header)
    print("-" * len(header))

    for prompt in all_prompts:
        row = f"{prompt:<{prompt_w}}"
        for model in models:
            stats = summary["models"][model].get(prompt)
            row += f"  {stats['accuracy']:>{col_w}.1%}" if stats else f"  {'N/A':>{col_w}}"
        print(row)

    print("\nBest model per prompt:")
    for prompt in all_prompts:
        best_model = max(
            models,
            key=lambda m: summary["models"][m].get(prompt, {}).get("accuracy", -1),
        )
        best_acc = summary["models"][best_model].get(prompt, {}).get("accuracy", 0)
        print(f"  {prompt:<{prompt_w}}  {best_model}  ({best_acc:.1%})")
    print()


def main() -> None:
    args = parse_args()

    # Normalise --model / --models into a single list
    models = args.models if args.models else [args.model]

    from vlm_prompt_runner.episode import resolve_episodes

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
    logger.info("Episodes: %d  |  Prompts: %d  |  Models: %s",
                len(episode_dirs), len(all_prompt_files), models)

    summary: dict = {
        "suite": args.suite,
        "level": args.level,
        "task": args.task,
        "models": {},
    }

    vlm_inputs_dir = Path(args.input_base)

    for model_key in models:
        summary["models"][model_key] = run_model(
            model_key=model_key,
            prompt_files=all_prompt_files,
            episode_dirs=episode_dirs,
            args=args,
            vlm_inputs_dir=vlm_inputs_dir,
        )

    Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results_out).write_text(json.dumps(summary, indent=2))
    logger.info("Results saved to %s", args.results_out)

    if len(models) == 1:
        _print_single_model_table(models[0], summary["models"][models[0]])
    else:
        _print_multi_model_table(summary)


if __name__ == "__main__":
    main()
