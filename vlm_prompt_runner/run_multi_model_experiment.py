#!/usr/bin/env python3
"""
Run all combinations of models × prompts, compute per-combo accuracy,
and output a comparison table.

Example — compare 3 API models on all prompts:
  python -m vlm_prompt_runner.run_multi_model_experiment \\
      --prompts-dir prompts/obstacle_id \\
      --models claude-sonnet-4-6 gpt-4.1 gemini-2.0-flash \\
      --suite safelibero_spatial --level I --task 0 \\
      --results-out vlm_prompt_runner/results/multimodel_comparison.json

Example — specific prompts only:
  python -m vlm_prompt_runner.run_multi_model_experiment \\
      --prompts-dir prompts/obstacle_id \\
      --prompts p3_candidate_list p5_cot_structured \\
      --models claude-sonnet-4-6 gpt-4.5-preview gemini-2.5-pro \\
      --suite safelibero_spatial --level I --task 0 \\
      --results-out vlm_prompt_runner/results/multimodel_phase2.json
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


def _model_slug(model_key: str) -> str:
    """Convert model key to filesystem-safe directory name."""
    return model_key.replace(".", "_").replace("-", "_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare multiple VLM models across prompts; report accuracy table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--prompts-dir", required=True,
                   help="Directory containing .md prompt files")
    p.add_argument("--models", nargs="+", required=True,
                   help=(
                       "One or more model keys: "
                       "claude-sonnet-4-6 | claude-opus-4-7 | "
                       "gpt-4.5-preview | gpt-4.1 | gpt-4.1-mini | "
                       "gemini-2.5-pro | gemini-2.0-flash | "
                       "qwen3-vl-8b | dry-run"
                   ))
    p.add_argument("--suite", required=True)
    p.add_argument("--level", required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--prompts", nargs="*", default=None,
                   help="Specific prompt stems to run (default: all *.md in --prompts-dir)")
    p.add_argument("--results-out", required=True,
                   help="Path to write the JSON results summary")
    p.add_argument("--input-base", default=str(DEFAULT_INPUT_BASE))
    p.add_argument("--output-base", default=str(_PROJECT_ROOT / "vlm_prompt_runner" / "outputs"),
                   help="Root for per-model output dirs (model slug appended automatically)")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="4-bit quantization for Qwen local models")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip episodes whose output.json already exists (default: True)")
    return p.parse_args()


def run_model(
    model_key: str,
    prompt_files: list[Path],
    episode_dirs: list[Path],
    args: argparse.Namespace,
    vlm_inputs_dir: Path,
) -> dict:
    """Run all prompts for one model. Returns {prompt_stem: {correct, total, accuracy}}."""
    from vlm_prompt_runner.accuracy import compute_accuracy
    from vlm_prompt_runner.backends import get_backend
    from vlm_prompt_runner.episode import output_path
    from vlm_prompt_runner.prompt_loader import load_prompt
    from vlm_prompt_runner.runner import run_episode

    slug = _model_slug(model_key)
    model_output_base = Path(args.output_base) / slug

    backend_kwargs: dict = {}
    if model_key.startswith("qwen"):
        backend_kwargs = {"load_in_4bit": args.load_in_4bit}

    logger.info("=== Model: %s ===", model_key)
    backend = get_backend(model_key, **backend_kwargs)

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
            if args.skip_existing and out.exists():
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

        stl_dir = model_output_base / stem
        acc = compute_accuracy(stl_dir, vlm_inputs_dir)
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


def print_comparison_table(summary: dict) -> None:
    models = list(summary["models"].keys())
    all_prompts = sorted({
        p for m in models for p in summary["models"][m]
    })

    if not models or not all_prompts:
        return

    col_w = max(12, *(len(m) for m in models))
    prompt_w = max(30, *(len(p) for p in all_prompts))

    header = f"{'Prompt':<{prompt_w}}" + "".join(f"  {m:>{col_w}}" for m in models)
    print()
    print("=== Multi-Model Accuracy Comparison ===")
    print(f"Suite: {summary['suite']}  |  Level: {summary['level']}  |  Task: {summary['task']}")
    print(header)
    print("-" * len(header))

    for prompt in all_prompts:
        row = f"{prompt:<{prompt_w}}"
        for model in models:
            stats = summary["models"][model].get(prompt)
            if stats:
                row += f"  {stats['accuracy']:>{col_w}.1%}"
            else:
                row += f"  {'N/A':>{col_w}}"
        print(row)

    print()
    print("Best model per prompt:")
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
                len(episode_dirs), len(all_prompt_files), args.models)

    summary: dict = {
        "suite": args.suite,
        "level": args.level,
        "task": args.task,
        "models": {},
    }

    vlm_inputs_dir = Path(args.input_base)

    for model_key in args.models:
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

    print_comparison_table(summary)


if __name__ == "__main__":
    main()
