import json
import re
from pathlib import Path
from typing import Optional

_STOP_WORDS = {
    "between", "and", "the", "a", "an", "in", "on", "of", "with",
    "from", "to", "at", "by", "for",
}


def normalize_name(name: str) -> set[str]:
    """Return meaningful tokens from an object name string."""
    name = re.sub(r"_\d+$", "", name)
    tokens = re.split(r"[_\s]+", name.lower())
    return {t for t in tokens if t and t not in _STOP_WORDS and t != "obstacle"}


def names_match(predicted: str, ground_truth: str) -> bool:
    """Return True if predicted and ground_truth refer to the same object."""
    return bool(normalize_name(predicted) & normalize_name(ground_truth))


def load_ground_truth(episode_dir: Path) -> Optional[str]:
    """Return obstacle.name from metadata.json in episode_dir, or None."""
    meta = episode_dir / "metadata.json"
    if not meta.exists():
        return None
    data = json.loads(meta.read_text())
    return data.get("obstacle", {}).get("name")


def load_prediction(episode_dir: Path) -> Optional[dict]:
    """Return parsed output.json from episode_dir, or None."""
    out = episode_dir / "output.json"
    if not out.exists():
        return None
    return json.loads(out.read_text())


def eval_episode(pred_ep_dir: Path, gt_ep_dir: Path) -> Optional[dict]:
    """Evaluate one episode. Returns None if ground truth is unavailable."""
    gt_name = load_ground_truth(gt_ep_dir)
    if gt_name is None:
        return None
    prediction = load_prediction(pred_ep_dir)
    if prediction is None:
        return {"correct": False, "predicted": None, "ground_truth": gt_name}
    pred_object = prediction.get("object", "")
    return {
        "correct": names_match(pred_object, gt_name),
        "predicted": pred_object,
        "ground_truth": gt_name,
    }


def _make_totals(correct: int, total: int) -> dict:
    rate = correct / total if total else 0.0
    return {"correct": correct, "total": total, "accuracy": round(rate, 4)}


def compute_accuracy(stl_dir: Path, vlm_inputs_dir: Path) -> dict:
    """Walk stl_dir for output.json files and build a nested accuracy report.

    Returns dict structured as:
      { suite: { level: { task: { "correct": int, "total": int, "episodes": [...],
                                  "_totals": {...} },
                          "_totals": {...} },
                 "_totals": {...} },
        "_totals": { "overall": {...} } }
    """
    results: dict = {}

    for output_json in sorted(stl_dir.rglob("output.json")):
        rel = output_json.parent.relative_to(stl_dir)
        parts = rel.parts
        if len(parts) != 4:
            continue
        suite, level, task, episode = parts

        gt_ep_dir = vlm_inputs_dir / suite / level / task / episode
        ep_result = eval_episode(output_json.parent, gt_ep_dir)
        if ep_result is None:
            continue

        ep_result["episode"] = episode
        (results
            .setdefault(suite, {})
            .setdefault(level, {})
            .setdefault(task, {"correct": 0, "total": 0, "episodes": []})
        )
        task_stats = results[suite][level][task]
        task_stats["correct"] += int(ep_result["correct"])
        task_stats["total"] += 1
        task_stats["episodes"].append(ep_result)

    # Roll up _totals at task → level → suite → overall
    overall_correct = overall_total = 0
    for suite, levels in results.items():
        suite_correct = suite_total = 0
        for level, tasks in levels.items():
            level_correct = level_total = 0
            for task, stats in tasks.items():
                c, t = stats["correct"], stats["total"]
                stats["_totals"] = _make_totals(c, t)
                level_correct += c
                level_total += t
            tasks["_totals"] = _make_totals(level_correct, level_total)
            suite_correct += level_correct
            suite_total += level_total
        levels["_totals"] = _make_totals(suite_correct, suite_total)
        overall_correct += suite_correct
        overall_total += suite_total

    results["_totals"] = {"overall": _make_totals(overall_correct, overall_total)}
    return results


def print_report(results: dict) -> None:
    """Print a human-readable accuracy report to stdout."""
    overall = results["_totals"]["overall"]
    print(f"\n=== Obstacle Identification Accuracy ===")
    print(f"Overall: {overall['correct']}/{overall['total']}  ({overall['accuracy']:.1%})\n")

    for suite, levels in results.items():
        if suite == "_totals":
            continue
        s = levels["_totals"]
        print(f"Suite [{suite}]: {s['correct']}/{s['total']}  ({s['accuracy']:.1%})")
        for level, tasks in levels.items():
            if level == "_totals":
                continue
            lv = tasks["_totals"]
            print(f"  Level [{level}]: {lv['correct']}/{lv['total']}  ({lv['accuracy']:.1%})")
            for task, stats in tasks.items():
                if task == "_totals":
                    continue
                ts = stats["_totals"]
                print(f"    Task [{task}]: {ts['correct']}/{ts['total']}  ({ts['accuracy']:.1%})")
                for ep in stats["episodes"]:
                    mark = "✓" if ep["correct"] else "✗"
                    print(f"      {mark} {ep['episode']}: pred={ep['predicted']}  gt={ep['ground_truth']}")
        print()


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Compute obstacle identification accuracy.")
    parser.add_argument(
        "--stl-dir",
        default="vlm_prompt_runner/stl_single_object",
        help="Root of the stl_single_object output tree",
    )
    parser.add_argument(
        "--vlm-inputs-dir",
        default="vlm_inputs",
        help="Root of the vlm_inputs tree containing metadata.json files",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write full results as JSON",
    )
    args = parser.parse_args()

    stl_dir = Path(args.stl_dir)
    vlm_inputs_dir = Path(args.vlm_inputs_dir)

    if not stl_dir.exists():
        print(f"ERROR: stl-dir not found: {stl_dir}", file=sys.stderr)
        sys.exit(1)
    if not vlm_inputs_dir.exists():
        print(f"ERROR: vlm-inputs-dir not found: {vlm_inputs_dir}", file=sys.stderr)
        sys.exit(1)

    results = compute_accuracy(stl_dir, vlm_inputs_dir)
    print_report(results)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nFull results written to {args.json_out}")
