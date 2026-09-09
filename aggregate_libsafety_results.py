"""
aggregate_libsafety_results.py

Combines per-task result JSONs (written by one-task-per-invocation eval runs,
see slurm/eval_fastwam_libsafety_full_v2.slurm) into a single suite-level
summary, matching the aggregate format already used for OpenVLA/OFT/pi0.5
(`*_all_tasks.json`).

Per-task invocation is used instead of a single --all_tasks call so that a
native (non-Python-catchable) crash in one task -- e.g. the MuJoCo/robosuite
EGL rendering abort seen in job 42161922 (SIGABRT inside
robosuite/utils/binding_utils.py:read_pixels, likely GPU-shared-partition
VRAM/EGL-context contention) -- only loses that one task's episodes instead
of killing the whole suite.

Handles both JSON schemas seen in this repo:
  - Fast-WAM: {"per_task": [{"task_index": i, ...}, ...], ...}
  - Cosmos:   {"task_0": {...}, "task_1": {...}, ..., "overall": {...}}

Usage:
    python aggregate_libsafety_results.py \
        --results_dir cosmos_benchmark_libsafety_full \
        --suites human_safety obstacle_avoidance obstacle_avoidance_human affordance
"""

import argparse
import glob
import json
import os
from statistics import mean, median


def extract_per_task_records(data):
    """Returns a dict {task_index: task_result_dict} from either schema."""
    records = {}
    if isinstance(data, dict) and "per_task" in data:
        for r in data["per_task"]:
            records[r["task_index"]] = r
    else:
        for key, val in data.items():
            if key.startswith("task_") and isinstance(val, dict) and "task_index" in val:
                records[val["task_index"]] = val
    return records


def aggregate_suite(results_dir, suite):
    suite_dir = os.path.join(results_dir, suite)
    if not os.path.isdir(suite_dir):
        return None, []

    all_records = {}
    for path in sorted(glob.glob(os.path.join(suite_dir, "*.json"))):
        if os.path.basename(path) == "AGGREGATE_SUMMARY.json":
            continue
        try:
            data = json.load(open(path))
        except Exception as e:
            print(f"  WARNING: failed to parse {path}: {e}")
            continue
        for task_index, record in extract_per_task_records(data).items():
            # Last write wins if a task somehow has multiple result files
            # (e.g. a retry that produced a partial file before a later
            # successful attempt overwrote it under a new timestamp).
            all_records[task_index] = record

    missing = sorted(set(range(15)) - set(all_records.keys()))

    total_episodes = sum(r["episodes"] for r in all_records.values())
    total_successes = sum(r["successes"] for r in all_records.values())
    total_collisions = sum(r["collisions"] for r in all_records.values())
    ets_values = []
    for r in all_records.values():
        ets_values.extend([r["ETS_mean"]] * r["episodes"])

    summary = {
        "suite": suite,
        "tasks_completed": len(all_records),
        "tasks_missing": missing,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_collisions": total_collisions,
        "TSR": round(total_successes / total_episodes, 4) if total_episodes else 0.0,
        "CAR": round((total_episodes - total_collisions) / total_episodes, 4) if total_episodes else 0.0,
        "ETS_mean": round(mean(ets_values), 2) if ets_values else 0.0,
        "ETS_median": round(median(ets_values), 2) if ets_values else 0.0,
        "per_task": [all_records[k] for k in sorted(all_records.keys())],
    }
    return summary, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument(
        "--suites", nargs="+",
        default=["human_safety", "obstacle_avoidance", "obstacle_avoidance_human", "affordance"],
    )
    args = parser.parse_args()

    print(f"{'Suite':<28} {'Episodes':>9} {'TSR':>8} {'CAR':>8} {'Missing tasks'}")
    print("-" * 80)
    for suite in args.suites:
        summary, missing = aggregate_suite(args.results_dir, suite)
        if summary is None:
            print(f"{suite:<28} (no results dir found)")
            continue
        out_path = os.path.join(args.results_dir, suite, "AGGREGATE_SUMMARY.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        missing_str = ",".join(str(m) for m in missing) if missing else "-"
        print(
            f"{suite:<28} {summary['total_episodes']:>9} "
            f"{summary['TSR']:>8.3f} {summary['CAR']:>8.3f}  {missing_str}"
        )


if __name__ == "__main__":
    main()
