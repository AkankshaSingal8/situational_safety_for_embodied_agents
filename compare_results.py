"""
Compare baseline vs FOL filter results.

Usage:
    python compare_results.py \
        --baseline baseline_benchmark \
        --fol fol_benchmark \
        [--suite safelibero_spatial] [--level I]
"""

import argparse, glob, json, os
from pathlib import Path


def load_results(results_dir: str, suite: str, level: str):
    """Load the most recent results JSON from a benchmark directory."""
    pattern = os.path.join(results_dir, suite, f"*level{level}*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        pattern2 = os.path.join(results_dir, "**", f"*{suite}*{level}*.json")
        files = sorted(glob.glob(pattern2, recursive=True))
    if not files:
        # Try to find any json
        files = sorted(glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True))
    if not files:
        return None
    latest = files[-1]
    print(f"  Loading: {latest}")
    with open(latest) as f:
        return json.load(f)


def extract_metrics(data: dict) -> dict:
    """Pull TSR, CAR, ETS from the results dict."""
    if data is None:
        return {}
    overall = data.get("overall", data)
    return {
        "TSR": overall.get("TSR", overall.get("tsr", overall.get("task_success_rate", None))),
        "CAR": overall.get("CAR", overall.get("car", overall.get("collision_avoidance_rate", None))),
        "ETS": overall.get("ETS_mean", overall.get("ets_mean", overall.get("mean_steps", None))),
        "episodes": overall.get("total_episodes", None),
        "fol_enabled": overall.get("fol_filter_enabled", False),
        "fol_level": overall.get("fol_level", None),
    }


def print_comparison(baseline, fol):
    print("\n" + "="*60)
    print("RESULTS COMPARISON: Baseline vs FOL Safety Filter")
    print("="*60)
    print(f"{'Metric':<20} {'Baseline':>12} {'FOL Filter':>12} {'Delta':>10}")
    print("-"*60)

    for metric in ["TSR", "CAR", "ETS"]:
        b = baseline.get(metric)
        f = fol.get(metric)
        if b is not None and f is not None:
            delta = f - b
            arrow = "↑" if (delta > 0 and metric in ("TSR", "CAR")) or \
                          (delta < 0 and metric == "ETS") else "↓"
            print(f"  {metric:<18} {b:>12.3f} {f:>12.3f} {delta:>+9.3f} {arrow}")
        else:
            print(f"  {metric:<18} {str(b):>12} {str(f):>12}")

    print("-"*60)
    print(f"\n  FOL level: {fol.get('fol_level', 'N/A')}")
    print(f"  Episodes: baseline={baseline.get('episodes')}, fol={fol.get('episodes')}")

    # Key assessment
    car_b = baseline.get("CAR", 0) or 0
    car_f = fol.get("CAR", 0) or 0
    tsr_b = baseline.get("TSR", 0) or 0
    tsr_f = fol.get("TSR", 0) or 0

    print("\n" + "="*60)
    if car_f > car_b + 0.1:
        print(f"✓ CAR improved by {car_f-car_b:.1%} — FOL filter is working!")
    elif car_f > car_b:
        print(f"~ CAR improved by {car_f-car_b:.1%} — modest improvement")
    else:
        print(f"✗ CAR did not improve ({car_b:.3f} → {car_f:.3f}) — needs tuning")

    if tsr_f >= tsr_b - 0.05:
        print(f"✓ TSR maintained ({tsr_b:.3f} → {tsr_f:.3f}) — filter not over-constraining")
    else:
        print(f"⚠ TSR dropped by {tsr_b-tsr_f:.1%} — filter may be too aggressive")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="baseline_benchmark")
    parser.add_argument("--fol", default="fol_benchmark")
    parser.add_argument("--suite", default="safelibero_spatial")
    parser.add_argument("--level", default="I")
    args = parser.parse_args()

    print(f"Comparing: {args.baseline} vs {args.fol}")
    print(f"Suite: {args.suite}  Level: {args.level}")

    b_data = load_results(args.baseline, args.suite, args.level)
    f_data = load_results(args.fol, args.suite, args.level)

    if b_data is None:
        print(f"Baseline results not found in {args.baseline}")
    if f_data is None:
        print(f"FOL results not found in {args.fol}")

    if b_data and f_data:
        print_comparison(extract_metrics(b_data), extract_metrics(f_data))
    else:
        print("One or both result files missing — jobs may still be running.")
        if b_data:
            print("Baseline results:")
            print(json.dumps(extract_metrics(b_data), indent=2))
        if f_data:
            print("FOL results:")
            print(json.dumps(extract_metrics(f_data), indent=2))
