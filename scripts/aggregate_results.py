"""
aggregate_results.py

Reads OpenVLA and Pi0.5 SafeLIBERO benchmark result JSONs and generates:
  1. summary_table.{md,csv}    — 8-row × 6-metric summary (suite × level)
  2. {model}_{suite}_{level}_per_task.{md,csv} — per-task table for each condition

Usage:
  python scripts/aggregate_results.py
  python scripts/aggregate_results.py \\
      --openvla-dir results/safelibero/policy_baselines/openvla_oft \\
      --pi05-dir    results/safelibero/policy_baselines/pi05 \\
      --output-dir  results/tables
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


SUITES = ["safelibero_spatial", "safelibero_object", "safelibero_goal", "safelibero_long"]
SUITE_LABELS = {
    "safelibero_spatial": "SPATIAL",
    "safelibero_object":  "OBJECT",
    "safelibero_goal":    "GOAL",
    "safelibero_long":    "LONG",
}
LEVELS = ["I", "II"]


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate SafeLIBERO benchmark results")
    p.add_argument("--openvla-dir", default="results/safelibero/policy_baselines/openvla_oft")
    p.add_argument("--pi05-dir",    default="results/safelibero/policy_baselines/pi05")
    p.add_argument("--output-dir",  default="results/tables")
    return p.parse_args()


def latest_result_file(results_dir: Path) -> Optional[Path]:
    """Return the most recently written results_*.json in results_dir, or None."""
    candidates = sorted(results_dir.glob("results_*.json"))
    return candidates[-1] if candidates else None


def load_result(results_dir: Path, suite: str, level: str) -> Optional[Dict]:
    """Load the latest result JSON for a given (suite, level).

    Two layouts exist in results/ and both must be readable:

      nested: <suite>/<level>/results_*.json        (pi0.5, Cosmos, Fast-WAM)
      flat:   <suite>/results_*-level<LEVEL>-*.json (OpenVLA-OFT)

    Only the nested form was handled before, so every OpenVLA column in a
    regenerated table came out empty while the committed table showed numbers --
    i.e. the published tables could not be reproduced from the published data.
    """
    nested = results_dir / suite / level
    f = latest_result_file(nested)
    if f is None:
        # flat layout: the level is encoded in the filename as "-levelI-" / "-levelII-"
        suite_dir = results_dir / suite
        if suite_dir.is_dir():
            marker = f"-level{level}-"
            candidates = sorted(
                p for p in suite_dir.glob("results_*.json") if marker in p.name
            )
            f = candidates[-1] if candidates else None
    if f is None:
        return None
    with open(f) as fh:
        return json.load(fh)


def overall_metric(result: Optional[Dict], name: str):
    """Read an overall metric from either result schema.

    Two shapes exist in results/ and both are real:
      flat:   {"overall_TSR": ...}              (pi0.5, Cosmos, Fast-WAM)
      nested: {"overall": {"TSR": ...}}         (OpenVLA-OFT, the FOL driver)

    Only the flat form was read before, so every OpenVLA cell in a regenerated
    table was blank even when the data was present.
    """
    if not result:
        return None
    flat = result.get(f"overall_{name}")
    if flat is not None:
        return flat
    overall = result.get("overall")
    if isinstance(overall, dict):
        return overall.get(name)
    return None


def fmt_pct(v) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"


def fmt_float(v) -> str:
    if v is None:
        return "—"
    return f"{v:.1f}"


def write_summary_table(rows: List[Dict], output_dir: Path):
    """Write the 8-row summary table as .md and .csv."""
    md_lines = [
        "| Suite | Level | OpenVLA TSR | OpenVLA CAR | OpenVLA ETS | Pi0.5 TSR | Pi0.5 CAR | Pi0.5 ETS |",
        "|-------|-------|-------------|-------------|-------------|-----------|-----------|-----------|",
    ]
    csv_rows = [["Suite", "Level", "OpenVLA TSR", "OpenVLA CAR", "OpenVLA ETS",
                 "Pi0.5 TSR", "Pi0.5 CAR", "Pi0.5 ETS"]]

    for r in rows:
        md_lines.append(
            f"| {r['suite_label']} | {r['level']} "
            f"| {r['openvla_tsr']} | {r['openvla_car']} | {r['openvla_ets']} "
            f"| {r['pi05_tsr']} | {r['pi05_car']} | {r['pi05_ets']} |"
        )
        csv_rows.append([
            r["suite_label"], r["level"],
            r["openvla_tsr"], r["openvla_car"], r["openvla_ets"],
            r["pi05_tsr"],    r["pi05_car"],    r["pi05_ets"],
        ])

    md_path = output_dir / "summary_table.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {md_path}")

    csv_path = output_dir / "summary_table.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"Wrote {csv_path}")

    # Also print to terminal
    print()
    for line in md_lines:
        print(line)
    print()


def write_per_task_table(model: str, suite: str, level: str, result: dict, output_dir: Path):
    """Write the per-task table for one (model, suite, level) condition."""
    per_task = result.get("per_task", [])
    if not per_task:
        return

    suite_short = suite.replace("safelibero_", "")
    stem = f"{model}_{suite_short}_{level}"

    header = ["Task", "Description", "Episodes", "Successes", "Collisions", "TSR", "CAR", "ETS (mean)"]

    md_lines = [
        f"## {model.upper()} — {suite_short.upper()} Level {level}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---:"] * len(header)) + " |",
    ]
    csv_rows = [header]

    for t in per_task:
        tid = t["task_id"]
        desc = t["task_description"]
        eps = t["episodes"]
        succ = t["successes"]
        coll = t["collisions"]
        tsr = fmt_pct(t.get("TSR"))
        car = fmt_pct(t.get("CAR"))
        ets = fmt_float(t.get("ETS_mean"))

        md_lines.append(f"| {tid} | {desc} | {eps} | {succ} | {coll} | {tsr} | {car} | {ets} |")
        csv_rows.append([tid, desc, eps, succ, coll, tsr, car, ets])

    # Overall row
    ov_eps  = result.get("total_episodes", "—")
    ov_succ = result.get("total_successes", "—")
    ov_coll = result.get("total_collisions", "—")
    ov_tsr  = fmt_pct(overall_metric(result, "TSR"))
    ov_car  = fmt_pct(overall_metric(result, "CAR"))
    ov_ets  = fmt_float(overall_metric(result, "ETS_mean"))

    md_lines.append(f"| **Overall** | | **{ov_eps}** | **{ov_succ}** | **{ov_coll}** | **{ov_tsr}** | **{ov_car}** | **{ov_ets}** |")
    csv_rows.append(["Overall", "", ov_eps, ov_succ, ov_coll, ov_tsr, ov_car, ov_ets])

    md_path = output_dir / f"{stem}_per_task.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {md_path}")

    csv_path = output_dir / f"{stem}_per_task.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"Wrote {csv_path}")


def main():
    args = parse_args()
    repo_root = Path(__file__).parent.parent
    openvla_dir = repo_root / args.openvla_dir
    pi05_dir    = repo_root / args.pi05_dir
    output_dir  = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    missing = []

    for suite in SUITES:
        for level in LEVELS:
            ov = load_result(openvla_dir, suite, level)
            pi = load_result(pi05_dir,    suite, level)

            if ov is None:
                missing.append(f"OpenVLA  {suite} L{level}")
            if pi is None:
                missing.append(f"Pi0.5    {suite} L{level}")

            summary_rows.append({
                "suite_label": SUITE_LABELS[suite],
                "level": level,
                "openvla_tsr": fmt_pct(overall_metric(ov, "TSR")),
                "openvla_car": fmt_pct(overall_metric(ov, "CAR")),
                "openvla_ets": fmt_float(overall_metric(ov, "ETS_mean")),
                "pi05_tsr":    fmt_pct(overall_metric(pi, "TSR")),
                "pi05_car":    fmt_pct(overall_metric(pi, "CAR")),
                "pi05_ets":    fmt_float(overall_metric(pi, "ETS_mean")),
            })

            if ov:
                write_per_task_table("openvla", suite, level, ov, output_dir)
            if pi:
                write_per_task_table("pi05", suite, level, pi, output_dir)

    write_summary_table(summary_rows, output_dir)

    if missing:
        print("WARNING: Missing results for the following conditions (shown as '—'):")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
