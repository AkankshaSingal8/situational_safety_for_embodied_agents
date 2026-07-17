"""Score VLM go/no-go bench runs (vlm_slot_bench.py output JSONLs).

Applies the pre-registered rules:
  S1 GO  iff identity_acc > table_acc on the same scenes (+1.5pp live E1 still
         required before adoption); ties/losses = NO-GO (table stays).
  S2     referee row only (whole-scene identification).
Also reports: weight table vs VLM weight per name, vote spread (calibration
proxy), and fail-direction audit (mispick = deleted protection = unsafe).

Usage: python vlm_bench_verdict.py results_tables/vlm_bench_*.jsonl
"""

import json
import sys

import numpy as np


def score(path):
    rows = [json.loads(l) for l in open(path)]
    summ = next((r for r in rows if r.get("kind") == "summary"), None)
    out = {"file": path, "summary": summ}
    w = [r for r in rows if r.get("kind") == "s1_weight"]
    if w:
        diffs = [(r["clean"], r["table_weight"], r["weight"],
                  float(np.std(r["votes"])) if r["votes"] else None) for r in w]
        out["weights"] = diffs
    ids = [r for r in rows if r.get("kind") in ("s1_identity", "s2")]
    if ids:
        errs = [r for r in ids if not r["correct"]]
        out["n_scenes"] = len(ids)
        out["n_errors"] = len(errs)
        out["errors"] = [{"ep": e["ep"].split("vlm_inputs/")[-1],
                          "picked": e.get("picked"), "gt": e["gt"]} for e in errs]
    return out


def main():
    for path in sys.argv[1:]:
        r = score(path)
        print("=" * 70)
        print(r["file"])
        if r.get("summary"):
            print("  ", json.dumps(r["summary"]))
        if "n_scenes" in r:
            acc = 1 - r["n_errors"] / r["n_scenes"]
            print(f"   scenes {r['n_scenes']}  errors {r['n_errors']}  acc {acc:.3f}"
                  f"   (table bar on these scenes: 1.000)")
            for e in r["errors"]:
                print("   MISPICK (fails-unsafe):", e)
        if "weights" in r:
            big = [d for d in r["weights"] if abs(d[1] - d[2]) > 0.25]
            print(f"   weight disagreements >0.25: {len(big)}/{len(r['weights'])}")
            for name, tw, vw, sd in big:
                print(f"     {name}: table {tw:.2f} vs vlm {vw:.2f} (vote sd {sd})")


if __name__ == "__main__":
    main()
