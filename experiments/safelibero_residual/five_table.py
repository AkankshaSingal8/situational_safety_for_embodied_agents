"""TSR / CAR / SR (%) on the 10 random Spatial runs in runs_v2/five_runs.txt (5 level I, 5 level II), per arm.

    python five_table.py
pi0.5, ex7, ex8 come from their full Spatial runs (same runs, same seed); the API arms from runs_v2/five/spatial/<arm>.
"""
import glob
import json

RUNS = [tuple(x.split(":")) for x in open("runs_v2/five_runs.txt").read().strip().split(";")]
ARMS = [("pi0.5", "runs_v2/spatial/test/pi05/*"), ("ex7", "runs_v2/spatial/test/ex7last/true/*"),
        ("ex8", "runs_v2/spatial/test/ex8s6k/true/*"),
        ("Fable 5.2 full control", "runs_v2/five/spatial/fable_base"),
        ("Fable 5.2 residual", "runs_v2/five/spatial/fable_res"),
        ("Astra 6 full control", "runs_v2/five/spatial/astra_base"),
        ("Astra 6 residual", "runs_v2/five/spatial/astra_res")]


def load(pattern):
    out = {}
    for f in glob.glob(pattern + "/part_*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if "error" not in r:
                out[(r["level"], str(r["task"]), str(r["ep"]))] = r
    return out


def main():
    print("| Method | Level | Runs | TSR | CAR | SR |\n|---|---|---|---|---|---|")
    for name, pat in ARMS:
        data = load(pat)
        for lv in ("I", "II"):
            rs = [data[k] for k in RUNS if k[0] == lv and k in data]
            if not rs:
                print(f"| {name} | {lv} | 0/5 | pending | pending | pending |")
                continue
            n = len(rs)
            t = sum(r["success"] for r in rs)
            c = sum(not r["collide"] for r in rs)
            s = sum(r["success"] and not r["collide"] for r in rs)
            print(f"| {name} | {lv} | {n}/5 | {100*t/n:.0f} | {100*c/n:.0f} | {100*s/n:.0f} |")
    print("\nRuns (level:task:episode):", "; ".join(":".join(k) for k in RUNS))


if __name__ == "__main__":
    main()
