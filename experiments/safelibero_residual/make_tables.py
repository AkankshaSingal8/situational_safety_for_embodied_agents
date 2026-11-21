"""Print the 4 ex2 eval tables (markdown) from eval_sl.py outputs. Cell = TSR / CAR / SR in %.

    python make_tables.py [--ex ex2_1gpu_bs64] [--arm true]

Table 1a  official SafeLIBERO episodes 0-49            runs_v2/<s>/test/<arm dir>/eps{0-19,20-39,40-49}
Table 1b  official episodes 0-39 (training layouts)    same files, ep < 40
Table 2   25 unseen GSO objects x 4 runs, layouts 0-39 runs_v2/<s>/scenes_newobj25/<arm dir>/all
Table 3   benchmark obstacles, held-out layouts 40-49  runs_v2/<s>/scenes_newlayout/<arm dir>/all
TSR = task success, CAR = no obstacle collision, SR = success and no collision.
"""
import argparse
import glob
import json

SUITES = ["spatial", "object", "goal", "long"]


def load(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(r["level"], r["task"], r.get("scene", r.get("ep")), r.get("ep"))] = r
    return list(out.values())


def cell(rs):
    n = len(rs)
    if not n:
        return "pending", 0
    t = 100 * sum(r["success"] for r in rs) / n
    c = 100 * sum(not r["collide"] for r in rs) / n
    s = 100 * sum(r["success"] and not r["collide"] for r in rs) / n
    return f"{t:.1f} / {c:.1f} / {s:.1f}", n


def table(title, sub, arms, keep=lambda r: True, extra=None):
    hdr = ["Suite", "Level"] + (["Objects x runs"] if extra else []) + ["Runs"] + [a for a, _ in arms]
    lines = [f"### {title}", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for s in SUITES:
        data = {a: [r for r in load(f"runs_v2/{s}/{sub}/{d}/*/part_*.jsonl") if keep(r)] for a, d in arms}
        for L in ("I", "II"):
            cells, n = [], 0
            for a, _ in arms:
                txt, k = cell([r for r in data[a] if r["level"] == L])
                cells.append(txt)
                n = max(n, k)
            lines.append("| " + " | ".join([s.capitalize(), L] + ([extra] if extra else []) + [str(n)] + cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ex", default="ex2_1gpu_bs64")
    ap.add_argument("--arm", default="true")
    a = ap.parse_args()
    arms = [("pi0.5", "pi05"), ("ex2", f"{a.ex}/{a.arm}")]
    print(table("Table 1a: official episodes 0-49", "test", arms))
    print(table("Table 1b: official episodes 0-39 (training layouts)", "test", arms, keep=lambda r: r["ep"] < 40))
    print(table("Table 2: 25 unseen objects, training layouts 0-39", "scenes_newobj25", arms, extra="25 x 4"))
    print(table("Table 3: benchmark obstacles, held-out layouts 40-49", "scenes_newlayout", arms))


if __name__ == "__main__":
    main()
