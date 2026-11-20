"""Pick ONE residual config per base from validation data only, pooled over suites (SCALEUP.md section 3).

    python select_config.py --base flow --suites spatial,object,goal,long

Criterion (declared before any result): closed-loop validation safe success on official episodes 20-24
(text residual, correct sentence, training seed 0), pooled over every suite's episodes. Configs within
2 pp of the best are tied and broken by the mean offline validation R2 (held-out layouts 20-24).
Writes runs_scale/<suite>/<base>/selected.json for every suite (the same config everywhere) and prints the table.
Reads nothing from runs_scale/*/*/test.
"""
import argparse
import glob
import json
import os

CONFIGS = [f"s{s}_lr{lr}" for s in (25000, 50000) for lr in ("3e-4", "1e-4")]


def rows(d):
    out = {}
    for f in glob.glob(os.path.join(d, "part_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            k = (r["level"], r["task"], r["ep"])
            out.setdefault(k, r)  # first row per episode wins (compare.py rule)
    return list(out.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="flow")
    ap.add_argument("--suites", default="spatial,object,goal,long")
    a = ap.parse_args()
    suites = a.suites.split(",")
    table = {}
    for c in CONFIGS:
        n = ss = tsr = car = 0
        r2 = []
        for s in suites:
            root = f"runs_scale/{s}/{a.base}"
            assert "test" not in c
            rs = rows(f"{root}/val/{c}_true")
            assert len(rs) == 40, f"{root}/val/{c}_true has {len(rs)} episodes, expected 40 (4 tasks x 2 levels x 5)"
            n += len(rs)
            ss += sum(r["success"] and r.get("collide") is False for r in rs)
            tsr += sum(r["success"] for r in rs)
            car += sum(r.get("collide") is False for r in rs)
            fit = json.load(open(f"{root}/{c}/text_img_seed0/fit.json"))
            r2.append(fit["val"]["R2"])
        table[c] = dict(n=n, safe_success=ss / n, tsr=tsr / n, car=car / n, val_R2=sum(r2) / len(r2))
    best = max(v["safe_success"] for v in table.values())
    tied = [c for c, v in table.items() if v["safe_success"] >= best - 0.02]
    pick = max(tied, key=lambda c: table[c]["val_R2"])
    for c, v in table.items():
        print(f"{c:18s} n={v['n']}  safe&succ {v['safe_success']:.3f}  TSR {v['tsr']:.3f}  CAR {v['car']:.3f}  "
              f"valR2 {v['val_R2']:.3f}{'  <- selected' if c == pick else ''}")
    for s in suites:
        json.dump(dict(config=pick, criterion="pooled val safe success (eps 20-24), ties<=2pp by val R2",
                       table=table), open(f"runs_scale/{s}/{a.base}/selected.json", "w"), indent=1)


if __name__ == "__main__":
    main()
