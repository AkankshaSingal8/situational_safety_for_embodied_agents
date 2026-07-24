"""Assemble the LIBERO-Safety first-rows grid from ls_rows/ results JSONs.

Prints per (suite, level): TSR / SafeRate (1 - violation_rate) for each arm,
plus symbolic-arm identification accuracy and refusal counts. Latest results
file per cell wins.
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).parents[1] / "ls_rows"
ARMS = ["baseline", "guided_gt", "guided_symbolic", "refusal_baseline", "refusal"]


def latest(cell_dir):
    files = sorted(cell_dir.glob("results_*.json"))
    return json.load(open(files[-1])) if files else None


def ident_acc(cell_dir):
    files = sorted(cell_dir.glob("episodes_*.jsonl"))
    if not files:
        return None
    n = ok = 0
    for line in open(files[-1]):
        r = json.loads(line)
        if r.get("ident_correct") is not None:
            n += 1
            ok += int(r["ident_correct"])
    return f"{ok}/{n}" if n else None


def main():
    rows = {}
    for arm in ARMS:
        for cell in sorted(ROOT.glob(f"{arm}/*/L*")):
            r = latest(cell)
            if r is None:
                continue
            key = (cell.parent.name, cell.name)
            rows.setdefault(key, {})[arm] = {
                "TSR": round(100 * r["overall_TSR"], 1),
                "Safe": round(100 * (1 - r["overall_violation_rate"]), 1),
                "refused": r.get("refused_tasks", 0),
                "ident": ident_acc(cell) if arm == "guided_symbolic" else None,
            }
    hdr = f"{'cell':<26}" + "".join(f"{a:>22}" for a in ARMS)
    print(hdr)
    for key in sorted(rows):
        line = f"{key[0][:20]:<20} {key[1]:<5}"
        for a in ARMS:
            v = rows[key].get(a)
            if v is None:
                line += f"{'-':>22}"
            else:
                s = f"{v['TSR']}/{v['Safe']}"
                if v["ident"]:
                    s += f" id{v['ident']}"
                if v["refused"]:
                    s += f" R{v['refused']}"
                line += f"{s:>22}"
        print(line)


if __name__ == "__main__":
    main()
