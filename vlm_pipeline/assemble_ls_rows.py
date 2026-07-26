"""Assemble the LIBERO-Safety first-rows grid from ls_rows/ results JSONs.

Prints per (suite, level): TSR / SafeRate (1 - violation_rate) for each arm,
plus symbolic-arm identification accuracy and refusal counts. Latest results
file per cell wins.
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).parents[1] / "ls_rows"
ARMS = ["baseline", "guided_gt", "guided_symbolic", "guided_fol",
        "guided_nogt_fol", "guided_nogt_sym", "prompt_baseline",
        "prompt_guided", "refusal_baseline", "refusal"]

# Paper (arXiv 2606.23686) Table 3, pi0.5 row, SR = success with ZERO
# violations (violation terminates the episode as failure). Suite mapping:
# obstacle_avoidance=TSA, human_safety=HRI, obstacle_avoidance_human=FSHOA,
# affordance=AAG. SSR track (refusal): best model RR 80/72/36.
PAPER_PI05 = {
    "obstacle_avoidance": {"L0": 58.0, "L1": 62.7, "L2": 56.7},
    "human_safety": {"L0": 84.7, "L1": 88.7, "L2": 83.3},
    "obstacle_avoidance_human": {"L0": 55.3, "L1": 58.7, "L2": 51.3},
    "affordance": {"L0": 78.7, "L1": 59.3, "L2": 35.3},
}


def safe_sr(cell_dir):
    """Paper-style SR: success AND no violation, per episode (dedup by
    (task, ep), last occurrence wins — episode logs append across reruns)."""
    files = sorted(cell_dir.glob("episodes_*.jsonl"))
    if not files:
        return None
    eps = {}
    for line in open(files[-1]):
        r = json.loads(line)
        eps[(r["task"], r["ep"])] = r
    if not eps:
        return None
    ok = sum(1 for r in eps.values() if r["success"] and not r["violation"])
    return round(100 * ok / len(eps), 1)


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
                "SafeSR": safe_sr(cell),
                "refused": r.get("refused_tasks", 0),
                "ident": ident_acc(cell) if arm in ("guided_symbolic", "guided_fol") else None,
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
                if v["SafeSR"] is not None:
                    s += f" S{v['SafeSR']}"
                if v["ident"]:
                    s += f" id{v['ident']}"
                if v["refused"]:
                    s += f" R{v['refused']}"
                line += f"{s:>22}"
        paper = PAPER_PI05.get(key[0], {}).get(key[1])
        if paper is not None:
            line += f"   paper={paper}"
        print(line)


if __name__ == "__main__":
    main()
