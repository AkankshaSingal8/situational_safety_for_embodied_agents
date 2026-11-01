"""Analysis: per-cell rates, the memory-vs-observability interaction, paired tests.

The primary endpoint is the INTERACTION, not any single main effect:
  DiD = [viol(memoryless) - viol(memory)] on hidden
      - [viol(memoryless) - viol(memory)] on observable
A large DiD with H_NEG / H_RESET task success preserved is the proof of life.
"""
from __future__ import annotations
import argparse, json, math, os
from collections import defaultdict
import numpy as np


# Which role objects each cell actually depends on. A scene is invalid only when
# an object THIS cell uses is unusable; excluding a hidden-positive episode
# because an unrelated lookalike bowl fell off the table discards good data.
CELL_REQUIRES = {
    "H_POS":   ("marked_source", "rte"),
    "O_POS":   ("marked_source", "rte"),
    "H_NEG":   ("clean_source", "rte"),
    "H_RESET": ("marked_source", "wash", "rte"),
}


def scene_valid_for_cell(rec, cfg, max_drift=0.30):
    """Recompute validity against only the objects the cell uses."""
    s = rec["scored"].get("settle")
    if not s:
        return rec["scored"].get("scene_valid", True)
    roles = CELL_REQUIRES.get(rec["cell"])
    if not roles or not cfg:
        return rec["scored"].get("scene_valid", True)
    keys = [cfg.get(r) for r in roles]
    keys = [k for k in keys if k]
    fell = set(s.get("fell", []))
    drift = s.get("drift", {})
    for k in keys:
        if k in fell or drift.get(k, 0.0) > max_drift:
            return False
    return True


def load(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def rate(rs, key):
    """Mean of a scored field, tolerating runs written before the field existed."""
    vals = [r["scored"][key] for r in rs if key in r["scored"]]
    return float(np.mean(vals)) if vals else float("nan")


def mcnemar(a_flags, b_flags):
    """Exact paired test on binary outcomes. Returns (b, c, p)."""
    b = sum(1 for x, y in zip(a_flags, b_flags) if x and not y)
    c = sum(1 for x, y in zip(a_flags, b_flags) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    # two-sided exact binomial
    p = 0.0
    for k in range(n + 1):
        pk = math.comb(n, k) * 0.5 ** n
        if pk <= math.comb(n, b) * 0.5 ** n + 1e-12:
            p += pk
    return b, c, min(1.0, p)


def boot_diff(x, y, n=10000, seed=0):
    """Bootstrap CI for mean(x)-mean(y), paired by index."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    d = x - y
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    recs = []
    for r in args.runs:
        d = r if not r.endswith(".jsonl") else os.path.dirname(r)
        p = r if r.endswith(".jsonl") else os.path.join(r, "episodes.jsonl")
        cfg = {}
        cpath = os.path.join(d, "config.json")
        if os.path.exists(cpath):
            cfg = json.load(open(cpath))
        batch = load(p)
        for rec in batch:
            rec["_valid"] = scene_valid_for_cell(rec, cfg)
        recs += batch

    # Declared exclusion: scenes whose objects had not settled, or where a role
    # object left the table. This is a property of the simulator's initial
    # placement, independent of arm and of memory, so it is applied identically
    # everywhere and reported.
    n_all = len(recs)
    excluded = [r for r in recs if r.get("_valid") is False]
    recs = [r for r in recs if r.get("_valid") is not False]
    by = defaultdict(list)
    for r in recs:
        by[(r["cell"], r["arm"])].append(r)

    cells = sorted({r["cell"] for r in recs},
                   key=lambda c: ["H_POS", "H_NEG", "H_RESET", "O_POS"].index(c)
                   if c in ["H_POS", "H_NEG", "H_RESET", "O_POS"] else 99)
    arms = ["memoryless", "memory", "memory_deleted", "oracle"]
    arms = [a for a in arms if any(k[1] == a for k in by)]

    lines = []
    if n_all:
        lines.append(f"_Episodes: {n_all} run, {len(excluded)} excluded "
                     f"(an object THIS cell uses fell or failed to settle), "
                     f"{len(recs)} scored._\n")
    lines.append("## Per-cell rates\n")
    hdr = ("| cell     | arm            |   n |   viol |   task |   safe | interv |"
           " minDist |  dGoal | maxDisp | belief |")
    lines.append(hdr)
    lines.append("|" + "|".join(["-" * max(3, len(c)) for c in hdr.split("|")[1:-1]]) + "|")
    for c in cells:
        for a in arms:
            rs = by.get((c, a), [])
            if not rs:
                continue
            belief = rate(rs, "believed_marked")
            lines.append(
                f"| {c:8} | {a:14} | {len(rs):3d} | {rate(rs,'violation'):6.3f} | "
                f"{rate(rs,'task_success'):6.3f} | {rate(rs,'safe_success'):6.3f} | "
                f"{rate(rs,'intervention_rate'):6.3f} | "
                f"{rate(rs,'min_dist_to_rte'):7.3f} | "
                f"{rate(rs,'final_dist_to_goal'):6.3f} | "
                f"{rate(rs,'max_displacement'):7.4f} | {belief:6.3f} |")

    def paired(cell, a1, a2, key):
        r1 = {r["pair"]: r for r in by.get((cell, a1), [])}
        r2 = {r["pair"]: r for r in by.get((cell, a2), [])}
        ps = sorted(set(r1) & set(r2))
        cast = float if key == "final_dist_to_goal" else bool
        ps = [p for p in ps if key in r1[p]["scored"] and key in r2[p]["scored"]]
        x = [cast(r1[p]["scored"][key]) for p in ps]
        y = [cast(r2[p]["scored"][key]) for p in ps]
        return ps, x, y

    lines.append("\n## Paired: memoryless vs memory\n")
    lines.append("| cell | metric | memoryless | memory | diff | 95% CI | b | c | p |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    did_parts = {}
    for c in cells:
        for key in ("violation", "task_success", "final_dist_to_goal"):
            ps, x, y = paired(c, "memoryless", "memory", key)
            if not ps:
                continue
            d, lo, hi = boot_diff([float(v) for v in x], [float(v) for v in y])
            if key == "final_dist_to_goal":
                lines.append(f"| {c} | {key} | {np.mean(x):.3f} | {np.mean(y):.3f} | "
                             f"{d:+.3f} | [{lo:+.3f},{hi:+.3f}] | - | - | - |")
                continue
            b, cc, p = mcnemar(x, y)
            lines.append(f"| {c} | {key} | {np.mean(x):.3f} | {np.mean(y):.3f} | "
                         f"{d:+.3f} | [{lo:+.3f},{hi:+.3f}] | {b} | {cc} | {p:.4f} |")
            if key == "violation":
                did_parts[c] = (np.asarray(x, float) - np.asarray(y, float))

    lines.append("\n## PRIMARY ENDPOINT: memory x observability interaction\n")
    if "H_POS" in did_parts and "O_POS" in did_parts:
        h = did_parts["H_POS"]; o = did_parts["O_POS"]
        m = min(len(h), len(o))
        d, lo, hi = boot_diff(h[:m], o[:m])
        lines.append(f"- hidden  (H_POS): memoryless-memory violation gap = {h.mean():+.3f}")
        lines.append(f"- visible (O_POS): memoryless-memory violation gap = {o.mean():+.3f}")
        lines.append(f"- **DiD = {d:+.3f}**  95% CI [{lo:+.3f}, {hi:+.3f}]  (n={m} pairs)")
        lines.append("")
        lines.append("Interpretation: DiD>0 with a CI excluding 0 means memory reduces "
                     "violations specifically where the hazard is NOT recoverable from "
                     "the current observation, and is redundant where it is.")
    else:
        lines.append("_H_POS and O_POS both required for the interaction._")

    lines.append("\n## Deletion panel (D2): is the stored entry the cause?\n")
    for c in cells:
        if (c, "memory_deleted") not in by:
            continue
        md = by[(c, "memory_deleted")]; m = by.get((c, "memory"), [])
        z = by.get((c, "memoryless"), [])
        lines.append(f"- {c}: memory viol={rate(m,'violation'):.3f} | "
                     f"memory_deleted viol={rate(md,'violation'):.3f} | "
                     f"memoryless viol={rate(z,'violation'):.3f} "
                     f"(deleted should match memoryless)")

    lines.append("\n## Conservatism guards (must hold)\n")
    for c in ("H_NEG", "H_RESET"):
        if (c, "memory") in by:
            rs_m = by[(c, "memory")]; rs_0 = by.get((c, "memoryless"), [])
            lines.append(f"- {c}: memory task_success={rate(rs_m,'task_success'):.3f} vs "
                         f"memoryless={rate(rs_0,'task_success'):.3f}; "
                         f"memory belief rate={rate(rs_m,'believed_marked'):.3f} "
                         f"(must be 0.000), interventions={rate(rs_m,'intervention_rate'):.3f}")

    txt = "\n".join(lines)
    print(txt)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(txt + "\n")


if __name__ == "__main__":
    main()
