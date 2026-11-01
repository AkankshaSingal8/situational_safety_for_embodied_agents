"""Aggregate v2 rollouts (both splits) into tables + paired statistics.
    python analyze_v2.py --evals runs_v2/eval_train runs_v2/eval_test --out runs_v2
writes <out>/results_v2.json and <out>/results_v2.md
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.texts import (TRAIN_A, TRAIN_B, TRAIN_AB, HELDOUT_A, HELDOUT_B, HELDOUT2_A, HELDOUT2_B,  # noqa: E402
                       COMBINED, COMBINED_FAR)
from src.traj import _path_normal  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--evals", nargs="+", required=True)
ap.add_argument("--data_dir", default="data_v2")
ap.add_argument("--out", default="runs_v2")
args = ap.parse_args()


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(rs, f):
    k = sum(bool(f(r)) for r in rs)
    lo, hi = wilson(k, len(rs))
    return dict(k=k, n=len(rs), p=k / max(len(rs), 1), lo=lo, hi=hi)


F = {
    "viol_A": lambda r: r["viol_A"],
    "viol_B": lambda r: r["viol_B"],
    "success": lambda r: r["success"],
    "A_ok&succ": lambda r: (not r["viol_A"]) and r["success"],
    "B_ok&succ": lambda r: (not r["viol_B"]) and r["success"],
    "AB_ok&succ": lambda r: (not r["viol_A"]) and (not r["viol_B"]) and r["success"],
}


def p_or_none(rs):
    """both-ok & success rate, or None when the arm has no rollouts (never a fake 0)."""
    return rate(rs, F["AB_ok&succ"])["p"] if rs else None


def nfmt(v):
    return "n/a" if v is None else f"{v:.2f}"


def fmt(x):
    return f"{x['p']:.3f} [{x['lo']:.3f},{x['hi']:.3f}]"


def mcnemar(by, conds1, conds2, f):
    """Exact McNemar on flow sample 0. conds1/conds2: equal-length lists of conditions,
    paired element-wise and by scene (pooling over several sentences pairs (sentence, scene))."""
    b = c = n = 0
    for c1, c2 in zip(conds1, conds2):
        a = {r["scene"]: bool(f(r)) for r in by.get(c1, []) if r["sample"] == 0}
        o = {r["scene"]: bool(f(r)) for r in by.get(c2, []) if r["sample"] == 0}
        for k in set(a) & set(o):
            n += 1
            b += a[k] and not o[k]
            c += o[k] and not a[k]
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return dict(n=n, b=b, c=c, p=float(p))


Z = np.load(os.path.join(args.data_dir, "text_emb.npz"), allow_pickle=True)
E = dict(zip(Z["texts"], Z["emb"]))
TA = np.stack([E[t] for t in TRAIN_A]); TB = np.stack([E[t] for t in TRAIN_B])
NC = len(COMBINED)
ARMS3 = [("main|combined", "combined sentence, fed directly (main)"),
         ("compose|combined", "velocity composition (main)"),
         ("stack_AB|combined", "clauses in turn, A then B (main)"),
         ("stack_BA|combined", "clauses in turn, B then A (main)"),
         ("sum|combined", "two corrections added (main)"),
         ("shuffled|combined", "shuffled-text model"),
         ("main_ab|combined", "trained on combined phrasings (main_ab)"),
         ("main_ab|trainAB", "main_ab, its own training combined phrasings"),
         ("main|far", "far set: sentence fed directly (main)"),
         ("stack_AB|far", "far set: clauses in turn, A then B (main)"),
         ("stack_BA|far", "far set: clauses in turn, B then A (main)"),
         ("sum|far", "far set: two corrections added (main)"),
         ("main_ab|far", "far set: trained on combined phrasings (main_ab)")]

RES, MD = {}, ["# Residual-flow POC v2 results\n"]
for ev in args.evals:
    R = [json.loads(l) for l in open(os.path.join(ev, "rollouts.jsonl"))]
    split = R[0]["split"]
    by = defaultdict(list)
    for r in R:
        by[r["cond"]].append(r)
    base_exec = {r["scene"]: np.array(r["exec_grip"]) for r in by["base"]}

    def summ(rs):
        out = {k: rate(rs, f) for k, f in F.items()}
        out["exec_dev_pos_m"] = float(np.mean([np.linalg.norm(np.array(r["exec_grip"]) - base_exec[r["scene"]], axis=-1).mean() for r in rs]))
        out["corr_pos_mean_m"] = float(np.mean([r["corr_pos_mean"] for r in rs]))
        out["corr_rot_mean_deg"] = float(np.mean([r["corr_rot_mean_deg"] for r in rs]))
        out["max_tilt_deg"] = float(np.mean([r["max_tilt_deg"] for r in rs]))
        out["mean_track_err_m"] = float(np.mean([r["mean_track_err"] for r in rs]))
        return out

    S = {c: summ(rs) for c, rs in by.items()}
    res = dict(split=split, n_rollouts=len(R), n_scenes=len(base_exec), summary=S, tests={})
    MD.append(f"\n# Split: {split} ({len(base_exec)} scenes, {len(R)} rollouts)\n")

    # ---- control + Exp 1
    MD.append("## Control check and Exp 1 (training phrasings)\n")
    MD.append("| condition | n | viol A | viol B | success | A-ok & succ | B-ok & succ | corr pos (m) | corr rot (deg) |")
    MD.append("|---|---|---|---|---|---|---|---|---|")
    for c in ["base", "oracle_A", "oracle_B", "oracle_AB", "main|A", "main|B", "main|null", "notext",
              "shuffled|A", "shuffled|B", "mean_A", "mean_B", "main_ab|A", "main_ab|B", "main_ab|null"]:
        if c in S:
            s = S[c]
            MD.append(f"| {c} | {s['success']['n']} | {fmt(s['viol_A'])} | {fmt(s['viol_B'])} | {fmt(s['success'])} | "
                      f"{fmt(s['A_ok&succ'])} | {fmt(s['B_ok&succ'])} | {s['corr_pos_mean_m']:.3f} | {s['corr_rot_mean_deg']:.1f} |")
    MD.append("\n| paired comparison (sample 0, by scene) | metric | n | b | c | p |")
    MD.append("|---|---|---|---|---|---|")
    for c1, c2, m in [("main|A", "shuffled|A", "A_ok&succ"), ("main|A", "notext", "A_ok&succ"),
                      ("main|A", "mean_A", "A_ok&succ"), ("main|B", "shuffled|B", "B_ok&succ"),
                      ("main|B", "notext", "B_ok&succ"), ("main|B", "mean_B", "B_ok&succ"),
                      ("main|A", "main|B", "A_ok&succ"), ("main|B", "main|A", "B_ok&succ")]:
        if c1 in by and c2 in by:
            t = mcnemar(by, [c1], [c2], F[m]); res["tests"][f"{c1} vs {c2} [{m}]"] = t
            MD.append(f"| {c1} vs {c2} | {m} | {t['n']} | {t['b']} | {t['c']} | {t['p']:.2g} |")
    # detour side agreement with the oracle
    W = np.load(os.path.join(ev, "waypoints.npz"), allow_pickle=True)
    SPD = np.load(os.path.join(args.data_dir, f"{split}.npz"), allow_pickle=True)
    agree = []
    for mj, r in zip(W["meta"], W["residual"]):
        m = json.loads(mj)
        if m["cond"] != "main|A":
            continue
        n = _path_normal(SPD["scenes"][m["scene"]])     # detour side = sign along the path normal
        agree.append(np.sign(r[20, :2] @ n) == np.sign(SPD["dA"][m["scene"], 20, :2] @ n))
    res["detour_side_agreement"] = float(np.mean(agree))
    MD.append(f"\nmain|A detours on the oracle's side on {100 * np.mean(agree):.1f}% of samples.\n")

    # ---- Exp 2
    MD.append("## Exp 2 (unseen paraphrases, model main)\n")
    MD.append("| set | phrasing | class | max cos A | max cos B | target ok & succ | other still violated | corr pos (m) | corr rot (deg) |")
    MD.append("|---|---|---|---|---|---|---|---|---|")
    ex2 = {}
    for tag, LA, LB in (("heldout", HELDOUT_A, HELDOUT_B), ("heldout2", HELDOUT2_A, HELDOUT2_B)):
        for cls, L in (("A", LA), ("B", LB)):
            other = "B" if cls == "A" else "A"
            for j, t in enumerate(L):
                c = f"main|{tag}_{cls}{j}"
                if c not in by:
                    continue
                rs = by[c]
                tok = rate(rs, F[f"{cls}_ok&succ"]); oth = rate(rs, F[f"viol_{other}"])
                ex2[c] = dict(text=t, set=tag, cls=cls, cos_A=float((TA @ E[t]).max()), cos_B=float((TB @ E[t]).max()),
                              target_ok_succ=tok, other_violated=oth, corr_pos=S[c]["corr_pos_mean_m"], corr_rot=S[c]["corr_rot_mean_deg"])
                MD.append(f"| {tag} | {t} | {cls} | {ex2[c]['cos_A']:.3f} | {ex2[c]['cos_B']:.3f} | {fmt(tok)} | {fmt(oth)} | "
                          f"{S[c]['corr_pos_mean_m']:.3f} | {S[c]['corr_rot_mean_deg']:.1f} |")
            pooled = [r for c in by if c.startswith(f"main|{tag}_{cls}") for r in by[c]]
            res[f"exp2_{tag}_{cls}"] = rate(pooled, F[f"{cls}_ok&succ"])
    res["exp2"] = ex2
    MD.append("")
    for tag in ("heldout", "heldout2"):
        for cls in ("A", "B"):
            MD.append(f"- pooled {tag} {cls}: {fmt(res[f'exp2_{tag}_{cls}'])} (training phrasings: {fmt(S[f'main|{cls}'][f'{cls}_ok&succ'])})")

    # ---- Exp 3
    MD.append("\n## Exp 3 (combined instruction; the six COMBINED sentences are unseen by every model)\n")
    MD.append("| arm | n | viol A | viol B | success | both ok & succ |")
    MD.append("|---|---|---|---|---|---|")
    res["exp3"] = {}
    for pre, lab in ARMS3:
        rs = [r for c in by if c.startswith(pre) for r in by[c]]
        if not rs:
            continue
        s = {k: rate(rs, f) for k, f in F.items()}
        res["exp3"][pre] = dict(label=lab, **s)
        MD.append(f"| {lab} | {len(rs)} | {fmt(s['viol_A'])} | {fmt(s['viol_B'])} | {fmt(s['success'])} | {fmt(s['AB_ok&succ'])} |")
    MD.append("\nPer sentence, both ok & success:\n")
    MD.append("| sentence | max cos to TRAIN_AB | direct | compose | stack A→B | stack B→A | sum | main_ab |")
    MD.append("|---|---|---|---|---|---|---|---|")
    res["exp3_per_sentence"] = {}
    for j, (t, _) in enumerate(COMBINED):
        row = {pre: p_or_none(by.get(f"{pre}{j}", []))
               for pre in ("main|combined", "compose|combined", "stack_AB|combined", "stack_BA|combined", "sum|combined", "main_ab|combined")}
        res["exp3_per_sentence"][t] = row
        MD.append(f"| {t} | {float((np.stack([E[x] for x in TRAIN_AB]) @ E[t]).max()):.3f} | "
                  + " | ".join(nfmt(v) for v in row.values()) + " |")
    TAB = np.stack([E[t] for t in TRAIN_AB])
    MD.append("\nFar set (worded away from main_ab's training phrasings), both ok & success:\n")
    MD.append("| sentence | max cos to TRAIN_AB | direct | stack A→B | stack B→A | sum | main_ab |")
    MD.append("|---|---|---|---|---|---|---|")
    res["exp3_far_per_sentence"] = {}
    for j, (t, _) in enumerate(COMBINED_FAR):
        row = {pre: p_or_none(by.get(f"{pre}{j}", []))
               for pre in ("main|far", "stack_AB|far", "stack_BA|far", "sum|far", "main_ab|far")}
        cos = float((TAB @ E[t]).max())
        res["exp3_far_per_sentence"][t] = dict(cos_trainAB=cos, **row)
        MD.append(f"| {t} | {cos:.3f} | " + " | ".join(nfmt(v) for v in row.values()) + " |")
    res["exp3_cos_trainAB"] = {t: float((TAB @ E[t]).max()) for t, _ in COMBINED}
    MD.append("\nPaired, pooled over the six sentences ((sentence, scene) pairs, sample 0; pairs share scenes, so p is optimistic):\n")
    MD.append("| comparison | n | b | c | p |")
    MD.append("|---|---|---|---|---|")
    for p1, p2 in (("main_ab|far", "main|far"), ("main_ab|far", "stack_AB|far"),
                   ("main_ab|combined", "main|combined"), ("stack_AB|combined", "main|combined"),
                   ("main_ab|combined", "stack_AB|combined"), ("main_ab|combined", "sum|combined"),
                   ("main|combined", "shuffled|combined")):
        J = range(len(COMBINED_FAR) if p1.endswith("|far") else NC)
        t = mcnemar(by, [f"{p1}{j}" for j in J], [f"{p2}{j}" for j in J], F["AB_ok&succ"])
        if t["n"]:
            res["tests"][f"{p1} vs {p2} [AB_ok&succ]"] = t
            MD.append(f"| {p1} vs {p2} | {t['n']} | {t['b']} | {t['c']} | {t['p']:.2g} |")
    RES[split] = res

# ---- train vs test side by side
if len(RES) > 1:
    L = ["\n# Train vs test, headline numbers\n", "| metric | " + " | ".join(RES) + " |", "|---|" + "---|" * len(RES)]

    def row(name, get):
        L.append(f"| {name} | " + " | ".join(fmt(get(RES[s])) for s in RES) + " |")

    row("base viol A", lambda r: r["summary"]["base"]["viol_A"])
    row("base viol B", lambda r: r["summary"]["base"]["viol_B"])
    row("oracle A+B both ok & succ", lambda r: r["summary"]["oracle_AB"]["AB_ok&succ"])
    row("Exp1 main A text: A ok & succ", lambda r: r["summary"]["main|A"]["A_ok&succ"])
    row("Exp1 main A text: B still violated", lambda r: r["summary"]["main|A"]["viol_B"])
    row("Exp1 main B text: B ok & succ", lambda r: r["summary"]["main|B"]["B_ok&succ"])
    row("Exp1 main B text: A still violated", lambda r: r["summary"]["main|B"]["viol_A"])
    row("Exp1 no-text: A ok & succ", lambda r: r["summary"]["notext"]["A_ok&succ"])
    row("Exp1 shuffled A text: A ok & succ", lambda r: r["summary"]["shuffled|A"]["A_ok&succ"])
    row("Exp1 mean A residual: A ok & succ", lambda r: r["summary"]["mean_A"]["A_ok&succ"])
    for tag in ("heldout", "heldout2"):
        for cls in ("A", "B"):
            row(f"Exp2 {tag} {cls} pooled", lambda r, k=f"exp2_{tag}_{cls}": r[k])
    for pre, lab in ARMS3:
        if all(pre in RES[s]["exp3"] for s in RES):
            row(f"Exp3 {lab}", lambda r, p=pre: r["exp3"][p]["AB_ok&succ"])
    if all("main_ab|A" in RES[s]["summary"] for s in RES):
        row("main_ab A text: A ok & succ", lambda r: r["summary"]["main_ab|A"]["A_ok&succ"])
        row("main_ab A text: B still violated", lambda r: r["summary"]["main_ab|A"]["viol_B"])
        row("main_ab B text: B ok & succ", lambda r: r["summary"]["main_ab|B"]["B_ok&succ"])
        row("main_ab B text: A still violated", lambda r: r["summary"]["main_ab|B"]["viol_A"])
    MD[1:1] = L

json.dump(RES, open(os.path.join(args.out, "results_v2.json"), "w"), indent=1)
open(os.path.join(args.out, "results_v2.md"), "w").write("\n".join(MD) + "\n")
print("\n".join(MD))
