"""Aggregate rollouts into the three experiment tables + paired statistics.
    python analyze.py --eval runs/eval
writes <eval>/results.json and <eval>/results.md
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.texts import TRAIN_A, TRAIN_B, HELDOUT_A, HELDOUT_B, COMBINED  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--eval", default="runs/eval")
args = ap.parse_args()

R = [json.loads(l) for l in open(os.path.join(args.eval, "rollouts.jsonl"))]
by = defaultdict(list)
for r in R:
    by[r["cond"]].append(r)

base_exec = {r["scene"]: np.array(r["exec_grip"]) for r in by["base"]}
base_tilt = {r["scene"]: np.array(r["exec_tilt"]) for r in by["base"]}


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


def summarize(cond):
    rs = by[cond]
    out = {k: rate(rs, f) for k, f in F.items()}
    dev = [np.linalg.norm(np.array(r["exec_grip"]) - base_exec[r["scene"]], axis=-1).mean() for r in rs]
    out["exec_dev_pos_m"] = float(np.mean(dev))
    out["corr_pos_mean_m"] = float(np.mean([r["corr_pos_mean"] for r in rs]))
    out["corr_rot_mean_deg"] = float(np.mean([r["corr_rot_mean_deg"] for r in rs]))
    out["max_tilt_deg"] = float(np.mean([r["max_tilt_deg"] for r in rs]))
    out["min_laptop_clear_m"] = float(np.mean([r["min_laptop_clearance"] for r in rs]))
    out["mean_track_err_m"] = float(np.mean([r["mean_track_err"] for r in rs]))
    return out


def scene_vec(cond, f, sample=0):
    d = {r["scene"]: bool(f(r)) for r in by[cond] if r["sample"] == sample}
    return d


def mcnemar(c1, c2, f, sample=0):
    """Paired by scene on a single flow sample. b = c1 good & c2 bad."""
    a, b_ = scene_vec(c1, f, sample), scene_vec(c2, f, sample)
    keys = sorted(set(a) & set(b_))
    b = sum(a[k] and not b_[k] for k in keys)
    c = sum(b_[k] and not a[k] for k in keys)
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return dict(n=len(keys), b=b, c=c, p=float(p), c1=c1, c2=c2)


S = {c: summarize(c) for c in by}
res = dict(summary=S, tests={})


def fmt(x):
    return f"{x['p']:.2f} [{x['lo']:.2f},{x['hi']:.2f}]"


md = []
md.append("# Residual-flow POC results\n")
md.append(f"Rollouts: {len(R)}; test scenes: {len(base_exec)}. Rates are over all flow samples; "
          "brackets are 95% Wilson CIs. Paired tests use flow sample 0, paired by scene (exact McNemar).\n")

# ---------------- Exp 1 ----------------
rows1 = ["base", "oracle_A", "oracle_B", "oracle_AB", "main|A", "main|B", "main|null",
         "notext", "shuffled|A", "shuffled|B", "mean_A", "mean_B"]
md.append("## Exp 1 - text-conditioned correction (training phrasings, unseen scenes)\n")
md.append("| condition | n | viol A (over laptop) | viol B (tilt>15deg) | task success | A-ok & success | B-ok & success | exec. dev. from base (m) | corr. pos (m) | corr. rot (deg) |")
md.append("|---|---|---|---|---|---|---|---|---|---|")
for c in rows1:
    if c not in S:
        continue
    s = S[c]
    md.append(f"| {c} | {s['viol_A']['n']} | {fmt(s['viol_A'])} | {fmt(s['viol_B'])} | {fmt(s['success'])} | "
              f"{fmt(s['A_ok&succ'])} | {fmt(s['B_ok&succ'])} | {s['exec_dev_pos_m']:.3f} | {s['corr_pos_mean_m']:.3f} | {s['corr_rot_mean_deg']:.1f} |")

tests1 = [
    ("main|A", "shuffled|A", "A_ok&succ"), ("main|A", "notext", "A_ok&succ"), ("main|A", "mean_A", "A_ok&succ"),
    ("main|B", "shuffled|B", "B_ok&succ"), ("main|B", "notext", "B_ok&succ"), ("main|B", "mean_B", "B_ok&succ"),
    # swap: same scene, same model, only the text changes
    ("main|A", "main|B", "A_ok&succ"), ("main|B", "main|A", "B_ok&succ"),
]
md.append("\nPaired tests (b = first condition better on that scene, c = second better):\n")
md.append("| comparison | metric | n | b | c | p |")
md.append("|---|---|---|---|---|---|")
for c1, c2, m in tests1:
    if c1 in by and c2 in by:
        t = mcnemar(c1, c2, F[m])
        res["tests"][f"{c1} vs {c2} [{m}]"] = t
        md.append(f"| {c1} vs {c2} | {m} | {t['n']} | {t['b']} | {t['c']} | {t['p']:.2g} |")

# swap specificity matrix
if "main|A" in S and "main|B" in S:
    md.append("\nSwap specificity (same model, same scenes; only the instruction changes):\n")
    md.append("| instruction | P(viol A) | P(viol B) | corr. pos (m) | corr. rot (deg) |")
    md.append("|---|---|---|---|---|")
    for c in ("main|A", "main|B", "main|null"):
        s = S[c]
        md.append(f"| {c.split('|')[1]} | {fmt(s['viol_A'])} | {fmt(s['viol_B'])} | {s['corr_pos_mean_m']:.3f} | {s['corr_rot_mean_deg']:.1f} |")
    # per-scene detour side agreement with the oracle (shows the A correction is scene-dependent, not a lookup)
    import numpy as _np
    TE = _np.load("data/test.npz", allow_pickle=True)
    W = _np.load(os.path.join(args.eval, "waypoints.npz"), allow_pickle=True)
    metas = [json.loads(m) for m in W["meta"]]
    agree = []
    for m, r in zip(metas, W["residual"]):
        if m["cond"] == "main|A":
            agree.append(_np.sign(r[20, 1]) == _np.sign(TE["dA"][m["scene"], 20, 1]))
    if agree:
        res["detour_side_agreement_main|A"] = float(_np.mean(agree))
        md.append(f"\nDetour side chosen by main|A agrees with the oracle's side on {100*_np.mean(agree):.1f}% of samples "
                  f"(oracle side is +y on {100*_np.mean(TE['dA'][:, 20, 1] > 0):.0f}% of test scenes).\n")

# ---------------- Exp 2 ----------------
Z = np.load("data/text_emb.npz", allow_pickle=True)
E = dict(zip(Z["texts"], Z["emb"]))
TA = np.stack([E[t] for t in TRAIN_A]); TB = np.stack([E[t] for t in TRAIN_B])
md.append("\n## Exp 2 - unseen paraphrases\n")
md.append("| phrasing | class | max cos to train A | max cos to train B | P(target constraint ok) | P(other constraint still violated) | task success | target-ok & success |")
md.append("|---|---|---|---|---|---|---|---|")
ex2 = {}
for cls, L in (("A", HELDOUT_A), ("B", HELDOUT_B)):
    for j, t in enumerate(L):
        c = f"main|heldout_{cls}{j}"
        if c not in S:
            continue
        rs = by[c]
        ok = rate(rs, lambda r: not r[f"viol_{cls}"])
        other = "B" if cls == "A" else "A"
        keep = rate(rs, lambda r: r[f"viol_{other}"])
        both = rate(rs, lambda r: (not r[f"viol_{cls}"]) and r["success"])
        ca, cb = float((TA @ E[t]).max()), float((TB @ E[t]).max())
        ex2[c] = dict(text=t, cls=cls, cos_A=ca, cos_B=cb, target_ok=ok, other_violated=keep, target_ok_succ=both, success=S[c]["success"])
        md.append(f"| {t} | {cls} | {ca:.3f} | {cb:.3f} | {fmt(ok)} | {fmt(keep)} | {fmt(S[c]['success'])} | {fmt(both)} |")
res["exp2"] = ex2
# pooled held-out vs training phrasings
for cls in ("A", "B"):
    held = [r for c in by if c.startswith(f"main|heldout_{cls}") for r in by[c]]
    tr = [r for r in by.get(f"main|{cls}", [])]
    if held and tr:
        f = F[f"{cls}_ok&succ"]
        res[f"exp2_pooled_{cls}"] = dict(heldout=rate(held, f), train=rate(tr, f))
        md.append(f"\nPooled {cls}: held-out phrasings {fmt(rate(held, f))} vs training phrasings {fmt(rate(tr, f))} ({cls}-ok & success).")

# ---------------- Exp 3 ----------------
md.append("\n\n## Exp 3 - unseen combined instruction A+B\n")
md.append("| condition | instruction | viol A | viol B | task success | both ok & success | corr. pos (m) | corr. rot (deg) |")
md.append("|---|---|---|---|---|---|---|---|")
rows3 = ["oracle_AB", "main|A", "main|B", "mean_A+mean_B"]
for j in range(len(COMBINED)):
    rows3 += [f"main|combined{j}", f"compose|combined{j}", f"shuffled|combined{j}"]
for c in rows3:
    if c not in S:
        continue
    s = S[c]
    txt = by[c][0].get("text") or ""
    md.append(f"| {c} | {txt} | {fmt(s['viol_A'])} | {fmt(s['viol_B'])} | {fmt(s['success'])} | {fmt(s['AB_ok&succ'])} | {s['corr_pos_mean_m']:.3f} | {s['corr_rot_mean_deg']:.1f} |")
for kind in ("main", "compose", "shuffled"):
    pooled = [r for c in by if c.startswith(f"{kind}|combined") for r in by[c]]
    if pooled:
        res[f"exp3_pooled_{kind}"] = {k: rate(pooled, f) for k, f in F.items()}
        md.append(f"\nPooled {kind}|combined*: both-ok&success {fmt(res[f'exp3_pooled_{kind}']['AB_ok&succ'])}, "
                  f"viol A {fmt(res[f'exp3_pooled_{kind}']['viol_A'])}, viol B {fmt(res[f'exp3_pooled_{kind}']['viol_B'])}, "
                  f"success {fmt(res[f'exp3_pooled_{kind}']['success'])}")
md.append("\nPaired tests, Exp 3 (AB-ok & success):\n")
md.append("| comparison | n | b | c | p |")
md.append("|---|---|---|---|---|")
for j in range(len(COMBINED)):
    for c1, c2 in ((f"main|combined{j}", f"shuffled|combined{j}"), (f"compose|combined{j}", f"main|combined{j}"),
                   (f"main|combined{j}", "main|A"), (f"main|combined{j}", "main|B")):
        if c1 in by and c2 in by:
            t = mcnemar(c1, c2, F["AB_ok&succ"])
            res["tests"][f"{c1} vs {c2} [AB_ok&succ]"] = t
            md.append(f"| {c1} vs {c2} | {t['n']} | {t['b']} | {t['c']} | {t['p']:.2g} |")

json.dump(res, open(os.path.join(args.eval, "results.json"), "w"), indent=1)
open(os.path.join(args.eval, "results.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
