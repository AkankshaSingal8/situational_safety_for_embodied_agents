"""Two independent checks on the published figure, run on a compute node.

1. Palette validation. The dataviz skill's validator is a Node script and Node is
   not installed on this cluster, so the same six checks are computed here:
   OKLab lightness and chroma, CVD simulation (Machado severity 1.0) for the
   three dichromacies, pairwise dE in OKLab x100, and WCAG contrast.

2. Figure-number re-extraction. Every value shown on the page is recomputed from
   the raw episode logs and diffed against what was published, including the
   six-row walkthrough for campaign B pair 0 and the identical-twin claim
   widened from one pair to all pairs.
"""
from __future__ import annotations
import json, math, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# ----------------------------------------------------------------- palette ---
def srgb2lin(c):
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def hex2lin(h):
    h = h.lstrip('#')
    return [srgb2lin(int(h[i:i + 2], 16)) for i in (0, 2, 4)]

def lin2oklab(r, g, b):
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = [x ** (1 / 3) if x > 0 else -((-x) ** (1 / 3)) for x in (l, m, s)]
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)

CVD = {
 'protan': [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998]],
 'deutan': [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881]],
 'tritan': [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
            [0.004733, 0.691367, 0.303900]]}

def simulate(lin, k):
    m = CVD[k]
    return [sum(m[i][j] * lin[j] for j in range(3)) for i in range(3)]

def dE(a, b):
    A, B = lin2oklab(*a), lin2oklab(*b)
    return 100 * math.sqrt(sum((A[i] - B[i]) ** 2 for i in range(3)))

def contrast(a, b):
    L = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    la, lb = L(a), L(b)
    if la < lb: la, lb = lb, la
    return (la + 0.05) / (lb + 0.05)

def check_palette(name, hexes, surface, labels):
    print("=" * 74); print(name, " surface", surface)
    lins = [hex2lin(h) for h in hexes]; surf = hex2lin(surface)
    print("\n  colour                  hex       OKLab L  chroma  contrast")
    low = []
    for h, l, lab in zip(hexes, lins, labels):
        L, a, b = lin2oklab(*l); C = math.hypot(a, b); cr = contrast(l, surf)
        if cr < 3.0: low.append(lab)
        print(f"  {lab:22s} {h}   {L:.3f}   {C:.3f}   {cr:.2f}")
    print("\n  adjacent pair                        normal   protan  deutan  tritan")
    wn, wc = 99, 99
    for i in range(len(hexes) - 1):
        n = dE(lins[i], lins[i + 1]); wn = min(wn, n)
        ds = {k: dE(simulate(lins[i], k), simulate(lins[i + 1], k)) for k in CVD}
        wc = min(wc, min(ds.values()))
        pair = f"{labels[i].split(' (')[0]} / {labels[i+1].split(' (')[0]}"
        print(f"  {pair:36s} {n:6.1f}   " +
              "  ".join(f"{ds[k]:5.1f}" for k in ('protan', 'deutan', 'tritan')))
    print(f"\n  worst adjacent normal-vision dE {wn:5.1f}  floor 15  -> "
          f"{'PASS' if wn >= 15 else 'FAIL'}")
    verdict = 'PASS' if wc >= 8 else ('WARN (6-8, needs secondary encoding)' if wc >= 6 else 'FAIL')
    print(f"  worst adjacent CVD dE           {wc:5.1f}  target 8  -> {verdict}")
    if low:
        print(f"  relief rule applies to: {', '.join(low)}")
        print("  -> every bar on the page carries a direct value label and a table view exists")
    return wn, wc

LABELS = ["No memory (blue)", "With memory (orange)", "Oracle (aqua)", "Deleted (yellow)"]
check_palette("LIGHT MODE", ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], "#fcfcfb", LABELS)
check_palette("DARK MODE", ["#3987e5", "#d95926", "#199e70", "#c98500"], "#1a1a19", LABELS)

# ------------------------------------------------------------ figure numbers ---
REQ = {"H_POS": ("marked_source", "rte"), "O_POS": ("marked_source", "rte"),
       "H_NEG": ("clean_source", "rte"), "H_RESET": ("marked_source", "wash", "rte")}

def load(dirs):
    out = []
    for d in dirs:
        cfg = json.load(open(f"{d}/config.json"))
        for line in open(f"{d}/episodes.jsonl"):
            r = json.loads(line)
            s = r["scored"].get("settle")
            keep = True
            if s:
                keys = [cfg.get(x) for x in REQ[r["cell"]]]
                keep = not any(k in set(s.get("fell", [])) or s["drift"].get(k, 0) > 0.30
                               for k in keys if k)
            if keep:
                r["_dir"] = d
                out.append(r)
    return out

print("\n" + "=" * 74); print("PUBLISHED NUMBERS vs RECOMPUTED (campaign B)")
B = load(["runs/libero_v2", "runs/libero_v2b"])
by = defaultdict(list)
for r in B:
    by[(r["cell"], r["arm"])].append(r)

PUBLISHED = {  # exactly what the artifact shows
 ("H_POS", "memoryless"): (0.750, 0.875, 0.156, 0.000),
 ("H_POS", "memory"):     (0.281, 0.719, 0.500, 0.257),
 ("H_POS", "memory_deleted"): (0.750, 0.875, 0.156, 0.000),
 ("H_NEG", "memoryless"): (0.000, 0.906, 0.906, 0.000),
 ("H_NEG", "memory"):     (0.000, 0.906, 0.906, 0.000),
 ("H_RESET", "memoryless"): (0.062, 0.875, 0.812, 0.000),
 ("H_RESET", "memory"):   (0.031, 0.875, 0.844, 0.005),
 ("O_POS", "memoryless"): (0.281, 0.719, 0.500, 0.257),
 ("O_POS", "memory"):     (0.281, 0.719, 0.500, 0.257),
}
mean = lambda rs, k: sum(r["scored"][k] for r in rs) / len(rs)
bad = 0
print(f"  {'cell/arm':30s} {'published':>10s} {'recomputed':>11s}  metric")
for (cell, arm), pub in PUBLISHED.items():
    rs = by.get((cell, arm), [])
    if not rs:
        print(f"  {cell+'/'+arm:30s}  NO DATA"); bad += 1; continue
    got = (mean(rs, "violation"), mean(rs, "task_success"),
           mean(rs, "safe_success"), mean(rs, "intervention_rate"))
    for p, g, nm in zip(pub, got, ("violation", "task", "safe", "interv")):
        if abs(p - g) > 0.0015:
            print(f"  {cell+'/'+arm:30s} {p:10.3f} {g:11.3f}  {nm}  <-- MISMATCH")
            bad += 1
print(f"  n per cell/arm: {sorted({len(v) for v in by.values()})}")
print(f"  MISMATCHES: {bad}" if bad else "  ALL PUBLISHED CELL VALUES MATCH")

# --------------------------------------------------- pair 0 walkthrough rows ---
print("\n" + "=" * 74); print("WALKTHROUGH ROWS (campaign B, pair 0) as published")
for cell, arm in [("H_POS", "memoryless"), ("H_POS", "memory"),
                  ("H_POS", "memory_deleted"), ("H_POS", "oracle"),
                  ("H_NEG", "memoryless"), ("H_NEG", "memory")]:
    rs = [r for r in B if r["cell"] == cell and r["arm"] == arm
          and r["pair"] == 0 and r["_dir"] == "runs/libero_v2"]
    if not rs:
        print(f"  {cell}/{arm}: pair 0 excluded or absent"); continue
    s = rs[0]["scored"]
    print(f"  {cell:8s}/{arm:15s} marked={s['truly_marked']!s:5s} "
          f"belief={s['believed_marked']!s:5s} mind={s['min_dist_to_rte']:.3f} "
          f"disp={s.get('max_displacement',float('nan')):.4f} viol={s['violation']!s:5s} "
          f"task={s['task_success']}")

# ------------------------------------------- twin claim widened to all pairs ---
print("\n" + "=" * 74)
print("TWIN CHECK: is the no-memory arm identical between H_POS and H_NEG?")
pairs = sorted({r["pair"] for r in B})
same = tot = 0
worst = 0.0
for d in ("runs/libero_v2", "runs/libero_v2b"):
    for p in pairs:
        a = [r for r in B if r["_dir"] == d and r["pair"] == p
             and r["cell"] == "H_POS" and r["arm"] == "memoryless"]
        b = [r for r in B if r["_dir"] == d and r["pair"] == p
             and r["cell"] == "H_NEG" and r["arm"] == "memoryless"]
        if not a or not b:
            continue
        tot += 1
        ta, tb = a[0]["scored"]["traj"], b[0]["scored"]["traj"]
        if len(ta) == len(tb):
            dmax = max((abs(x - y) for ra, rb in zip(ta, tb) for x, y in zip(ra, rb)),
                       default=0.0)
            worst = max(worst, dmax)
            if dmax < 1e-9:
                same += 1
print(f"  pairs compared: {tot}")
print(f"  bit-identical no-memory trajectories: {same}/{tot}")
print(f"  largest coordinate difference seen: {worst:.3e} m")
print("  -> without memory the robot behaves the same whether or not the gripper is marked"
      if same == tot and tot else "  -> NOT uniformly identical, see above")
print("\nDONE")
