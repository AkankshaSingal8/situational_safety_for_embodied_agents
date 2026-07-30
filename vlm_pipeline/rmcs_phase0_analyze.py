"""
rmcs_phase0_analyze.py — RMCS Phase 0 gate analysis (diagnostics only).

Reads k_diag_*.jsonl files written by run_guided_safelibero_pi05_eval.py
(--log_k_diag) and computes, per (cell, task_id, ep) episode:

  - stall_fraction: fraction of replans whose net EEF displacement over the
    trailing 40 control steps is < 0.015 m (replan cadence is 5 steps, so
    this compares against the record ~8 replans earlier via the `t` field).
  - diversity_in_stall: fraction of stalled replans whose K candidate prefix
    displacements span >= 2 signatures. Signature = (xy single-linkage
    cluster id at 0.03 m threshold, side bit). Side bit = sign of the 2D
    cross product (obstacle_xy - eef_xy) x cand_disp_xy for the nearest
    logged obstacle (omitted when no obstacles are logged).
  - revisit_rate: among stalled replans with a previous stalled replan in
    the same episode, fraction whose SELECTED candidate matches the
    previously selected one (same side bit AND selected cand_disp xy within
    0.03 m).

Pre-registered gate (pooled over stalled replans):
  GO   iff diversity_in_stall >= 0.25 AND revisit_rate >= 0.50
  NO-GO otherwise
  INVALID if pooled n_stalled < 20 (never a verdict on thin data).

Usage:
  python rmcs_phase0_analyze.py --diag_glob "rmcs_phase0/*/*/*/k_diag_*.jsonl" \
      --out rmcs_phase0/gate.json
  python rmcs_phase0_analyze.py --self_test

Pure numpy + stdlib.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

STALL_WINDOW_STEPS = 40      # control steps for the trailing-displacement test
STALL_DISP_M = 0.015         # net EEF displacement threshold (m)
LINKAGE_M = 0.03             # single-linkage threshold on cand_disp xy (m)
REVISIT_MATCH_M = 0.03       # selected-disp xy match threshold (m)
MIN_STALLED_POOLED = 20      # below this: verdict INVALID
GATE_DIVERSITY = 0.25
GATE_REVISIT = 0.50


def load_records(diag_glob):
    """Return {cell: {(task_id, ep): [record, ...]}} with records sorted by t.

    Cell id = directory of the jsonl file (results_dir), which under the
    Phase 0 harness is <results_output_dir>/<cell>/<suite>/<level>/.
    """
    cells = {}
    files = sorted(glob.glob(diag_glob))
    for f in files:
        cell = os.path.dirname(f)
        eps = cells.setdefault(cell, {})
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec.get("task_id", -1), rec.get("ep", -1))
                eps.setdefault(key, []).append(rec)
    for eps in cells.values():
        for recs in eps.values():
            recs.sort(key=lambda r: r.get("t", 0))
    return cells, files


def _single_linkage_clusters(xy, thresh):
    """Union-find single-linkage clustering of (K, 2) points; returns (K,) ids."""
    k = xy.shape[0]
    parent = list(range(k))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(k):
        for j in range(i + 1, k):
            if np.linalg.norm(xy[i] - xy[j]) <= thresh:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    return np.array([find(i) for i in range(k)])


def _side_bits(rec):
    """Per-candidate side bit vs the nearest logged obstacle, or None.

    Side bit = sign of cross2d(obstacle_xy - eef_xy, cand_disp_xy).
    """
    obstacles = rec.get("obstacles") or []
    if not obstacles:
        return None
    eef = np.asarray(rec["eef"], dtype=float)
    obs = np.asarray(obstacles, dtype=float)
    nearest = obs[np.argmin(np.linalg.norm(obs[:, :2] - eef[None, :2], axis=1))]
    v = nearest[:2] - eef[:2]
    disp = np.asarray(rec["cand_disp"], dtype=float)[:, :2]
    cross = v[0] * disp[:, 1] - v[1] * disp[:, 0]
    return np.sign(cross).astype(int)


def _signatures(rec):
    """Return list of (cluster_id, side_bit) per candidate (side_bit may be None)."""
    disp = np.asarray(rec["cand_disp"], dtype=float)[:, :2]
    clusters = _single_linkage_clusters(disp, LINKAGE_M)
    sides = _side_bits(rec)
    if sides is None:
        return [(int(c), None) for c in clusters]
    return [(int(c), int(s)) for c, s in zip(clusters, sides)]


def analyze_episode(recs):
    """Per-episode metrics from time-sorted replan records.

    Returns dict with n_replans, n_stalled, n_diverse_stalled,
    n_revisit_eligible, n_revisit.
    """
    n = len(recs)
    eefs = [np.asarray(r["eef"], dtype=float) for r in recs]
    ts = [int(r.get("t", 0)) for r in recs]

    stalled = [False] * n
    for i in range(n):
        # latest earlier record at least STALL_WINDOW_STEPS control steps back
        j = None
        for jj in range(i - 1, -1, -1):
            if ts[i] - ts[jj] >= STALL_WINDOW_STEPS:
                j = jj
                break
        if j is None:
            continue
        stalled[i] = bool(np.linalg.norm(eefs[i] - eefs[j]) < STALL_DISP_M)

    n_stalled = 0
    n_diverse = 0
    n_rev_elig = 0
    n_rev = 0
    prev_sel = None  # (side_bit, sel_disp_xy) at previous stalled replan
    for i in range(n):
        if not stalled[i]:
            continue
        n_stalled += 1
        sigs = _signatures(recs[i])
        if len(set(sigs)) >= 2:
            n_diverse += 1
        bi = int(recs[i].get("best_idx", -1))
        disp = np.asarray(recs[i]["cand_disp"], dtype=float)
        if 0 <= bi < disp.shape[0]:
            sides = _side_bits(recs[i])
            sel_side = None if sides is None else int(sides[bi])
            sel_xy = disp[bi, :2]
            if prev_sel is not None:
                n_rev_elig += 1
                p_side, p_xy = prev_sel
                if (sel_side == p_side
                        and np.linalg.norm(sel_xy - p_xy) <= REVISIT_MATCH_M):
                    n_rev += 1
            prev_sel = (sel_side, sel_xy)
    return {
        "n_replans": n,
        "n_stalled": n_stalled,
        "n_diverse_stalled": n_diverse,
        "n_revisit_eligible": n_rev_elig,
        "n_revisit": n_rev,
    }


def _summarize(counts):
    n_rep = sum(c["n_replans"] for c in counts)
    n_st = sum(c["n_stalled"] for c in counts)
    n_div = sum(c["n_diverse_stalled"] for c in counts)
    n_re = sum(c["n_revisit_eligible"] for c in counts)
    n_rv = sum(c["n_revisit"] for c in counts)
    return {
        "n_replans": n_rep,
        "n_stalled": n_st,
        "stall_fraction": (n_st / n_rep) if n_rep else 0.0,
        "diversity_in_stall": (n_div / n_st) if n_st else 0.0,
        "revisit_rate": (n_rv / n_re) if n_re else 0.0,
        "n_revisit_eligible": n_re,
    }


def run_analysis(diag_glob, out_path):
    cells, files = load_records(diag_glob)
    summary = {"diag_glob": diag_glob, "n_files": len(files), "cells": {}}
    all_counts = []
    for cell, eps in sorted(cells.items()):
        counts = [analyze_episode(recs) for recs in eps.values()]
        all_counts.extend(counts)
        summary["cells"][cell] = _summarize(counts)
        summary["cells"][cell]["n_episodes"] = len(eps)
    pooled = _summarize(all_counts)
    summary["pooled"] = pooled

    if pooled["n_stalled"] < MIN_STALLED_POOLED:
        verdict = "INVALID"
        note = (f"pooled n_stalled={pooled['n_stalled']} < {MIN_STALLED_POOLED}: "
                "too few stalled replans for a verdict; the harness needs more "
                "episodes (or the cells simply do not stall).")
    elif (pooled["diversity_in_stall"] >= GATE_DIVERSITY
          and pooled["revisit_rate"] >= GATE_REVISIT):
        verdict = "GO"
        note = "diversity and revisit gates both met (pooled over stalled replans)."
    else:
        verdict = "NO-GO"
        fails = []
        if pooled["diversity_in_stall"] < GATE_DIVERSITY:
            fails.append(f"diversity_in_stall {pooled['diversity_in_stall']:.3f} "
                         f"< {GATE_DIVERSITY} (nothing to select)")
        if pooled["revisit_rate"] < GATE_REVISIT:
            fails.append(f"revisit_rate {pooled['revisit_rate']:.3f} "
                         f"< {GATE_REVISIT} (no loop to break)")
        note = "; ".join(fails)
    summary["gate"] = {
        "verdict": verdict,
        "note": note,
        "thresholds": {"diversity_in_stall": GATE_DIVERSITY,
                       "revisit_rate": GATE_REVISIT,
                       "min_stalled_pooled": MIN_STALLED_POOLED},
    }

    print(json.dumps(summary, indent=2))
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote {out_path}")
    print(f"\nVERDICT: {verdict} — {note}")
    return summary


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    import tempfile

    def make_records(task_id, ep, n_replans, moving, k=2):
        """Stationary episodes stall after t>=40; moving ones never do."""
        recs = []
        for i in range(n_replans):
            t = 5 * i
            eef = [0.1 + (0.02 * i if moving else 0.0), 0.1, 1.0]
            recs.append({
                "task_id": task_id, "ep": ep, "t": t, "eef": eef,
                # two xy modes 0.10 m apart -> 2 clusters; obstacle straight
                # ahead in +y -> opposite side bits -> 2 signatures
                "cand_disp": [[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]][:k],
                "cand_min_margin": [0.02] * k,
                "best_idx": 0,
                "obstacles": [[0.1, 0.2, 1.0]],
            })
        return recs

    with tempfile.TemporaryDirectory() as td:
        cell_dir = os.path.join(td, "cellA", "suite", "I")
        os.makedirs(cell_dir)
        path = os.path.join(cell_dir, "k_diag_test.jsonl")

        # --- Case 1: thin data -> INVALID guard -------------------------------
        # ep 0: 13 stationary replans (t=0..60); stall window needs t>=40,
        # so replans at t=40..60 (5 of them) are stalled.
        # ep 1: moving -> zero stalled.
        with open(path, "w") as f:
            for r in make_records(0, 0, 13, moving=False):
                f.write(json.dumps(r) + "\n")
            for r in make_records(0, 1, 13, moving=True):
                f.write(json.dumps(r) + "\n")
        s = run_analysis(os.path.join(td, "*", "*", "*", "k_diag_*.jsonl"), None)
        assert s["pooled"]["n_stalled"] == 5, s["pooled"]
        assert s["gate"]["verdict"] == "INVALID", s["gate"]
        # stalled detection: moving episode contributed nothing
        counts_moving = analyze_episode(make_records(0, 1, 13, moving=True))
        assert counts_moving["n_stalled"] == 0, counts_moving
        # 2-signature diversity: every stalled replan spans 2 signatures
        assert s["pooled"]["diversity_in_stall"] == 1.0, s["pooled"]
        # revisit matching: 4 eligible stalled replans, all re-select sig of
        # best_idx=0 (same side bit, identical xy)
        assert s["pooled"]["revisit_rate"] == 1.0, s["pooled"]

        # --- Case 2: enough stalled replans -> GO -----------------------------
        # 33 stationary replans -> stalled from t>=40 (25 stalled).
        with open(path, "w") as f:
            for r in make_records(0, 0, 33, moving=False):
                f.write(json.dumps(r) + "\n")
        out = os.path.join(td, "gate.json")
        s = run_analysis(os.path.join(td, "*", "*", "*", "k_diag_*.jsonl"), out)
        assert s["pooled"]["n_stalled"] == 25, s["pooled"]
        assert s["gate"]["verdict"] == "GO", s["gate"]
        assert os.path.exists(out)

        # --- Case 3: no diversity, no revisit -> NO-GO ------------------------
        # single cluster (identical candidates), alternating selected side via
        # obstacles is impossible with identical disp, so break revisit by
        # alternating xy displacement > 0.03 between consecutive replans.
        with open(path, "w") as f:
            for i in range(33):
                d = [[0.05, 0.0, 0.0], [0.055, 0.0, 0.0]] if i % 2 == 0 \
                    else [[-0.05, 0.0, 0.0], [-0.055, 0.0, 0.0]]
                f.write(json.dumps({
                    "task_id": 0, "ep": 0, "t": 5 * i, "eef": [0.1, 0.1, 1.0],
                    "cand_disp": d, "cand_min_margin": [0.02, 0.02],
                    "best_idx": 0, "obstacles": [[0.1, 0.2, 1.0]],
                }) + "\n")
        s = run_analysis(os.path.join(td, "*", "*", "*", "k_diag_*.jsonl"), None)
        assert s["gate"]["verdict"] == "NO-GO", s["gate"]
        assert s["pooled"]["diversity_in_stall"] == 0.0, s["pooled"]
        assert s["pooled"]["revisit_rate"] == 0.0, s["pooled"]

        # --- Case 4: records without obstacles (side bit skipped) -------------
        rec = {"task_id": 0, "ep": 0, "t": 0, "eef": [0.1, 0.1, 1.0],
               "cand_disp": [[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]],
               "best_idx": 0, "obstacles": []}
        sigs = _signatures(rec)
        assert len(set(sigs)) == 2 and all(s[1] is None for s in sigs), sigs

    print("SELF_TEST_OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--diag_glob", type=str, default=None,
                    help="Glob pattern for k_diag_*.jsonl files.")
    ap.add_argument("--out", type=str, default=None,
                    help="Path for the summary/gate JSON.")
    ap.add_argument("--self_test", action="store_true",
                    help="Run the synthetic self-test and exit.")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.diag_glob:
        ap.error("--diag_glob is required (or use --self_test)")
    run_analysis(args.diag_glob, args.out)


if __name__ == "__main__":
    main()
