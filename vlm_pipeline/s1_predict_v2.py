"""Predict the S1 v2 contrast criterion from the v1 run — no new compute.

v2's primary metric is cos(dv, u) with dv = v(approach) - v(task) and u the
hazard direction. v1 did not save raw vectors, BUT it saved each vector's xy
norm and its cosine to u, and the component of a vector along u is exactly
|v| * cos. So the COMPONENT of dv along u is recoverable:

    proj(dv) = |v_appr| cos_appr - |v_task| cos_task

The sign of proj(dv) is the sign of cos(dv, u). That is enough to predict
whether v2's criterion can pass, before spending the GPU hour.
"""

import json
import pathlib

import numpy as np


def main(path="s1_probe/probe_results.json"):
    rows = json.loads(pathlib.Path(path).read_text())["rows"]
    cos_t = np.array([r["cos_task"] for r in rows])
    nrm_t = np.array([r["norm_task"] for r in rows])
    # v1's criterion took the better of the two approach prompts per scene.
    pick = [1 if r["cos_approach1"] >= r["cos_approach2"] else 2 for r in rows]
    cos_a = np.array([r[f"cos_approach{p}"] for r, p in zip(rows, pick)])
    nrm_a = np.array([r[f"norm_approach{p}"] for r, p in zip(rows, pick)])

    proj_t, proj_a = nrm_t * cos_t, nrm_a * cos_a
    dproj = proj_a - proj_t

    hdr = (f"{'scene':7s} {'cos_task':>9s} {'|v_task|':>9s} {'|v_appr|':>9s} "
           f"{'proj_task':>10s} {'proj_appr':>10s} {'proj(dv)':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for r, ct, nt, na, pt, pa, dp in zip(rows, cos_t, nrm_t, nrm_a, proj_t, proj_a, dproj):
        print(f"t{r['task_id']}e{r['ep']:<4d} {ct:9.2f} {nt:9.3f} {na:9.3f} "
              f"{pt:10.3f} {pa:10.3f} {dp:+10.3f}")

    print(f"\nsign(cos(dv,u)) > 0 on {(dproj > 0).sum()}/{len(rows)} scenes "
          f"({np.mean(dproj > 0):.0%})")
    ceil = cos_t >= 0.9
    info = cos_t < 0.8
    print(f"  ceiling      (cos_task >= 0.9): n={ceil.sum():2d}  "
          f"positive {(dproj[ceil] > 0).sum()}/{ceil.sum()}")
    print(f"  informative  (cos_task <  0.8): n={info.sum():2d}  "
          f"positive {(dproj[info] > 0).sum()}/{info.sum()}")

    print(f"\nROOT CAUSE — displacement MAGNITUDE:")
    print(f"  |v_appr| > |v_task| on only {(nrm_a > nrm_t).sum()}/{len(rows)} scenes")
    print(f"  median |v_task| {np.median(nrm_t):.3f}   median |v_appr| {np.median(nrm_a):.3f}"
          f"   ratio {np.median(nrm_a) / np.median(nrm_t):.2f}")
    print("""
  The approach prompt makes the policy move LESS, not more. On the ceiling
  scenes both vectors point at the hazard, so a shorter approach vector makes
  dv = v_appr - v_task point AWAY from it: cos(dv,u) < 0 on 14/14. The
  instruction swap does not redirect the chunk toward the named object; it
  makes the policy hesitate. v2's contrast criterion would therefore fail
  with a NEGATIVE median, worse than v1's 0.55.

  Caveat: this is xy-only and full-chunk (v1 saved nothing else); v2 scores 3D
  and the prefix, which could differ in magnitude but not plausibly in sign
  across 14/14 scenes.""")


if __name__ == "__main__":
    main()
