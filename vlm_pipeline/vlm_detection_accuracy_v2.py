"""
GT-free obstacle localization validation — v2.

Changes vs v1:
  1. Qwen canonical dense-grounding prompt in BOTH views (v1 under-detected:
     usually 1 box/image).
  2. Cross-view association by MINIMAL RAY GAP (epipolar consistency) instead
     of referring-expression name matching (v1 killer stage: top-down wrist
     appearance broke language lookup).  Names come from agentview only.
  3. Per-stage logging for funnel analysis.

Pipeline: dense-detect both views → rays from bbox centers → greedy min-gap
association → triangulated 3D candidates (agentview names) → ¬MENTIONED →
NEAREST_TO_PATH → obstacle pick + position.  GT used only for scoring.
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from fol_safety_filter.vlm_client import VLMClient  # noqa: E402


def project(p, K, Einv):
    pc = Einv @ np.append(p, 1.0)
    if pc[2] <= 1e-6:
        return None
    return K[0, 2] + K[0, 0] * pc[0] / pc[2], K[1, 2] + K[1, 1] * pc[1] / pc[2]


def make_ray(u, v, K, E):
    d_cam = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    d = E[:3, :3] @ d_cam
    return E[:3, 3], d / np.linalg.norm(d)


def closest_point(c1, d1, c2, d2):
    w0 = c1 - c2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d_, e_ = d1 @ w0, d2 @ w0
    den = a * c - b * b
    if abs(den) < 1e-9:
        return None, None
    t1 = (b * e_ - c * d_) / den
    t2 = (a * e_ - b * d_) / den
    p1, p2 = c1 + t1 * d1, c2 + t2 * d2
    return (p1 + p2) / 2, float(np.linalg.norm(p1 - p2))


def gt_bbox(seg, uv, max_px):
    best = None
    for sid in np.unique(seg):
        mask = seg == sid
        n = int(mask.sum())
        if not (50 < n < max_px):
            continue
        ys, xs = np.where(mask)
        d = float(np.hypot(xs - uv[0], ys - uv[1]).min())
        if d < 15 and (best is None or n < best[1]):
            best = ([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], n)
    return best[0] if best else None


def iou(a, b):
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# Qwen2.5-VL canonical dense grounding phrasing
DETECT_PROMPT = (
    'Outline the position of each object on the table (dishes, containers, '
    'bottles, boxes, food items, appliances, books — every distinct item, '
    'but NOT the robot arm, table, or walls) and output all the coordinates '
    'in JSON format:\n'
    '[{"bbox_2d": [x1, y1, x2, y2], "label": "<object name>"}]'
)


def parse_detections(raw, img_wh=512):
    m = re.search(r'\[[\s\S]*\]', raw or "")
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        bb = it.get("bbox_2d") or it.get("bbox")
        nm = it.get("label") or it.get("name")
        if not (isinstance(bb, list) and len(bb) == 4 and nm):
            continue
        bb = [float(x) for x in bb]
        mx = max(bb)
        if mx > img_wh * 1.05:
            bb = [x * img_wh / 1000.0 for x in bb] if mx <= 1000 else \
                 [x * img_wh / mx for x in bb]
        out.append({"name": str(nm).lower().strip(), "bbox": bb})
    return out


def mentioned(name, task):
    words = [w for w in name.lower().split() if len(w) > 3]
    return any(w in task.lower() for w in words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_dir", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--model_tag", default="unknown")
    args = ap.parse_args()

    client = VLMClient()
    assert client.backend_name != "mock", "VLM server not reachable"
    print(f"backend: {client.backend_name}")

    rows = []
    metas = sorted(glob.glob(os.path.join(args.episodes_dir, "**", "metadata.json"),
                             recursive=True))
    for mi, mf in enumerate(metas):
        ep = Path(mf).parent
        meta = json.load(open(mf))
        task = meta["task_description"]
        gt_pos = np.array(meta["obstacle"]["position"])
        gt_name = meta["obstacle"]["name"]
        cp = json.load(open(ep / "camera_params.json"))

        dets = {}
        cams = {}
        gtb = {}
        for view, rgbf, segf, mp in [
            ("agentview", "agentview_rgb.png", "agentview_seg.npy", 40000),
            ("robot0_eye_in_hand", "eye_in_hand_rgb.png", "eye_in_hand_seg.npy", 130000),
        ]:
            K = np.array(cp[view]["intrinsic"]); E = np.array(cp[view]["extrinsic"])
            cams[view] = (K, E)
            img = np.array(Image.open(ep / rgbf).convert("RGB"))
            raw = client.infer(DETECT_PROMPT, [img], max_tokens=1024)
            dets[view] = parse_detections(raw)
            uv = project(gt_pos, K, np.linalg.inv(E))
            gtb[view] = gt_bbox(np.load(ep / segf).squeeze(), uv, mp) \
                if uv and 0 <= uv[0] < 512 and 0 <= uv[1] < 512 else None

        # rays per detection
        def rays_of(view):
            K, E = cams[view]
            return [(d, make_ray((d["bbox"][0]+d["bbox"][2])/2,
                                 (d["bbox"][1]+d["bbox"][3])/2, K, E))
                    for d in dets[view]]

        ra = rays_of("agentview")
        rb = rays_of("robot0_eye_in_hand")

        # greedy min-gap association (each agentview det ↔ best EIH det)
        cands3d = []
        used_b = set()
        pairs = []
        for i, (da, (ca, dda)) in enumerate(ra):
            for j, (db, (cb, ddb)) in enumerate(rb):
                p, gap = closest_point(ca, dda, cb, ddb)
                if p is None:
                    continue
                pairs.append((gap, i, j, p))
        for gap, i, j, p in sorted(pairs):
            if i in {x[0] for x in cands3d} or j in used_b:
                continue
            if gap > 0.08:
                continue
            if not (-0.6 < p[0] < 0.6 and -0.5 < p[1] < 0.8 and 0.7 < p[2] < 1.5):
                continue
            cands3d.append((i, j, p, gap))
            used_b.add(j)

        cands = [{"name": ra[i][0]["name"], "bbox": ra[i][0]["bbox"],
                  "pos": p, "gap": gap} for i, j, p, gap in cands3d]
        unmen = [c for c in cands if not mentioned(c["name"], task)]

        # NEAREST_TO_PATH
        rs = meta["robot_state"]
        if isinstance(rs, str):
            eef = np.array(eval(re.search(r"'eef_pos': (\[[^\]]*\])", rs).group(1)))
        else:
            eef = np.array(rs["eef_pos"])
        spos, tpos = eef[:2], np.array([0.0, 0.15])
        seg_v = tpos - spos
        L2 = float(seg_v @ seg_v)

        def dpath(p):
            q = np.array(p[:2])
            t = float(np.clip((q - spos) @ seg_v / L2, 0, 1)) if L2 > 1e-9 else 0.0
            return float(np.linalg.norm(q - (spos + t * seg_v)))

        pick = min(unmen, key=lambda c: dpath(c["pos"])) if unmen else None
        pos_err = float(np.linalg.norm(pick["pos"] - gt_pos)) if pick else None
        det_iou_a = max((iou(d["bbox"], gtb["agentview"]) for d in dets["agentview"]),
                        default=0.0)

        row = {
            "episode": str(ep).split("safelibero_spatial/")[-1],
            "gt": gt_name,
            "n_det_agent": len(dets["agentview"]),
            "n_det_eih": len(dets["robot0_eye_in_hand"]),
            "agent_names": [d["name"] for d in dets["agentview"]],
            "det_iou_agent": round(det_iou_a, 2),
            "n_matched3d": len(cands),
            "n_unmentioned": len(unmen),
            "picked": pick["name"] if pick else None,
            "pos_err_cm": round(pos_err * 100, 1) if pos_err is not None else None,
        }
        rows.append(row)
        print(f"[{mi+1}/{len(metas)}] gt={gt_name.replace('_obstacle_1',''):18s} "
              f"detA={len(dets['agentview'])} detE={len(dets['robot0_eye_in_hand'])} "
              f"3d={len(cands)} unmen={len(unmen)} picked={row['picked']} "
              f"err={row['pos_err_cm']}cm")

    errs = [r["pos_err_cm"] for r in rows if r["pos_err_cm"] is not None]
    ok15 = [e for e in errs if e < 15]
    summary = {
        "model": args.model_tag,
        "episodes": len(rows),
        "localized": len(errs),
        "within_15cm": len(ok15),
        "pos_err_median_cm": float(np.median(errs)) if errs else None,
        "pos_err_max_cm": float(np.max(errs)) if errs else None,
        "rows": rows,
    }
    print(f"\n== {args.model_tag}: localized {len(errs)}/{len(rows)}, "
          f"within15cm {len(ok15)}/{len(rows)}, median={summary['pos_err_median_cm']}cm")
    json.dump(summary, open(args.output_json, "w"), indent=1)
    print("Wrote", args.output_json)


if __name__ == "__main__":
    main()
