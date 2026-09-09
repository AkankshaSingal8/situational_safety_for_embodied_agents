"""
End-to-end GT-free obstacle localization validation.

Pipeline under test (per saved episode):
  1. Qwen open-vocab detection on agentview  → [{name, bbox}]
  2. ¬MENTIONED_IN_TASK filter on DETECTED names → unmentioned candidates
  3. Referring-expression bbox for each candidate in eye-in-hand view
  4. Two-view triangulation (calibrated rays, bbox centers) → 3D estimates
  5. Obstacle = NEAREST_TO_PATH among candidates (eef_start → target ray)

Metrics vs GT: obstacle identity accuracy, per-view bbox IoU, 3D position error.
GT is used ONLY for scoring — never inside the pipeline.
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


# ---------------- geometry (validated conventions) ----------------

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


# ---------------- VLM prompts ----------------

DETECT_PROMPT = (
    'Detect every distinct object sitting on the table in this robot workspace '
    'image. Ignore the robot arm, the table itself, and the walls.\n'
    'Output ONLY a JSON list, one entry per object:\n'
    '[{"name": "<short object name>", "bbox_2d": [x1, y1, x2, y2]}]'
)


def locate_prompt(name):
    return (
        f'Locate the {name} in this image. If it is not visible, output [].\n'
        f'Output ONLY JSON: [{{"name": "{name}", "bbox_2d": [x1, y1, x2, y2]}}]'
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
        nm = it.get("name") or it.get("label")
        if not (isinstance(bb, list) and len(bb) == 4 and nm):
            continue
        bb = [float(x) for x in bb]
        # Qwen2.5-VL emits absolute pixel coords for its (smart-resized) input;
        # rescale if coords exceed the true image size, or if they look 0-1000
        # normalized on a 512px image.
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

        views = {}
        for view, rgbf, segf, mp in [
            ("agentview", "agentview_rgb.png", "agentview_seg.npy", 40000),
            ("robot0_eye_in_hand", "eye_in_hand_rgb.png", "eye_in_hand_seg.npy", 130000),
        ]:
            K = np.array(cp[view]["intrinsic"]); E = np.array(cp[view]["extrinsic"])
            img = np.array(Image.open(ep / rgbf).convert("RGB"))
            seg = np.load(ep / segf).squeeze()
            uv = project(gt_pos, K, np.linalg.inv(E))
            gbb = gt_bbox(seg, uv, mp) if uv and 0 <= uv[0] < 512 and 0 <= uv[1] < 512 else None
            views[view] = dict(K=K, E=E, img=img, gt_bbox=gbb)

        # 1. detection on agentview
        raw = client.infer(DETECT_PROMPT, [views["agentview"]["img"]], max_tokens=768)
        dets = parse_detections(raw)

        # 2. unmentioned candidates
        cands = [d for d in dets if not mentioned(d["name"], task)]

        # score: which detection covers the GT obstacle bbox?
        gt_abb = views["agentview"]["gt_bbox"]
        det_hit = max((iou(d["bbox"], gt_abb) for d in dets), default=0.0)
        cand_hit = max((iou(d["bbox"], gt_abb) for d in cands), default=0.0)

        # 3+4. locate each candidate in EIH, triangulate
        ests = []
        for d in cands[:4]:
            raw2 = client.infer(locate_prompt(d["name"]),
                                [views["robot0_eye_in_hand"]["img"]], max_tokens=256)
            loc = parse_detections(raw2)
            if not loc:
                continue
            b1, b2 = d["bbox"], loc[0]["bbox"]
            r1 = make_ray((b1[0]+b1[2])/2, (b1[1]+b1[3])/2,
                          views["agentview"]["K"], views["agentview"]["E"])
            r2 = make_ray((b2[0]+b2[2])/2, (b2[1]+b2[3])/2,
                          views["robot0_eye_in_hand"]["K"], views["robot0_eye_in_hand"]["E"])
            p, gap = closest_point(*r1, *r2)
            if p is None or gap > 0.25:
                continue
            if not (-0.6 < p[0] < 0.6 and -0.5 < p[1] < 0.8 and 0.7 < p[2] < 1.5):
                continue  # outside workspace → reject
            ests.append({"name": d["name"], "pos": p.tolist(), "gap": gap,
                         "eih_iou": iou(b2, views["robot0_eye_in_hand"]["gt_bbox"])})

        # 5. NEAREST_TO_PATH among estimates
        rs = meta["robot_state"]
        if isinstance(rs, str):
            eef = np.array(eval(re.search(r"'eef_pos': (\[[^\]]*\])", rs).group(1)))
        else:
            eef = np.array(rs["eef_pos"])
        tpos = np.array([0.0, 0.15])  # unknown target xy → workspace center proxy
        spos = eef[:2]
        seg_v = tpos - spos
        L2 = float(seg_v @ seg_v)

        def dpath(p):
            q = np.array(p[:2])
            t = float(np.clip((q - spos) @ seg_v / L2, 0, 1)) if L2 > 1e-9 else 0.0
            return float(np.linalg.norm(q - (spos + t * seg_v)))

        pick = min(ests, key=lambda e: dpath(e["pos"])) if ests else None
        pos_err = float(np.linalg.norm(np.array(pick["pos"]) - gt_pos)) if pick else None

        row = {
            "episode": str(ep).split("safelibero_spatial/")[-1],
            "gt": gt_name,
            "n_detections": len(dets),
            "det_names": [d["name"] for d in dets],
            "n_unmentioned": len(cands),
            "det_iou_vs_gt": round(det_hit, 3),
            "cand_iou_vs_gt": round(cand_hit, 3),
            "picked": pick["name"] if pick else None,
            "pos_err_cm": round(pos_err * 100, 1) if pos_err is not None else None,
        }
        rows.append(row)
        print(f"[{mi+1}/{len(metas)}] gt={gt_name.replace('_obstacle_1',''):18s} "
              f"dets={len(dets)} unmentioned={len(cands)} "
              f"iou={det_hit:.2f} picked={row['picked']} err={row['pos_err_cm']}cm")

    errs = [r["pos_err_cm"] for r in rows if r["pos_err_cm"] is not None]
    ok = [r for r in rows if r["pos_err_cm"] is not None and r["pos_err_cm"] < 15]
    summary = {
        "model": args.model_tag,
        "episodes": len(rows),
        "localized": len(errs),
        "within_15cm": len(ok),
        "pos_err_median_cm": float(np.median(errs)) if errs else None,
        "pos_err_max_cm": float(np.max(errs)) if errs else None,
        "rows": rows,
    }
    print(f"\n== {args.model_tag}: localized {len(errs)}/{len(rows)}, "
          f"within15cm {len(ok)}/{len(rows)}, "
          f"median_err={summary['pos_err_median_cm']}cm")
    json.dump(summary, open(args.output_json, "w"), indent=1)
    print("Wrote", args.output_json)


if __name__ == "__main__":
    main()
