"""Offline GroundingDINO detector validation for the no-GT tier's last
privilege (gt_seg pixel masks -> open-vocabulary detector).

For each saved episode capture (vlm_inputs/*), run GroundingDINO on
agentview_rgb.png with the active obstacle's cleaned name as the text prompt,
and compare the top box against the GT segmentation bbox:
- hit rate (IoU > 0.3), median/p90 center offset in px, no-detection rate.

Known data caveat: saved seg arrays have inconsistent vertical orientation
across tasks (runtime unaffected). We score IoU for both orientations and
keep the better one, recording which was used.

Run inside the gdino_cpu venv:
  /ocean/projects/cis250185p/asingal/envs/gdino_cpu/bin/python \
    vlm_pipeline/gdino_offline_validate.py --limit_per_task 3
"""

import argparse
import glob
import json
import pathlib
import re

import numpy as np

GDINO_DIR = pathlib.Path("/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/vlsa-aegis/GroundingDINO")


def clean_name(name):
    name = re.sub(r"_obstacle", "", name)
    name = re.sub(r"_\d+$", "", name)
    return name.replace("_", " ").strip()


def project_gt_pixel(pos_w, cam):
    """World point -> pixel via saved camera params (robosuite convention:
    extrinsic T maps camera->world, +z forward — inverse of percep_obstacle's
    _backproject)."""
    K = np.asarray(cam["intrinsic"], dtype=np.float64)
    T = np.asarray(cam["extrinsic"], dtype=np.float64)
    p_cam = np.linalg.inv(T) @ np.array([*pos_w, 1.0])
    if abs(p_cam[2]) < 1e-6:
        return None
    uv = K @ (p_cam[:3] / p_cam[2])
    return float(uv[0]), float(uv[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs_root", default="/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/vlm_inputs")
    ap.add_argument("--suites", nargs="+", default=["safelibero_spatial"])
    ap.add_argument("--limit_per_task", type=int, default=3)
    ap.add_argument("--box_threshold", type=float, default=0.35)
    ap.add_argument("--text_threshold", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output_json", default="results_tables/gdino_offline_validate.json")
    args = ap.parse_args()

    import torch
    from PIL import Image
    import groundingdino.datasets.transforms as T
    from groundingdino.util.box_ops import box_cxcywh_to_xyxy
    from groundingdino.util.inference import load_model, predict

    model = load_model(str(GDINO_DIR / "GroundingDINO_SwinT_OGC.py"),
                       str(GDINO_DIR / "groundingdino_swint_ogc.pth"),
                       device=args.device)
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rows = []
    for suite in args.suites:
        for task_dir in sorted(glob.glob(f"{args.inputs_root}/{suite}/level_*/task_*")):
            eps = sorted(glob.glob(f"{task_dir}/episode_*"))[: args.limit_per_task]
            for ep in eps:
                ep = pathlib.Path(ep)
                try:
                    ob = json.load(open(ep / "obstacle.json"))["active_obstacle"]
                    obstacle, gt_pos = ob["name"], ob["position"]
                    cam = json.load(open(ep / "camera_params.json"))["agentview"]
                    rgb = np.array(Image.open(ep / "agentview_rgb.png").convert("RGB"))
                except Exception as e:  # noqa: BLE001
                    print(f"skip {ep}: {e}")
                    continue
                prompt = clean_name(obstacle) + " ."
                img_t, _ = transform(Image.fromarray(rgb), None)
                boxes, logits, phrases = predict(
                    model=model, image=img_t, caption=prompt,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold, device=args.device)
                h, w = rgb.shape[:2]
                row = {"ep": str(ep), "obstacle": obstacle, "prompt": prompt,
                       "n_det": int(len(boxes))}
                if len(boxes):
                    uv = project_gt_pixel(gt_pos, cam)
                    if uv is not None:
                        all_xyxy = (box_cxcywh_to_xyxy(boxes)
                                    * torch.tensor([w, h, w, h])).tolist()
                        # saved captures have inconsistent orientation across
                        # tasks (known data bug; runtime renders are clean) —
                        # score all 4 orientations of the GT pixel, keep the
                        # one with the most box containments.
                        orients = {"asis": uv, "h": (w - uv[0], uv[1]),
                                   "v": (uv[0], h - uv[1]),
                                   "hv": (w - uv[0], h - uv[1])}
                        best_o, contains = None, [False] * len(all_xyxy)
                        for o, (gx, gy) in orients.items():
                            c = [b[0] <= gx <= b[2] and b[1] <= gy <= b[3]
                                 for b in all_xyxy]
                            if sum(c) > sum(contains):
                                best_o, contains = o, c
                        gx, gy = orients[best_o or "asis"]
                        row["orientation"] = best_o
                        k = int(torch.argmax(logits))
                        cx, cy = ((all_xyxy[k][0] + all_xyxy[k][2]) / 2,
                                  (all_xyxy[k][1] + all_xyxy[k][3]) / 2)
                        row.update({
                            "any_box_hit": any(contains),
                            "top_box_hit": contains[k],
                            "top_center_off_px": float(np.hypot(cx - gx, cy - gy)),
                            "top_logit": float(logits[k]),
                            "hit_logits": [round(float(l), 3)
                                           for l, c in zip(logits, contains) if c],
                        })
                rows.append(row)
                print(row)

    n = len(rows)
    summary = {
        "n": n,
        "no_detection": sum(r["n_det"] == 0 for r in rows),
        "any_box_hit_rate": sum(r.get("any_box_hit", False) for r in rows) / n if n else 0.0,
        "top_box_hit_rate": sum(r.get("top_box_hit", False) for r in rows) / n if n else 0.0,
        "mean_n_det": float(np.mean([r["n_det"] for r in rows])) if rows else 0.0,
    }
    print("SUMMARY", json.dumps(summary))
    out = pathlib.Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
