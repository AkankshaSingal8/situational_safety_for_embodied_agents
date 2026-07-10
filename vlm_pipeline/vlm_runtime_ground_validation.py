"""
Validate the RUNTIME visual grounder (fol_safety_filter.visual_grounder) under
eval-identical conditions: 256px images in raw (unrotated) orientation +
proprioception, exactly as the filter receives them.  Single source of truth —
tests the deployed class, not a reimplementation.  GT used only for scoring.
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

from fol_safety_filter.visual_grounder import VisualObstacleGrounder  # noqa: E402
from fol_safety_filter.vlm_client import VLMClient  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_dir", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--img_res", type=int, default=256)
    args = ap.parse_args()

    client = VLMClient()
    assert client.backend_name != "mock"
    g = VisualObstacleGrounder(client)

    rows = []
    metas = sorted(glob.glob(os.path.join(args.episodes_dir, "**", "metadata.json"),
                             recursive=True))
    for mi, mf in enumerate(metas):
        ep = Path(mf).parent
        meta = json.load(open(mf))
        gt = np.array(meta["obstacle"]["position"])

        def load_raw(fname):
            img = Image.open(ep / fname).convert("RGB")
            if args.img_res != img.size[0]:
                img = img.resize((args.img_res, args.img_res))
            arr = np.array(img)
            return arr[::-1, ::-1].copy()  # saved→raw orientation (grounder re-rotates)

        agent_raw = load_raw("agentview_rgb.png")
        eih_raw = load_raw("eye_in_hand_rgb.png")

        rs = meta["robot_state"]
        if isinstance(rs, str):
            eef = np.array(eval(re.search(r"'eef_pos': (\[[^\]]*\])", rs).group(1)))
            quat = np.array(eval(re.search(r"'eef_quat': (\[[^\]]*\])", rs).group(1)))
        else:
            eef, quat = np.array(rs["eef_pos"]), np.array(rs["eef_quat"])

        res = g.ground(agent_raw, eih_raw, meta["task_description"], eef, quat)
        if res is None:
            rows.append({"gt": meta["obstacle"]["name"], "picked": None})
            print(f"[{mi+1}/{len(metas)}] gt={meta['obstacle']['name']}: NONE")
            continue
        err = float(np.linalg.norm(np.asarray(res["pos"]) - gt))
        xy = float(np.linalg.norm(np.asarray(res["pos"])[:2] - gt[:2]))
        rows.append({"gt": meta["obstacle"]["name"], "picked": res["name"],
                     "method": res["method"], "err_cm": round(err*100, 1),
                     "xy_cm": round(xy*100, 1)})
        print(f"[{mi+1}/{len(metas)}] gt={meta['obstacle']['name'].replace('_obstacle_1',''):18s} "
              f"picked={res['name']:16s} {res['method']:18s} err={err*100:.1f}cm xy={xy*100:.1f}cm")

    errs = [r["err_cm"] for r in rows if r.get("err_cm") is not None]
    ok = [e for e in errs if e < 15]
    print(f"\n== runtime-path@{args.img_res}px: localized {len(errs)}/{len(rows)}, "
          f"<15cm {len(ok)}/{len(rows)}, median={np.median(errs) if errs else None}")
    json.dump({"rows": rows, "localized": len(errs), "within15": len(ok),
               "median_cm": float(np.median(errs)) if errs else None},
              open(args.output_json, "w"), indent=1)


if __name__ == "__main__":
    main()
