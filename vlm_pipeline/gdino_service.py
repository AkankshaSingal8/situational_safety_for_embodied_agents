"""Persistent GroundingDINO detection service (runs in the gdino_py310 env).

Protocol: JSON lines on stdin/stdout. Model loads once, then:
  request : {"image": "<png path>", "phrase": "moka pot"}
  response: {"boxes": [[x0,y0,x1,y1], ...], "logits": [...], "phrases": [...]}
Prints "READY" on stdout after model load. Any error → {"error": "..."}.

Started by the eval client as a subprocess (see percep_obstacle.make_gdino_detector);
keeps the heavyweight torch env fully out of the eval process.
"""

import json
import sys

GDINO_DIR = "/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/vlsa-aegis/GroundingDINO"


def main():
    import numpy as np
    import torch
    from PIL import Image
    import groundingdino.datasets.transforms as T
    from groundingdino.util.box_ops import box_cxcywh_to_xyxy
    from groundingdino.util.inference import load_model, predict

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(f"{GDINO_DIR}/GroundingDINO_SwinT_OGC.py",
                       f"{GDINO_DIR}/groundingdino_swint_ogc.pth", device=device)
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            rgb = np.array(Image.open(req["image"]).convert("RGB"))
            h, w = rgb.shape[:2]
            img_t, _ = transform(Image.fromarray(rgb), None)
            boxes, logits, phrases = predict(
                model=model, image=img_t,
                caption=req["phrase"].rstrip(" .") + " .",
                box_threshold=float(req.get("box_threshold", 0.30)),
                text_threshold=float(req.get("text_threshold", 0.25)),
                device=device)
            xyxy = (box_cxcywh_to_xyxy(boxes) * torch.tensor([w, h, w, h])).tolist()
            print(json.dumps({"boxes": xyxy,
                              "logits": [float(l) for l in logits],
                              "phrases": phrases}), flush=True)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
