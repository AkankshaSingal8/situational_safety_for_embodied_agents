"""Minimal REST server wrapping an off-the-shelf OpenVLA checkpoint.

Run inside the `oopsieverse_openvla_srv` conda env:

    conda activate oopsieverse_openvla_srv
    export HF_HOME=/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/hf
    python oopsieverse_eval/servers/openvla_server.py \
        --checkpoint openvla/openvla-7b-finetuned-libero-10 \
        --unnorm_key libero_10 \
        --port 8100

POST /act with a json-numpy-encoded body:
    {"image": <H,W,3 uint8 array>, "instruction": "pick up the egg"}
returns a json-numpy-encoded 7-dim action array
`[dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]`, gripper in the checkpoint's
own training convention (0=close, 1=open per OpenVLA's RLDS/LIBERO data) --
NOT yet converted to any particular sim's action convention. Callers (e.g.
`eval/run_robocasa_openvla_eval.py`) are responsible for any downstream
gripper-sign flip or action-scale adaptation needed for their sim.
"""

from __future__ import annotations

import argparse
import logging

import json_numpy

json_numpy.patch()  # registers numpy array (de)serialization for json.dumps/loads

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openvla_server")

INSTRUCTION_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


class OpenVLAServer:
    def __init__(self, checkpoint: str, unnorm_key: str, attn_implementation: str = "flash_attention_2"):
        self.unnorm_key = unnorm_key
        logger.info("Loading processor for %s", checkpoint)
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

        logger.info("Loading model for %s (attn_implementation=%s)", checkpoint, attn_implementation)
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                checkpoint,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to("cuda:0")
        except Exception:
            logger.warning("Falling back to attn_implementation='sdpa' (flash-attn unavailable/failed).")
            self.model = AutoModelForVision2Seq.from_pretrained(
                checkpoint,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to("cuda:0")

        available_keys = list(getattr(self.model, "norm_stats", {}).keys())
        logger.info("Model loaded. Available unnorm_stats keys: %s", available_keys)
        if unnorm_key not in available_keys:
            raise ValueError(
                f"unnorm_key={unnorm_key!r} not found in checkpoint's norm_stats "
                f"(available: {available_keys})"
            )

    def predict(self, image: np.ndarray, instruction: str) -> np.ndarray:
        img = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        prompt = INSTRUCTION_TEMPLATE.format(instruction=instruction)
        inputs = self.processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)
        action = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        return np.asarray(action, dtype=np.float32)


def build_app(server: OpenVLAServer) -> FastAPI:
    app = FastAPI()

    @app.post("/act")
    async def act(request: Request):
        payload = json_numpy.loads(await request.body())
        image = payload["image"]
        instruction = payload["instruction"]
        try:
            action = server.predict(image, instruction)
        except Exception:
            logger.exception("Inference failed")
            return JSONResponse(status_code=500, content={"error": "inference_failed"})
        return JSONResponse(content=json_numpy.dumps(action))

    @app.get("/health")
    async def health():
        return {"status": "ok", "unnorm_key": server.unnorm_key}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="openvla/openvla-7b-finetuned-libero-10")
    parser.add_argument("--unnorm_key", default="libero_10")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    server = OpenVLAServer(checkpoint=args.checkpoint, unnorm_key=args.unnorm_key)
    app = build_app(server)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
