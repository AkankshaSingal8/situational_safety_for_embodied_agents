#!/usr/bin/env python3
"""
qwen_vlm_server.py

Persistent HTTP server: loads Qwen2.5-VL once at startup, serves
per-chunk VLM inference requests over localhost.

Must run in the 'qwen' conda environment:
    conda activate qwen
    CUDA_VISIBLE_DEVICES=1 python qwen_vlm_server.py --port 5001 --model qwen2.5-vl-7b

Endpoints:
    GET  /health   → {"status": "ok", "model": "<model_key>"}
    POST /infer    → body: {"obs_folder": "...", "method": "m1",
                            "num_votes": 1, "dry_run": false}
                   ← {"single": {"description": ..., "end_object": ..., "objects": [...]}}
"""
import argparse
import logging
import os
import sys

from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen_vlm_worker as _worker
from qwen_vlm_worker import DRY_RUN_RESPONSE, DRY_RUN_RESULT, QWEN_MODELS  # noqa: F401 (DRY_RUN_RESULT kept for back-compat)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
_model = None
_processor = None
_loaded_model_key = None


def _load_model(model_key: str) -> None:
    global _model, _processor, _loaded_model_key
    if _model is not None:
        if _loaded_model_key != model_key:
            logger.warning(
                f"Requested model '{model_key}' but '{_loaded_model_key}' already loaded — ignoring."
            )
        return
    import torch
    from transformers import AutoProcessor

    model_id = QWEN_MODELS[model_key]
    logger.info(f"Loading {model_id} onto available GPUs ...")
    _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Pick the right model class (mirrors load_qwen_model in qwen_vlm_worker)
    if model_key.startswith("qwen2.5"):
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    elif model_key.startswith("qwen2"):
        from transformers import Qwen2VLForConditionalGeneration as ModelCls
    else:
        from transformers import AutoModelForImageTextToText as ModelCls

    dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    _model = ModelCls.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto"
    )
    _model.eval()
    _loaded_model_key = model_key
    logger.info(f"Model ready: {model_id}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": _loaded_model_key})


@app.route("/infer", methods=["POST"])
def infer():
    import time as _time
    t0 = _time.time()
    body = request.get_json(force=True)
    obs_folder = body.get("obs_folder", "")
    method = body.get("method", "m1")
    num_votes = int(body.get("num_votes", 1))
    dry_run = bool(body.get("dry_run", False))

    if dry_run:
        if not obs_folder:
            logger.warning("dry_run=True with empty obs_folder — returning placeholder")
        else:
            logger.info(f"dry_run=True; returning placeholder for {obs_folder}")
        return jsonify(DRY_RUN_RESPONSE)

    if not os.path.isdir(obs_folder):
        logger.warning(f"obs_folder not found: {obs_folder}")
        return jsonify({"error": f"obs_folder not found: {obs_folder}"}), 400

    if _model is None or _processor is None:
        logger.error("Model not loaded")
        return jsonify({"error": "Model not loaded"}), 503

    try:
        # Load episode data from folder
        episode = _worker.load_episode(obs_folder)

        # Dispatch to the correct method function
        process_fn = _worker.METHOD_FN.get(method)
        if process_fn is None:
            return jsonify({"error": f"Unknown method: {method}. "
                                      f"Choose from {list(_worker.METHOD_FN.keys())}"}), 400

        result = process_fn(
            episode=episode,
            model=_model,
            processor=_processor,
            num_votes=num_votes,
            max_new_tokens=256,
            dry_run=False,
        )
        # Strip raw VLM log from response (can be large)
        result.pop("_vlm_log", None)
        logger.info(f"Infer complete in {_time.time() - t0:.1f}s for {obs_folder}")
        return jsonify({"single": result})

    except Exception as exc:
        logger.exception(f"Inference error for {obs_folder}: {exc}")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persistent Qwen VLM inference server")
    parser.add_argument("--port", type=int, default=5001, help="Port to listen on")
    parser.add_argument(
        "--model", default="qwen2.5-vl-7b",
        choices=list(QWEN_MODELS.keys()),
        help="Qwen model key (see QWEN_MODELS in qwen_vlm_worker.py)",
    )
    args = parser.parse_args()
    _load_model(args.model)
    app.run(host="0.0.0.0", port=args.port, threaded=False)
