#!/usr/bin/env python3
"""
qwen_vlm_worker.py

Standalone VLM inference worker that runs inside the "qwen" conda environment.
Called via subprocess from the SafeLIBERO evaluation script (which runs in a
different conda env).

Usage (called automatically by semantic_cbf_filter.py):
    conda run -n qwen python qwen_vlm_worker.py \
        --input_json  /tmp/cbf_vlm_queries.json \
        --output_json /tmp/cbf_vlm_results.json \
        --model qwen2.5-vl-7b

Input JSON format:
    {
        "model": "qwen2.5-vl-7b",
        "queries": [
            {"id": "spatial_0_above", "prompt": "...", "image_path": null},
            {"id": "behavioral_0",   "prompt": "...", "image_path": "/tmp/scene.png"},
            ...
        ]
    }

Output JSON format:
    {
        "model": "qwen2.5-vl-7b",
        "results": [
            {"id": "spatial_0_above", "response": "unsafe. The water may spill..."},
            {"id": "behavioral_0",   "response": "caution. Fragile object nearby..."},
            ...
        ]
    }
"""

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [QwenWorker] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Supported Qwen models ────────────────────────────────────────────────────

QWEN_MODELS = {
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3-vl-8b":   "Qwen/Qwen3-VL-8B",
}


def load_qwen_model(model_key: str, device: str = "auto", load_in_4bit: bool = False):
    """Load Qwen2.5-VL or Qwen3-VL model and processor."""
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    hf_id = QWEN_MODELS[model_key]
    logger.info(f"Loading model: {hf_id}")

    # Resolve dtype
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    load_kwargs = dict(torch_dtype=dtype, device_map=device)

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
        )

    t0 = time.time()
    model = Qwen2VLForConditionalGeneration.from_pretrained(hf_id, **load_kwargs)
    processor = AutoProcessor.from_pretrained(hf_id)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    return model, processor


def run_query(model, processor, prompt: str, image_path: str = None,
              max_new_tokens: int = 256) -> str:
    """Run a single VLM query and return the text response."""
    import torch
    from PIL import Image as PILImage

    # Build chat message
    content = []
    pil_img = None
    if image_path and os.path.exists(image_path):
        pil_img = PILImage.open(image_path).convert("RGB")
        content.append({"type": "image", "image": pil_img})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    # Apply chat template
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize
    if pil_img is not None:
        inputs = processor(
            text=[prompt_text], images=[pil_img],
            return_tensors="pt", padding=True,
        )
    else:
        inputs = processor(
            text=[prompt_text], return_tensors="pt", padding=True,
        )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode only the new tokens
    gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    return response


def main():
    parser = argparse.ArgumentParser(description="Qwen VLM inference worker")
    parser.add_argument("--input_json", required=True,
                        help="Path to input JSON file with queries")
    parser.add_argument("--output_json", required=True,
                        help="Path to write output JSON file with results")
    parser.add_argument("--model", default="qwen2.5-vl-7b",
                        choices=list(QWEN_MODELS.keys()),
                        help="Qwen model to use")
    parser.add_argument("--device", default="auto",
                        help="Device for model (auto, cuda:0, cpu)")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load with 4-bit quantization")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    # Read input queries
    logger.info(f"Reading queries from: {args.input_json}")
    with open(args.input_json, "r") as f:
        input_data = json.load(f)

    model_key = input_data.get("model", args.model)
    queries = input_data.get("queries", [])
    logger.info(f"Received {len(queries)} queries for model '{model_key}'")

    if not queries:
        # Write empty results and exit
        with open(args.output_json, "w") as f:
            json.dump({"model": model_key, "results": []}, f)
        return

    # Load model
    model, processor = load_qwen_model(
        model_key, device=args.device, load_in_4bit=args.load_in_4bit
    )

    # Process each query
    results = []
    for i, query in enumerate(queries):
        qid = query.get("id", f"query_{i}")
        prompt = query["prompt"]
        image_path = query.get("image_path", None)

        logger.info(f"  [{i+1}/{len(queries)}] id={qid}")

        try:
            response = run_query(
                model, processor, prompt,
                image_path=image_path,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            logger.error(f"  Query {qid} failed: {e}")
            response = "error"

        results.append({"id": qid, "response": response})
        logger.info(f"    → {response[:80]}{'...' if len(response) > 80 else ''}")

    # Write output
    output_data = {"model": model_key, "results": results}
    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results written to: {args.output_json}")


if __name__ == "__main__":
    main()