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


def _load_model(model_key: str, load_in_4bit: bool = False) -> None:
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
    logger.info(f"Loading {model_id} onto available GPUs (4bit={load_in_4bit}) ...")
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
    load_kwargs = dict(torch_dtype=dtype, device_map="auto")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
        )
    _model = ModelCls.from_pretrained(model_id, **load_kwargs)
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


@app.route("/infer_fol", methods=["POST"])
def infer_fol():
    """
    FOL-specific endpoint for real-time obstacle identification.

    Body: {
        "images_b64": ["<base64 PNG>", ...],   # agentview + eye_in_hand
        "task_description": "put the bowl on the stove",
        "candidates": ["moka pot", "plate", ...],  # cleaned names, no 'obstacle' label
        "num_votes": 3
    }
    Response: {
        "obstacle": "moka pot",
        "physical_properties": ["IS_SPILLABLE"],
        "confidence": 0.9
    }
    """
    import base64
    import collections
    import tempfile
    import time as _time

    t0 = _time.time()
    body = request.get_json(force=True)
    images_b64 = body.get("images_b64", [])
    task_description = body.get("task_description", "")
    candidates = body.get("candidates", [])
    num_votes = int(body.get("num_votes", 3))

    if _model is None or _processor is None:
        return jsonify({"error": "Model not loaded"}), 503

    # Decode base64 images to temp PNG files
    img_paths = []
    tmp_files = []
    for i, b64_str in enumerate(images_b64):
        if not b64_str:
            continue
        try:
            img_data = base64.b64decode(b64_str)
            tmp = tempfile.NamedTemporaryFile(suffix=f"_fol_{i}.png", delete=False)
            tmp.write(img_data)
            tmp.close()
            img_paths.append(tmp.name)
            tmp_files.append(tmp.name)
        except Exception as e:
            logger.warning(f"[infer_fol] Failed to decode image {i}: {e}")

    candidates_text = "\n".join(f"- {c}" for c in candidates)

    obstacle_prompt = (
        f"You are a robot safety monitor analyzing a manipulation scene.\n"
        f"Task: {task_description}\n\n"
        f"Objects present:\n{candidates_text}\n\n"
        f"Which ONE object should the robot arm avoid? It is an obstacle, not the task target.\n"
        f"Reply with ONLY the object name from the list above, or 'none' if no obstacle visible."
    )

    try:
        # Prompt 1: obstacle identification (N votes)
        obstacle_votes = []
        for _ in range(num_votes):
            resp = _worker.run_vlm_query(
                _model, _processor, obstacle_prompt,
                image_paths=img_paths, max_new_tokens=64,
            )
            obstacle_votes.append(resp.strip().lower())

        obstacle_name = (
            collections.Counter(obstacle_votes).most_common(1)[0][0]
            if obstacle_votes else "none"
        )

        # Match against candidate list (longest-substring)
        matched_obstacle = None
        if obstacle_name not in ("none", ""):
            # Exact match first
            for cand in candidates:
                if cand.lower() == obstacle_name:
                    matched_obstacle = cand
                    break
            # Substring match
            if matched_obstacle is None:
                for cand in candidates:
                    if cand.lower() in obstacle_name or obstacle_name in cand.lower():
                        matched_obstacle = cand
                        break

        # Prompt 2: physical properties (only if obstacle found)
        physical_props = []
        if matched_obstacle:
            prop_prompt = (
                f"You are a robot safety monitor.\n"
                f"The obstacle in this scene is: {matched_obstacle}\n"
                f"Does it have any of these properties?\n"
                f"- IS_SPILLABLE: contains liquid (bowl, cup, pot with liquid)\n"
                f"- IS_FRAGILE: breaks if hit (glass, egg, ceramic)\n"
                f"- IS_HOT: dangerous temperature (candle, heater, stove)\n"
                f"Reply with applicable property names comma-separated, or 'none'."
            )
            prop_votes = []
            for _ in range(num_votes):
                resp = _worker.run_vlm_query(
                    _model, _processor, prop_prompt,
                    image_paths=img_paths, max_new_tokens=64,
                )
                prop_votes.append(resp.strip().upper())

            for prop in ["IS_SPILLABLE", "IS_FRAGILE", "IS_HOT"]:
                count = sum(1 for v in prop_votes if prop in v)
                if count >= (num_votes // 2 + 1):
                    physical_props.append(prop)

        confidence = sum(1 for v in obstacle_votes if v not in ("none", "")) / max(num_votes, 1)
        logger.info(
            f"[infer_fol] obstacle='{matched_obstacle}' props={physical_props} "
            f"conf={confidence:.2f} in {_time.time()-t0:.1f}s"
        )

        return jsonify({
            "obstacle": matched_obstacle or "none",
            "physical_properties": physical_props,
            "confidence": confidence,
        })

    except Exception as exc:
        logger.exception(f"[infer_fol] error: {exc}")
        return jsonify({"error": str(exc)}), 500

    finally:
        for f in tmp_files:
            try:
                import os as _os
                _os.unlink(f)
            except OSError:
                pass


@app.route("/infer_fol_v5", methods=["POST"])
def infer_fol_v5():
    """
    FOL v5 endpoint — single structured call returning task_target, goal_location,
    obstacles (multi), and all_properties.

    Body: {
        "images_b64": ["<base64 PNG>", ...],
        "task_description": "pick up the black bowl between the plate and the ramekin",
        "candidates": ["moka pot", "plate", "black bowl", "ramekin"],
        "parsed_target": "black bowl"   # optional, pre-parsed from description
    }
    Response: {
        "task_target": "black bowl",
        "goal_location": null,
        "obstacles": [{"name": "moka pot", "properties": ["IS_HEAVY"], "avoidance_radius": 0.20}],
        "all_properties": {"moka pot": ["IS_HEAVY"], "plate": ["IS_FRAGILE"], ...},
        "confidence": 0.9
    }
    """
    import base64
    import json
    import re
    import tempfile
    import time as _time

    t0 = _time.time()
    body = request.get_json(force=True)
    images_b64 = body.get("images_b64", [])
    task_description = body.get("task_description", "")
    candidates = body.get("candidates", [])
    parsed_target = body.get("parsed_target") or ""

    if _model is None or _processor is None:
        return jsonify({"error": "Model not loaded"}), 503

    img_paths = []
    tmp_files = []
    for i, b64_str in enumerate(images_b64):
        if not b64_str:
            continue
        try:
            img_data = base64.b64decode(b64_str)
            tmp = tempfile.NamedTemporaryFile(suffix=f"_fol_v5_{i}.png", delete=False)
            tmp.write(img_data)
            tmp.close()
            img_paths.append(tmp.name)
            tmp_files.append(tmp.name)
        except Exception as e:
            logger.warning(f"[infer_fol_v5] Failed to decode image {i}: {e}")

    candidates_text = "\n".join(f"- {c}" for c in candidates)
    target_hint = (
        f"TASK TARGET (robot must pick this up): {parsed_target}\n"
        f"Do NOT list the task target as an obstacle."
        if parsed_target else ""
    )

    prompt = (
        f"You are a robot safety monitor. Analyze the robot workspace image(s).\n\n"
        f"ROBOT TASK: {task_description}\n"
        f"CANDIDATE OBJECTS IN SCENE:\n{candidates_text}\n\n"
        f"{target_hint}\n\n"
        f"Identify safety-relevant information. Return ONLY valid JSON, no other text:\n"
        f'{{\n'
        f'  "task_target": "<exact name from candidates above, or null>",\n'
        f'  "goal_location": "<where robot places/puts the object, exact name or null>",\n'
        f'  "obstacles": [\n'
        f'    {{"name": "<exact name from candidates>", '
        f'"properties": ["IS_HEAVY"|"IS_HOT"|"IS_FRAGILE"|"IS_SPILLABLE"|"IS_SHARP"], '
        f'"avoidance_radius": 0.10}}\n'
        f'  ],\n'
        f'  "all_properties": {{"<name>": ["PROP", ...], ...}},\n'
        f'  "confidence": 0.9\n'
        f'}}\n\n'
        f"Property definitions:\n"
        f"- IS_FRAGILE: ceramic/glass/egg — easily broken; avoidance_radius 0.12-0.15\n"
        f"- IS_SPILLABLE: full open container with liquid; avoidance_radius 0.15-0.20\n"
        f"- IS_HOT: stove/candle/heater — burn hazard; avoidance_radius 0.20-0.28\n"
        f"- IS_SHARP: knife/scissors — cut hazard; avoidance_radius 0.12-0.18\n"
        f"- IS_HEAVY: over 2kg cast/iron; avoidance_radius 0.15-0.22\n\n"
        f"Rules:\n"
        f"- task_target and goal_location MUST NOT be in obstacles\n"
        f"- obstacles are objects the robot should avoid, NOT the pickup target\n"
        f"- If no hazardous obstacles, use \"obstacles\": []\n"
        f"- avoidance_radius: 0.10=minimal, 0.20=standard, 0.30=very dangerous"
    )

    default_response = {
        "task_target": parsed_target or None,
        "goal_location": None,
        "obstacles": [],
        "all_properties": {},
        "confidence": 0.0,
    }

    try:
        raw = _worker.run_vlm_query(
            _model, _processor, prompt,
            image_paths=img_paths, max_new_tokens=512,
        )
        logger.info(f"[infer_fol_v5] Raw VLM response: {raw[:300]}")

        # Extract JSON from response (Qwen may wrap in markdown code blocks)
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            logger.warning(f"[infer_fol_v5] No JSON found in response")
            return jsonify(default_response)

        data = json.loads(json_match.group(0))

        # Validate and sanitize
        task_target = data.get("task_target")
        if task_target and task_target not in candidates:
            # fuzzy match
            task_target = next(
                (c for c in candidates if c.lower() in str(task_target).lower()
                 or str(task_target).lower() in c.lower()), None
            )

        goal_location = data.get("goal_location")
        if goal_location and goal_location not in candidates:
            goal_location = next(
                (c for c in candidates if c.lower() in str(goal_location).lower()
                 or str(goal_location).lower() in c.lower()), None
            )

        obstacles = []
        for obs_entry in data.get("obstacles", []):
            name = obs_entry.get("name", "")
            if name == task_target or name == goal_location:
                logger.info(f"[infer_fol_v5] Skipping obstacle='{name}' (matches target/goal)")
                continue
            if name not in candidates:
                name = next(
                    (c for c in candidates if c.lower() in name.lower()
                     or name.lower() in c.lower()), None
                )
            if name:
                props = [p for p in obs_entry.get("properties", [])
                         if p in ("IS_FRAGILE", "IS_SPILLABLE", "IS_HOT", "IS_SHARP", "IS_HEAVY")]
                radius = float(obs_entry.get("avoidance_radius", 0.18))
                radius = max(0.10, min(0.30, radius))
                obstacles.append({"name": name, "properties": props, "avoidance_radius": radius})

        all_properties = {}
        for k, v in data.get("all_properties", {}).items():
            matched = next(
                (c for c in candidates if c.lower() == k.lower()
                 or c.lower() in k.lower() or k.lower() in c.lower()), None
            )
            if matched:
                all_properties[matched] = [
                    p for p in (v if isinstance(v, list) else [])
                    if p in ("IS_FRAGILE", "IS_SPILLABLE", "IS_HOT", "IS_SHARP", "IS_HEAVY")
                ]

        confidence = float(data.get("confidence", 0.7))

        result = {
            "task_target": task_target,
            "goal_location": goal_location,
            "obstacles": obstacles,
            "all_properties": all_properties,
            "confidence": confidence,
        }

        logger.info(
            f"[infer_fol_v5] target='{task_target}' goal='{goal_location}' "
            f"obstacles={[o['name'] for o in obstacles]} conf={confidence:.2f} "
            f"in {_time.time()-t0:.1f}s"
        )
        return jsonify(result)

    except json.JSONDecodeError as e:
        logger.warning(f"[infer_fol_v5] JSON parse error: {e} | raw={raw[:200]}")
        return jsonify(default_response)
    except Exception as exc:
        logger.exception(f"[infer_fol_v5] error: {exc}")
        return jsonify({"error": str(exc)}), 500
    finally:
        for f in tmp_files:
            try:
                import os as _os
                _os.unlink(f)
            except OSError:
                pass


@app.route("/infer_v6", methods=["POST"])
def infer_v6():
    """
    Generic text+image inference for v6 multi-prompt CoT chain.

    Body: {"prompt": "...", "images_b64": ["<base64 PNG>", ...], "max_new_tokens": 512}
    Response: {"text": "<model output>"}
    """
    import base64
    import tempfile
    import time as _time

    t0 = _time.time()
    body = request.get_json(force=True)
    prompt = body.get("prompt", "")
    images_b64 = body.get("images_b64", []) or []
    max_new_tokens = int(body.get("max_new_tokens", 512))

    if _model is None or _processor is None:
        return jsonify({"error": "Model not loaded"}), 503

    tmp_paths = []
    try:
        for b64 in images_b64:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.write(base64.b64decode(b64))
            tmp.close()
            tmp_paths.append(tmp.name)

        text = _worker.run_vlm_query(
            _model, _processor, prompt,
            image_paths=tmp_paths,
            max_new_tokens=max_new_tokens,
        )
        logger.info(
            f"[infer_v6] {len(images_b64)} imgs, prompt={len(prompt)} chars "
            f"→ {len(text)} chars in {_time.time() - t0:.1f}s"
        )
        return jsonify({"text": text})
    except Exception as exc:
        logger.exception(f"[infer_v6] Error: {exc}")
        return jsonify({"error": str(exc)}), 500
    finally:
        for p in tmp_paths:
            try:
                import os as _os
                _os.unlink(p)
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persistent Qwen VLM inference server")
    parser.add_argument("--port", type=int, default=5001, help="Port to listen on")
    parser.add_argument(
        "--model", default="qwen2.5-vl-7b",
        choices=list(QWEN_MODELS.keys()),
        help="Qwen model key (see QWEN_MODELS in qwen_vlm_worker.py)",
    )
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load model in 4-bit quantization (~8GB VRAM, fits alongside OpenVLA)")
    args = parser.parse_args()
    _load_model(args.model, load_in_4bit=args.load_in_4bit)
    app.run(host="0.0.0.0", port=args.port, threaded=False)
