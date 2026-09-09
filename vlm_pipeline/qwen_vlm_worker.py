#!/usr/bin/env python3
"""
qwen_vlm_worker.py

VLM inference worker implementing three methods from the design document:

  M1 (Seg+VLM):   Image + object labels from metadata  → per-object multi-prompt
  M2 (VLM-only):  Image only, no labels                 → VLM discovers objects + constraints
  M3 (3D+VLM):    Image + object labels + 3D positions  → per-object multi-prompt with spatial context

All three methods output the same benchmark-compatible JSON format.

Usage — single episode:
    python qwen_vlm_worker.py \
        --method m1 \
        --input_folder vlm_inputs/safelibero_spatial/level_I/task_0/episode_00 \
        --output_json results/m1_task0_ep00.json

Usage — batch (all episodes under a suite directory):
    python qwen_vlm_worker.py \
        --method m1 \
        --input_dir vlm_inputs/safelibero_spatial \
        --output_json results/m1_spatial_all.json

Usage — dry run (no GPU, placeholder responses):
    python qwen_vlm_worker.py \
        --method m1 \
        --input_folder vlm_inputs/safelibero_spatial/level_I/task_0/episode_00 \
        --output_json /tmp/test.json \
        --dry_run

Output JSON format (matches prompt_tuning_benchmark_set):
    {
      "level_I/task_0/episode_00": {
        "description": "pick up the black bowl ...",
        "end_object": "black bowl",
        "objects": [
          ["moka_pot_obstacle_1", ["above", "around in front of"]],
          ["plate_1", []],
          ["end_effector", ["caution", "rotation lock"]]
        ]
      },
      ...
    }
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

QWEN_MODELS = {
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2-vl-7b":   "Qwen/Qwen2-VL-7B-Instruct",
    "qwen3-vl-8b":   "Qwen/Qwen3-VL-8B",
}

SPATIAL_RELATIONS = ["above", "below", "around in front of", "around behind"]

METHODS = ["m1", "m2", "m3"]

# Full HTTP response body for dry-run mode. Matches live /infer response shape.
DRY_RUN_RESPONSE = {
    "single": {
        "description": "dry run placeholder",
        "end_object": "object",
        "objects": [],
    }
}
# Keep backwards-compatible alias
DRY_RUN_RESULT = DRY_RUN_RESPONSE


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading and inference
# ═══════════════════════════════════════════════════════════════════════════════

def load_qwen_model(model_key: str, device: str = "auto", load_in_4bit: bool = False):
    """Load Qwen VLM model and processor.

    Selects the correct model class based on the model key:
        qwen2.5-vl-*  →  Qwen2_5_VLForConditionalGeneration
        qwen2-vl-*    →  Qwen2VLForConditionalGeneration
        qwen3-vl-*    →  AutoModelForImageTextToText (catches future classes)
    Using the wrong class causes MISSING/UNEXPECTED weight errors because
    the ViT MLP architecture differs between Qwen2-VL and Qwen2.5-VL.
    """
    import torch
    from transformers import AutoProcessor

    hf_id = QWEN_MODELS[model_key]
    logger.info(f"Loading model: {hf_id}  (key={model_key})")

    # Pick the right model class
    if model_key.startswith("qwen2.5"):
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    elif model_key.startswith("qwen2"):
        from transformers import Qwen2VLForConditionalGeneration as ModelCls
    else:
        # Qwen3-VL and future models — let transformers auto-detect
        from transformers import AutoModelForImageTextToText as ModelCls

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    load_kwargs = dict(torch_dtype=dtype, device_map=device)

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
        )

    t0 = time.time()
    model = ModelCls.from_pretrained(hf_id, **load_kwargs)
    processor = AutoProcessor.from_pretrained(hf_id)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s  (class={ModelCls.__name__})")
    return model, processor


def run_vlm_query(model, processor, prompt: str,
                  image_paths: list = None, max_new_tokens: int = 256) -> str:
    """Run a single VLM query with one or more images. Returns raw text response."""
    import torch
    from PIL import Image as PILImage

    content = []
    pil_images = []

    # Attach images
    if image_paths:
        for p in image_paths:
            if p and os.path.exists(p):
                img = PILImage.open(p).convert("RGB")
                pil_images.append(img)
                content.append({"type": "image", "image": img})

    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    if pil_images:
        inputs = processor(text=[prompt_text], images=pil_images,
                           return_tensors="pt", padding=True)
    else:
        inputs = processor(text=[prompt_text], return_tensors="pt", padding=True)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Response parsers
# ═══════════════════════════════════════════════════════════════════════════════

def parse_spatial_response(response: str) -> bool:
    """Returns True if the VLM says the relationship is UNSAFE."""
    r = response.lower()
    if any(k in r for k in ["unsafe", "not safe", "dangerous", "should not", "risk"]):
        return True
    if r.startswith("safe") or "it is safe" in r:
        return False
    # Default to unsafe (conservative)
    return True


def parse_caution_response(response: str) -> bool:
    """Returns True if VLM recommends caution."""
    r = response.lower()
    return any(k in r for k in ["caution", "careful", "slowly", "yes", "should be cautious"])


def parse_rotation_response(response: str) -> bool:
    """Returns True if VLM says rotation should be constrained."""
    r = response.lower()
    return any(k in r for k in ["constrained", "locked", "upright", "spill", "pour", "yes"])


def parse_m2_object_discovery(response: str) -> list:
    """Parse M2's free-form object discovery response into a list of dicts.

    Tries to extract JSON from the response. Falls back to line-by-line parsing.
    Expected format from VLM:
        [{"object": "plate", "unsafe_relationships": ["above"]}, ...]
    """
    # Try JSON extraction
    import re
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # Fallback: try to find JSON object blocks
    results = []
    for m in re.finditer(r'\{[^}]+\}', response):
        try:
            obj = json.loads(m.group())
            if "object" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Episode metadata helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_episode(folder: str) -> dict:
    """Load metadata and resolve image paths for one episode folder."""
    folder = Path(folder)
    meta_path = folder / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {folder}")

    with open(meta_path) as f:
        metadata = json.load(f)

    return {
        "metadata": metadata,
        "agentview": str(folder / "agentview_rgb.png"),
        "eye_in_hand": str(folder / "eye_in_hand_rgb.png"),
        "folder": str(folder),
    }


def get_scene_objects(metadata: dict) -> list:
    """Extract scene object names, filtering out far-away/invalid entries."""
    objects = metadata.get("objects", {})
    robot_prefixes = ("robot0_", "gripper")
    scene = []
    for name, info in objects.items():
        if any(name.startswith(p) for p in robot_prefixes):
            continue
        pos = info.get("position", [0, 0, 0])
        # Skip objects placed far outside workspace
        if any(abs(p) > 5.0 for p in pos):
            continue
        scene.append(name)
    return scene


def infer_manipulated_object(metadata: dict) -> str:
    """Extract the manipulated object name from the task description."""
    desc = metadata.get("task_description", "").lower()
    for phrase in ["pick up the ", "move the ", "put the ", "place the ",
                   "push the ", "put both the "]:
        if phrase in desc:
            rest = desc.split(phrase, 1)[1]
            for stop in [" and ", " on ", " to ", " between ", " into ",
                         " from ", " near ", " in "]:
                if stop in rest:
                    rest = rest.split(stop, 1)[0]
                    break
            return rest.strip()
    return "object"


# ═══════════════════════════════════════════════════════════════════════════════
# Method 1 — Segmentation + VLM → Predicates
#
# The VLM receives RGB images and explicit object labels (from metadata).
# Per-object multi-prompt queries for spatial, caution, and rotation.
# This mirrors Brunke et al.'s multi-prompt strategy with a VLM replacing
# their text-only LLM.
# ═══════════════════════════════════════════════════════════════════════════════

M1_SPATIAL_PROMPT = """\
Look at the image of a robotic manipulation scene.

A robotic manipulator is holding "{manip}" in its end-effector.
There is an object called "{obj}" in the scene.

Is it unsafe for the robot to move the "{manip}" {rel} the "{obj}"?
Consider risks like collision, spilling, contamination, or damage.

Answer ONLY "safe" or "unsafe", followed by a one-sentence reason."""

M1_CAUTION_PROMPT = """\
Look at the image of a robotic manipulation scene.

A robotic manipulator is holding "{manip}" in its end-effector.
There is an object called "{obj}" in the scene.

Should the robot exercise extra caution (move more slowly) when the \
end-effector is near "{obj}"?

Answer ONLY "caution" or "no_caution", followed by a one-sentence reason."""

M1_ROTATION_PROMPT = """\
A robotic manipulator is holding "{manip}" in its end-effector.

Should the end-effector orientation be constrained (rotation locked) to \
prevent the contents from s+pilling or the object from being damaged?

Answer ONLY "constrained_rotation" or "free_rotation", followed by a \
one-sentence reason."""


def process_m1(episode: dict, model, processor, num_votes: int,
               max_new_tokens: int, dry_run: bool) -> dict:
    """M1: Image + object labels → per-object multi-prompt."""
    metadata = episode["metadata"]
    images = [episode["agentview"], episode["eye_in_hand"]]
    manip = infer_manipulated_object(metadata)
    scene_objects = get_scene_objects(metadata)
    task_desc = metadata.get("task_description", "")

    objects_out = []
    vlm_log = []

    for obj_name in scene_objects:
        constraints = []

        # ── Spatial ──
        for rel in SPATIAL_RELATIONS:
            prompt = M1_SPATIAL_PROMPT.format(manip=manip, obj=obj_name, rel=rel)
            votes = []
            for v in range(num_votes):
                if dry_run:
                    resp = f"unsafe. dry run placeholder (vote {v})"
                else:
                    resp = run_vlm_query(model, processor, prompt,
                                         image_paths=images,
                                         max_new_tokens=max_new_tokens)
                votes.append(resp)
                vlm_log.append({"type": "spatial", "object": obj_name,
                                "relation": rel, "vote": v,
                                "prompt": prompt, "response": resp})

            unsafe_votes = sum(parse_spatial_response(r) for r in votes)
            if unsafe_votes > num_votes / 2:
                constraints.append(rel)

        objects_out.append([obj_name, constraints])

    # ── Caution (per-object, but stored on end_effector) ──
    ee_constraints = []
    any_caution = False
    for obj_name in scene_objects:
        prompt = M1_CAUTION_PROMPT.format(manip=manip, obj=obj_name)
        votes = []
        for v in range(num_votes):
            if dry_run:
                resp = f"caution. dry run placeholder (vote {v})"
            else:
                resp = run_vlm_query(model, processor, prompt,
                                     image_paths=images,
                                     max_new_tokens=max_new_tokens)
            votes.append(resp)
            vlm_log.append({"type": "caution", "object": obj_name,
                            "vote": v, "prompt": prompt, "response": resp})

        caution_votes = sum(parse_caution_response(r) for r in votes)
        if caution_votes > num_votes / 2:
            any_caution = True

    if any_caution:
        ee_constraints.append("caution")

    # ── Rotation ──
    prompt = M1_ROTATION_PROMPT.format(manip=manip)
    votes = []
    for v in range(num_votes):
        if dry_run:
            resp = f"constrained_rotation. dry run placeholder (vote {v})"
        else:
            resp = run_vlm_query(model, processor, prompt,
                                 image_paths=[episode["eye_in_hand"]],
                                 max_new_tokens=max_new_tokens)
        votes.append(resp)
        vlm_log.append({"type": "rotation", "vote": v,
                        "prompt": prompt, "response": resp})

    rot_votes = sum(parse_rotation_response(r) for r in votes)
    if rot_votes > num_votes / 2:
        ee_constraints.append("rotation lock")

    objects_out.append(["end_effector", ee_constraints])

    return {
        "description": task_desc,
        "end_object": manip,
        "scene_objects": scene_objects,
        "objects": objects_out,
        "_vlm_log": vlm_log,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Method 2 — VLM-only → Predicates
#
# The VLM receives ONLY RGB images and the task description.
# No object labels are provided. The VLM must:
#   1. Discover what objects are in the scene
#   2. Reason about which spatial relationships are unsafe
#   3. Determine caution and rotation constraints
# This tests the VLM as a joint perceiver + reasoner.
# ═══════════════════════════════════════════════════════════════════════════════

M2_DISCOVERY_PROMPT = """\
Look at this image of a robotic manipulation scene on a tabletop.

The robot's task is: "{task_desc}"

The robot is holding "{manip}" in its end-effector.

1. List every distinct object you can see on the table (exclude the robot arm \
itself and the held object).
2. For each object, decide which spatial relationships would be UNSAFE if the \
robot moved the "{manip}" into that relationship with the object.
   Possible spatial relationships: above, below, around in front of, around behind

Respond ONLY with a JSON array, no other text:
[
  {{"object": "<name>", "unsafe_relationships": ["<rel1>", ...]}},
  ...
]
If an object has no unsafe relationships, use an empty list."""

M2_CAUTION_PROMPT = """\
Look at this image. The robot is holding "{manip}".

Given the objects visible in the scene, should the robot exercise extra \
caution (move more slowly) when manipulating this object?

Answer ONLY "caution" or "no_caution", followed by a one-sentence reason."""

M2_ROTATION_PROMPT = M1_ROTATION_PROMPT  # Same as M1 — object-centric, no scene needed


def process_m2(episode: dict, model, processor, num_votes: int,
               max_new_tokens: int, dry_run: bool) -> dict:
    """M2: Image only → VLM discovers objects + constraints."""
    metadata = episode["metadata"]
    images = [episode["agentview"], episode["eye_in_hand"]]
    manip = infer_manipulated_object(metadata)
    task_desc = metadata.get("task_description", "")

    vlm_log = []

    # ── Object discovery + spatial (combined prompt, majority vote) ──
    all_discoveries = []
    for v in range(num_votes):
        prompt = M2_DISCOVERY_PROMPT.format(task_desc=task_desc, manip=manip)
        if dry_run:
            resp = json.dumps([
                {"object": "plate_1", "unsafe_relationships": ["above"]},
                {"object": "obstacle_moka_pot_1", "unsafe_relationships": ["above", "around in front of"]},
            ])
        else:
            resp = run_vlm_query(model, processor, prompt,
                                 image_paths=images,
                                 max_new_tokens=max_new_tokens)

        vlm_log.append({"type": "discovery", "vote": v,
                        "prompt": prompt, "response": resp})
        parsed = parse_m2_object_discovery(resp)
        all_discoveries.append(parsed)

    # Merge votes: for each object seen in any vote, tally relationship votes
    obj_rel_votes = {}  # obj_name -> {rel -> count}
    obj_seen_count = {}  # obj_name -> how many votes saw it
    for discovery in all_discoveries:
        seen_this_vote = set()
        for entry in discovery:
            name = entry.get("object", "unknown")
            rels = entry.get("unsafe_relationships", [])
            if name not in obj_rel_votes:
                obj_rel_votes[name] = {}
                obj_seen_count[name] = 0
            if name not in seen_this_vote:
                obj_seen_count[name] += 1
                seen_this_vote.add(name)
            for r in rels:
                r_norm = r.lower().strip()
                obj_rel_votes[name][r_norm] = obj_rel_votes[name].get(r_norm, 0) + 1

    # Keep objects seen in majority of votes; keep relations with majority votes
    threshold = num_votes / 2
    objects_out = []
    for obj_name, rel_counts in obj_rel_votes.items():
        if obj_seen_count[obj_name] <= threshold:
            continue  # Object not consistently detected
        constraints = [r for r, cnt in rel_counts.items() if cnt > threshold]
        objects_out.append([obj_name, constraints])

    # ── Caution ──
    ee_constraints = []
    prompt = M2_CAUTION_PROMPT.format(manip=manip)
    votes = []
    for v in range(num_votes):
        if dry_run:
            resp = "caution. the held object may be fragile"
        else:
            resp = run_vlm_query(model, processor, prompt,
                                 image_paths=images,
                                 max_new_tokens=max_new_tokens)
        votes.append(resp)
        vlm_log.append({"type": "caution", "vote": v,
                        "prompt": prompt, "response": resp})

    if sum(parse_caution_response(r) for r in votes) > threshold:
        ee_constraints.append("caution")

    # ── Rotation ──
    prompt = M2_ROTATION_PROMPT.format(manip=manip)
    votes = []
    for v in range(num_votes):
        if dry_run:
            resp = "constrained_rotation. may spill"
        else:
            resp = run_vlm_query(model, processor, prompt,
                                 image_paths=[episode["eye_in_hand"]],
                                 max_new_tokens=max_new_tokens)
        votes.append(resp)
        vlm_log.append({"type": "rotation", "vote": v,
                        "prompt": prompt, "response": resp})

    if sum(parse_rotation_response(r) for r in votes) > threshold:
        ee_constraints.append("rotation lock")

    objects_out.append(["end_effector", ee_constraints])

    return {
        "description": task_desc,
        "end_object": manip,
        "objects": objects_out,
        "_vlm_log": vlm_log,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Method 3 — 3D context + VLM → Predicates
#
# The VLM receives RGB images AND 3D spatial context (object positions and
# approximate sizes from metadata.json). The prompt includes textual
# descriptions like "plate at position (0.30, 0.10, 0.82)".
# This tests whether grounding the VLM with metric spatial information
# improves predicate accuracy.
# ═══════════════════════════════════════════════════════════════════════════════

M3_SPATIAL_PROMPT = """\
Look at the image of a robotic manipulation scene.

A robotic manipulator is holding "{manip}" in its end-effector.
The end-effector is currently at position ({ee_x:.3f}, {ee_y:.3f}, {ee_z:.3f}).

There is an object called "{obj}" at position ({ox:.3f}, {oy:.3f}, {oz:.3f}).

Is it unsafe for the robot to move the "{manip}" {rel} the "{obj}"?
Consider the 3D positions, risks of collision, spilling, contamination, or damage.

Answer ONLY "safe" or "unsafe", followed by a one-sentence reason."""

M3_CAUTION_PROMPT = """\
Look at the image of a robotic manipulation scene.

A robotic manipulator is holding "{manip}" in its end-effector at position \
({ee_x:.3f}, {ee_y:.3f}, {ee_z:.3f}).

The following objects are in the scene:
{objects_list}

Should the robot exercise extra caution (move more slowly) when the \
end-effector is near any of these objects?

Answer ONLY "caution" or "no_caution", followed by a one-sentence reason."""

M3_ROTATION_PROMPT = M1_ROTATION_PROMPT  # Same — object-centric


def format_objects_list_3d(metadata: dict, scene_objects: list) -> str:
    """Format scene objects with 3D positions for prompt."""
    objects_info = metadata.get("objects", {})
    lines = []
    for name in scene_objects:
        info = objects_info.get(name, {})
        pos = info.get("position", [0, 0, 0])
        lines.append(f"  - \"{name}\" at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    return "\n".join(lines) if lines else "  (none)"


def process_m3(episode: dict, model, processor, num_votes: int,
               max_new_tokens: int, dry_run: bool) -> dict:
    """M3: Image + 3D positions → per-object multi-prompt with spatial context."""
    metadata = episode["metadata"]
    images = [episode["agentview"], episode["eye_in_hand"]]
    manip = infer_manipulated_object(metadata)
    scene_objects = get_scene_objects(metadata)
    task_desc = metadata.get("task_description", "")
    objects_info = metadata.get("objects", {})

    # End-effector position
    robot_state = metadata.get("robot_state", {})
    ee_pos = robot_state.get("eef_pos", [0, 0, 0])

    vlm_log = []
    objects_out = []

    for obj_name in scene_objects:
        constraints = []
        obj_pos = objects_info.get(obj_name, {}).get("position", [0, 0, 0])

        # ── Spatial ──
        for rel in SPATIAL_RELATIONS:
            prompt = M3_SPATIAL_PROMPT.format(
                manip=manip, obj=obj_name, rel=rel,
                ee_x=ee_pos[0], ee_y=ee_pos[1], ee_z=ee_pos[2],
                ox=obj_pos[0], oy=obj_pos[1], oz=obj_pos[2],
            )
            votes = []
            for v in range(num_votes):
                if dry_run:
                    resp = f"unsafe. dry run placeholder (vote {v})"
                else:
                    resp = run_vlm_query(model, processor, prompt,
                                         image_paths=images,
                                         max_new_tokens=max_new_tokens)
                votes.append(resp)
                vlm_log.append({"type": "spatial", "object": obj_name,
                                "relation": rel, "vote": v,
                                "prompt": prompt, "response": resp})

            if sum(parse_spatial_response(r) for r in votes) > num_votes / 2:
                constraints.append(rel)

        objects_out.append([obj_name, constraints])

    # ── Caution ──
    ee_constraints = []
    obj_list_str = format_objects_list_3d(metadata, scene_objects)
    prompt = M3_CAUTION_PROMPT.format(
        manip=manip, objects_list=obj_list_str,
        ee_x=ee_pos[0], ee_y=ee_pos[1], ee_z=ee_pos[2],
    )
    votes = []
    for v in range(num_votes):
        if dry_run:
            resp = "caution. dry run"
        else:
            resp = run_vlm_query(model, processor, prompt,
                                 image_paths=images,
                                 max_new_tokens=max_new_tokens)
        votes.append(resp)
        vlm_log.append({"type": "caution", "vote": v,
                        "prompt": prompt, "response": resp})

    if sum(parse_caution_response(r) for r in votes) > num_votes / 2:
        ee_constraints.append("caution")

    # ── Rotation ──
    prompt = M3_ROTATION_PROMPT.format(manip=manip)
    votes = []
    for v in range(num_votes):
        if dry_run:
            resp = "constrained_rotation. dry run"
        else:
            resp = run_vlm_query(model, processor, prompt,
                                 image_paths=[episode["eye_in_hand"]],
                                 max_new_tokens=max_new_tokens)
        votes.append(resp)
        vlm_log.append({"type": "rotation", "vote": v,
                        "prompt": prompt, "response": resp})

    if sum(parse_rotation_response(r) for r in votes) > num_votes / 2:
        ee_constraints.append("rotation lock")

    objects_out.append(["end_effector", ee_constraints])

    return {
        "description": task_desc,
        "end_object": manip,
        "scene_objects": scene_objects,
        "objects": objects_out,
        "_vlm_log": vlm_log,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════════════════

METHOD_FN = {
    "m1": process_m1,
    "m2": process_m2,
    "m3": process_m3,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Folder discovery
# ═══════════════════════════════════════════════════════════════════════════════

def discover_episodes(input_dir: str) -> list:
    """Find all episode folders under input_dir (any nesting depth).

    Returns list of (relative_key, absolute_path) tuples sorted by key.
    E.g. ("level_I/task_0/episode_00", "/abs/path/to/episode_00")
    """
    input_dir = Path(input_dir)
    episodes = []
    for meta in sorted(input_dir.rglob("metadata.json")):
        ep_folder = meta.parent
        rel = ep_folder.relative_to(input_dir)
        episodes.append((str(rel), str(ep_folder)))
    return episodes


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VLM inference for semantic safety predicates (M1/M2/M3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Method
    parser.add_argument("--method", required=True, choices=METHODS,
                        help="m1 = Seg+VLM labels, m2 = VLM-only, m3 = 3D+VLM")

    # Input (mutually exclusive: single folder or batch directory)
    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument("--input_folder",
                           help="Single episode folder (must contain metadata.json)")
    input_grp.add_argument("--input_dir",
                           help="Root directory to scan recursively for episode folders")

    # Output
    parser.add_argument("--output_json", required=True,
                        help="Path to write results JSON")
    parser.add_argument("--save_vlm_log", action="store_true",
                        help="Include raw VLM responses in output (under _vlm_log)")

    # Model
    parser.add_argument("--model", default="qwen2.5-vl-7b",
                        choices=list(QWEN_MODELS.keys()))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=256)

    # Voting
    parser.add_argument("--num_votes", type=int, default=1,
                        help="Majority voting rounds per query (Brunke et al. use 5)")

    # Dry run
    parser.add_argument("--dry_run", action="store_true",
                        help="Placeholder responses, no GPU (for testing)")

    args = parser.parse_args()

    # ── Resolve episodes ──
    if args.input_folder:
        episodes = [("single", args.input_folder)]
    else:
        episodes = discover_episodes(args.input_dir)
        if not episodes:
            logger.error(f"No episode folders found under {args.input_dir}")
            sys.exit(1)
        logger.info(f"Found {len(episodes)} episodes under {args.input_dir}")

    # ── Load model ──
    model, processor = None, None
    if not args.dry_run:
        model, processor = load_qwen_model(
            args.model, device=args.device, load_in_4bit=args.load_in_4bit,
        )

    # ── Process ──
    process_fn = METHOD_FN[args.method]
    all_results = {}
    t_start = time.time()

    for i, (key, folder) in enumerate(episodes):
        logger.info(f"[{i+1}/{len(episodes)}] {key}")
        try:
            episode = load_episode(folder)
        except FileNotFoundError as e:
            logger.warning(f"  Skipping: {e}")
            continue

        result = process_fn(
            episode, model, processor,
            num_votes=args.num_votes,
            max_new_tokens=args.max_new_tokens,
            dry_run=args.dry_run,
        )

        # Strip VLM log unless requested
        if not args.save_vlm_log:
            result.pop("_vlm_log", None)

        all_results[key] = result
        logger.info(f"  objects: {len(result['objects'])}  "
                     f"manip: {result['end_object']}")

    elapsed = time.time() - t_start
    logger.info(f"Processed {len(all_results)} episodes in {elapsed:.1f}s "
                f"({elapsed / max(len(all_results), 1):.1f}s/ep)")

    # ── Write output ──
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()