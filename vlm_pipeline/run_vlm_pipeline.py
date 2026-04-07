#!/usr/bin/env python3
"""
run_vlm_pipeline.py

VLM-based semantic safety predicate extraction pipeline.

Reads RGB images from vlm_inputs/safelibero_spatial, queries the Qwen VLM
with three prompt types (spatial, caution, rotation), and saves structured
JSON outputs in the prompt_tuning_benchmark_set format.

Usage:
    # Dummy mode (no GPU, generates placeholder outputs for testing)
    python run_vlm_pipeline.py --dummy

    # Real VLM inference
    python run_vlm_pipeline.py --model qwen2.5-vl-7b

    # Specific episode
    python run_vlm_pipeline.py --model qwen2.5-vl-7b --episode 0
"""

import argparse
import glob
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VLMPipeline] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "vlm_inputs" / "safelibero_spatial"
OUTPUT_DIR = BASE_DIR / "vlm_outputs"

SPATIAL_RELATIONSHIPS = ["above", "below", "around in front of", "around behind"]

SCENE_DESCRIPTION = (
    "A robotic manipulator is holding a {end_object} in its end-effector. "
    "The robotic manipulator is navigating atop a desk."
)

# ── Prompt Templates ─────────────────────────────────────────────────────────

SPATIAL_PROMPT = """\
Scene description: {description}

The robot is holding a {end_object} in its end-effector.
Consider the object "{scene_object}" in the scene.

For each of the following spatial relationships, answer YES or NO — is it \
unsafe for the robot to move the {end_object} {relationships_list} the \
{scene_object}?

Spatial relationships: {relationships}

Respond ONLY in this exact JSON format, no other text:
{{"object": "{scene_object}", "unsafe_relationships": ["<relationship1>", ...]}}

If none are unsafe, respond with:
{{"object": "{scene_object}", "unsafe_relationships": []}}"""

CAUTION_PROMPT = """\
Scene description: {description}

The robot is holding a {end_object}. Consider the safety of manipulating \
this object.

Should the robot exercise extra caution (move more slowly, be more careful) \
when moving the {end_object}? Answer based on the physical properties of \
the held object — is it fragile, contains liquid, is sharp, flammable, etc.?

Respond ONLY in this exact JSON format, no other text:
{{"object": "end_effector", "caution": true, "reason": "..."}}
or
{{"object": "end_effector", "caution": false, "reason": "..."}}"""

ROTATION_PROMPT = """\
The robot is holding a {end_object} in its end-effector.

Should the end-effector orientation be constrained (i.e., rotation locked) \
to prevent the contents of {end_object} from spilling or the object from \
being damaged?

Consider: would tilting or rotating this object cause problems (spilling, \
falling, breaking)?

Respond ONLY in this exact JSON format, no other text:
{{"rotation_lock": true, "reason": "..."}}
or
{{"rotation_lock": false, "reason": "..."}}"""


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class EpisodeData:
    """All data for a single episode."""
    level: str
    task_id: int
    episode_idx: int
    task_description: str
    agentview_rgb_path: str
    eye_in_hand_rgb_path: str
    backview_rgb_path: Optional[str]
    metadata: dict
    visible_objects: List[str] = field(default_factory=list)


@dataclass
class SpatialResult:
    object_name: str
    unsafe_relationships: List[str]


@dataclass
class CautionResult:
    caution: bool
    reason: str


@dataclass
class RotationResult:
    rotation_lock: bool
    reason: str


@dataclass
class EpisodeResults:
    """Full VLM results for one episode."""
    episode_data: EpisodeData
    held_object: str
    spatial: List[SpatialResult]
    caution: CautionResult
    rotation: RotationResult
    raw_responses: Dict[str, List[str]] = field(default_factory=dict)


# ── Episode Discovery ────────────────────────────────────────────────────────

def discover_episodes(input_dir: Path, level: str = None,
                      task_id: int = None,
                      episode_idx: int = None) -> List[EpisodeData]:
    """Find all episodes under the input directory."""
    episodes = []

    for level_dir in sorted(input_dir.iterdir()):
        if not level_dir.is_dir():
            continue
        level_name = level_dir.name  # e.g. "level_I"
        if level and level_name != f"level_{level}":
            continue

        for task_dir in sorted(level_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            tid = int(task_dir.name.split("_")[1])
            if task_id is not None and tid != task_id:
                continue

            for ep_dir in sorted(task_dir.iterdir()):
                if not ep_dir.is_dir():
                    continue
                eidx = int(ep_dir.name.split("_")[1])
                if episode_idx is not None and eidx != episode_idx:
                    continue

                metadata_path = ep_dir / "metadata.json"
                agentview_path = ep_dir / "agentview_rgb.png"
                eye_in_hand_path = ep_dir / "eye_in_hand_rgb.png"
                backview_path = ep_dir / "backview_rgb.png"

                if not metadata_path.exists() or not agentview_path.exists():
                    logger.warning(f"Skipping incomplete episode: {ep_dir}")
                    continue

                with open(metadata_path) as f:
                    metadata = json.load(f)

                # Extract visible objects (filter out objects far from workspace)
                visible = _extract_visible_objects(metadata)

                episodes.append(EpisodeData(
                    level=level_name,
                    task_id=tid,
                    episode_idx=eidx,
                    task_description=metadata.get("task_description", ""),
                    agentview_rgb_path=str(agentview_path),
                    eye_in_hand_rgb_path=str(eye_in_hand_path),
                    backview_rgb_path=str(backview_path) if backview_path.exists() else None,
                    metadata=metadata,
                    visible_objects=visible,
                ))

    logger.info(f"Discovered {len(episodes)} episodes")
    return episodes


def _extract_visible_objects(metadata: dict) -> List[str]:
    """Extract object names that are actually visible in the scene.

    Filters out objects placed far away (position > 5m from origin),
    robot parts, and fixture elements.
    """
    objects = metadata.get("objects", {})
    visible = []
    for name, info in objects.items():
        pos = info.get("position", [0, 0, 0])
        # Filter out objects placed far away (off-screen obstacles)
        if any(abs(p) > 5.0 for p in pos):
            continue
        # Skip robot/fixture naming patterns
        if "robot" in name or "cabinet" in name or "stove" in name:
            continue
        visible.append(name)
    return visible


def _infer_held_object(metadata: dict) -> str:
    """Infer what the robot is holding from the task description.

    For the initial scene (before grasping), we treat the target object
    from the task description as the planned held object.
    """
    desc = metadata.get("task_description", "").lower()

    # Common held object patterns from SafeLIBERO
    held_object_patterns = {
        "bowl": "black bowl",
        "cup": "cup of water",
        "candle": "lit candle",
        "knife": "knife",
        "sponge": "dry sponge",
        "bottle": "bottle",
        "plate": "plate",
        "mug": "coffee mug",
    }

    # Try to find what's being picked up
    pick_match = re.search(r"pick up (?:the |a )?([\w\s]+?)(?:\s+(?:and|from|between|on))", desc)
    if pick_match:
        target = pick_match.group(1).strip()
        return target

    # Fallback: check for known objects
    for key, obj_name in held_object_patterns.items():
        if key in desc:
            return obj_name

    return "object"


# ── VLM Interface ────────────────────────────────────────────────────────────

class VLMInterface:
    """Wrapper for Qwen VLM inference."""

    def __init__(self, model_key: str = "qwen2.5-vl-7b", dummy: bool = False,
                 load_in_4bit: bool = False, num_votes: int = 1):
        self.model_key = model_key
        self.dummy = dummy
        self.num_votes = num_votes
        self.model = None
        self.processor = None

        if not dummy:
            self._load_model(load_in_4bit)

    def _load_model(self, load_in_4bit: bool):
        """Load the Qwen VLM model."""
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        QWEN_MODELS = {
            "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
            "qwen3-vl-8b": "Qwen/Qwen3-VL-8B",
        }

        hf_id = QWEN_MODELS[self.model_key]
        logger.info(f"Loading VLM: {hf_id}")

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        load_kwargs = dict(torch_dtype=dtype, device_map="auto")

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
            )

        t0 = time.time()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(hf_id, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(hf_id)
        logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    def query(self, prompt: str, image_paths: List[str],
              max_new_tokens: int = 512) -> str:
        """Run a single VLM query with images."""
        if self.dummy:
            return self._dummy_response(prompt)

        import torch
        from PIL import Image as PILImage

        # Build message content
        content = []
        pil_images = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                img = PILImage.open(img_path).convert("RGB")
                pil_images.append(img)
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if pil_images:
            inputs = self.processor(
                text=[prompt_text], images=pil_images,
                return_tensors="pt", padding=True,
            )
        else:
            inputs = self.processor(
                text=[prompt_text], return_tensors="pt", padding=True,
            )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        return response

    def query_with_voting(self, prompt: str, image_paths: List[str],
                          max_new_tokens: int = 512) -> List[str]:
        """Run a query N times for majority voting."""
        responses = []
        for i in range(self.num_votes):
            resp = self.query(prompt, image_paths, max_new_tokens)
            responses.append(resp)
        return responses

    def _dummy_response(self, prompt: str) -> str:
        """Generate a plausible dummy response based on prompt type."""
        prompt_lower = prompt.lower()

        if "spatial relationships" in prompt_lower or "unsafe_relationships" in prompt_lower:
            # Extract object name from prompt
            obj_match = re.search(r'object "([^"]+)"', prompt)
            obj_name = obj_match.group(1) if obj_match else "unknown"
            # Dummy: mark some relationships as unsafe for interesting objects
            unsafe = []
            if any(kw in obj_name.lower() for kw in ["laptop", "camera", "book"]):
                unsafe = ["above"]
            return json.dumps({"object": obj_name, "unsafe_relationships": unsafe})

        elif "caution" in prompt_lower:
            # Check if the held object sounds dangerous
            has_danger = any(kw in prompt_lower for kw in
                            ["water", "candle", "knife", "soup", "fish tank", "bottle", "sugar"])
            return json.dumps({
                "object": "end_effector",
                "caution": has_danger,
                "reason": "dummy: object requires careful handling" if has_danger else "dummy: safe object"
            })

        elif "rotation" in prompt_lower:
            needs_lock = any(kw in prompt_lower for kw in
                             ["water", "candle", "soup", "fish tank", "plate", "bottle", "sugar"])
            return json.dumps({
                "rotation_lock": needs_lock,
                "reason": "dummy: contents may spill" if needs_lock else "dummy: no spill risk"
            })

        return '{"error": "unknown prompt type"}'


# ── Response Parsing ─────────────────────────────────────────────────────────

def parse_json_response(response: str) -> dict:
    """Extract JSON from a VLM response that may contain extra text."""
    # Try direct parse first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in the response
    # Look for outermost { ... }
    brace_depth = 0
    start = None
    for i, ch in enumerate(response):
        if ch == '{':
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                try:
                    return json.loads(response[start:i + 1])
                except json.JSONDecodeError:
                    start = None

    logger.warning(f"Could not parse JSON from response: {response[:200]}")
    return {}


def majority_vote_spatial(responses: List[str], object_name: str) -> SpatialResult:
    """Majority vote across spatial relationship responses."""
    relationship_votes = {r: 0 for r in SPATIAL_RELATIONSHIPS}

    for resp in responses:
        parsed = parse_json_response(resp)
        unsafe = parsed.get("unsafe_relationships", [])
        for rel in unsafe:
            rel_lower = rel.lower().strip()
            # Normalize: "under" -> "below", etc.
            if rel_lower == "under":
                rel_lower = "below"
            if rel_lower in relationship_votes:
                relationship_votes[rel_lower] += 1

    threshold = len(responses) / 2.0
    unsafe_rels = [r for r, count in relationship_votes.items() if count > threshold]

    return SpatialResult(object_name=object_name, unsafe_relationships=unsafe_rels)


def majority_vote_caution(responses: List[str]) -> CautionResult:
    """Majority vote for caution flag."""
    caution_votes = 0
    reasons = []
    for resp in responses:
        parsed = parse_json_response(resp)
        if parsed.get("caution", False):
            caution_votes += 1
            reasons.append(parsed.get("reason", ""))

    threshold = len(responses) / 2.0
    is_caution = caution_votes > threshold
    reason = reasons[0] if reasons else "no caution needed"
    return CautionResult(caution=is_caution, reason=reason)


def majority_vote_rotation(responses: List[str]) -> RotationResult:
    """Majority vote for rotation lock."""
    lock_votes = 0
    reasons = []
    for resp in responses:
        parsed = parse_json_response(resp)
        if parsed.get("rotation_lock", False):
            lock_votes += 1
            reasons.append(parsed.get("reason", ""))

    threshold = len(responses) / 2.0
    is_locked = lock_votes > threshold
    reason = reasons[0] if reasons else "no rotation constraint needed"
    return RotationResult(rotation_lock=is_locked, reason=reason)


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run_episode(vlm: VLMInterface, episode: EpisodeData,
                held_object: str = None) -> EpisodeResults:
    """Run the full VLM predicate extraction pipeline for one episode."""
    if held_object is None:
        held_object = _infer_held_object(episode.metadata)

    description = SCENE_DESCRIPTION.format(end_object=held_object)
    image_paths = [episode.agentview_rgb_path, episode.eye_in_hand_rgb_path]
    if episode.backview_rgb_path is not None:
        image_paths.append(episode.backview_rgb_path)
    raw_responses = {"spatial": [], "caution": [], "rotation": []}

    logger.info(f"Processing episode {episode.episode_idx} | "
                f"held={held_object} | objects={episode.visible_objects}")

    # 1. Spatial constraints — one query per visible scene object
    spatial_results = []
    for obj_name in episode.visible_objects:
        prompt = SPATIAL_PROMPT.format(
            description=description,
            end_object=held_object,
            scene_object=obj_name,
            relationships_list="/".join(SPATIAL_RELATIONSHIPS),
            relationships=json.dumps(SPATIAL_RELATIONSHIPS),
        )
        responses = vlm.query_with_voting(prompt, image_paths)
        raw_responses["spatial"].extend(responses)
        result = majority_vote_spatial(responses, obj_name)
        spatial_results.append(result)
        logger.info(f"  Spatial [{obj_name}]: unsafe={result.unsafe_relationships}")

    # 2. Caution constraint — one query for the held object
    caution_prompt = CAUTION_PROMPT.format(
        description=description,
        end_object=held_object,
    )
    caution_responses = vlm.query_with_voting(caution_prompt, image_paths)
    raw_responses["caution"] = caution_responses
    caution_result = majority_vote_caution(caution_responses)
    logger.info(f"  Caution: {caution_result.caution} ({caution_result.reason})")

    # 3. Rotation constraint — one query for the held object
    rotation_prompt = ROTATION_PROMPT.format(end_object=held_object)
    rotation_responses = vlm.query_with_voting(rotation_prompt, image_paths)
    raw_responses["rotation"] = rotation_responses
    rotation_result = majority_vote_rotation(rotation_responses)
    logger.info(f"  Rotation lock: {rotation_result.rotation_lock} ({rotation_result.reason})")

    return EpisodeResults(
        episode_data=episode,
        held_object=held_object,
        spatial=spatial_results,
        caution=caution_result,
        rotation=rotation_result,
        raw_responses=raw_responses,
    )


# ── Output Formatting ────────────────────────────────────────────────────────

def format_benchmark_output(results: List[EpisodeResults]) -> Tuple[dict, dict, dict]:
    """Convert episode results into benchmark-format JSON dicts.

    Returns (spatial_json, caution_json, rotation_json) matching the format
    in prompt_tuning_benchmark_set/.
    """
    spatial_json = {}
    caution_json = {}
    rotation_json = {}

    for res in results:
        # Key by episode for now (level_task_episode)
        key = f"{res.episode_data.level}_task{res.episode_data.task_id}_ep{res.episode_data.episode_idx}"
        held = res.held_object
        desc = SCENE_DESCRIPTION.format(end_object=held)

        # Spatial: list of [object_name, [unsafe_relationships]]
        spatial_objects = []
        for sp in res.spatial:
            spatial_objects.append([sp.object_name, sp.unsafe_relationships])

        spatial_json[key] = {
            "description": desc,
            "end_object": held,
            "objects": spatial_objects,
        }

        # Caution
        caution_constraints = ["caution"] if res.caution.caution else []
        caution_json[key] = {
            "description": desc,
            "end_object": held,
            "objects": [["end_effector", caution_constraints]],
        }

        # Rotation
        rotation_constraints = ["rotation lock"] if res.rotation.rotation_lock else []
        rotation_json[key] = {
            "description": desc,
            "end_object": held,
            "objects": [["end_effector", rotation_constraints]],
        }

    return spatial_json, caution_json, rotation_json


def save_results(results: List[EpisodeResults], output_dir: Path,
                 tag: str = "vlm"):
    """Save results as three benchmark-format JSON files + raw responses."""
    output_dir.mkdir(parents=True, exist_ok=True)

    spatial_json, caution_json, rotation_json = format_benchmark_output(results)

    spatial_path = output_dir / f"{tag}_spatial.json"
    caution_path = output_dir / f"{tag}_caution.json"
    rotation_path = output_dir / f"{tag}_rotation.json"

    with open(spatial_path, "w") as f:
        json.dump(spatial_json, f, indent=2)
    logger.info(f"Saved spatial results: {spatial_path}")

    with open(caution_path, "w") as f:
        json.dump(caution_json, f, indent=2)
    logger.info(f"Saved caution results: {caution_path}")

    with open(rotation_path, "w") as f:
        json.dump(rotation_json, f, indent=2)
    logger.info(f"Saved rotation results: {rotation_path}")

    # Also save full raw responses for analysis
    raw_path = output_dir / f"{tag}_raw_responses.json"
    raw_data = {}
    for res in results:
        key = f"{res.episode_data.level}_task{res.episode_data.task_id}_ep{res.episode_data.episode_idx}"
        raw_data[key] = {
            "held_object": res.held_object,
            "task_description": res.episode_data.task_description,
            "visible_objects": res.episode_data.visible_objects,
            "raw_responses": res.raw_responses,
        }
    with open(raw_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    logger.info(f"Saved raw responses: {raw_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VLM semantic safety predicate extraction pipeline"
    )
    parser.add_argument("--model", default="qwen2.5-vl-7b",
                        choices=["qwen2.5-vl-7b", "qwen3-vl-8b"],
                        help="Qwen VLM model to use")
    parser.add_argument("--dummy", action="store_true",
                        help="Use dummy VLM responses (no GPU needed)")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load model with 4-bit quantization")
    parser.add_argument("--num_votes", type=int, default=1,
                        help="Number of voting rounds per query (default 1, paper uses 5)")
    parser.add_argument("--input_dir", type=str, default=str(INPUT_DIR),
                        help="Input directory with episode data")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for results")
    parser.add_argument("--level", type=str, default=None,
                        help="Filter by safety level (e.g. 'I')")
    parser.add_argument("--task", type=int, default=None,
                        help="Filter by task ID")
    parser.add_argument("--episode", type=int, default=None,
                        help="Filter by episode index")
    parser.add_argument("--held_object", type=str, default=None,
                        help="Override held object name (auto-detected from metadata)")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Max number of episodes to process")
    parser.add_argument("--tag", type=str, default="vlm",
                        help="Output file tag/prefix")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Discover episodes
    episodes = discover_episodes(
        input_dir,
        level=args.level,
        task_id=args.task,
        episode_idx=args.episode,
    )

    if not episodes:
        logger.error("No episodes found. Check --input_dir path.")
        sys.exit(1)

    if args.max_episodes:
        episodes = episodes[:args.max_episodes]

    # Initialize VLM
    vlm = VLMInterface(
        model_key=args.model,
        dummy=args.dummy,
        load_in_4bit=args.load_in_4bit,
        num_votes=args.num_votes,
    )

    # Run pipeline
    all_results = []
    for i, episode in enumerate(episodes):
        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {i+1}/{len(episodes)}: "
                     f"{episode.level}/task_{episode.task_id}/episode_{episode.episode_idx:02d}")
        logger.info(f"Task: {episode.task_description}")
        logger.info(f"{'='*60}")

        result = run_episode(vlm, episode, held_object=args.held_object)
        all_results.append(result)

    # Save outputs
    tag = args.tag if not args.dummy else f"{args.tag}_dummy"
    save_results(all_results, output_dir, tag=tag)

    logger.info(f"\nPipeline complete. Processed {len(all_results)} episodes.")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
