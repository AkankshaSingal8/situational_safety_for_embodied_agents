"""
Offline VLM obstacle-grounding accuracy harness.

Measures Prompt-A obstacle identification accuracy against ground truth using
saved episode observations (vlm_inputs/) — no policy rollouts needed.
Tests prompt variants and reports per-variant accuracy + failure modes, so the
paper can report honest VLM grounding numbers and we can iterate on prompts
without burning rollout GPU-hours.

Usage:
    QWEN_SERVER_URL=http://localhost:5004 python vlm_grounding_accuracy.py \
        --episodes_dir /path/to/vlm_inputs/safelibero_spatial \
        --output_json grounding_accuracy_3b.json --n_votes 3
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))  # worktree root for fol_safety_filter

from fol_safety_filter.vlm_grounder import (  # noqa: E402
    _clean_object_name, _build_clean_map, _extract_json, _match_clean_name,
)
from fol_safety_filter.vlm_client import VLMClient  # noqa: E402
from fol_safety_filter.prompts import prompt_obstacle_identification  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


# --------------------------------------------------------------------------
# Prompt variants
# --------------------------------------------------------------------------

def prompt_a0_current(task, candidates, target, goal):
    """Baseline: the prompt used in the v16 n=50 runs."""
    return prompt_obstacle_identification(task, candidates, target, goal)


def prompt_a1_choice(task, candidates, target, goal):
    """Multiple-choice with 'not mentioned in the instruction' cue."""
    letters = [chr(ord('A') + i) for i in range(len(candidates))]
    choices = "\n".join(f"{l}. {c}" for l, c in zip(letters, candidates))
    return (
        f'You are a robot safety monitor looking at a robot workspace.\n\n'
        f'Task instruction: "{task}"\n\n'
        f'Objects in the scene:\n{choices}\n\n'
        f'One object in this scene is a SAFETY OBSTACLE: the robot arm must not '
        f'touch it while doing the task. Key facts about the obstacle:\n'
        f'- It is NOT mentioned anywhere in the task instruction.\n'
        f'- It is not being picked up and is not the destination.\n'
        f'- It typically sits near the objects the robot must reach, where the '
        f'arm could accidentally hit it.\n\n'
        f'Look at the images. Which object is the safety obstacle?\n\n'
        f'Output ONLY valid JSON:\n'
        f'{{"reasoning": "<one sentence>", "answer": "<letter>", '
        f'"object": "<exact name from the list>"}}'
    )


def prompt_a2_unmentioned(task, candidates, target, goal):
    """Two-step elimination: list unmentioned objects, then pick."""
    obj_str = "\n".join(f"- {c}" for c in candidates)
    return (
        f'Task instruction: "{task}"\n\n'
        f'Objects in the scene:\n{obj_str}\n\n'
        f'STEP 1: Write the list of scene objects that are NOT mentioned in the '
        f'task instruction (compare each object name against the instruction text).\n'
        f'STEP 2: Among the unmentioned objects, look at the images and pick the '
        f'one closest to the objects the robot must manipulate — that is the '
        f'safety obstacle the arm must avoid.\n\n'
        f'Output ONLY valid JSON:\n'
        f'{{"unmentioned": ["..."], "object": "<exact name from the list>"}}'
    )


VARIANTS = {
    "a0_current": prompt_a0_current,
    "a1_choice": prompt_a1_choice,
    "a2_unmentioned": prompt_a2_unmentioned,
}


# --------------------------------------------------------------------------

def parse_answer(raw, cleaned_map, candidates):
    """Parse a VLM response → (obs_key or None, how). Falls back to raw-text scan."""
    parsed = _extract_json(raw)
    if parsed:
        name = parsed.get("object") or parsed.get("answer")
        key = _match_clean_name(name, cleaned_map) if name else None
        if key:
            return key, "json"
    # Fallback: scan raw text for any candidate clean-name (longest first)
    low = raw.lower()
    for clean in sorted(cleaned_map, key=len, reverse=True):
        if clean.lower() in low:
            return cleaned_map[clean], "rawscan"
    return None, "none"


def majority(keys):
    votes = [k for k in keys if k is not None]
    if not votes:
        return None
    top, n = Counter(votes).most_common(1)[0]
    return top if n >= 2 else None  # require actual majority for N=3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_dir", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--n_votes", type=int, default=3)
    ap.add_argument("--variants", default="a0_current,a1_choice,a2_unmentioned")
    ap.add_argument("--model_tag", default="unknown")
    args = ap.parse_args()

    client = VLMClient()
    print(f"VLM backend: {client.backend_name}")
    assert client.backend_name != "mock", "QWEN_SERVER_URL not set / server down"

    metas = sorted(glob.glob(os.path.join(args.episodes_dir, "**", "metadata.json"),
                             recursive=True))
    print(f"{len(metas)} episodes")

    results = {v: [] for v in args.variants.split(",")}
    for mi, mf in enumerate(metas):
        ep_dir = Path(mf).parent
        meta = json.load(open(mf))
        task = meta["task_description"]
        objects = meta["objects"]
        if isinstance(objects, str):
            objects = eval(objects)  # older metadata stores dict repr
        # Same position-sanity filter as the live filter's _extract_objects:
        # drops obstacle assets parked outside the workspace (|x|,|y| >= 1.5).
        candidate_keys = [
            k for k, v in objects.items()
            if v["position"][2] > -0.1
            and abs(v["position"][0]) < 1.5 and abs(v["position"][1]) < 1.5
        ]
        gt = meta["obstacle"]["name"]
        cleaned_map = _build_clean_map(candidate_keys)
        clean_candidates = list(cleaned_map.keys())
        # target/goal heuristics: same parse the filter uses (approximate: None)
        target = goal = None

        images = []
        for name in ["agentview_rgb.png", "eye_in_hand_rgb.png"]:
            p = ep_dir / name
            if p.exists():
                images.append(np.array(Image.open(p).convert("RGB")))

        for vname in results:
            prompt = VARIANTS[vname](task, clean_candidates, target, goal)
            picks, hows = [], []
            for _ in range(args.n_votes):
                try:
                    raw = client.infer(prompt, images, max_tokens=384)
                except Exception as e:
                    raw = f"<error: {e}>"
                key, how = parse_answer(raw, cleaned_map, clean_candidates)
                picks.append(key)
                hows.append(how)
            final = majority(picks)
            results[vname].append({
                "episode": str(ep_dir).split("safelibero_spatial/")[-1],
                "gt": gt, "picked": final, "votes": picks, "parse": hows,
                "correct": final == gt,
            })
            print(f"[{mi+1}/{len(metas)}] {vname}: gt={gt} picked={final} "
                  f"{'OK' if final == gt else 'X'} (parse={hows})")

    summary = {"model": args.model_tag, "n_votes": args.n_votes, "variants": {}}
    for vname, rows in results.items():
        n = len(rows)
        acc = sum(r["correct"] for r in rows) / max(n, 1)
        none_rate = sum(r["picked"] is None for r in rows) / max(n, 1)
        parse_fail = sum(h == "none" for r in rows for h in r["parse"]) / max(n * args.n_votes, 1)
        summary["variants"][vname] = {
            "accuracy": acc, "abstain_rate": none_rate,
            "vote_parse_fail_rate": parse_fail, "episodes": rows,
        }
        print(f"== {vname}: accuracy={acc:.2f} abstain={none_rate:.2f} "
              f"parse_fail={parse_fail:.2f}")

    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
