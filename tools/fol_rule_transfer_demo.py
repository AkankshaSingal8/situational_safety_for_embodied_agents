#!/usr/bin/env python
"""Do the rule records transfer to scenes they were not written for?

The records in ``fol_safety_filter/safety_memory/`` name no object and no task.
This script tests whether that buys anything: it loads the library **once**,
prints each record's content hash, and then evaluates activation across all 11
payload scenes of ``prompt_tuning_benchmark_set/`` without editing a single
record between scenes. The same hash at the first scene and the last is the
claim — the rules were not quietly re-authored per scene.

Scored against the benchmark's own annotations, so "it generalizes" is a
number rather than an assertion.

    python tools/fol_rule_transfer_demo.py \
        --out results/safelibero/fol/rule_memory

What this measures, exactly: **activation correctness on oracle facts.** The
scene facts come from a hand-declared fixture, not from perception, so this
says nothing about whether a VLM could supply them — that needs a separate
experiment against ``vlm_grounder.py``. It is not a task-success or
violation-rate result, it shares no table with TSR or CAR, and per the repo's
``CLAUDE.md`` it shares nothing with LIBERO-Safety SR either.

CPU only. No simulator, no GPU, no API calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from fol_safety_filter.primitives import ObjectState, RobotState  # noqa: E402
from fol_safety_filter.rule_memory import (  # noqa: E402
    SceneBindings,
    load_library,
    resolve_with_unknowns,
)

MEMORY_ROOT = os.path.join(_REPO_ROOT, "fol_safety_filter", "safety_memory")
FIXTURE = os.path.join(MEMORY_ROOT, "fixtures", "scene_facts.yaml")
BENCHMARK = os.path.join(_REPO_ROOT, "prompt_tuning_benchmark_set")

# Which annotation file is ground truth for which record. A record with no
# entry here is reported as an activation profile only, with no score, because
# inventing a ground truth for it would be scoring ourselves.
GROUND_TRUTH = {
    "spillable_rotation_lock": ("prompt_tuning_rotation.json", "rotation lock"),
}


# --------------------------------------------------------------------------- #
# Scene construction
# --------------------------------------------------------------------------- #

def _load_fixture() -> Dict:
    with open(FIXTURE) as fh:
        return yaml.safe_load(fh)


def _pose_for(record, scene: Dict) -> Tuple[Optional[str], Optional[str], str]:
    """Decide how to pose the robot so `record` is actually exercised.

    A fact-conditioned rule whose formula also has a geometric clause needs
    the robot placed somewhere that clause can be satisfied, or it binds and
    never fires and the demo reports nothing about it. The posing is chosen
    from the record's declared roles and effect — never from its expected
    answer — and is reported per scene so a reader can see what was arranged.

    Returns (payload_above, eef_near, description).
    """
    first_fragile = None
    for obj, obj_facts in (scene.get("object_facts") or {}).items():
        if obj_facts.get("IS_FRAGILE") is True:
            first_fragile = obj
            break

    if record.requires.predicate == "avoid_region_above":
        return first_fragile, None, (
            f"payload placed above {first_fragile}" if first_fragile
            else "neutral (no fragile object in scene)"
        )
    target = record.target_binding or ""
    if record.subject_binding == "eef" and target.startswith("objects_with_fact:"):
        return None, first_fragile, (
            f"end-effector placed beside {first_fragile}" if first_fragile
            else "neutral (no fragile object in scene)"
        )
    return None, None, "neutral"


def _scene_state(scene: Dict, payload_above: Optional[str] = None,
                 eef_near: Optional[str] = None):
    """Build (SceneBindings, RobotState) for one fixture scene.

    The base geometry is identical for every scene: the desk objects sit on a
    ring on the table and the payload is held above it. Only the *facts*
    differ, so a difference in activation **between scenes** is attributable
    to the facts and the rules rather than to the pose.

    `payload_above` / `eef_near` adjust the pose **per record**, identically
    across scenes, because a rule with a geometric clause otherwise binds and
    never fires and the sweep reports nothing about it. The choice comes from
    the record's declared roles (see `_pose_for`), not from the answer we
    want, and the report prints it for every cell.
    """
    objects: Dict[str, ObjectState] = {}
    facts: Dict[str, Dict[str, Optional[bool]]] = {}

    names = sorted(scene.get("object_facts") or {})
    for i, name in enumerate(names):
        angle = 2.0 * np.pi * i / max(len(names), 1)
        objects[name] = ObjectState(
            name=name,
            pos=np.array([0.25 * np.cos(angle), 0.25 * np.sin(angle), 0.85]),
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            bbox_half=np.full(3, 0.04),
        )
        facts[name] = scene["object_facts"][name]

    payload = scene.get("payload")
    ee_pos = np.array([0.0, 0.0, 1.15])
    if eef_near is not None and eef_near in objects:
        # 0.10 m away: inside a 0.15 m radius, outside a 0.08 m hard zone.
        ee_pos = objects[eef_near].pos + np.array([0.10, 0.0, 0.0])
    if payload is not None:
        if payload_above is not None and payload_above in objects:
            ee_pos = objects[payload_above].pos + np.array([0.0, 0.0, 0.30])
        objects[payload] = ObjectState(
            name=payload,
            pos=ee_pos.copy(),
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            bbox_half=np.full(3, 0.04),
        )
        facts[payload] = scene["payload_facts"]

    for name, obj_facts in facts.items():
        for fact, value in obj_facts.items():
            setattr(objects[name], fact.lower(), value)

    def fact_lookup(obj_name: str, fact: str) -> Optional[bool]:
        return facts.get(obj_name, {}).get(fact)

    bindings = SceneBindings(
        obstacle_names=names,
        arm_checkpoints=[],
        grasped=payload,
        facts=fact_lookup,
        object_names=sorted(objects),
    )
    state = RobotState(
        ee_pos=ee_pos,
        ee_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        ee_vel=np.zeros(3),
        joint_pos=np.zeros(7),
        joint_vel=np.zeros(7),
        joint_limits_lo=np.full(7, -2.8),
        joint_limits_hi=np.full(7, 2.8),
        gripper_width=0.0 if payload else 0.08,
        objects=objects,
    )
    return bindings, state


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def _evaluate(library, fixture) -> Dict:
    scenes = fixture["scenes"]
    per_record: Dict[str, Dict] = {}

    for record in library.all_records:
        entry = {
            "revision": record.revision,
            "front_matter_hash": record.front_matter_hash,
            "effect": record.requires.predicate,
            "subject_binding": record.subject_binding,
            "target_binding": record.target_binding,
            "scenes": {},
        }
        for scene_name in sorted(scenes):
            scene = scenes[scene_name]
            above, near, posing = _pose_for(record, scene)
            bindings, state = _scene_state(
                scene, payload_above=above, eef_near=near
            )
            rules, unknowns = resolve_with_unknowns(record, bindings)
            fired = [r for r in rules if r.parsed.eval(state, r.bindings)]
            entry["scenes"][scene_name] = {
                "resolved": len(rules),
                "fired": len(fired),
                "bound_to": sorted({r.primary_object for r in fired
                                    if r.primary_object}),
                "subject": sorted({r.bindings.get("SUBJECT") for r in fired}),
                "unknown_facts": sorted(unknowns),
                "posing": posing,
            }
        per_record[record.id] = entry

    return per_record


def _score(record_id: str, per_scene: Dict) -> Optional[Dict]:
    spec = GROUND_TRUTH.get(record_id)
    if spec is None:
        return None
    filename, needle = spec
    with open(os.path.join(BENCHMARK, filename)) as fh:
        gt_doc = json.load(fh)

    truth = {
        name
        for name, entry in gt_doc.items()
        if any(needle in rel for _obj, rels in entry["objects"] for rel in rels)
    }
    fired = {name for name, res in per_scene.items() if res["fired"] > 0}
    scored = set(gt_doc) & set(per_scene)

    tp = sorted(fired & truth & scored)
    fp = sorted((fired - truth) & scored)
    fn = sorted((truth - fired) & scored)
    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else None
    recall = len(tp) / (len(tp) + len(fn)) if (tp or fn) else None
    return {
        "ground_truth_file": filename,
        "scenes_scored": len(scored),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _markdown(report: Dict) -> str:
    lines: List[str] = []
    lines.append("# Rule transfer: activation correctness on oracle facts")
    lines.append("")
    lines.append(
        "**These are activation-correctness numbers on a hand-declared oracle "
        "fact fixture. They are not task-success or violation-rate results, "
        "they say nothing about perception, and they share no table with "
        "SafeLIBERO TSR/CAR or LIBERO-Safety SR.**"
    )
    lines.append("")
    lines.append(
        f"The library was loaded once and not edited between scenes; "
        f"{report['n_scenes']} scenes, {report['n_records']} records."
    )
    lines.append("")

    lines.append("## Records, loaded once")
    lines.append("")
    lines.append("| record | rev | effect | subject | target | hash |")
    lines.append("|---|---|---|---|---|---|")
    for rid, entry in report["records"].items():
        lines.append(
            f"| `{rid}` | {entry['revision']} | `{entry['effect']}` | "
            f"`{entry['subject_binding']}` | "
            f"`{entry['target_binding'] or '—'}` | "
            f"`{entry['front_matter_hash'][:12]}` |"
        )
    lines.append("")
    lines.append(
        "Identical hashes were printed before the first scene and after the "
        "last, so no record was re-authored per scene."
    )
    lines.append("")

    lines.append("## Activation per scene")
    lines.append("")
    scene_names = report["scene_names"]
    header = "| record | " + " | ".join(scene_names) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(scene_names) + 1))
    for rid, entry in report["records"].items():
        cells = []
        for name in scene_names:
            res = entry["scenes"][name]
            cells.append("●" if res["fired"] else ("·" if res["resolved"] else "–"))
        lines.append(f"| `{rid}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "● fired · bound but correctly silent – nothing to bind to in this scene"
    )
    lines.append("")
    lines.append("### How the robot was posed")
    lines.append("")
    lines.append(
        "A rule with a geometric clause has to be given a pose it can be "
        "satisfied at, or it binds and never fires and the sweep says nothing "
        "about it. The pose is derived from each record's declared roles, not "
        "from its expected answer, and is the same for every scene:"
    )
    lines.append("")
    for rid, entry in report["records"].items():
        posings = sorted({s["posing"].split(" placed ")[0] + " placed …"
                          if " placed " in s["posing"] else s["posing"]
                          for s in entry["scenes"].values()})
        lines.append(f"- `{rid}`: {', '.join(posings)}")
    lines.append("")

    lines.append("## Binding transfer")
    lines.append("")
    lines.append(
        "The same record binding to different objects across scenes, with no "
        "edit, is the property under test:"
    )
    lines.append("")
    for rid, entry in report["records"].items():
        bound = {}
        for name in scene_names:
            for obj in entry["scenes"][name]["bound_to"]:
                bound.setdefault(obj, []).append(name)
        if bound:
            targets = ", ".join(
                f"`{obj}` ({len(scenes)} scene{'s' if len(scenes) != 1 else ''})"
                for obj, scenes in sorted(bound.items())
            )
            lines.append(f"- `{rid}` bound to {targets}")
    lines.append("")

    scored = {k: v for k, v in report["scores"].items() if v}
    if scored:
        lines.append("## Scored against the benchmark annotations")
        lines.append("")
        lines.append("| record | scenes | precision | recall | misses |")
        lines.append("|---|---|---|---|---|")
        for rid, sc in scored.items():
            prec = "—" if sc["precision"] is None else f"{sc['precision']:.2f}"
            rec = "—" if sc["recall"] is None else f"{sc['recall']:.2f}"
            misses = ", ".join(f"`{m}`" for m in sc["false_negatives"]) or "none"
            lines.append(
                f"| `{rid}` | {sc['scenes_scored']} | {prec} | {rec} | {misses} |"
            )
        lines.append("")
        lines.append(
            "Records without a row have no benchmark annotation to score "
            "against; inventing one would be scoring ourselves. Their "
            "activation profile is above."
        )
        lines.append("")

    lines.append("## Manifest toggle")
    lines.append("")
    lines.append(
        f"Default-loaded: {', '.join('`' + r + '`' for r in report['enabled'])}. "
        f"Withheld: {', '.join('`' + r + '`' for r in report['disabled'])}. "
        "Switching a record on is a manifest edit; no filter code changes."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=os.path.join(_REPO_ROOT, "results", "safelibero", "fol",
                             "rule_memory"),
        help="directory for transfer_report.{json,md}",
    )
    args = ap.parse_args(argv)

    fixture = _load_fixture()
    library = load_library(MEMORY_ROOT)

    hashes_before = {r.id: r.front_matter_hash for r in library.all_records}
    per_record = _evaluate(library, fixture)
    # Reloaded from disk after the whole sweep: same bytes, or the "loaded
    # once, never re-authored" claim is false.
    hashes_after = {
        r.id: r.front_matter_hash for r in load_library(MEMORY_ROOT).all_records
    }
    if hashes_before != hashes_after:
        print("ERROR: record hashes changed during the sweep", file=sys.stderr)
        return 2

    scores = {rid: _score(rid, e["scenes"]) for rid, e in per_record.items()}
    report = {
        "n_scenes": len(fixture["scenes"]),
        "n_records": len(per_record),
        "scene_names": sorted(fixture["scenes"]),
        "records": per_record,
        "scores": scores,
        "enabled": [r.id for r in library.records],
        "disabled": [r.id for r in library.all_records
                     if r.id not in {e.id for e in library.records}],
        "fact_source": "oracle (hand-declared fixture), not perception",
        "caveat": (
            "Activation correctness only. Not task success, not violation "
            "rate, not a perception result. Shares no table with SafeLIBERO "
            "TSR/CAR or LIBERO-Safety SR."
        ),
    }

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, "transfer_report.json")
    md_path = os.path.join(args.out, "transfer_report.md")
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(md_path, "w") as fh:
        fh.write(_markdown(report))

    print(f"{report['n_records']} records x {report['n_scenes']} scenes")
    for rid, entry in per_record.items():
        n_fired = sum(1 for s in entry["scenes"].values() if s["fired"])
        sc = scores.get(rid)
        tail = ""
        if sc and sc["recall"] is not None:
            tail = (f"  recall={sc['recall']:.2f} "
                    f"precision={sc['precision']:.2f}"
                    if sc["precision"] is not None
                    else f"  recall={sc['recall']:.2f}")
        print(f"  {rid:26s} fired in {n_fired:2d}/{report['n_scenes']} "
              f"[{entry['front_matter_hash'][:12]}]{tail}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
