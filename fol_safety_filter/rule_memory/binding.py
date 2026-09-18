"""
FOL Safety Filter — role binding: records to `FOLRule` instances.

A record is scene-independent; a `FOLRule` is not. `resolve` lifts one record
over the current scene by enumerating its `roles.subject` and `roles.target`
bindings and emitting the cross product — one record over three obstacles yields
three rules with `bindings={"TARGET": obs_i}`. That lifting is what makes a
record authored once reusable in a scene it was never written for.

An empty enumeration (no obstacles, nothing grasped, no object carrying the
required fact) yields zero rules and is **not** an error: a rule with nothing to
apply to is vacuous, not broken.

Ordering is semantics, not style, because the downstream dispatcher mutates the
action in place. Each emitted rule therefore carries `target_index`,
`subject_rank` and `checkpoint_index`, and the dispatcher sorts by
`(target_index, subject_rank, checkpoint_index)` to reproduce the
obstacle-major nesting of the pre-decomposition loop. Rules are emitted in that
order already, but the attributes are what the dispatcher is required to sort
on — emission order is a convenience, not a contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..kb import FOLRule, parse_formula
from .schema import TARGET_FACT_PREFIX, InvalidRuleError, RuleRecord

# `subject_rank` values. The design fixes 0 for `eef` and 1 for
# `arm_checkpoints` so the EEF projection runs before the arm checkpoints for a
# given obstacle. `currently_grasped` is also 0: it is never ordered against the
# other two because it drives the speed and angular-rate channels, which resolve
# strictest-wins and so are order-independent.
SUBJECT_RANKS = {"eef": 0, "arm_checkpoints": 1, "currently_grasped": 0}


def _no_facts(obj_name: str, fact: str) -> Optional[bool]:
    """Default fact oracle: every fact is unknown."""
    return None


@dataclass
class SceneBindings:
    """Everything `resolve` needs to know about the current episode's scene."""

    obstacle_names: List[str] = field(default_factory=list)
    # Each entry has keys pos / warning_r / hard_r / push / name.
    arm_checkpoints: List[dict] = field(default_factory=list)
    grasped: Optional[str] = None
    # (object_name, FACT) -> True / False / None, where None means unknown.
    facts: Callable[[str, str], Optional[bool]] = _no_facts
    # Candidate universe for `objects_with_fact:` targets. The design's role
    # table says "each object whose fact F is true" without saying which objects
    # are candidates, so this makes it explicit; when None it falls back to the
    # obstacles plus whatever is grasped.
    object_names: Optional[List[str]] = None


# A subject instance: (symbol, label, subject_rank, checkpoint_index, checkpoint)
_Subject = Tuple[str, str, int, int, Optional[dict]]


def _enumerate_subjects(record: RuleRecord, scene: SceneBindings) -> List[_Subject]:
    binding = record.subject_binding
    rank = SUBJECT_RANKS[binding]

    if binding == "eef":
        return [("eef", "eef", rank, 0, None)]

    if binding == "arm_checkpoints":
        subjects: List[_Subject] = []
        for i, checkpoint in enumerate(scene.arm_checkpoints):
            name = str(checkpoint.get("name", f"arm_checkpoint_{i}"))
            subjects.append((name, name, rank, i, checkpoint))
        return subjects

    if binding == "currently_grasped":
        if scene.grasped is None:
            return []
        return [(scene.grasped, "currently_grasped", rank, 0, None)]

    raise InvalidRuleError(
        f"{record.source_path}: roles.subject.binding: unknown binding {binding!r}"
    )


def _fact_candidates(scene: SceneBindings) -> List[str]:
    if scene.object_names is not None:
        return list(scene.object_names)
    candidates = list(scene.obstacle_names)
    if scene.grasped is not None and scene.grasped not in candidates:
        candidates.append(scene.grasped)
    return candidates


def _enumerate_targets(record: RuleRecord,
                       scene: SceneBindings) -> Tuple[List[Optional[str]], List[str]]:
    """Return ``(target_names, unknown_fact_objects)``.

    A subject-only record yields the single target `None`, so the cross product
    below degenerates to one rule per subject.
    """
    binding = record.target_binding
    if binding is None:
        return [None], []

    if binding == "obstacles":
        # Given order, not sorted: the pre-decomposition loop iterated
        # `_obstacle_names` as grounded, and `target_index` must match it.
        return list(scene.obstacle_names), []

    if binding.startswith(TARGET_FACT_PREFIX):
        fact = binding[len(TARGET_FACT_PREFIX):]
        selected: List[str] = []
        unknown: List[str] = []
        for name in _fact_candidates(scene):
            value = scene.facts(name, fact)
            if value is None:
                unknown.append(name)
            elif value:
                selected.append(name)
        return sorted(selected), sorted(unknown)

    raise InvalidRuleError(
        f"{record.source_path}: roles.target.binding: unknown binding {binding!r}"
    )


def resolve_with_unknowns(
    record: RuleRecord, scene: SceneBindings
) -> Tuple[List[FOLRule], List[str]]:
    """Lift ``record`` over ``scene``, also reporting unknown facts.

    The second element lists the objects whose `objects_with_fact` lookup
    returned None. Those objects are **not** bound to any rule — an unknown fact
    is not a false one, and what to do about it is the record's
    `unknown_action`, decided by the caller rather than silently here.
    """
    subjects = _enumerate_subjects(record, scene)
    targets, unknown = _enumerate_targets(record, scene)
    if not subjects or not targets:
        return [], unknown

    effect = record.requires
    rules: List[FOLRule] = []
    for target_index, target_name in enumerate(targets):
        for symbol, label, rank, checkpoint_index, checkpoint in subjects:
            bindings: Dict[str, str] = {"SUBJECT": symbol}
            if target_name is not None:
                bindings["TARGET"] = target_name

            rule = FOLRule(
                name=f"{record.id}::{label}::{target_name or '-'}",
                formula=record.applies_when,
                parsed=parse_formula(record.applies_when),
                bindings=bindings,
                cbf_type=effect.predicate,
                cbf_params=dict(effect.params),
                violation_action=record.violation_action,
                primary_object=target_name,
                description=record.description,
                level=1,
            )
            # `FOLRule` is a plain (non-frozen, non-slotted) dataclass, so the
            # ordering metadata the dispatcher needs is attached directly.
            rule.record_id = record.id
            rule.target_index = target_index
            rule.subject_rank = rank
            rule.checkpoint_index = checkpoint_index
            if checkpoint is not None:
                rule.checkpoint = checkpoint
            rules.append(rule)

    return rules, unknown


def resolve(record: RuleRecord, scene: SceneBindings) -> List[FOLRule]:
    """Lift ``record`` over ``scene``; see `resolve_with_unknowns`."""
    return resolve_with_unknowns(record, scene)[0]
