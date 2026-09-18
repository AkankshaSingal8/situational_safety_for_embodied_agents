"""
FOL Safety Filter — reusable rule memory.

Safety behaviour authored as data instead of Python constants: each record is a
`SKILL.md` carrying one activation condition and one effect, validated strictly
(`schema.py`), loaded as an immutable snapshot (`store.py`), and lifted over the
current scene into `kb.FOLRule` instances (`binding.py`). Deleting a record is
meant to remove exactly one behaviour, which is only true if the geometry it
names is read from the record rather than hardcoded downstream.
"""

from __future__ import annotations

from .binding import (
    SceneBindings,
    register_checkpoint_pseudo_objects,
    resolve,
    resolve_with_unknowns,
)
from .schema import (
    EFFECT_PREDICATES,
    MAX_RECORD_BYTES,
    SCHEMA_VERSION,
    EffectSpec,
    InvalidRuleError,
    RuleRecord,
    load_record,
    parse_skill_md,
    validate,
)
from .store import RuleLibrary, install_rules, load_library

__all__ = [
    "EFFECT_PREDICATES",
    "EffectSpec",
    "InvalidRuleError",
    "MAX_RECORD_BYTES",
    "RuleLibrary",
    "RuleRecord",
    "SCHEMA_VERSION",
    "SceneBindings",
    "install_rules",
    "load_library",
    "load_record",
    "parse_skill_md",
    "register_checkpoint_pseudo_objects",
    "resolve",
    "resolve_with_unknowns",
    "validate",
]
