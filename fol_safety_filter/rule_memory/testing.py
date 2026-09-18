"""Test scaffolding for code that drives the filter below `certify`.

Several tests build a `FOLSafetyFilter` with `__new__` and poke `_apply_cbf`
directly. That worked while the geometry was hardcoded in `_apply_cbf`. It no
longer does: corrections now come from the rule records that fired, so a
filter whose knowledge base is empty produces no corrections at all.

That is the decomposition working, not a regression — but it makes installing
the memory a mandatory setup step for those tests. `install_memory` is that
step, routed through the same `rule_memory.install_rules` entry point the
filter and the parity harness use, so a test cannot end up with a different
notion of "the rules in force" than production.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from fol_safety_filter.kb import FOLKnowledgeBase
from fol_safety_filter.rule_memory import (
    SceneBindings,
    install_rules,
    register_checkpoint_pseudo_objects,
)


def install_memory(
    f,
    obstacles: Sequence[str],
    arm_checkpoints: Sequence[Dict] = (),
    grasped: Optional[str] = None,
    facts: Optional[Callable[[str, str], Optional[bool]]] = None,
    enabled_ids: Optional[Sequence[str]] = None,
):
    """Give `f` a knowledge base populated from the real rule store.

    `enabled_ids=None` uses the manifest's defaults (the two spatial records).
    Pass an explicit list to test a single record in isolation, or to test that
    removing one removes exactly one behaviour.
    """
    if not hasattr(f, "kb") or f.kb is None:
        f.kb = FOLKnowledgeBase()
    scene = SceneBindings(
        obstacle_names=list(obstacles),
        arm_checkpoints=list(arm_checkpoints),
        grasped=grasped,
        facts=facts or (lambda _obj, _fact: None),
        object_names=list(obstacles),
    )
    return install_rules(f.kb, scene, enabled_ids=enabled_ids)


def register_checkpoints(state, checkpoints: Sequence[Dict]) -> None:
    """Make arm checkpoints visible to the predicate layer (see binding.py)."""
    register_checkpoint_pseudo_objects(state, list(checkpoints))
