"""The fixed, human-written safety rule.

ONE rule, written by a human, never learned, never edited at runtime:

    MARKED(x) AND RTE(y)  ->  keep_away(x, y, margin)

`ground()` binds the free variables against the entities actually present in a
scene, using a belief source (memory, oracle, or a memoryless stub). The rule
text is identical across every experimental arm; only the belief source differs.
That is what makes the comparison a test of memory rather than of rule quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from .ledger import ROLE_RTE, ROLE_TOOL

PREDICATE = "MARKED"


@dataclass(frozen=True)
class KeepAway:
    """A grounded constraint instance handed to the action filter."""
    tool: str
    target: str
    margin: float
    provenance: str     # which belief source activated it

    def as_dict(self) -> dict:
        return {"type": "keep_away", "tool": self.tool, "target": self.target,
                "margin": self.margin, "provenance": self.provenance}


# A belief source answers exactly one question: is this instance marked?
BeliefSource = Callable[[str], bool]


def ground(roles: Dict[str, str], believes_marked: BeliefSource,
           margin: float, provenance: str) -> List[KeepAway]:
    """Instantiate the fixed rule against the current scene.

    Returns one KeepAway per (marked tool, RTE target) pair. An empty list is a
    legal, meaningful result: it means the antecedent is not believed to hold.
    """
    tools = [o for o, r in roles.items() if r == ROLE_TOOL]
    targets = [o for o, r in roles.items() if r == ROLE_RTE]
    out: List[KeepAway] = []
    for tool in sorted(tools):
        if not believes_marked(tool):
            continue
        for target in sorted(targets):
            out.append(KeepAway(tool=tool, target=target, margin=margin,
                                provenance=provenance))
    return out
