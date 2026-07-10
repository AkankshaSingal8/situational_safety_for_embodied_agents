"""
FOL Safety Filter — Knowledge Base

Holds Level 1 primitives, Level 2 composed rules, and Level 3 novel predicates.
Evaluates the active constraint set at 20 Hz from a RobotState snapshot.

Formula mini-language (case-insensitive):
    NEAR(eef, obj, 0.15)
    IS_LIT(candle) AND IS_FLAMMABLE(napkin)
    IS_SPILLABLE(cup) AND HOLDING(eef, cup)
    NOT NEAR_EDGE(eef)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .primitives import PREDICATE_REGISTRY, RobotState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AST nodes for the formula mini-language
# ---------------------------------------------------------------------------

class _Expr:
    def eval(self, state: RobotState, bindings: Dict[str, str]) -> bool:
        raise NotImplementedError


@dataclass
class _Atom(_Expr):
    name: str
    args: List[Any]  # mix of str (object names) and float (literal params)

    def eval(self, state: RobotState, bindings: Dict[str, str]) -> bool:
        fn = PREDICATE_REGISTRY.get(self.name.upper())
        if fn is None:
            logger.warning(f"Unknown predicate: {self.name}")
            return False
        # Resolve variable bindings (e.g. A → "candle")
        resolved = []
        for a in self.args:
            if isinstance(a, str) and a in bindings:
                resolved.append(bindings[a])
            else:
                resolved.append(a)
        try:
            return bool(fn(state, *resolved))
        except Exception as e:
            logger.debug(f"Predicate {self.name}({resolved}): {e}")
            return False


@dataclass
class _Not(_Expr):
    child: _Expr

    def eval(self, state: RobotState, bindings: Dict[str, str]) -> bool:
        return not self.child.eval(state, bindings)


@dataclass
class _And(_Expr):
    children: List[_Expr]

    def eval(self, state: RobotState, bindings: Dict[str, str]) -> bool:
        return all(c.eval(state, bindings) for c in self.children)


@dataclass
class _Or(_Expr):
    children: List[_Expr]

    def eval(self, state: RobotState, bindings: Dict[str, str]) -> bool:
        return any(c.eval(state, bindings) for c in self.children)


# ---------------------------------------------------------------------------
# Simple recursive-descent parser
# ---------------------------------------------------------------------------

class _Parser:
    """Parse FOL formula strings into _Expr ASTs."""

    def __init__(self, text: str):
        # Tokenize: split on whitespace/parens/commas, keep tokens
        self._tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]*\.?[0-9]+|\(|\)|,", text)
        self._pos = 0

    def _peek(self) -> Optional[str]:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self, expected: Optional[str] = None) -> str:
        tok = self._tokens[self._pos]
        if expected and tok.upper() != expected.upper():
            raise ValueError(f"Expected {expected!r}, got {tok!r}")
        self._pos += 1
        return tok

    def parse(self) -> _Expr:
        expr = self._parse_or()
        if self._pos < len(self._tokens):
            raise ValueError(f"Unexpected token: {self._peek()!r}")
        return expr

    def _parse_or(self) -> _Expr:
        left = self._parse_and()
        children = [left]
        while self._peek() and self._peek().upper() == "OR":
            self._consume("OR")
            children.append(self._parse_and())
        return _Or(children) if len(children) > 1 else left

    def _parse_and(self) -> _Expr:
        left = self._parse_not()
        children = [left]
        while self._peek() and self._peek().upper() == "AND":
            self._consume("AND")
            children.append(self._parse_not())
        return _And(children) if len(children) > 1 else left

    def _parse_not(self) -> _Expr:
        if self._peek() and self._peek().upper() == "NOT":
            self._consume("NOT")
            return _Not(self._parse_atom())
        return self._parse_atom()

    def _parse_atom(self) -> _Expr:
        if self._peek() == "(":
            self._consume("(")
            expr = self._parse_or()
            self._consume(")")
            return expr
        name = self._consume()
        args: List[Any] = []
        if self._peek() == "(":
            self._consume("(")
            while self._peek() and self._peek() != ")":
                tok = self._peek()
                if tok == ",":
                    self._consume(",")
                    continue
                # Consume once, then try numeric conversion
                self._consume()
                try:
                    args.append(float(tok))
                except ValueError:
                    args.append(tok)
            self._consume(")")
        return _Atom(name, args)


def parse_formula(formula: str) -> _Expr:
    return _Parser(formula).parse()


# ---------------------------------------------------------------------------
# Rule and knowledge base data structures
# ---------------------------------------------------------------------------

@dataclass
class FOLRule:
    name: str
    formula: str
    parsed: _Expr
    # Variable bindings e.g. {"A": "candle", "B": "napkin"}
    bindings: Dict[str, str]
    # CBF to build when rule is active
    cbf_type: str                  # "spatial" | "velocity" | "rotation" | "boolean" | "precondition"
    cbf_params: Dict[str, Any]
    violation_action: str          # "block" | "slow" | "avoid" | "ask"
    # Which objects are involved (for CBF construction)
    primary_object: Optional[str] = None
    description: str = ""
    level: int = 2                 # 1=primitive, 2=composed, 3=novel


@dataclass
class ActiveConstraint:
    """A rule that evaluated to True at the current timestep."""
    rule: FOLRule
    primary_object: Optional[str]
    obj_pos: Optional[np.ndarray]  # 3D position for spatial CBFs
    obj_quat: Optional[np.ndarray]
    obj_bbox: Optional[np.ndarray] # half-extents


@dataclass
class NovelPredicate:
    name: str
    grounding_query: str           # "Is {obj} X?"
    cbf_type: str
    effects: Dict[str, Any]
    cached_objects: Dict[str, bool] = field(default_factory=dict)


class FOLKnowledgeBase:
    """
    Holds all rules and evaluates the active constraint set each timestep.

    Usage:
        kb = FOLKnowledgeBase()
        kb.add_rules_from_primitives(["moka_pot_obstacle", "plate"])
        kb.add_composed_rules(vlm_rules_json)
        active = kb.evaluate(state)
    """

    def __init__(self, rules: Optional[List["FOLRule"]] = None):
        self.rules: List[FOLRule] = list(rules) if rules else []
        self.novel_predicates: Dict[str, NovelPredicate] = {}
        self._eval_count = 0

    def clear(self):
        self.rules.clear()
        self.novel_predicates.clear()
        self._eval_count = 0

    # ------------------------------------------------------------------ #
    # Rule population
    # ------------------------------------------------------------------ #

    def add_rule(self, rule: FOLRule):
        self.rules.append(rule)

    def add_rules_for_objects(self, object_names: List[str],
                               ee_name: str = "eef"):
        """Add Level-1 collision-avoidance rules for each detected object."""
        from .primitives import PREDICATE_CBF_TEMPLATES

        for obj in object_names:
            # Basic spatial avoidance: robot shouldn't crash into any object
            formula = f"NEAR({ee_name}, {obj}, 0.12)"
            try:
                parsed = parse_formula(formula)
            except Exception:
                continue

            tpl = PREDICATE_CBF_TEMPLATES.get("NEAR")
            rule = FOLRule(
                name=f"AVOID_{obj.upper()}",
                formula=formula,
                parsed=parsed,
                bindings={},
                cbf_type=tpl.cbf_type if tpl else "spatial",
                cbf_params=tpl.params.copy() if tpl else {},
                violation_action=tpl.violation_action if tpl else "avoid",
                primary_object=obj,
                description=f"Keep EEF away from {obj}",
                level=1,
            )
            self.rules.append(rule)

    def add_composed_rules(self, rules_json: List[Dict]):
        """
        Add Level-2 rules from VLM output JSON.
        Each dict must have: name, formula, bindings, cbf_type, cbf_params,
                             violation_action, primary_object.
        """
        for r in rules_json:
            try:
                parsed = parse_formula(r["formula"])
            except Exception as e:
                logger.warning(f"Could not parse rule {r.get('name')}: {e}")
                continue
            rule = FOLRule(
                name=r["name"],
                formula=r["formula"],
                parsed=parsed,
                bindings=r.get("bindings", {}),
                cbf_type=r.get("cbf_type", "spatial"),
                cbf_params=r.get("cbf_params", {}),
                violation_action=r.get("violation_action", "avoid"),
                primary_object=r.get("primary_object"),
                description=r.get("description", ""),
                level=r.get("level", 2),
            )
            self.rules.append(rule)
            logger.info(f"[KB] Added rule: {rule.name}: {rule.formula}")

    def add_workspace_safety_rules(self):
        """Add Level-1 workspace boundary rules.

        IN_WORKSPACE() and NEAR_EDGE() have no positional object args —
        they always operate on the EEF position from state.ee_pos.
        Omit 'eef' to avoid the arg being misinterpreted as a margin value.
        """
        templates = [
            ("WORKSPACE_BOUNDS", "NOT IN_WORKSPACE()", "boolean",
             {"block_axes": ["x", "y", "z"]}, "block"),
            ("NEAR_EDGE_GUARD", "NEAR_EDGE()", "boolean",
             {"block_axes": ["x", "y"]}, "slow"),
        ]
        for name, formula, cbf_type, params, action in templates:
            try:
                parsed = parse_formula(formula)
            except Exception:
                continue
            self.rules.append(FOLRule(
                name=name,
                formula=formula,
                parsed=parsed,
                bindings={},
                cbf_type=cbf_type,
                cbf_params=params,
                violation_action=action,
                level=1,
            ))

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(self, state: RobotState) -> List[ActiveConstraint]:
        """
        Evaluate all rules against current state.
        Returns list of ActiveConstraint for rules that fire.
        Called at ~20 Hz.
        """
        self._eval_count += 1
        active: List[ActiveConstraint] = []

        for rule in self.rules:
            try:
                fires = rule.parsed.eval(state, rule.bindings)
            except Exception as e:
                logger.debug(f"Rule {rule.name} eval error: {e}")
                fires = False

            if fires:
                # Collect object state for CBF construction
                obj_pos = obj_quat = obj_bbox = None
                if rule.primary_object and rule.primary_object in state.objects:
                    o = state.objects[rule.primary_object]
                    obj_pos = o.pos.copy()
                    obj_quat = o.quat.copy()
                    obj_bbox = o.bbox_half.copy()

                active.append(ActiveConstraint(
                    rule=rule,
                    primary_object=rule.primary_object,
                    obj_pos=obj_pos,
                    obj_quat=obj_quat,
                    obj_bbox=obj_bbox,
                ))

        if self._eval_count % 100 == 0 or (active and self._eval_count % 10 == 0):
            names = [c.rule.name for c in active]
            logger.debug(f"[KB t={self._eval_count}] active={names or 'none'}")

        return active

    def summary(self) -> str:
        lines = [f"FOLKnowledgeBase: {len(self.rules)} rules"]
        for r in self.rules:
            lines.append(f"  [{r.level}] {r.name}: {r.formula}")
        return "\n".join(lines)
