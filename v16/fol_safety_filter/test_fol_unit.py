"""
Unit tests for the FOL safety filter (no MuJoCo/LIBERO dependencies).

Run with:
    python -m pytest fol_safety_filter/test_fol_unit.py -v
Or standalone:
    python fol_safety_filter/test_fol_unit.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vlm_pipeline"))

import numpy as np
import math

# ── imports ────────────────────────────────────────────────────────────────────

from fol_safety_filter.primitives import (
    ObjectState, RobotState,
    NEAR, ABOVE, BELOW, SUPPORTS, MOVING_FAST, GRIPPING, HOLDING,
    IS_FRAGILE, IS_WET, IS_SPILLABLE, IS_LIT, IS_FLAMMABLE,
    TILTED, EEF_TILTED, NEAR_EDGE, IN_WORKSPACE,
    PREDICATE_REGISTRY,
)
from fol_safety_filter.kb import FOLKnowledgeBase, parse_formula


# ── helpers ─────────────────────────────────────────────────────────────────────

def make_state(ee_pos=(0.0, 0.0, 0.9), objects=None) -> RobotState:
    return RobotState(
        ee_pos=np.array(ee_pos, dtype=float),
        ee_quat=np.array([0., 0., 0., 1.]),
        ee_vel=np.zeros(3),
        joint_pos=np.zeros(7),
        joint_vel=np.zeros(7),
        joint_limits_lo=np.full(7, -3.0),
        joint_limits_hi=np.full(7, 3.0),
        gripper_width=0.5,
        objects=objects or {},
    )


def make_obj(name, pos=(0.0, 0.0, 0.85), **flags) -> ObjectState:
    obj = ObjectState(
        name=name,
        pos=np.array(pos, dtype=float),
        quat=np.array([0., 0., 0., 1.]),
        bbox_half=np.array([0.04, 0.04, 0.04]),
    )
    for k, v in flags.items():
        setattr(obj, k, v)
    return obj


# ── spatial predicate tests ────────────────────────────────────────────────────

def test_near_true():
    state = make_state(ee_pos=(0.0, 0.0, 0.9),
                       objects={"cup": make_obj("cup", (0.05, 0.0, 0.9))})
    assert NEAR(state, "eef", "cup", 0.15), "EEF is 0.05m from cup — should be NEAR"


def test_near_false():
    state = make_state(ee_pos=(0.0, 0.0, 0.9),
                       objects={"cup": make_obj("cup", (0.30, 0.0, 0.9))})
    assert not NEAR(state, "eef", "cup", 0.15), "EEF is 0.30m from cup — should NOT be NEAR"


def test_above():
    state = make_state(objects={
        "bowl": make_obj("bowl", (0.0, 0.0, 0.90)),
        "table": make_obj("table", (0.0, 0.0, 0.81)),
    })
    assert ABOVE(state, "bowl", "table")
    assert not ABOVE(state, "table", "bowl")


def test_supports():
    state = make_state(objects={
        "plate": make_obj("plate", (0.0, 0.0, 0.83)),
        "bowl":  make_obj("bowl",  (0.0, 0.0, 0.92)),
    })
    assert SUPPORTS(state, "plate", "bowl"), "plate supports bowl"
    assert not SUPPORTS(state, "bowl", "plate"), "bowl does not support plate"


def test_moving_fast_true():
    state = make_state()
    state.ee_vel = np.array([0.0, 0.0, 0.40])  # fast downward
    assert MOVING_FAST(state, 0.25)


def test_moving_fast_false():
    state = make_state()
    state.ee_vel = np.array([0.0, 0.0, 0.10])
    assert not MOVING_FAST(state, 0.25)


def test_in_workspace():
    state = make_state(ee_pos=(0.0, 0.0, 1.0))
    assert IN_WORKSPACE(state), "Centre of workspace should be IN_WORKSPACE"


def test_near_edge():
    state = make_state(ee_pos=(0.49, 0.0, 1.0))  # near x boundary
    assert NEAR_EDGE(state, edge_margin=0.05)


def test_tilted():
    # 45-degree tilt around x-axis
    angle = math.radians(45)
    quat = np.array([math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)])
    state = make_state(objects={"cup": make_obj("cup", quat=quat)})
    state.objects["cup"].quat = quat
    assert TILTED(state, "cup", threshold_deg=20.0)


# ── physical-state predicate tests ───────────────────────────────────────────────

def test_is_fragile():
    state = make_state(objects={"egg": make_obj("egg", is_fragile=True)})
    assert IS_FRAGILE(state, "egg")


def test_is_spillable():
    obj = make_obj("cup", is_full=True, is_open=True)
    obj.is_spillable = True
    state = make_state(objects={"cup": obj})
    assert IS_SPILLABLE(state, "cup")


def test_is_spillable_not_open():
    obj = make_obj("thermos", is_full=True, is_open=False)
    obj.is_spillable = False
    state = make_state(objects={"thermos": obj})
    assert not IS_SPILLABLE(state, "thermos")


# ── formula parser tests ────────────────────────────────────────────────────────

def test_parse_simple():
    f = parse_formula("NEAR(eef, cup, 0.15)")
    assert f is not None


def test_parse_and():
    f = parse_formula("IS_LIT(candle) AND IS_FLAMMABLE(napkin)")
    assert f is not None


def test_parse_not():
    f = parse_formula("NOT IN_WORKSPACE(eef)")
    assert f is not None


def test_parse_complex():
    f = parse_formula("IS_SPILLABLE(cup) AND HOLDING(eef, cup) AND MOVING_FAST(eef, 0.1)")
    assert f is not None


# ── KB evaluation tests ────────────────────────────────────────────────────────

def test_kb_near_rule_fires():
    kb = FOLKnowledgeBase()
    kb.add_rules_for_objects(["obstacle"])
    # EEF very close to obstacle
    state = make_state(ee_pos=(0.0, 0.0, 0.9),
                       objects={"obstacle": make_obj("obstacle", (0.05, 0.0, 0.9))})
    active = kb.evaluate(state)
    names = [a.rule.name for a in active]
    assert any("OBSTACLE" in n for n in names), f"Expected AVOID_OBSTACLE, got {names}"


def test_kb_near_rule_does_not_fire_far():
    kb = FOLKnowledgeBase()
    kb.add_rules_for_objects(["obstacle"])
    state = make_state(ee_pos=(0.0, 0.0, 0.9),
                       objects={"obstacle": make_obj("obstacle", (0.5, 0.0, 0.9))})
    active = kb.evaluate(state)
    names = [a.rule.name for a in active]
    assert not any("OBSTACLE" in n for n in names), "Should not fire when far"


def test_predicate_registry_complete():
    required = [
        "NEAR", "ABOVE", "BELOW", "SUPPORTS", "IN_WORKSPACE",
        "NEAR_EDGE", "MOVING_FAST", "GRIPPING", "HOLDING",
        "IS_FRAGILE", "IS_WET", "IS_LIT", "IS_FULL", "IS_OPEN",
        "IS_FLAMMABLE", "IS_ABSORBENT", "IS_HUMAN", "HAS_SHARP_PART",
        "IS_HEAVY", "IS_SPILLABLE",
    ]
    for name in required:
        assert name in PREDICATE_REGISTRY, f"Missing predicate: {name}"


# ── runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed")
    if failed:
        sys.exit(1)
