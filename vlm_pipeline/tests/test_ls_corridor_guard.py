"""CPU unit tests for the LS runner's corridor-anchor / top-k guard helpers.

No GPU, no env creation, no mujoco/robosuite imports: the helpers under test
(`_corridor_anchor`, `_topk_guard`, `_extend_guidance`) are pure functions in
run_guided_libero_safety_eval.py (heavy imports live inside main()).
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import run_guided_libero_safety_eval as ls  # noqa: E402

EEF = np.array([0.0, 0.0, 1.0])

# Stubbed episode-start candidates (obs `{name}_pos` after workspace filter).
CANDS = {
    "banana_1": np.array([0.10, 0.10, 0.90]),
    "plate_1": np.array([0.20, 0.30, 0.90]),
    "moka_pot_obstacle": np.array([0.05, 0.08, 0.92]),
    "ball_obstacle": np.array([0.30, -0.20, 0.90]),
}


def test_corridor_anchor_picks_task_referenced_object():
    target, dest = ls._corridor_anchor("put the banana on the plate", CANDS)
    assert target == "banana_1"          # sanctioned goal object = anchor
    assert dest == "plate_1"             # latest head-noun = destination


def test_corridor_anchor_missing_entities_fail_safe():
    target, dest = ls._corridor_anchor(
        "press the button", {"moka_pot_obstacle": CANDS["moka_pot_obstacle"]})
    assert target is None                # no corridor granted


def test_gt_source_returns_all_checkrobotcontact_args():
    hazards = ["moka_pot_obstacle", "ball_obstacle"]
    guard = ls._topk_guard("gt", "put the banana on the plate",
                           CANDS, EEF, set(), hazards, k=1)
    assert guard == hazards              # today's behavior: ALL hazards, any k


def test_fol_topk2_two_guards_ordered_by_identity_score():
    from symbolic_identity import fol_obstacle_id
    desc = "put the banana on the plate"
    ranked = fol_obstacle_id(desc, CANDS, EEF, moving=set())
    g1 = ls._topk_guard("fol", desc, CANDS, EEF, set(), [], k=1)
    g2 = ls._topk_guard("fol", desc, CANDS, EEF, set(), [], k=2)
    assert len(g2) == 2
    assert g2 == ranked[:2]              # ordered by the identity score
    assert g1 == g2[:1]                  # k=1 unchanged (today's truncation)
    assert set(g2) == {"moka_pot_obstacle", "ball_obstacle"}


def test_symbolic_topk2_primary_unchanged_runnerup_appended():
    from symbolic_identity import scored_obstacle_id
    desc = "put the banana on the plate"
    picked = scored_obstacle_id(desc, CANDS, EEF, moving=set())
    g1 = ls._topk_guard("symbolic", desc, CANDS, EEF, set(), [], k=1)
    g2 = ls._topk_guard("symbolic", desc, CANDS, EEF, set(), [], k=2)
    assert g1 == [picked]                # k=1 identical to today
    assert g2[0] == picked               # primary never changes
    assert len(g2) == 2 and g2[1] != picked


def _base_payload():
    return {
        "enabled": 1.0,
        "eef_pos": np.asarray(EEF, dtype=np.float32),
        "obstacle_pos": np.asarray(CANDS["moka_pot_obstacle"], dtype=np.float32),
        "obstacle_radius": ls.obstacle_radius("moka_pot_obstacle"),
    }


def test_default_payload_shape_single_guard():
    # Defaults (guard_topk=1, corridor anchors as today): exactly today's keys
    # in today's insertion order — no obstacle2.
    out = ls._extend_guidance(_base_payload(), ["moka_pot_obstacle"],
                              CANDS.get, "banana_1", "plate_1")
    assert list(out) == ["enabled", "eef_pos", "obstacle_pos",
                         "obstacle_radius", "target_pos", "dest_pos"]
    assert np.allclose(out["target_pos"], CANDS["banana_1"])
    assert np.allclose(out["dest_pos"], CANDS["plate_1"])


def test_payload_two_guards_emits_obstacle2():
    out = ls._extend_guidance(_base_payload(),
                              ["moka_pot_obstacle", "ball_obstacle"],
                              CANDS.get, None, None)
    assert list(out) == ["enabled", "eef_pos", "obstacle_pos",
                         "obstacle_radius", "obstacle2"]
    assert set(out["obstacle2"]) == {"pos", "radius"}
    assert np.allclose(out["obstacle2"]["pos"], CANDS["ball_obstacle"])
    assert out["obstacle2"]["radius"] == ls.obstacle_radius("ball_obstacle")


def test_payload_missing_positions_fail_safe():
    # Unlocatable second guard / anchors simply add no keys.
    out = ls._extend_guidance(_base_payload(),
                              ["moka_pot_obstacle", "ghost"], CANDS.get,
                              "ghost_target", None)
    assert list(out) == ["enabled", "eef_pos", "obstacle_pos",
                         "obstacle_radius"]
