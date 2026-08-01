"""CPU unit tests for `--log_transitions` (Task 1, consequence-model data
collection, spec 2026-08-01).

`_transition_record` is a pure helper duplicated verbatim in both eval
clients (`run_guided_libero_safety_eval.py`, `run_guided_safelibero_pi05_eval.py`
— same pattern as `object_positions`/`obstacle_radius` elsewhere in this
codebase). Functional tests (schema/values, JSON round-trip) run against the
LS runner, whose module-level imports are all light enough for a CPU test
env (heavy imports — libero, openpi_client — live inside main()/run_eval()).
The SafeLIBERO runner imports `openpi_client` at MODULE level, which pulls in
`websockets` — not guaranteed present in this env — so its coverage here is
source-level only (no import): verbatim helper text + flag wiring, exactly
per the task brief's "source-level test that both clients expose the flag."
"""

import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))

import run_guided_libero_safety_eval as ls  # noqa: E402

SL_SRC = (_HERE.parents[1] / "run_guided_safelibero_pi05_eval.py").read_text()
LS_SRC = pathlib.Path(ls.__file__).read_text()


# --- _transition_record: schema/values -------------------------------------

def test_record_schema_and_values():
    entities = {"moka_pot_obstacle": np.array([0.1, 0.2, 0.81]),
                "plate": [0.3, -0.1, 0.82]}
    rec = ls._transition_record(
        task=3, ep=7, t=42,
        eef=np.array([0.05, 0.0, 0.95]),
        grip=np.array([0.02, 0.02]),
        action=np.array([0.01, -0.02, 0.0, 0.0, 0.0, 0.0, -1.0]),
        entities=entities,
    )
    assert rec["task"] == 3 and isinstance(rec["task"], int)
    assert rec["ep"] == 7 and isinstance(rec["ep"], int)
    assert rec["t"] == 42 and isinstance(rec["t"], int)
    assert rec["eef"] == [0.05, 0.0, 0.95]
    assert rec["grip"] == [0.02, 0.02]
    assert rec["action"] == [0.01, -0.02, 0.0, 0.0, 0.0, 0.0, -1.0]
    assert rec["entities"] == {
        "moka_pot_obstacle": [0.1, 0.2, 0.81],
        "plate": [0.3, -0.1, 0.82],
    }
    assert all(isinstance(v, list) for v in rec["entities"].values())
    assert all(isinstance(x, float) for v in rec["entities"].values() for x in v)


def test_record_empty_entities():
    rec = ls._transition_record(
        task=0, ep=0, t=0, eef=[0, 0, 0], grip=[0, 0],
        action=[0] * 7, entities={})
    assert rec["entities"] == {}


def test_record_all_fields_plain_python_types():
    # Guards against leaking numpy scalars/arrays into the record — the
    # caller writes this straight to JSONL with json.dumps.
    rec = ls._transition_record(
        task=np.int64(1), ep=np.int64(2), t=np.int64(3),
        eef=np.array([1.0, 2.0, 3.0]), grip=np.array([0.01, 0.02]),
        action=np.arange(7, dtype=np.float32),
        entities={"x": np.array([1.0, 2.0, 3.0])})
    assert type(rec["task"]) is int
    assert type(rec["ep"]) is int
    assert type(rec["t"]) is int
    assert all(type(x) is float for x in rec["eef"])
    assert all(type(x) is float for x in rec["grip"])
    assert all(type(x) is float for x in rec["action"])
    assert all(type(x) is float for x in rec["entities"]["x"])


# --- JSON round-trip ---------------------------------------------------------

def test_record_json_round_trip():
    rec = ls._transition_record(
        task=1, ep=2, t=3,
        eef=[0.1, 0.2, 0.3], grip=[0.01, 0.03],
        action=[0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 1.0],
        entities={"hand_obstacle": [0.4, 0.5, 0.6]})
    line = json.dumps(rec)
    back = json.loads(line)
    assert back == rec


# --- defaults-off no-op ------------------------------------------------------

def test_argparse_log_transitions_default_off():
    # Real source-level check (not a hand-set local): the flag must default
    # to False in argparse for BOTH clients — byte-identical legacy behavior
    # (payloads/results.json untouched) whenever it is left off.
    assert '"--log_transitions", action="store_true"' in LS_SRC
    assert '"--log_transitions", action="store_true"' in SL_SRC


def test_ls_record_building_gated_behind_flag():
    # The entity dict / _transition_record call site must be conditioned on
    # `args.log_transitions` so nothing is built (zero overhead) and no file
    # is opened when the flag is off. The gate appears TWICE: once in the
    # argparse `help=` prose, once as the actual wiring inside the env loop.
    assert "if args.log_transitions:" in LS_SRC
    gate_idx = LS_SRC.rindex("if args.log_transitions:")
    block = LS_SRC[gate_idx:gate_idx + 1200]
    assert "_transition_record(" in block
    assert "transitions_fh = open(" in block


def test_sl_record_building_gated_behind_flag():
    # Anchor on the env.step() call site (unique in the loop body) rather
    # than rindex-ing "if args.log_transitions:" — a second unrelated gate
    # added later anywhere below the loop must not silently retarget this
    # check.
    step_idx = SL_SRC.index('env.step(action.tolist())')
    block = SL_SRC[step_idx:step_idx + 3400]
    assert "if args.log_transitions:" in block
    assert "_transition_record(" in block
    assert "transitions_fh = open(" in block


def test_ls_results_metadata_records_flag():
    assert '"log_transitions": bool(args.log_transitions)' in LS_SRC


def test_sl_results_metadata_records_flag():
    assert '"log_transitions": bool(args.log_transitions)' in SL_SRC


# --- source-level: both clients expose the flag + helper --------------------

def test_both_clients_define_transition_record_helper():
    assert "def _transition_record(task, ep, t, eef, grip, action, entities):" in LS_SRC
    assert "def _transition_record(task, ep, t, eef, grip, action, entities):" in SL_SRC


def test_both_clients_expose_log_transitions_flag():
    for src in (LS_SRC, SL_SRC):
        idx = src.index('"--log_transitions", action="store_true"')
        assert 'action="store_true"' in src[idx:idx + 200]


def test_ls_filename_scheme_matches_spec():
    # <results_output_dir>/transitions_<suite-or-task>_<level>.jsonl
    assert 'f"transitions_{args.suite}_L{args.level}.jsonl"' in LS_SRC


def test_sl_filename_scheme_matches_spec():
    assert 'f"transitions_{args.task_suite_name}_{args.safety_level}.jsonl"' in SL_SRC


def test_sl_write_ordered_after_collision_check_before_done_break():
    # The write must be the LAST thing in the iteration that can still
    # raise, so a mid-iteration exception -> `continue` retry never re-logs
    # the same `t` and duplicates a (task, ep, t) key. Concretely: the write
    # site must come AFTER the collision-flag update and BEFORE `if done:`.
    collision_idx = SL_SRC.index('collide_flag = True')
    write_idx = SL_SRC.index('transitions_fh.write(json.dumps(_trec)')
    done_break_idx = SL_SRC.index('if done:\n                        break')
    assert collision_idx < write_idx < done_break_idx
