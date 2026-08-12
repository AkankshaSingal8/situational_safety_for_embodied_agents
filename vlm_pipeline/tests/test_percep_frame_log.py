"""CPU unit tests for `--log_percep_frames` (Phase 3 hazard-localizer
training-data logging): perception frames + GT entity positions captured at
every replan boundary.

Covers `_build_frame_record`, `_percep_frame_path` in
run_guided_libero_safety_eval.py — pure, no GPU, no env creation, no
mujoco/robosuite imports. Modeled on test_fol_predicates.py /
test_yield_regime.py.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import run_guided_libero_safety_eval as ls  # noqa: E402


def _mk_obs(**overrides):
    obs = {
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "agentview_depth": np.ones((256, 256), dtype=np.float32),
        "robot0_eye_in_hand_image": np.ones((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_depth": np.full((256, 256), 2.0, dtype=np.float32),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.9]),
        "moka_pot_obstacle_pos": np.array([0.3, -0.1, 0.85]),
        "plate_pos": np.array([0.0, 0.1, 0.81]),
        "robot0_gripper_qpos": np.array([0.02, 0.02]),  # not a *_pos entity
    }
    obs.update(overrides)
    return obs


_CAMS = {
    "agentview": {"K": np.eye(3), "T": np.eye(4)},
    "robot0_eye_in_hand": {"K": np.eye(3) * 2, "T": np.eye(4)},
}


# --- record contents / shapes / dtypes -------------------------------------

def test_record_contains_expected_keys_full_obs():
    rec, skipped = ls._build_frame_record(_mk_obs(), ["moka_pot_obstacle"], 42, _CAMS)
    assert skipped == []
    for key in ("agentview_rgb", "agentview_depth",
                "robot0_eye_in_hand_rgb", "robot0_eye_in_hand_depth",
                "eef_pos", "entity_names", "entity_pos", "t", "guard",
                "camera_K_agentview", "camera_T_agentview",
                "camera_K_robot0_eye_in_hand", "camera_T_robot0_eye_in_hand"):
        assert key in rec, key


def test_rgb_and_depth_dtypes_shapes():
    rec, _ = ls._build_frame_record(_mk_obs(), [], 0, _CAMS)
    assert rec["agentview_rgb"].dtype == np.uint8
    assert rec["agentview_rgb"].shape == (256, 256, 3)
    assert rec["agentview_depth"].dtype == np.float32
    assert rec["agentview_depth"].shape == (256, 256)
    assert rec["eef_pos"].shape == (3,)


def test_entities_are_all_scene_objects_not_just_hazards():
    rec, _ = ls._build_frame_record(_mk_obs(), ["moka_pot_obstacle"], 0, _CAMS)
    names = list(rec["entity_names"])
    assert set(names) == {"moka_pot_obstacle", "plate"}
    assert rec["entity_pos"].shape == (2, 3)
    # robot0_* and non-_pos keys must never leak into entities
    assert "robot0_gripper_qpos" not in names
    assert "robot0_eef" not in names


def test_guard_list_recorded():
    rec, _ = ls._build_frame_record(_mk_obs(), ["moka_pot_obstacle", "plate"], 7, _CAMS)
    assert list(rec["guard"]) == ["moka_pot_obstacle", "plate"]
    assert int(rec["t"]) == 7


def test_empty_guard_records_empty_array():
    rec, _ = ls._build_frame_record(_mk_obs(), [], 0, _CAMS)
    assert list(rec["guard"]) == []


# --- missing-field skip behavior --------------------------------------------

def test_missing_depth_key_absent_and_noted_in_skipped():
    obs = _mk_obs()
    del obs["agentview_depth"]
    del obs["robot0_eye_in_hand_depth"]
    rec, skipped = ls._build_frame_record(obs, [], 0, _CAMS)
    assert "agentview_depth" not in rec
    assert "robot0_eye_in_hand_depth" not in rec
    assert "agentview_depth" in skipped
    assert "robot0_eye_in_hand_depth" in skipped
    # unaffected fields still present
    assert "agentview_rgb" in rec


def test_missing_wrist_camera_entirely_absent_and_noted():
    obs = _mk_obs()
    del obs["robot0_eye_in_hand_image"]
    del obs["robot0_eye_in_hand_depth"]
    rec, skipped = ls._build_frame_record(obs, [], 0, _CAMS)
    assert "robot0_eye_in_hand_rgb" not in rec
    assert "robot0_eye_in_hand_rgb" in skipped


def test_no_entities_in_workspace_noted_in_skipped():
    obs = {"agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
           "robot0_eef_pos": np.array([0.0, 0.0, 0.9])}
    rec, skipped = ls._build_frame_record(obs, [], 0, {})
    assert "entities" in skipped
    assert "entity_names" not in rec


def test_missing_camera_matrices_noted_in_skipped():
    rec, skipped = ls._build_frame_record(_mk_obs(), [], 0, {"agentview": {}})
    assert "camera_K_agentview" in skipped
    assert "camera_T_agentview" in skipped
    assert "camera_K_agentview" not in rec


def test_never_crashes_on_sparse_obs():
    # Only the bare minimum -- must not raise.
    rec, skipped = ls._build_frame_record({}, None, 0, None)
    assert "eef_pos" in skipped
    assert "entities" in skipped
    assert list(rec["guard"]) == []


# --- filename layout ---------------------------------------------------------

def test_percep_frame_path_layout():
    p = ls._percep_frame_path("/tmp/frames", "obstacle_avoidance", 1, 3, 5, 2)
    assert p == pathlib.Path("/tmp/frames/obstacle_avoidance/L1/task3_ep5_replan2.npz")


def test_percep_frame_path_distinct_replans():
    p0 = ls._percep_frame_path("/tmp/frames", "human_safety", 0, 0, 0, 0)
    p1 = ls._percep_frame_path("/tmp/frames", "human_safety", 0, 0, 0, 1)
    assert p0 != p1
    assert p0.parent == p1.parent


# --- defaults-off ------------------------------------------------------------

def test_argparse_default_is_none():
    src = pathlib.Path(ls.__file__).read_text()
    assert '"--log_percep_frames", default=None' in src


def test_defaults_off_no_writer_object_created():
    # Mirrors the --log_transitions / --yield_regime source-level
    # defaults-off pattern: the writer/manifest state is created ONLY inside
    # an `if args.log_percep_frames:` guard, never unconditionally.
    src = pathlib.Path(ls.__file__).read_text()
    assert "percep_frame_manifest_skipped = None" in src
    assert "if args.log_percep_frames:" in src
    guard_idx = src.index("if args.log_percep_frames:\n        percep_frame_manifest_skipped = set()")
    assert guard_idx > 0
