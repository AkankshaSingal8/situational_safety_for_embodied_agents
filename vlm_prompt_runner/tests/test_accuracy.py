import json
from pathlib import Path

import pytest

from vlm_prompt_runner.accuracy import (
    compute_accuracy,
    eval_episode,
    load_ground_truth,
    load_prediction,
    names_match,
    normalize_name,
)


def test_normalize_strips_trailing_number():
    assert "moka_pot" not in normalize_name("moka_pot_1")  # returns set
    assert normalize_name("moka_pot_1") == {"moka", "pot"}


def test_normalize_strips_obstacle_suffix():
    assert normalize_name("moka_pot_obstacle_1") == {"moka", "pot"}


def test_normalize_strips_obstacle_no_number():
    assert normalize_name("wine_bottle_obstacle") == {"wine", "bottle"}


def test_normalize_removes_stop_words():
    tokens = normalize_name("black_bowl_between_plate_and_ramekin")
    assert "between" not in tokens
    assert "and" not in tokens
    assert "black" in tokens
    assert "bowl" in tokens


def test_names_match_same_core_words():
    assert names_match("moka_pot_1", "moka_pot_obstacle_1") is True


def test_names_match_one_word_overlap():
    assert names_match("moka_pot", "moka_pot_obstacle_1") is True


def test_names_match_no_overlap():
    assert names_match("glazed_rim_porcelain_ramekin_1", "moka_pot_obstacle_1") is False


def test_names_match_wine_bottle():
    assert names_match("wine_bottle", "wine_bottle_obstacle_1") is True


def test_names_match_different_objects():
    assert names_match("red_coffee_mug_obstacle_1", "wine_bottle_obstacle_1") is False


# ── Task 2: loaders and eval_episode ─────────────────────────────────────────

def test_load_ground_truth_returns_name(tmp_path):
    meta = tmp_path / "metadata.json"
    meta.write_text(json.dumps({
        "obstacle": {"name": "moka_pot_obstacle_1", "position": [0, 0, 0]}
    }))
    assert load_ground_truth(tmp_path) == "moka_pot_obstacle_1"


def test_load_ground_truth_missing_obstacle_key(tmp_path):
    meta = tmp_path / "metadata.json"
    meta.write_text(json.dumps({"task_description": "pick up cube"}))
    assert load_ground_truth(tmp_path) is None


def test_load_ground_truth_missing_file(tmp_path):
    assert load_ground_truth(tmp_path / "nonexistent") is None


def test_load_prediction_returns_object(tmp_path):
    out = tmp_path / "output.json"
    out.write_text(json.dumps({"object": "glazed_ramekin_1", "role": "obstacle"}))
    assert load_prediction(tmp_path) == {"object": "glazed_ramekin_1", "role": "obstacle"}


def test_load_prediction_missing_file(tmp_path):
    assert load_prediction(tmp_path / "nonexistent") is None


def test_eval_episode_correct(tmp_path):
    pred_ep = tmp_path / "pred" / "episode_00"
    gt_ep = tmp_path / "gt" / "episode_00"
    pred_ep.mkdir(parents=True)
    gt_ep.mkdir(parents=True)

    (pred_ep / "output.json").write_text(json.dumps({"object": "moka_pot_1", "role": "obstacle"}))
    (gt_ep / "metadata.json").write_text(json.dumps({"obstacle": {"name": "moka_pot_obstacle_1"}}))

    result = eval_episode(pred_ep, gt_ep)
    assert result["correct"] is True
    assert result["predicted"] == "moka_pot_1"
    assert result["ground_truth"] == "moka_pot_obstacle_1"


def test_eval_episode_incorrect(tmp_path):
    pred_ep = tmp_path / "pred" / "episode_00"
    gt_ep = tmp_path / "gt" / "episode_00"
    pred_ep.mkdir(parents=True)
    gt_ep.mkdir(parents=True)

    (pred_ep / "output.json").write_text(json.dumps({"object": "glazed_rim_ramekin_1", "role": "obstacle"}))
    (gt_ep / "metadata.json").write_text(json.dumps({"obstacle": {"name": "moka_pot_obstacle_1"}}))

    result = eval_episode(pred_ep, gt_ep)
    assert result["correct"] is False


def test_eval_episode_missing_gt_returns_none(tmp_path):
    pred_ep = tmp_path / "pred" / "episode_00"
    gt_ep = tmp_path / "gt" / "episode_00"
    pred_ep.mkdir(parents=True)
    gt_ep.mkdir(parents=True)  # no metadata.json

    (pred_ep / "output.json").write_text(json.dumps({"object": "moka_pot_1", "role": "obstacle"}))

    result = eval_episode(pred_ep, gt_ep)
    assert result is None


# ── Task 3: compute_accuracy ──────────────────────────────────────────────────

def _write_ep(pred_root, gt_root, suite, level, task, ep, pred_obj, gt_obj):
    pred_ep = pred_root / suite / level / task / ep
    gt_ep = gt_root / suite / level / task / ep
    pred_ep.mkdir(parents=True, exist_ok=True)
    gt_ep.mkdir(parents=True, exist_ok=True)
    (pred_ep / "output.json").write_text(json.dumps({"object": pred_obj, "role": "obstacle"}))
    (gt_ep / "metadata.json").write_text(json.dumps({"obstacle": {"name": gt_obj}}))


def test_compute_accuracy_basic(tmp_path):
    stl_dir = tmp_path / "stl"
    gt_dir = tmp_path / "gt"

    # 2 correct, 1 wrong
    _write_ep(stl_dir, gt_dir, "suite_A", "level_I", "task_0", "episode_00", "moka_pot_1", "moka_pot_obstacle_1")
    _write_ep(stl_dir, gt_dir, "suite_A", "level_I", "task_0", "episode_01", "wine_bottle_1", "wine_bottle_obstacle_1")
    _write_ep(stl_dir, gt_dir, "suite_A", "level_I", "task_0", "episode_02", "glazed_ramekin_1", "moka_pot_obstacle_1")

    results = compute_accuracy(stl_dir, gt_dir)

    task_stats = results["suite_A"]["level_I"]["task_0"]
    assert task_stats["correct"] == 2
    assert task_stats["total"] == 3
    assert len(task_stats["episodes"]) == 3

    overall = results["_totals"]["overall"]
    assert overall["correct"] == 2
    assert overall["total"] == 3


def test_compute_accuracy_per_level_and_suite(tmp_path):
    stl_dir = tmp_path / "stl"
    gt_dir = tmp_path / "gt"

    # level_I task_0: 1 correct
    _write_ep(stl_dir, gt_dir, "suite_A", "level_I", "task_0", "episode_00", "moka_pot_1", "moka_pot_obstacle_1")
    # level_I task_1: 0 correct
    _write_ep(stl_dir, gt_dir, "suite_A", "level_I", "task_1", "episode_00", "ramekin_1", "wine_bottle_obstacle_1")
    # level_II task_0: 1 correct
    _write_ep(stl_dir, gt_dir, "suite_A", "level_II", "task_0", "episode_00", "milk_1", "milk_obstacle_1")

    results = compute_accuracy(stl_dir, gt_dir)

    level_I = results["suite_A"]["level_I"]["_totals"]
    assert level_I["correct"] == 1
    assert level_I["total"] == 2

    suite_totals = results["suite_A"]["_totals"]
    assert suite_totals["correct"] == 2
    assert suite_totals["total"] == 3
