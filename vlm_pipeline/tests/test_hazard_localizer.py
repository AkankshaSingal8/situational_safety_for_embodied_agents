"""CPU unit tests for the Phase 3 hazard localizer (spec 2026-08-11):
`vlm_pipeline/hazard_localizer.py` -- dataset indexing, split-by-task,
projection round-trip, a training smoke step, and the G-L1 offline eval
report. All synthetic data, no GPU, no mujoco/robosuite, no real
`localizer_data/` dependency (that job may still be running).
"""

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import hazard_localizer as hl  # noqa: E402

torch = pytest.importorskip("torch")


IMG = 64


def _rand_invertible_T(seed):
    rng = np.random.RandomState(seed)
    # small rotation (near-identity) + translation, camera looking down +z
    theta = rng.uniform(-0.2, 0.2)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    t = rng.uniform(-0.1, 0.1, size=3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _mk_npz(path, entities, eef_pos=(0.1, 0.2, 0.9), img_size=IMG, K=None, T=None,
            guard=("moka_pot_obstacle",), t=0, include_rgb=True, include_cameras=True,
            include_entities=True):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {}
    if include_rgb:
        record["agentview_rgb"] = (np.random.RandomState(0).rand(img_size, img_size, 3)
                                    * 255).astype(np.uint8)
    record["eef_pos"] = np.asarray(eef_pos, dtype=np.float32)
    record["t"] = np.asarray(t)
    record["guard"] = np.array(list(guard), dtype="<U128")
    if include_entities:
        names = sorted(entities)
        record["entity_names"] = np.array(names)
        record["entity_pos"] = np.stack([np.asarray(entities[n], dtype=np.float32)
                                          for n in names])
    if include_cameras:
        if K is None:
            K = np.array([[img_size, 0, img_size / 2],
                           [0, img_size, img_size / 2],
                           [0, 0, 1]], dtype=np.float64)
        if T is None:
            T = np.eye(4)
        record["camera_K_agentview"] = K
        record["camera_T_agentview"] = T
    np.savez_compressed(path, **record)


def _default_entities():
    return {
        "moka_pot_obstacle": [0.05, -0.02, 1.2],
        "plate": [-0.03, 0.04, 1.0],
    }


# ---------------------------------------------------------------------------
# Projection round trip
# ---------------------------------------------------------------------------

class TestProjectionRoundTrip:
    def test_recovers_world_point_within_tolerance(self):
        K = np.array([[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1.0]])
        img_h = 64
        for seed in range(5):
            T = _rand_invertible_T(seed)
            p_w = np.array([0.05 * seed, -0.03 * seed, 1.0 + 0.1 * seed])
            proj = hl.project_world_to_pixel(p_w, K, T, img_h)
            assert proj is not None
            u, v, z = proj
            recovered = hl.backproject(u, v, z, K, T, img_h)
            assert np.allclose(recovered, p_w, atol=1e-4)

    def test_behind_camera_gives_none_or_negative_z(self):
        K = np.array([[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1.0]])
        T = np.eye(4)
        p_w = np.array([0.0, 0.0, -1.0])  # behind camera (T=identity => cam frame = world)
        proj = hl.project_world_to_pixel(p_w, K, T, 64)
        assert proj is None or proj[2] < 0

    @pytest.mark.parametrize("input_size", [32, 64])
    def test_full_pixel_chain_recovers_entity_position(self, tmp_path, input_size):
        """Discriminating test for the FULL chain a GPU training run depends
        on: index_entity_samples -> Dataset (orig->resized->heatmap scale
        conversions) -> a synthetic (non-degenerate, quadratic-bowl) heatmap
        + depth map -> predict_position's soft-argmax decode -> backproject.
        A one-hot spike heatmap would quantize to the nearest cell and mask
        a half-cell bias or a w/h swap; the quadratic bowl's softmax
        centroid recovers the exact target to high precision, so any wrong
        scale factor or axis swap in the chain shows up as a real error
        here -- independent of whether training itself converges."""
        img_size = 96  # deliberately != input_size, to exercise the resize ratio
        entity_pos = [0.04, -0.03, 1.15]
        K = np.array([[90.0, 0, img_size / 2 + 3],   # off-center principal
                       [0, 110.0, img_size / 2 - 5],  # point + non-square fx/fy
                       [0, 0, 1.0]])
        T = _rand_invertible_T(seed=3)

        _mk_npz(tmp_path / "s" / "L0" / "task1_ep0_replan0.npz",
                {"moka_pot_obstacle": entity_pos}, img_size=img_size, K=K, T=T)

        frame_files = hl.index_frame_files(tmp_path)
        samples = hl.index_entity_samples(frame_files)
        assert len(samples) == 1

        ds = hl.HazardFrameEntityDataset(samples, input_size=input_size)
        item = ds[0]

        heat = np.zeros((hl.HEATMAP_SIZE, hl.HEATMAP_SIZE), dtype=np.float64)
        ys, xs = np.mgrid[0:hl.HEATMAP_SIZE, 0:hl.HEATMAP_SIZE]
        heat = -((xs - item["u_hm"]) ** 2 + (ys - item["v_hm"]) ** 2) / 2.0  # quadratic bowl

        depth_mean, depth_std = 1.0, 0.5
        depth_norm = (item["depth"] - depth_mean) / depth_std
        depth_map = np.full((hl.HEATMAP_SIZE, hl.HEATMAP_SIZE), depth_norm, dtype=np.float64)

        pred = hl.predict_position(heat, depth_map, item["w_orig"], item["h_orig"],
                                    item["K"].numpy(), item["T"].numpy(), input_size,
                                    depth_mean, depth_std)
        assert np.allclose(pred, entity_pos, atol=1e-2), (pred, entity_pos)


# ---------------------------------------------------------------------------
# Frame indexing
# ---------------------------------------------------------------------------

class TestFrameIndexing:
    def test_parses_suite_level_task_ep_replan(self, tmp_path):
        _mk_npz(tmp_path / "obstacle_avoidance" / "L0" / "task2_ep0_replan0.npz",
                _default_entities())
        _mk_npz(tmp_path / "obstacle_avoidance_human" / "L1" / "task7_ep3_replan2.npz",
                _default_entities())
        files = hl.index_frame_files(tmp_path)
        assert len(files) == 2
        by_task = {f["task_id"]: f for f in files}
        assert by_task[2]["suite"] == "obstacle_avoidance"
        assert by_task[2]["level"] == "0"
        assert by_task[2]["ep"] == 0
        assert by_task[2]["replan"] == 0
        assert by_task[7]["suite"] == "obstacle_avoidance_human"
        assert by_task[7]["level"] == "1"
        assert by_task[7]["ep"] == 3
        assert by_task[7]["replan"] == 2

    def test_ignores_non_matching_files(self, tmp_path):
        (tmp_path / "junk").mkdir()
        (tmp_path / "junk" / "manifest.json").write_text("{}")
        files = hl.index_frame_files(tmp_path)
        assert files == []


# ---------------------------------------------------------------------------
# Entity sample indexing (projection filtering)
# ---------------------------------------------------------------------------

class TestEntitySampleIndexing:
    def test_onscreen_entities_indexed_offscreen_dropped(self, tmp_path):
        entities = {
            "moka_pot_obstacle": [0.0, 0.0, 1.0],   # near image center, in front
            "far_off_object": [100.0, 100.0, 1.0],  # projects way outside image
            "behind_camera_object": [0.0, 0.0, -1.0],  # z <= 0, dropped
        }
        _mk_npz(tmp_path / "s" / "L0" / "task1_ep0_replan0.npz", entities, T=np.eye(4))
        frame_files = hl.index_frame_files(tmp_path)
        samples = hl.index_entity_samples(frame_files)
        names = {s["entity_name"] for s in samples}
        assert "moka_pot_obstacle" in names
        assert "far_off_object" not in names
        assert "behind_camera_object" not in names

    def test_missing_camera_matrices_skips_frame(self, tmp_path):
        _mk_npz(tmp_path / "s" / "L0" / "task1_ep0_replan0.npz", _default_entities(),
                include_cameras=False)
        frame_files = hl.index_frame_files(tmp_path)
        samples = hl.index_entity_samples(frame_files)
        assert samples == []

    def test_missing_entities_skips_frame(self, tmp_path):
        _mk_npz(tmp_path / "s" / "L0" / "task1_ep0_replan0.npz", _default_entities(),
                include_entities=False)
        frame_files = hl.index_frame_files(tmp_path)
        samples = hl.index_entity_samples(frame_files)
        assert samples == []


# ---------------------------------------------------------------------------
# Split by task
# ---------------------------------------------------------------------------

class TestSplitByTask:
    def _mk_dataset(self, tmp_path, tasks=(1, 2, 3, 7, 12), eps_per_task=3):
        for task in tasks:
            for ep in range(eps_per_task):
                _mk_npz(tmp_path / "obstacle_avoidance" / "L0"
                        / f"task{task}_ep{ep}_replan0.npz", _default_entities(),
                        T=np.eye(4))
        frame_files = hl.index_frame_files(tmp_path)
        return hl.index_entity_samples(frame_files)

    def test_split_is_deterministic(self, tmp_path):
        samples = self._mk_dataset(tmp_path)
        train1, val1 = hl.split_by_task(samples, hl.DEFAULT_HELD_OUT_TASKS)
        train2, val2 = hl.split_by_task(samples, hl.DEFAULT_HELD_OUT_TASKS)
        assert [s["path"] + s["entity_name"] for s in train1] == \
            [s["path"] + s["entity_name"] for s in train2]
        assert [s["path"] + s["entity_name"] for s in val1] == \
            [s["path"] + s["entity_name"] for s in val2]

    def test_no_episode_leakage_across_split(self, tmp_path):
        samples = self._mk_dataset(tmp_path)
        train, val = hl.split_by_task(samples, hl.DEFAULT_HELD_OUT_TASKS)
        train_tasks = {s["task_id"] for s in train}
        val_tasks = {s["task_id"] for s in val}
        assert train_tasks.isdisjoint(val_tasks)
        # every episode of a held-out task is entirely in val, none in train
        held = set(hl.DEFAULT_HELD_OUT_TASKS)
        assert val_tasks == held & {s["task_id"] for s in samples}
        assert all(s["task_id"] in held for s in val)
        assert all(s["task_id"] not in held for s in train)

    def test_split_summary_json_serializable(self, tmp_path):
        samples = self._mk_dataset(tmp_path)
        train, val = hl.split_by_task(samples, hl.DEFAULT_HELD_OUT_TASKS)
        summary = hl.build_split_summary(train, val, hl.DEFAULT_HELD_OUT_TASKS)
        json.dumps(summary)  # must not raise
        assert summary["n_train"] == len(train)
        assert summary["n_val"] == len(val)


# ---------------------------------------------------------------------------
# Training smoke test
# ---------------------------------------------------------------------------

class TestTrainingSmoke:
    def _mk_dataset(self, tmp_path, tasks=(1, 2, 3, 7), eps_per_task=2):
        rng = np.random.RandomState(0)
        for task in tasks:
            for ep in range(eps_per_task):
                entities = {
                    "moka_pot_obstacle": (rng.uniform(-0.05, 0.05, size=3)
                                           + [0.0, 0.0, 1.2]).tolist(),
                    "plate": (rng.uniform(-0.05, 0.05, size=3) + [0.0, 0.0, 1.0]).tolist(),
                }
                _mk_npz(tmp_path / "data" / "obstacle_avoidance" / "L0"
                        / f"task{task}_ep{ep}_replan0.npz", entities, T=np.eye(4))
        return tmp_path / "data"

    def test_train_runs_two_epochs_and_checkpoints(self, tmp_path):
        data_dir = self._mk_dataset(tmp_path)
        out_dir = tmp_path / "out"
        ap = hl.build_argparser()
        args = ap.parse_args([
            "train", "--data", str(data_dir), "--out", str(out_dir),
            "--held_out_tasks", "7", "--epochs", "2", "--patience", "100",
            "--batch_size", "4", "--input_size", "32", "--num_workers", "0",
        ])
        hl.train_main(args)

        assert (out_dir / "model.pt").exists()
        assert (out_dir / "split.json").exists()
        assert (out_dir / "norm_stats.json").exists()
        assert (out_dir / "train_history.json").exists()

        with open(out_dir / "train_history.json") as f:
            hist = json.load(f)
        assert len(hist["history"]) == 2
        for row in hist["history"]:
            assert np.isfinite(row["train_loss"])

        with open(out_dir / "split.json") as f:
            split = json.load(f)
        assert split["held_out_task_ids"] == [7]

    def test_model_loads_back_and_predicts(self, tmp_path):
        data_dir = self._mk_dataset(tmp_path)
        out_dir = tmp_path / "out"
        ap = hl.build_argparser()
        args = ap.parse_args([
            "train", "--data", str(data_dir), "--out", str(out_dir),
            "--held_out_tasks", "7", "--epochs", "1", "--patience", "100",
            "--batch_size", "4", "--input_size", "32", "--num_workers", "0",
        ])
        hl.train_main(args)
        model, ckpt = hl.load_model(out_dir)
        assert isinstance(model, hl.HazardLocalizerNet)


# ---------------------------------------------------------------------------
# G-L1 eval
# ---------------------------------------------------------------------------

class TestEval:
    def _mk_dataset(self, tmp_path, tasks=(1, 2, 3, 7), eps_per_task=2):
        rng = np.random.RandomState(1)
        for task in tasks:
            for ep in range(eps_per_task):
                entities = {
                    "moka_pot_obstacle": (rng.uniform(-0.05, 0.05, size=3)
                                           + [0.0, 0.0, 1.2]).tolist(),
                    "plate": (rng.uniform(-0.05, 0.05, size=3) + [0.0, 0.0, 1.0]).tolist(),
                }
                _mk_npz(tmp_path / "data" / "obstacle_avoidance" / "L0"
                        / f"task{task}_ep{ep}_replan0.npz", entities, T=np.eye(4))
        return tmp_path / "data"

    def test_eval_writes_report_with_required_keys(self, tmp_path):
        data_dir = self._mk_dataset(tmp_path)
        out_dir = tmp_path / "out"
        ap = hl.build_argparser()
        train_args = ap.parse_args([
            "train", "--data", str(data_dir), "--out", str(out_dir),
            "--held_out_tasks", "7", "--epochs", "1", "--patience", "100",
            "--batch_size", "4", "--input_size", "32", "--num_workers", "0",
        ])
        hl.train_main(train_args)

        eval_args = ap.parse_args([
            "eval", "--data", str(data_dir), "--model", str(out_dir),
            "--batch_size", "4", "--num_workers", "0",
        ])
        report = hl.eval_main(eval_args)

        report_path = out_dir / "g_l1_report.json"
        assert report_path.exists()
        with open(report_path) as f:
            on_disk = json.load(f)

        for key in ("overall", "per_suite", "per_entity", "gate_pass",
                    "gate_median_m", "held_out_task_ids"):
            assert key in on_disk

        assert on_disk["overall"]["n"] > 0
        assert on_disk["held_out_task_ids"] == [7]
        assert on_disk["gate_median_m"] == hl.GATE_MEDIAN_M
        assert report == on_disk
