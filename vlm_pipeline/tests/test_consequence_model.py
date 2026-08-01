"""CPU tests for `consequence_model.py` (Task 2, consequence-steering-code
plan, spec 2026-08-01).

Torch-dependent tests are skipped (not failed) in environments without
torch, via `pytest.importorskip("torch")` at module scope for the training
smoke test, and per-function guards elsewhere -- the pure dataset/labeling/
baseline/split logic has no torch dependency and always runs.
"""

import json
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import consequence_model as cm  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures: tiny synthetic transitions_*.jsonl files
# --------------------------------------------------------------------------

def _write_episode(path, task, ep, n_steps, entity_name="hazard_a",
                    entity_start=(0.5, 0.5, 0.9), entity_vel=(0.0, 0.0, 0.0),
                    eef_start=(0.0, 0.0, 0.9), action_xyz=(0.01, 0.0, 0.0),
                    grip_open=(0.02, 0.02)):
    """Writes n_steps records for one episode: constant action -> eef moves
    by action_xyz each step (post-step convention), entity drifts by
    entity_vel each step. grip is constant (open)."""
    eef = np.array(eef_start, dtype=np.float64)
    entity = np.array(entity_start, dtype=np.float64)
    lines = []
    for t in range(n_steps):
        eef = eef + np.array(action_xyz)
        entity = entity + np.array(entity_vel)
        rec = {
            "task": task, "ep": ep, "t": t,
            "eef": eef.tolist(), "grip": list(grip_open),
            "action": list(action_xyz) + [0.0, 0.0, 0.0, -1.0],
            "entities": {entity_name: entity.tolist()},
        }
        lines.append(json.dumps(rec))
    path.write_text("\n".join(lines) + "\n")


def _append_episode(path, **kwargs):
    existing = path.read_text() if path.exists() else ""
    _write_to = path.parent / "___tmp.jsonl"
    _write_episode(_write_to, **kwargs)
    new = _write_to.read_text()
    _write_to.unlink()
    path.write_text(existing + new)


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    # LS-style filename, 25 steps so multiple H=10 windows exist per episode.
    p = d / "transitions_obstacle_avoidance_L0.jsonl"
    _append_episode(p, task=0, ep=0, n_steps=25)
    _append_episode(p, task=0, ep=1, n_steps=25)
    return d


@pytest.fixture
def multi_episode_dir(tmp_path):
    d = tmp_path / "data2"
    d.mkdir()
    p = d / "transitions_obstacle_avoidance_L0.jsonl"
    for ep in range(6):
        _append_episode(p, task=0, ep=ep, n_steps=15,
                         eef_start=(0.0, 0.0, 0.9 + 0.001 * ep))
    q = d / "transitions_safelibero_spatial_I.jsonl"
    for ep in range(4):
        _append_episode(q, task=1, ep=ep, n_steps=15,
                         eef_start=(1.0, 0.0, 0.9 + 0.001 * ep))
    return d


# --------------------------------------------------------------------------
# Domain inference
# --------------------------------------------------------------------------

def test_infer_domain_safelibero():
    assert cm.infer_domain(pathlib.Path("x/transitions_safelibero_spatial_I.jsonl")) == "SL"


def test_infer_domain_ls():
    assert cm.infer_domain(pathlib.Path("x/transitions_obstacle_avoidance_L0.jsonl")) == "LS"


# --------------------------------------------------------------------------
# Dataset parsing
# --------------------------------------------------------------------------

def test_load_episodes_groups_and_sorts(data_dir):
    episodes = cm.load_episodes(data_dir)
    assert len(episodes) == 2
    keys = {(e.task, e.ep) for e in episodes}
    assert keys == {(0, 0), (0, 1)}
    for e in episodes:
        ts = [r["t"] for r in e.records]
        assert ts == sorted(ts)
        assert e.domain == "LS"


def test_entity_order_deterministic_and_capped(data_dir):
    episodes = cm.load_episodes(data_dir)
    ep = episodes[0]
    order = ep.entity_order()
    assert order == sorted(set(order))
    assert len(order) <= cm.E


def test_entity_order_caps_at_E(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    p = d / "transitions_obstacle_avoidance_L0.jsonl"
    eef = [0.0, 0.0, 0.9]
    lines = []
    for t in range(3):
        rec = {"task": 0, "ep": 0, "t": t, "eef": eef, "grip": [0.02, 0.02],
               "action": [0.0] * 7,
               "entities": {f"e{i}": [0.0, 0.0, 0.0] for i in range(12)}}
        lines.append(json.dumps(rec))
    p.write_text("\n".join(lines))
    episodes = cm.load_episodes(d)
    order = episodes[0].entity_order()
    assert len(order) == cm.E
    assert order == sorted(f"e{i}" for i in range(12))[:cm.E]


# --------------------------------------------------------------------------
# Sample construction: padding / mask
# --------------------------------------------------------------------------

def test_build_samples_padding_and_mask(data_dir):
    episodes = cm.load_episodes(data_dir)
    samples = cm.build_samples(episodes)
    assert len(samples) > 0
    s = samples[0]
    assert s.entity_rel.shape == (cm.E, 3)
    assert s.entity_mask.shape == (cm.E,)
    assert s.action_window.shape == (cm.H, 3)
    assert s.path_target.shape == (cm.H, 3)
    # one entity present -> slot 0 mask=1, rest 0
    assert s.entity_mask.sum() == 1.0
    assert np.all(s.entity_rel[1:] == 0.0)


def test_build_samples_window_count_matches_tail_drop(data_dir):
    episodes = cm.load_episodes(data_dir)
    ep = [e for e in episodes if e.ep == 0][0]
    samples = cm.build_samples([ep])
    n = len(ep.records)
    # anchors i in [1, n-1], need i + H - 1 <= n - 1 -> i <= n - H
    expected = max(0, (n - cm.H) - 1 + 1)
    assert len(samples) == expected


def test_build_samples_short_episode_drops_all(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    p = d / "transitions_obstacle_avoidance_L0.jsonl"
    _write_episode(p, task=0, ep=0, n_steps=5)  # shorter than H=10
    episodes = cm.load_episodes(d)
    samples = cm.build_samples(episodes)
    assert samples == []


# --------------------------------------------------------------------------
# Label computation vs hand-computed values
# --------------------------------------------------------------------------

def test_contact_label_hand_computed(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    p = d / "transitions_obstacle_avoidance_L0.jsonl"
    # eef stationary at (0,0,0.9) (zero action); hazard fixed 0.02 away
    # (< CONTACT_DIST=0.05) for every step in the window -> CONTACT should
    # be 1 at every anchor.
    _write_episode(p, task=0, ep=0, n_steps=12,
                    eef_start=(0.0, 0.0, 0.9), action_xyz=(0.0, 0.0, 0.0),
                    entity_name="hazard_a", entity_start=(0.02, 0.0, 0.9),
                    entity_vel=(0.0, 0.0, 0.0))
    episodes = cm.load_episodes(d)
    samples = cm.build_samples(episodes)
    s = samples[0]
    slot = s.entity_names.index("hazard_a")
    assert s.contact_mask[slot] == 1.0
    assert s.contact_target[slot] == 1.0


def test_contact_label_negative_when_far(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    p = d / "transitions_obstacle_avoidance_L0.jsonl"
    _write_episode(p, task=0, ep=0, n_steps=12,
                    eef_start=(0.0, 0.0, 0.9), action_xyz=(0.0, 0.0, 0.0),
                    entity_name="hazard_a", entity_start=(5.0, 5.0, 5.0),
                    entity_vel=(0.0, 0.0, 0.0))
    episodes = cm.load_episodes(d)
    samples = cm.build_samples(episodes)
    s = samples[0]
    slot = s.entity_names.index("hazard_a")
    assert s.contact_mask[slot] == 1.0
    assert s.contact_target[slot] == 0.0


def test_holding_label_band(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    p = d / "transitions_obstacle_avoidance_L0.jsonl"
    _write_episode(p, task=0, ep=0, n_steps=12, grip_open=(0.02, 0.02))  # width=0.04 -> in band
    episodes = cm.load_episodes(d)
    s = cm.build_samples(episodes)[0]
    assert s.holding_target == 1.0

    q = d / "transitions_obstacle_avoidance_L0b.jsonl"
    _write_episode(q, task=0, ep=1, n_steps=12, grip_open=(0.0, 0.0))  # width=0 -> not in band
    episodes2 = cm.load_episodes(d)
    ep1 = [e for e in episodes2 if e.ep == 1][0]
    s2 = cm.build_samples([ep1])[0]
    assert s2.holding_target == 0.0


def test_progress_label_hand_computed(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    p = d / "transitions_obstacle_avoidance_L0.jsonl"
    # eef starts (0,0,0.9), moves +x 0.01/step, target fixed (1.0,0,0.9).
    # anchor i=1: s_1 = record[0].eef = (0.01,0,0.9); dist0 = 0.99
    # window end index i+H-1 = 1+9 = 10 -> record[10].eef = (0.11,0,0.9);
    # dist1 = 0.89 -> progress = 0.99-0.89 = 0.10
    _write_episode(p, task=0, ep=0, n_steps=15,
                    eef_start=(0.0, 0.0, 0.9), action_xyz=(0.01, 0.0, 0.0),
                    entity_name="target_obj", entity_start=(1.0, 0.0, 0.9),
                    entity_vel=(0.0, 0.0, 0.0))
    episodes = cm.load_episodes(d)
    target_map = {0: "target_obj"}
    samples = cm.build_samples(episodes, target_map)
    s = samples[0]
    assert s.progress_mask == 1.0
    assert abs(s.progress_target - 0.10) < 1e-9


def test_progress_label_masked_without_target_map(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    p = d / "transitions_obstacle_avoidance_L0.jsonl"
    _write_episode(p, task=0, ep=0, n_steps=15,
                    entity_name="target_obj", entity_start=(1.0, 0.0, 0.9))
    episodes = cm.load_episodes(d)
    samples = cm.build_samples(episodes)  # no target_map
    assert all(s.progress_mask == 0.0 for s in samples)


def test_progress_label_masked_when_task_unmapped(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    p = d / "transitions_obstacle_avoidance_L0.jsonl"
    _write_episode(p, task=5, ep=0, n_steps=15, entity_name="target_obj")
    episodes = cm.load_episodes(d)
    samples = cm.build_samples(episodes, {0: "target_obj"})  # task 5 unmapped
    assert all(s.progress_mask == 0.0 for s in samples)
    # contact/holding labels still present
    assert samples[0].contact_mask.sum() >= 1.0


# --------------------------------------------------------------------------
# Single-integrator baseline math
# --------------------------------------------------------------------------

def test_single_integrator_terminal_basic():
    window = np.tile(np.array([0.01, -0.02, 0.03]), (cm.H, 1))
    out = cm.single_integrator_terminal(window, translation_scale=2.0, cmd_clip=0.025)
    # clip: 0.01 unclipped, -0.02 unclipped, 0.03 -> clipped to 0.025
    expected_step = np.array([0.01, -0.02, 0.025]) * 2.0
    expected_terminal = expected_step * cm.H
    assert np.allclose(out, expected_terminal)


def test_single_integrator_terminal_batched():
    window = np.random.default_rng(0).uniform(-0.05, 0.05, size=(4, cm.H, 3))
    out = cm.single_integrator_terminal(window)
    assert out.shape == (4, 3)
    for k in range(4):
        single = cm.single_integrator_terminal(window[k])
        assert np.allclose(out[k], single)


# --------------------------------------------------------------------------
# Split determinism
# --------------------------------------------------------------------------

def test_split_deterministic(multi_episode_dir):
    episodes = cm.load_episodes(multi_episode_dir)
    train1, held1 = cm.split_episodes(episodes, seed=42)
    train2, held2 = cm.split_episodes(episodes, seed=42)
    assert train1 == train2
    assert held1 == held2
    assert set(train1).isdisjoint(set(held1))
    assert set(train1) | set(held1) == {e.key for e in episodes}


def test_split_different_seed_can_differ(multi_episode_dir):
    episodes = cm.load_episodes(multi_episode_dir)
    _, held_a = cm.split_episodes(episodes, seed=1)
    _, held_b = cm.split_episodes(episodes, seed=2)
    assert held_a != held_b or len(episodes) < 3  # extremely unlikely to tie


def test_split_save_load_roundtrip(multi_episode_dir, tmp_path):
    episodes = cm.load_episodes(multi_episode_dir)
    train, held = cm.split_episodes(episodes, seed=7)
    split_path = tmp_path / "split.json"
    cm.save_split(split_path, train, held)
    train2, held2 = cm.load_split(split_path)
    assert train2 == train
    assert held2 == held


# --------------------------------------------------------------------------
# AUC / Spearman helpers (no torch needed)
# --------------------------------------------------------------------------

def test_roc_auc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert cm._roc_auc(scores, labels) == 1.0


def test_roc_auc_worst_case():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([0, 0, 1, 1])
    assert cm._roc_auc(scores, labels) == 0.0


def test_roc_auc_chance():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 1, 0, 1])
    assert abs(cm._roc_auc(scores, labels) - 0.5) < 1e-9


def test_spearman_monotonic():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    assert abs(cm._spearman(a, b) - 1.0) < 1e-9


def test_spearman_inverse():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([40.0, 30.0, 20.0, 10.0])
    assert abs(cm._spearman(a, b) - (-1.0)) < 1e-9


# --------------------------------------------------------------------------
# Normalization stats
# --------------------------------------------------------------------------

def test_norm_stats_json_roundtrip(data_dir):
    episodes = cm.load_episodes(data_dir)
    samples = cm.build_samples(episodes)
    norm = cm.compute_norm_stats(samples)
    d = norm.to_json()
    norm2 = cm.NormStats.from_json(d)
    assert np.allclose(norm.rel_mean, norm2.rel_mean)
    assert np.allclose(norm.path_std, norm2.path_std)
    assert np.allclose(norm.action_std, norm2.action_std)


# --------------------------------------------------------------------------
# No `_obstacle_` substring in decision logic (module source check)
# --------------------------------------------------------------------------

def test_no_obstacle_name_substring_in_module():
    src = pathlib.Path(cm.__file__).read_text()
    assert "_obstacle_" not in src


# --------------------------------------------------------------------------
# Torch-dependent: import guard + training smoke
# --------------------------------------------------------------------------

def test_module_imports_without_torch_available_flag():
    # This just checks the module exposes the flag consistently; it does not
    # require torch to be absent (both cases pass this assertion).
    assert isinstance(cm.TORCH_AVAILABLE, bool)


def test_torch_unavailable_functions_raise_cleanly(monkeypatch):
    monkeypatch.setattr(cm, "TORCH_AVAILABLE", False)
    with pytest.raises(RuntimeError):
        cm._require_torch()


def test_model_param_count_under_1M():
    pytest.importorskip("torch")
    model = cm.ConsequenceModel()
    n_params = cm.count_parameters(model)
    assert 0 < n_params < 1_000_000


def test_smoke_train_loss_decreases_and_g1_emitted(tmp_path):
    pytest.importorskip("torch")
    d = tmp_path / "data"
    d.mkdir()
    p = d / "transitions_obstacle_avoidance_L0.jsonl"
    rng = np.random.default_rng(0)

    # Deliberately construct two hazard-distance regimes so G1 metric (a)
    # (near-entity terminal error) and (b) (CONTACT AUC, needs both classes)
    # are actually MEASURABLE in this smoke test, not just NaN-safe. "close"
    # episodes: hazard ~0.028 m from a near-stationary eef -> inside
    # CONTACT_DIST=0.05 -> contact-positive, and inside NEAR_ENTITY_RADIUS
    # (0.25 m). "medium" episodes: hazard ~0.15 m away -> outside
    # CONTACT_DIST (contact-negative) but still inside NEAR_ENTITY_RADIUS.
    # Near-zero action noise keeps eef essentially stationary so the anchor
    # distance category holds for the whole window.
    for ep in range(8):
        close = ep % 2 == 0
        offset = (0.02, 0.02, 0.0) if close else (0.15, 0.0, 0.0)
        _append_episode(
            p, task=0, ep=ep, n_steps=20,
            eef_start=(0.0, 0.0, 0.9),
            action_xyz=tuple(rng.uniform(-0.002, 0.002, size=3)),
            entity_name="hazard_a",
            entity_start=(0.0 + offset[0], 0.0 + offset[1], 0.9 + offset[2]),
            grip_open=(0.02, 0.02) if ep % 2 == 0 else (0.0, 0.0),
        )
    episodes = cm.load_episodes(d)
    # Manual split (not cm.split_episodes) so BOTH hazard-distance regimes
    # are guaranteed present in the held-out set regardless of RNG seed --
    # split-fairness itself is covered by test_split_* above.
    held_eps = [e for e in episodes if e.ep in (0, 1, 2, 3)]  # 2 close + 2 medium
    train_eps = [e for e in episodes if e.ep in (4, 5, 6, 7)]
    target_map = {0: "hazard_a"}
    train_samples = cm.build_samples(train_eps, target_map)
    held_samples = cm.build_samples(held_eps, target_map)
    assert len(train_samples) > 0 and len(held_samples) > 0

    norm = cm.compute_norm_stats(train_samples)
    out_dir = tmp_path / "out"
    histories = cm.train_ensemble(
        train_samples, held_samples, norm, out_dir,
        n_members=1, max_epochs=2, patience=5)
    assert len(histories) == 1
    losses = histories[0]["train_losses"]
    assert len(losses) == 2
    assert losses[-1] <= losses[0] + 1e-6  # non-increasing across 2 epochs

    models, norm2 = cm.load_ensemble(out_dir, n_members=1)
    report = cm.evaluate_g1(held_samples, models, norm2)
    g1_report_path = out_dir / "g1_report.json"
    g1_report_path.write_text(json.dumps(report, indent=2))

    assert (out_dir / "consequence_model_member0.pt").exists()
    assert g1_report_path.exists()
    assert "pass" in report and "err_ratio" in report and "auc" in report \
        and "spearman" in report
    assert report["n_near_entity_samples"] > 0
    assert report["n_contact_positives"] > 0
    assert report["n_slots_scored"] > 0
    assert math.isfinite(report["err_ratio"]), report
    assert math.isfinite(report["auc"]), report
    assert isinstance(report["unmeasurable"], bool)
