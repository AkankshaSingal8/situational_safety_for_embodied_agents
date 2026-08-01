"""CPU unit tests for the LS runner's --exec_parity helpers.

Covers the authors'-harness execution-parity seams added 2026-08-01:
`_parity_controller_config` / `_install_parity_controller` (vendored
robosuite-1.4 OSC gains), `_plan_actions` (raw passthrough vs the legacy
ACTION_TO_CMD conversion, byte-identical when the flag is off), and
`_parity_images` (client-side resize_with_pad 224, the baseline client's
exact image path). No GPU, no env creation, no mujoco/robosuite imports —
the robosuite monkeypatch is exercised against a stub module.
"""

import pathlib
import sys
import types

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
# openpi_client (used by _parity_images) ships inside the worktree.
sys.path.insert(
    0, str(_HERE.parents[2] / "openpi_guided" / "packages" / "openpi-client" / "src"))

import run_guided_libero_safety_eval as ls  # noqa: E402

STOCK_OSC = {
    "type": "OSC_POSE",
    "input_max": 1,
    "input_min": -1,
    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    "kp": 150,
    "damping_ratio": 1,
    "impedance_mode": "fixed",
    "kp_limits": [0, 300],
    "control_delta": True,
    "ramp_ratio": 0.2,
}


# --- controller config ------------------------------------------------------

def test_parity_config_applies_vendored_gains():
    out = ls._parity_controller_config(STOCK_OSC)
    assert out["output_max"] == [2] * 6
    assert out["output_min"] == [-2] * 6
    assert out["kp"] == 750
    assert out["kp_limits"] == [0, 1000]


def test_parity_config_pure_and_preserves_other_keys():
    snapshot = dict(STOCK_OSC)
    out = ls._parity_controller_config(STOCK_OSC)
    assert STOCK_OSC == snapshot          # input never mutated
    for k in ("type", "input_max", "input_min", "damping_ratio",
              "impedance_mode", "control_delta", "ramp_ratio"):
        assert out[k] == STOCK_OSC[k]     # non-gain keys pass through


def test_install_parity_controller_patches_and_is_idempotent():
    stub = types.ModuleType("robosuite")
    calls = []

    def load_controller_config(default_controller="OSC_POSE"):
        calls.append(default_controller)
        if default_controller == "OSC_POSE":
            return dict(STOCK_OSC)
        return {"type": default_controller}

    stub.load_controller_config = load_controller_config
    saved = sys.modules.get("robosuite")
    sys.modules["robosuite"] = stub
    try:
        ls._install_parity_controller()
        cfg = stub.load_controller_config(default_controller="OSC_POSE")
        assert cfg["output_max"] == [2] * 6 and cfg["kp"] == 750
        # Non-OSC_POSE configs pass through unpatched.
        assert stub.load_controller_config(
            default_controller="JOINT_VELOCITY") == {"type": "JOINT_VELOCITY"}
        # Idempotent: second install must not double-wrap.
        patched = stub.load_controller_config
        ls._install_parity_controller()
        assert stub.load_controller_config is patched
    finally:
        if saved is not None:
            sys.modules["robosuite"] = saved
        else:
            del sys.modules["robosuite"]


# --- action path ------------------------------------------------------------

CHUNK = np.array([
    [0.012, -0.02, 0.018, 0.03, -0.05, 0.02, 1.0],    # typical LS metric step
    [0.08, -0.08, 0.0, 0.6, 0.0, 0.0, -1.0],          # would clip post-conversion
])


def test_plan_actions_default_byte_identical_to_legacy():
    # Flag off: exactly the historical expression, element for element.
    legacy = np.clip(CHUNK * ls.ACTION_TO_CMD, -1.0, 1.0)
    out = ls._plan_actions(CHUNK)
    assert np.array_equal(out, legacy)


def test_plan_actions_raw_actions_clips_only():
    out = ls._plan_actions(CHUNK, raw_actions=True)
    assert np.array_equal(out, np.clip(CHUNK, -1.0, 1.0))


def test_plan_actions_exec_parity_is_raw_passthrough():
    # Parity: NO conversion and NO client clip (robosuite clips internally),
    # matching the baseline harness's env.step(action) passthrough.
    big = np.array([[1.5, -1.5, 0.02, 0.0, 0.0, 0.0, 1.0]])
    assert np.array_equal(ls._plan_actions(CHUNK, exec_parity=True), CHUNK)
    assert np.array_equal(ls._plan_actions(big, exec_parity=True), big)


def test_plan_actions_exec_parity_supersedes_raw_actions():
    big = np.array([[1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    out = ls._plan_actions(big, exec_parity=True, raw_actions=True)
    assert np.array_equal(out, big)       # unclipped -> parity won


# --- image path -------------------------------------------------------------

def test_parity_images_square_matches_reference_resize():
    # 256x256 -> 224x224: aspect preserved, so resize_with_pad == plain PIL
    # bilinear resize with zero padding. Compare byte-exactly against an
    # independent PIL reference (the op the baseline client performs).
    from PIL import Image
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    out_img, out_wrist = ls._parity_images(img, wrist)
    for src, out in ((img, out_img), (wrist, out_wrist)):
        assert out.shape == (224, 224, 3) and out.dtype == np.uint8
        ref = np.asarray(
            Image.fromarray(src).resize((224, 224), resample=Image.BILINEAR))
        assert np.array_equal(out, ref)


def test_parity_images_pads_non_square_with_zeros():
    img = np.full((128, 256, 3), 200, dtype=np.uint8)   # 2:1 wide
    out, _ = ls._parity_images(img, img)
    assert out.shape == (224, 224, 3)
    # Scaled to 224x112, centered: 56 zero rows top and bottom.
    assert (out[:56] == 0).all() and (out[-56:] == 0).all()
    assert (out[56:168] > 0).any()


def test_parity_images_noop_when_already_target_size():
    img = np.random.default_rng(1).integers(
        0, 256, size=(224, 224, 3), dtype=np.uint8)
    out, _ = ls._parity_images(img, img)
    assert np.array_equal(out, img)
