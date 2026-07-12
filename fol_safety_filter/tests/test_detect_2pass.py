"""Tests for the FOL_DETECT_2PASS upper-half second detection pass."""

import json

import numpy as np
import pytest

from fol_safety_filter.visual_grounder import VisualObstacleGrounder


class _TwoScaleClient:
    """Full image (512): only a 'stove' box. Upscaled upper half (512 from
    256): stove AND the pot that merges with it at full scale."""

    def __init__(self):
        self.calls = []

    def infer(self, prompt, images, max_tokens=512):
        img = images[0]
        self.calls.append(img.shape)
        if img.mean() > 150:  # upscaled upper-half crop is uniformly bright
            return json.dumps([
                {"bbox_2d": [40, 40, 240, 240], "label": "stove"},
                {"bbox_2d": [100, 20, 160, 120], "label": "moka pot"},
            ])
        return json.dumps([{"bbox_2d": [20, 20, 120, 120], "label": "stove"}])


def _img():
    im = np.zeros((512, 512, 3), dtype=np.uint8)
    im[:256] = 200  # bright upper half so the fake client can tell crops apart
    return im


def test_2pass_off_by_default(monkeypatch):
    monkeypatch.delenv("FOL_DETECT_2PASS", raising=False)
    g = VisualObstacleGrounder(_TwoScaleClient())
    dets = g._detect_two_pass(_img(), votes=1)
    assert len(dets) == 1
    assert dets[0]["name" if "name" in dets[0] else "label"] == "stove"


def test_2pass_adds_remapped_upper_detection(monkeypatch):
    monkeypatch.setenv("FOL_DETECT_2PASS", "1")
    g = VisualObstacleGrounder(_TwoScaleClient())
    dets = g._detect_two_pass(_img(), votes=1)
    labels = [d.get("label") or d.get("name") for d in dets]
    assert "moka pot" in labels
    pot = next(d for d in dets if (d.get("label") or d.get("name")) == "moka pot")
    # crop bbox [100,20,160,120] halves back to full-image coords
    assert pot["bbox"] == [50.0, 10.0, 80.0, 60.0]


def test_2pass_dedups_overlapping(monkeypatch):
    monkeypatch.setenv("FOL_DETECT_2PASS", "1")
    g = VisualObstacleGrounder(_TwoScaleClient())
    dets = g._detect_two_pass(_img(), votes=1)
    # upscaled stove [40,40,240,240] -> [20,20,120,120] == full-image stove;
    # IoU 1.0 must dedup, leaving exactly stove + pot
    assert len(dets) == 2
