"""Fast correctness checks (no simulator, CPU). Run inside a job: python checks.py"""
import numpy as np
import torch

import oracle as O
import texts as TX
from eval_sl import naming
from policy import BasePolicy, ResidualFlow, ResidualFlowImg, random_shift
from train_base import build_chunk_index


def test_chunk_index():
    ep = np.array([0, 0, 0, 1, 1, 2])
    idx = build_chunk_index(ep)
    assert idx.shape == (6, 16)
    assert (idx[0, :3] == [0, 1, 2]).all() and (idx[0, 3:] == 2).all()
    assert (idx[3, :2] == [3, 4]).all() and (idx[3, 2:] == 4).all()
    assert (idx[5] == 5).all()


def test_shift():
    x = torch.rand(4, 3, 32, 32)
    y = random_shift(x, pad=4)
    assert y.shape == x.shape
    assert torch.allclose(random_shift(x, pad=0), x)


def test_oracle_clears_box():
    cfg = dict(O.DEFAULT, use_arm=False)
    lo, hi = np.array([0.0, -0.05, 0.90]), np.array([0.10, 0.05, 1.00])
    p0 = np.array([-0.15, 0.0, 0.97])
    A = np.zeros((16, 7)); A[:, 0] = 1.0; A[:, 6] = 1  # straight +x through the box at box height
    offs, rad = np.zeros((1, 3)), np.array([cfg["r_tip"]])
    A2, prof = O.correct_chunk(A, p0, offs, rad, [(lo, hi)], cfg)
    assert np.abs(prof).max() > 0, "oracle did not react to a path through the box"
    path = p0 + np.cumsum(A2[:, :3] * O.GAIN, 0)
    m = np.array([cfg["m_xy"], cfg["m_xy"], cfg["m_z"]])
    for q in path:
        d = np.maximum(np.maximum(lo - m - q, q - hi - m), 0)
        assert np.linalg.norm(d) >= cfg["r_tip"] - 0.012, (q, d)  # 1 step of clipping slack
    assert (A2[:, 3:] == A[:, 3:]).all()
    # a path far from the box is untouched
    A3, prof3 = O.correct_chunk(A, p0 + np.array([0, 0.4, 0]), offs, rad, [(lo, hi)], cfg)
    assert not prof3.any() and (A3 == A).all()


def test_texts_and_naming():
    tr = set(sum((TX.train_texts(o) for o in TX.OBJ), []))
    he = set(sum((TX.held_texts(o) for o in TX.OBJ), []))
    assert not tr & he
    bank = TX.TextBank()
    assert np.allclose(bank(TX.NULL), 0) and abs(np.linalg.norm(bank(TX.train_texts(TX.OBJ[0])[0])) - 1) < 1e-4
    present = ["moka_pot_obstacle_1"]
    assert naming("true", present, 3) == (present, "don't hit the moka pot")
    n, t = naming("wrong", present, 3)
    assert n[0] != present[0] and n[0] in TX.OBJ
    assert naming("none", present, 0) == ([], TX.NULL)
    assert naming("held", present, 1)[1] in TX.held_texts(present[0])


def test_models_forward():
    b = BasePolicy(10, pretrained=False)
    ag = torch.randint(0, 255, (2, 128, 128, 3), dtype=torch.uint8)
    loss = b.loss(ag, ag, torch.zeros(2, 9), torch.tensor([0, 3]), torch.zeros(2, 16, 7))
    x, f = b.sample(ag, ag, torch.zeros(2, 9), torch.tensor([0, 3]), steps=2, return_feats=True)
    assert x.shape == (2, 16, 7) and f.shape == (2, 1152) and torch.isfinite(loss)
    for td in (768, 0):
        r = ResidualFlow(feat_dim=1152, text_dim=td)
        assert torch.isfinite(r.loss(f, x, torch.zeros(2, 768), torch.zeros(2, 16, 3)))
        assert r.sample(f, x, torch.zeros(2, 768), steps=2).shape == (2, 16, 3)

    # image residual: init from the base trunks, map size, guided sampling shape
    for td in (768, 0):
        ri = ResidualFlowImg(feat_dim=1152, text_dim=td)
        ri.init_from_base(b.state_dict())
        assert torch.equal(ri.enc_a.trunk[0].weight, b.enc_a.trunk[0].weight)
        assert ri._map(ri.enc_a, ag, False).shape == (2, 512, 4, 4)
        assert torch.isfinite(ri.loss_img(ag, ag, f, x, torch.randn(2, 768), torch.zeros(2, 16, 3)))
        assert ri.sample_img(ag, ag, f, x, torch.zeros(2, 768), steps=2).shape == (2, 16, 3)
    r = ResidualFlow(feat_dim=1152, text_dim=768)
    assert r.sample(f, x, torch.randn(2, 768), steps=2, guide=2.0).shape == (2, 16, 3)
    ri = ResidualFlowImg(feat_dim=1152, text_dim=768, instr_dim=768)  # instruction read as text
    ins = torch.randn(2, 768)
    assert torch.isfinite(ri.loss_img(ag, ag, f, x, torch.randn(2, 768), torch.zeros(2, 16, 3), instr=ins))
    assert ri.sample_img(ag, ag, f, x, torch.zeros(2, 768), steps=2, instr=ins).shape == (2, 16, 3)



def test_object_lists_aligned():
    import sl_env
    import texts as TX
    assert TX.OBJ == sl_env.OBSTACLES and TX.ORIG6 == sl_env.OBSTACLES[:6]
    assert len(set(TX.NOUN_TRAIN.values())) == len(TX.OBJ) and len(set(TX.NOUN_HELD.values())) == len(TX.OBJ)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
    print("ALL CHECKS PASSED")
