"""Conditional flow-matching model over residual action sequences.

v_theta(x_t, t | s, A_base, E_text(T)) with rectified-flow interpolation
x_t = (1-t) x0 + t x1, x0 ~ N(0, I), x1 = normalized residual.
The empty instruction is a learned null token.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .traj import K

X_DIM = K * 6
S_DIM = 10
TXT_DIM = 768


def t_embed(t, dim=128):
    half = dim // 2
    f = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * f[None] * 1000.0
    return torch.cat([a.sin(), a.cos()], -1)


class Block(nn.Module):
    def __init__(self, h, c):
        super().__init__()
        self.ln = nn.LayerNorm(h, elementwise_affine=False)
        self.film = nn.Linear(c, 2 * h)
        self.ff = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))

    def forward(self, x, c):
        g, b = self.film(c).chunk(2, -1)
        return x + self.ff(self.ln(x) * (1 + g) + b)


class FlowNet(nn.Module):
    def __init__(self, h=1024, n_blocks=6, use_text=True):
        super().__init__()
        self.use_text = use_text
        self.txt = nn.Sequential(nn.Linear(TXT_DIM, 256), nn.SiLU(), nn.Linear(256, 256))
        self.null = nn.Parameter(torch.zeros(256))
        self.tm = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 256))
        self.sm = nn.Sequential(nn.Linear(S_DIM, 128), nn.SiLU(), nn.Linear(128, 128))
        self.bm = nn.Sequential(nn.Linear(X_DIM, 512), nn.SiLU(), nn.Linear(512, 256))
        c = 256 + 256 + 128 + 256
        self.cond = nn.Sequential(nn.Linear(c, 512), nn.SiLU(), nn.Linear(512, 512))
        self.inp = nn.Linear(X_DIM + X_DIM + S_DIM, h)
        self.blocks = nn.ModuleList([Block(h, 512) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, X_DIM))
        nn.init.zeros_(self.out[1].weight); nn.init.zeros_(self.out[1].bias)

    def forward(self, x, t, s, base, txt, is_null):
        """x, base: (B, X_DIM) normalized; s: (B, S_DIM) normalized;
        txt: (B, 768); is_null: (B,) bool."""
        te = self.txt(txt)
        null_mask = is_null[:, None] if self.use_text else torch.ones_like(is_null[:, None])
        te = torch.where(null_mask, self.null.expand_as(te), te)
        c = self.cond(torch.cat([te, self.tm(t_embed(t)), self.sm(s), self.bm(base)], -1))
        h = self.inp(torch.cat([x, base, s], -1))
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(h)


@torch.no_grad()
def sample(model, s, base, conds, n_steps=32, generator=None, weights=None):
    """Euler integration. `conds` is a list of (txt, is_null) pairs.
    One cond = ordinary conditional sample. Several conds = inference-time
    composition  v = v_null + sum_i w_i (v_i - v_null)."""
    B = s.shape[0]
    x = torch.randn(B, X_DIM, device=s.device, generator=generator)
    null_txt = torch.zeros(B, TXT_DIM, device=s.device)
    null_flag = torch.ones(B, dtype=torch.bool, device=s.device)
    weights = weights or [1.0] * len(conds)
    for i in range(n_steps):
        t = torch.full((B,), i / n_steps, device=s.device)
        if len(conds) == 1:
            v = model(x, t, s, base, *conds[0])
        else:
            v0 = model(x, t, s, base, null_txt, null_flag)
            v = v0.clone()
            for w, (tx, nl) in zip(weights, conds):
                v = v + w * (model(x, t, s, base, tx, nl) - v0)
        x = x + v / n_steps
    return x
