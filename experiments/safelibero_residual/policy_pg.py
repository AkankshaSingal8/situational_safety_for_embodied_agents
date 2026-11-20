"""Residual flow head over frozen backbone tokens (backbones A / B / C share this head).

  tokens    (T, K)   frozen tokens: C = fused PaliGemma (image + sentence), B = SigLIP-So400m, A = SigLIP-base
  tok_mask  (T,)     valid tokens
  sent      (S,)     sentence embedding for A/B (frozen SigLIP-base text tower); C has none (sentence is in tokens)
  instr     (S,)     instruction embedding for A/B
  state     (8,)     pi0.5 state, standardised
  base      (10,7)   pi0.5 chunk, standardised
Output: flow velocity over the xyz correction (10,3); 1-D U-Net head with zero-initialised output (policy.py).
"""
import torch
import torch.nn as nn

from policy import FlowUNet1D, euler_sample, flow_loss


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, o), nn.GELU(), nn.Linear(o, o))


class ResidualFlowTok(nn.Module):
    def __init__(self, tok_dim, n_tok, text_dim=0, d=256, horizon=10, act_dim=7, state_dim=8, res_dims=3,
                 layers=3, heads=4, dropout=0.1, tok_drop=0.1):
        super().__init__()
        self.H, self.R, self.text_dim, self.tok_drop = horizon, res_dims, text_dim, tok_drop
        self.tok = nn.Sequential(nn.LayerNorm(tok_dim), nn.Linear(tok_dim, d))
        self.pos = nn.Parameter(torch.randn(n_tok, d) * 0.02)
        self.state = mlp(state_dim, d)
        self.base = mlp(horizon * act_dim, d)
        self.cls = nn.Parameter(torch.zeros(d))
        n_small = 3
        if text_dim:
            self.sent = nn.Linear(text_dim, d)
            self.null_sent = nn.Parameter(torch.zeros(d))
            self.instr = nn.Linear(text_dim, d)
            n_small += 2
        self.type_emb = nn.Parameter(torch.randn(n_small, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256))
        self.head = FlowUNet1D(res_dims, 256, dims=(64, 128, 256))

    def cond(self, tokens, tok_mask, state_n, base_n, sent=None, has_sent=None, instr=None):
        B = tokens.shape[0]
        x = self.tok(tokens.float()) + self.pos[: tokens.shape[1]]
        mask = tok_mask.bool()
        if self.training and self.tok_drop > 0:
            mask = mask & (torch.rand(mask.shape, device=mask.device) > self.tok_drop)
        small = [self.cls.expand(B, -1), self.state(state_n), self.base(base_n.flatten(1))]
        if self.text_dim:
            h = has_sent.float().view(B, 1)
            small += [h * self.sent(sent) + (1 - h) * self.null_sent, self.instr(instr)]
        small = torch.stack(small, 1) + self.type_emb
        keep = torch.cat([torch.ones(B, small.shape[1], dtype=torch.bool, device=x.device), mask], 1)
        z = self.enc(torch.cat([small, x], 1), src_key_padding_mask=~keep)
        return self.out(z[:, 0])

    def loss(self, d_n, **kw):
        return flow_loss(self.head, d_n, self.cond(**kw))

    @torch.no_grad()
    def sample(self, steps=10, n=1, **kw):
        c = self.cond(**kw)
        xs = [euler_sample(self.head, c, (c.shape[0], self.H, self.R), steps) for _ in range(n)]
        return torch.stack(xs).mean(0)
