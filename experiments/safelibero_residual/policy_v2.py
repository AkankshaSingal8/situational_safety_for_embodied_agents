"""v2 residual flow: frozen SigLIP image tokens + frozen SigLIP text embeddings -> small transformer -> flow head.

Inputs per chunk
  agent, wrist     (49,768) each   frozen SigLIP-base patch tokens (7x7 pooled), cached by cache_feats_v2.py
  sentence         (768,)          frozen SigLIP text embedding of the safety sentence; has_sent=0 -> learned null token
  instruction      (768,)          frozen SigLIP text embedding of the task instruction
  state            (8,)            pi0.5 state (eef pos, axis-angle, gripper qpos), standardised
  base chunk       (10,7)          pi0.5's chunk in env units, standardised
Output: flow velocity over dA (10,3), the xyz correction added to the base chunk (target scaled by `scale`).
Trainable: token projections, a 2-layer transformer over the 103 tokens, and the 1-D U-Net flow head
(zero-initialised output, from policy.py).
"""
import torch
import torch.nn as nn

from policy import FlowUNet1D, euler_sample, flow_loss


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, o), nn.GELU(), nn.Linear(o, o))


class ResidualFlowV2(nn.Module):
    def __init__(self, tok_dim=768, text_dim=768, d=256, horizon=10, act_dim=7, state_dim=8, res_dims=3,
                 n_tok=49, layers=2, heads=4):
        super().__init__()
        self.H, self.R = horizon, res_dims
        self.tok = nn.Sequential(nn.LayerNorm(tok_dim), nn.Linear(tok_dim, d))
        self.pos = nn.Parameter(torch.randn(2, n_tok, d) * 0.02)          # per camera, per patch
        self.sent = nn.Linear(text_dim, d)
        self.null_sent = nn.Parameter(torch.zeros(d))
        self.instr = nn.Linear(text_dim, d)
        self.state = mlp(state_dim, d)
        self.base = mlp(horizon * act_dim, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.type_emb = nn.Parameter(torch.randn(5, d) * 0.02)           # cls, sentence, instruction, state, base
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256))
        self.head = FlowUNet1D(res_dims, 256, dims=(64, 128, 256))

    def cond(self, agent, wrist, sent, has_sent, instr, state_n, base_n):
        B = agent.shape[0]
        img = torch.cat([self.tok(agent.float()) + self.pos[0], self.tok(wrist.float()) + self.pos[1]], 1)
        h = has_sent.float().view(B, 1)
        s = h * self.sent(sent) + (1 - h) * self.null_sent
        small = torch.stack([self.cls.expand(B, -1, -1)[:, 0], s, self.instr(instr), self.state(state_n),
                             self.base(base_n.flatten(1))], 1) + self.type_emb
        x = self.enc(torch.cat([small, img], 1))
        return self.out(x[:, 0])

    def loss(self, d_n, **kw):
        return flow_loss(self.head, d_n, self.cond(**kw))

    @torch.no_grad()
    def sample(self, steps=10, n=1, **kw):
        """Mean of n flow samples (n=1: one sample)."""
        c = self.cond(**kw)
        xs = [euler_sample(self.head, c, (c.shape[0], self.H, self.R), steps) for _ in range(n)]
        return torch.stack(xs).mean(0)
