"""Image flow-matching policy (base) and the text-conditioned residual flow.

Base: two ResNet-18 encoders (agentview, wrist; ImageNet init) + proprio + task embedding
-> condition; a 1-D temporal U-Net with FiLM predicts the rectified-flow velocity of a
16-step action chunk. Same recipe family as Diffusion Policy, with flow matching instead of
DDPM.

Residual: same flow head over the chunk correction dA (xyz dims only), conditioned on the
base policy's frozen features, the base chunk, and a sentence embedding (BGE-base CLS).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def random_shift(x, pad=8):
    """x: (B,3,H,W) float. DrQ-style random shift."""
    B, C, H, W = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    oy = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    rows = (oy.view(B, 1) + torch.arange(H, device=x.device).view(1, H))  # B,H
    cols = (ox.view(B, 1) + torch.arange(W, device=x.device).view(1, W))  # B,W
    x = x[torch.arange(B, device=x.device).view(B, 1, 1, 1), torch.arange(C, device=x.device).view(1, C, 1, 1),
          rows.view(B, 1, H, 1), cols.view(B, 1, 1, W)]
    return x


class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        self.trunk = nn.Sequential(*list(r.children())[:-1])  # -> (B,512,1,1)

    def forward(self, img_u8, aug):
        x = img_u8.permute(0, 3, 1, 2).float() / 255.0
        if aug:
            x = random_shift(x)
        x = (x - IMNET_MEAN.to(x.device)) / IMNET_STD.to(x.device)
        return self.trunk(x).flatten(1)


def sinusoidal(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.view(-1, 1) * 1000 * f.view(1, -1)
    return torch.cat([a.sin(), a.cos()], -1)


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, cond_dim, k=5):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.GroupNorm(8, cout), nn.Mish())
        self.c2 = nn.Sequential(nn.Conv1d(cout, cout, k, padding=k // 2), nn.GroupNorm(8, cout), nn.Mish())
        self.film = nn.Linear(cond_dim, 2 * cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, c):
        h = self.c1(x)
        s, b = self.film(c).unsqueeze(-1).chunk(2, 1)
        h = h * (1 + s) + b
        return self.c2(h) + self.skip(x)


class FlowUNet1D(nn.Module):
    """Velocity field v(x_t, t | cond) over a (H, D) chunk."""

    def __init__(self, act_dim, cond_dim, dims=(128, 256, 512), tdim=128):
        super().__init__()
        self.tmlp = nn.Sequential(nn.Linear(tdim, 256), nn.Mish(), nn.Linear(256, 256))
        self.tdim = tdim
        cd = cond_dim + 256
        self.inp = nn.Conv1d(act_dim, dims[0], 1)
        self.down = nn.ModuleList()
        ch = dims[0]
        for d in dims:
            self.down.append(nn.ModuleList([ResBlock1D(ch, d, cd), ResBlock1D(d, d, cd)]))
            ch = d
        self.mid = nn.ModuleList([ResBlock1D(ch, ch, cd), ResBlock1D(ch, ch, cd)])
        self.up = nn.ModuleList()
        for d in reversed(dims):
            self.up.append(nn.ModuleList([ResBlock1D(ch + d, d, cd), ResBlock1D(d, d, cd)]))
            ch = d
        self.out = nn.Conv1d(ch, act_dim, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, t, cond):
        # x: (B,H,D); keep temporal length fixed (no striding) since H is only 16
        c = torch.cat([cond, self.tmlp(sinusoidal(t, self.tdim))], -1)
        h = self.inp(x.transpose(1, 2))
        skips = []
        for b1, b2 in self.down:
            h = b2(b1(h, c), c); skips.append(h)
        for m in self.mid:
            h = m(h, c)
        for b1, b2 in self.up:
            h = torch.cat([h, skips.pop()], 1)
            h = b2(b1(h, c), c)
        return self.out(h).transpose(1, 2)


class BasePolicy(nn.Module):
    def __init__(self, n_tasks, prop_dim=9, act_dim=7, horizon=16, pretrained=True):
        super().__init__()
        self.enc_a = ImageEncoder(pretrained)
        self.enc_w = ImageEncoder(pretrained)
        self.prop = nn.Sequential(nn.Linear(prop_dim, 64), nn.Mish(), nn.Linear(64, 64))
        self.task = nn.Embedding(n_tasks, 64)
        self.fuse = nn.Sequential(nn.Linear(512 * 2 + 64 + 64, 512), nn.Mish(), nn.Linear(512, 256))
        self.head = FlowUNet1D(act_dim, 256)
        self.H, self.D = horizon, act_dim

    def features(self, agent, wrist, prop_n, task, aug=False):
        return torch.cat([self.enc_a(agent, aug), self.enc_w(wrist, aug), self.prop(prop_n), self.task(task)], -1)

    def cond(self, feats):
        return self.fuse(feats)

    def loss(self, agent, wrist, prop_n, task, act_n):
        c = self.cond(self.features(agent, wrist, prop_n, task, aug=True))
        return flow_loss(self.head, act_n, c)

    @torch.no_grad()
    def sample(self, agent, wrist, prop_n, task, steps=10, return_feats=False):
        feats = self.features(agent, wrist, prop_n, task, aug=False)
        c = self.cond(feats)
        x = euler_sample(self.head, c, (agent.shape[0], self.H, self.D), steps)
        return (x, feats) if return_feats else x


def flow_loss(head, x1, c):
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device)
    xt = (1 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * x1
    return F.mse_loss(head(xt, t, c), x1 - x0)


def euler_sample(head, c, shape, steps):
    x = torch.randn(shape, device=c.device)
    for i in range(steps):
        t = torch.full((shape[0],), i / steps, device=c.device)
        x = x + head(x, t, c) / steps
    return x


class ResidualFlow(nn.Module):
    """p(dA | base features, base chunk, sentence). dA lives on the xyz dims of the chunk.
    text_dim=0 gives the no-text control."""

    def __init__(self, feat_dim=512 * 2 + 64 + 64, horizon=16, act_dim=7, text_dim=768, res_dims=3):
        super().__init__()
        self.text_dim = text_dim
        self.fproj = nn.Sequential(nn.Linear(feat_dim, 256), nn.Mish())
        self.bproj = nn.Sequential(nn.Linear(horizon * act_dim, 256), nn.Mish())
        self.tproj = nn.Sequential(nn.Linear(text_dim, 256), nn.Mish()) if text_dim else None
        cd = 256 * (3 if text_dim else 2)
        self.fuse = nn.Sequential(nn.Linear(cd, 512), nn.Mish(), nn.Linear(512, 256))
        self.head = FlowUNet1D(res_dims, 256, dims=(64, 128, 256))
        self.H, self.R = horizon, res_dims

    def cond(self, feats, base_n, text):
        parts = [self.fproj(feats), self.bproj(base_n.flatten(1))]
        if self.tproj is not None:
            parts.append(self.tproj(text))
        return self.fuse(torch.cat(parts, -1))

    def loss(self, feats, base_n, text, d_n):
        return flow_loss(self.head, d_n, self.cond(feats, base_n, text))

    @torch.no_grad()
    def sample(self, feats, base_n, text, steps=10, guide=1.0):
        """guide != 1: classifier-free guidance on the sentence, v = v_null + w (v_text - v_null),
        with the null sentence (zero vector) as the unconditional branch."""
        c = self.cond(feats, base_n, text)
        if guide == 1.0 or self.tproj is None:
            return euler_sample(self.head, c, (feats.shape[0], self.H, self.R), steps)
        c0 = self.cond(feats, base_n, torch.zeros_like(text))
        x = torch.randn((feats.shape[0], self.H, self.R), device=c.device)
        for i in range(steps):
            t = torch.full((x.shape[0],), i / steps, device=c.device)
            v0 = self.head(x, t, c0)
            x = x + (v0 + guide * (self.head(x, t, c) - v0)) / steps
        return x


class TextSpatialPool(nn.Module):
    """Multi-head attention pooling over a (B,C,h,w) feature map with a sentence-derived query
    (or a learned query when there is no text): lets the residual look at the named object."""

    def __init__(self, cin=512, text_dim=768, heads=4, dh=64):
        super().__init__()
        self.h, self.dh = heads, dh
        self.k = nn.Conv2d(cin, heads * dh, 1)
        self.v = nn.Conv2d(cin, heads * dh, 1)
        self.q = nn.Linear(text_dim, heads * dh) if text_dim else None
        self.q0 = nn.Parameter(torch.zeros(heads * dh))

    def forward(self, fmap, text):
        B = fmap.shape[0]
        k = self.k(fmap).flatten(2).view(B, self.h, self.dh, -1)          # B,h,dh,N
        v = self.v(fmap).flatten(2).view(B, self.h, self.dh, -1)
        q = self.q0.expand(B, -1) + (self.q(text) if self.q is not None else 0)
        q = q.view(B, self.h, self.dh, 1)
        att = torch.softmax((k * q).sum(2) / math.sqrt(self.dh), -1)    # B,h,N
        return (v * att.unsqueeze(2)).sum(-1).flatten(1)                # B,h*dh


class ResidualFlowImg(ResidualFlow):
    """ResidualFlow plus its own trainable image encoders (agentview, wrist; initialised from the
    base policy's trunks), keeping the 4x4 layer-4 map for text-conditioned attention pooling."""

    def __init__(self, feat_dim=1152, horizon=16, act_dim=7, text_dim=768, res_dims=3):
        super().__init__(feat_dim, horizon, act_dim, text_dim, res_dims)
        self.enc_a, self.enc_w = ImageEncoder(False), ImageEncoder(False)
        self.pool_a, self.pool_w = TextSpatialPool(512, text_dim), TextSpatialPool(512, text_dim)
        self.gproj = nn.Sequential(nn.Linear(512 * 2 + 256 * 2, 256), nn.Mish())
        cd = 256 * (4 if text_dim else 3)
        self.fuse = nn.Sequential(nn.Linear(cd, 512), nn.Mish(), nn.Linear(512, 256))

    def init_from_base(self, base_state):
        for name in ("enc_a", "enc_w"):
            sd = {k[len(name) + 1:]: v for k, v in base_state.items() if k.startswith(name + ".")}
            getattr(self, name).load_state_dict(sd)

    def freeze_bn(self):
        """Put encoder BatchNorm layers in eval mode (running stats fixed) and keep them there."""
        self._bn_frozen = True
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "_bn_frozen", False):
            for mod in list(self.enc_a.modules()) + list(self.enc_w.modules()):
                if isinstance(mod, nn.BatchNorm2d):
                    mod.eval()
        return self

    def encoder_params(self):
        return list(self.enc_a.parameters()) + list(self.enc_w.parameters())

    def _map(self, enc, img_u8, aug):
        x = img_u8.permute(0, 3, 1, 2).float() / 255.0
        if aug:
            x = random_shift(x)
        x = (x - IMNET_MEAN.to(x.device)) / IMNET_STD.to(x.device)
        return enc.trunk[:-1](x)                                        # B,512,4,4 at 128 px

    def cond_img(self, agent, wrist, feats, base_n, text, aug=False):
        ma, mw = self._map(self.enc_a, agent, aug), self._map(self.enc_w, wrist, aug)
        t = text if self.text_dim else None
        g = torch.cat([ma.mean((2, 3)), mw.mean((2, 3)), self.pool_a(ma, t), self.pool_w(mw, t)], -1)
        parts = [self.fproj(feats), self.bproj(base_n.flatten(1)), self.gproj(g)]
        if self.tproj is not None:
            parts.append(self.tproj(text))
        return self.fuse(torch.cat(parts, -1))

    def loss_img(self, agent, wrist, feats, base_n, text, d_n):
        return flow_loss(self.head, d_n, self.cond_img(agent, wrist, feats, base_n, text, aug=True))

    @torch.no_grad()
    def sample_img(self, agent, wrist, feats, base_n, text, steps=10):
        c = self.cond_img(agent, wrist, feats, base_n, text)
        return euler_sample(self.head, c, (feats.shape[0], self.H, self.R), steps)
