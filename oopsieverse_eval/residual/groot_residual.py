"""GR00T-N1.6 residual expert: the OopsieVerse port of ex2/ex8_lora (experiments/safelibero_residual/res_expert.py).

ex8_lora on pi0.5                          -> here, on GR00T-N1.6-3B (the base policy itself)
  frozen pi05_base PaliGemma + rank-16 LoRA  -> frozen Eagle VLM backbone + rank-16 LoRA (peft)
  fine-tuned pi05_base action expert          -> fine-tuned GR00T DiT action head (state/action enc/dec, DiT)
  zero-init cond_proj(base chunk, state)      -> zero-init cond_proj(base xyz chunk) added to the action tokens;
                                                 state enters through GR00T's own state encoder
  flow matching on the xyz residual           -> GR00T's flow matching on the xyz residual (3 of the 128 padded
                                                 action slots; loss masked to those slots)
  prompt "<instruction>. <safety sentence>"   -> same
Residual = executed xyz - GR00T base xyz (env action units), scaled by 1/std of non-zero residuals.
"""
import math

import numpy as np
import torch
import torch.nn as nn

EMB = "ROBOCASA_PANDA_OMRON"
MODEL = "nvidia/GR00T-N1.6-3B"
STATE_KEYS = ("end_effector_position_relative", "end_effector_rotation_relative", "gripper_qpos",
              "base_position", "base_rotation")
STATE_DIMS = (3, 4, 2, 3, 4)
CAMS = ("side_0", "side_1", "wrist_0")
H = 16
GATE_DIM = 3  # padded action slot (flow loss uses only 0:3) read as the correction-needed logit, as res_expert.gate_logit

SENTENCES = {
    "open_single_door": "Be gentle with the microwave door.",
    "turn_on_microwave": "Press the microwave button gently.",
    "counter_to_microwave": "Handle the cup and the microwave gently.",
    "turn_on_stove": "Turn the stove knob gently.",
}


def prompt(instr, sentence=""):
    instr = str(instr).strip()
    return f"{instr}. {sentence}" if sentence else instr


def split_state(s):
    out, i = {}, 0
    for k, d in zip(STATE_KEYS, STATE_DIMS):
        out[k] = np.asarray(s[i:i + d], np.float32)[None]  # (T=1, D)
        i += d
    return out


class GrootResidual(nn.Module):
    def __init__(self, device="cuda", lora_r=16):
        super().__init__()
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        pol = Gr00tPolicy(embodiment_tag=EmbodimentTag[EMB], model_path=MODEL, device=device, strict=False)
        self.processor, self.collate_fn, self.emb_tag = pol.processor, pol.collate_fn, pol.embodiment_tag
        self.lang_key = pol.language_key
        self.model = pol.model.float()
        head = self.model.action_head
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in head.parameters():
            p.requires_grad_(True)
        # LoRA on the backbone's language model (attention + MLP), like openpi's gemma_2b_lora.
        from peft import LoraConfig, get_peft_model
        lm_owner = self.model.backbone.model
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_r, lora_dropout=0.0,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        lm_owner.language_model = get_peft_model(lm_owner.language_model, cfg)
        self.cond_proj = nn.Linear(3, head.input_embedding_dim)
        nn.init.zeros_(self.cond_proj.weight); nn.init.zeros_(self.cond_proj.bias)
        self.to(device)
        self.device = device
        self.scale = 1.0     # residual scale (set from data stats)
        self.b_scale = 1.0   # base xyz scale

    # ── inputs ──────────────────────────────────────────────────────────
    def encode(self, images, states, texts):
        """images: list of {cam: (H,W,3) uint8}; states: list of 16-d arrays; texts: list of str."""
        from gr00t.data.types import MessageType, VLAStepData
        proc = []
        for im, st, tx in zip(images, states, texts):
            sd = VLAStepData(images={f"res256_image_{c}": np.asarray(im[c], np.uint8)[None] for c in CAMS},
                             states=split_state(st), actions={}, text=tx, embodiment=self.emb_tag)
            proc.append(self.processor([{"type": MessageType.EPISODE_STEP.value, "content": sd}]))
        batch = self.collate_fn(proc)
        inputs = batch["inputs"] if "inputs" in batch else batch  # Gr00tPolicy calls model.get_action(**collated)
        return self.model.prepare_input(dict(inputs))  # moves to device / dtype

    def _features(self, bb_in, act_in):
        head = self.model.action_head
        bb_out = head.process_backbone_output(self.model.backbone(bb_in))
        state_feat = head.state_encoder(act_in.state, act_in.embodiment_id)
        return bb_out, state_feat

    def _velocity(self, bb_out, state_feat, emb_id, x, t, base):
        head = self.model.action_head
        t_disc = (t * head.num_timestep_buckets).long()
        a = head.action_encoder(x, t_disc, emb_id) + self.cond_proj(base)
        if head.config.add_pos_embed:
            a = a + head.position_embedding(torch.arange(a.shape[1], device=a.device))[None]
        sa = torch.cat((state_feat, a), dim=1)
        kw = dict(hidden_states=sa, encoder_hidden_states=bb_out.backbone_features,
                  encoder_attention_mask=bb_out.backbone_attention_mask, timestep=t_disc, return_all_hidden_states=True)
        if head.config.use_alternate_vl_dit:
            kw.update(image_mask=bb_out.image_mask, backbone_attention_mask=bb_out.backbone_attention_mask)
        out, _ = head.model(**kw)
        return head.action_decoder(out, emb_id)[:, -x.shape[1]:]

    def _pad(self, xyz):
        x = torch.zeros(xyz.shape[0], xyz.shape[1], self.model.action_head.action_dim, device=self.device)
        x[..., :3] = xyz
        return x

    def gate_logit(self, bb_out, sf, emb_id, base):
        """Correction-needed logit (B,) (port of res_expert.gate_logit): one head pass at t~1 with a zero residual,
        output slot GATE_DIM averaged over the H tokens. No new parameters."""
        x = torch.zeros(base.shape[0], base.shape[1], self.model.action_head.action_dim, device=self.device)
        t = torch.full((base.shape[0],), 0.999, device=self.device)
        return self._velocity(bb_out, sf, emb_id, x, t, base)[..., GATE_DIM].float().mean(-1)

    # ── training ────────────────────────────────────────────────────────
    def loss(self, images, states, texts, base_xyz, target_xyz, gate_w=0.0, gate_balance=True):
        """base_xyz, target_xyz: (B,16,3) env units; target = executed - base.
        + gate_w * BCE(gate_logit, target is non-zero), class-balanced within the batch when gate_balance."""
        bb_in, act_in = self.encode(images, states, texts)
        bb_out, sf = self._features(bb_in, act_in)
        x1 = self._pad(torch.as_tensor(target_xyz, device=self.device).float() * self.scale)
        base = torch.as_tensor(base_xyz, device=self.device).float() * self.b_scale
        noise = torch.randn_like(x1)
        t = self.model.action_head.sample_time(x1.shape[0], device=self.device, dtype=x1.dtype)[:, None, None]
        xt = (1 - t) * noise + t * x1
        v = self._velocity(bb_out, sf, act_in.embodiment_id, xt, t[:, 0, 0], base)
        loss = ((v - (x1 - noise))[..., :3] ** 2).mean()
        if gate_w:
            y = (torch.as_tensor(target_xyz, device=self.device).abs().amax(dim=(1, 2)) > 1e-6).float()
            z = self.gate_logit(bb_out, sf, act_in.embodiment_id, base)
            bce = nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")
            if gate_balance:
                pos, neg = y.sum().clamp(min=1), (1 - y).sum().clamp(min=1)
                gl = 0.5 * ((bce * y).sum() / pos + (bce * (1 - y)).sum() / neg)
            else:
                gl = bce.mean()
            loss = loss + gate_w * gl
            self._last_gate = (torch.sigmoid(z).detach().cpu().numpy(), y.cpu().numpy())
        return loss

    # ── inference ───────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, images, states, texts, base_xyz, steps=None, n=1, with_gate=False):
        """Mean of n Euler samples of the xyz residual (B,16,3) (res_expert.sample n); + gate prob (B,) if with_gate."""
        steps = steps or self.model.action_head.num_inference_timesteps
        bb_in, act_in = self.encode(images, states, texts)
        bb_out, sf = self._features(bb_in, act_in)
        base = torch.as_tensor(base_xyz, device=self.device).float() * self.b_scale
        outs = []
        for _ in range(n):
            x = torch.randn(base.shape[0], base.shape[1], self.model.action_head.action_dim, device=self.device)
            dt = 1.0 / steps
            for k in range(steps):
                t = torch.full((x.shape[0],), k * dt, device=self.device)
                x = x + dt * self._velocity(bb_out, sf, act_in.embodiment_id, x, t, base)
            outs.append(x[..., :3].float())
        r = (sum(outs) / n / self.scale).cpu().numpy()
        if with_gate:
            return r, torch.sigmoid(self.gate_logit(bb_out, sf, act_in.embodiment_id, base)).cpu().numpy()
        return r

    # ── checkpoint (trainable parts only) ───────────────────────────────
    def trainable_state(self):
        return {k: v.detach().cpu() for k, v in self.named_parameters() if v.requires_grad}

    def save(self, path, extra=None):
        torch.save(dict(state=self.trainable_state(), scale=self.scale, b_scale=self.b_scale, extra=extra or {}), path)

    def load(self, path):
        ck = torch.load(path, map_location="cpu")
        missing = self.load_state_dict(ck["state"], strict=False)
        self.scale, self.b_scale = ck["scale"], ck["b_scale"]
        return ck
