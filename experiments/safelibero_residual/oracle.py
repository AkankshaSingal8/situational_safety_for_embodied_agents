"""Geometric detour oracle for "don't hit the <object>" (privileged sim state).

Given the base chunk A (H x 7, env units) and the current state, predict the end-effector
path p_j = p0 + g * cumsum(A[:j+1, :3]) with the per-axis gain g fitted on the LIBERO demos
(corr 0.95). The robot is approximated by checkpoints that translate rigidly with the
end-effector over one chunk: fingertips/held bowl (a ring around the eef), the hand and
the wrist links. Each checkpoint is a sphere; the named object is its world AABB inflated
by a margin. Where a checkpoint would enter the inflated box, the eef is displaced either
straight up or sideways away from the box (whichever is smaller, unless mode forces one),
the displacement profile is smoothed into a bump, and the chunk's xyz deltas are adjusted so
the executed path follows it. Rotation and gripper are never touched.

The correction depends on which object the sentence names, so the same scene gives
different targets for different sentences; naming an object that is absent gives zero.
"""
import json

import numpy as np

import sl_env

GAIN = np.array([0.0107, 0.0120, 0.0111])  # m per action unit per step, fitted on the demos
ARM_BODIES = ["robot0_right_hand", "robot0_link7", "robot0_link6", "robot0_link5"]
DEFAULT = dict(m_xy=0.02, m_z=0.02, r_tip=0.03, r_ring=0.055, ring_drop=0.03, r_arm=0.06,
               mode="auto", ramp=3, max_shift=0.15, use_arm=True)


def body_pos(sim, name):
    return sim.data.body_xpos[sim.model.body_name2id(name)].copy()


def checkpoints(sim, p0, cfg):
    """(K,3) offsets from the eef and (K,) radii."""
    offs, rad = [np.zeros(3)], [cfg["r_tip"]]
    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        offs.append(np.array([np.cos(ang) * cfg["r_ring"] * 0.7, np.sin(ang) * cfg["r_ring"] * 0.7, -cfg["ring_drop"]]))
        rad.append(cfg["r_ring"] * 0.3 + 0.01)
    if cfg["use_arm"]:
        for b in ARM_BODIES:
            offs.append(body_pos(sim, b) - p0); rad.append(cfg["r_arm"])
    return np.array(offs), np.array(rad)


def exit_dist(q, u, blo, bhi):
    """Distance along unit u (no zero components needed) for point q to leave box [blo, bhi]."""
    t = np.inf
    for ax in range(3):
        if u[ax] > 1e-9:
            t = min(t, (bhi[ax] - q[ax]) / u[ax])
        elif u[ax] < -1e-9:
            t = min(t, (blo[ax] - q[ax]) / u[ax])
    return max(t, 0.0)


def required_shift(q, r, lo, hi, cfg, perp):
    """Smallest eef shift (3,) taking sphere (q, r) out of the inflated box, or zeros.
    Candidates: straight up, or sideways along +-perp (perpendicular to the direction of motion),
    so the detour goes over or around the object instead of stalling in front of it."""
    m = np.array([cfg["m_xy"], cfg["m_xy"], cfg["m_z"]])
    blo, bhi = lo - m - r, hi + m + r
    if np.any(q <= blo) or np.any(q >= bhi):
        return np.zeros(3)
    up = np.array([0.0, 0.0, 1.0]) * exit_dist(q, np.array([0.0, 0.0, 1.0]), blo, bhi)
    sides = [sgn * perp * exit_dist(q, sgn * perp, blo, bhi) for sgn in (1, -1)]
    side = min(sides, key=np.linalg.norm)
    if cfg["mode"] == "up":
        return up
    if cfg["mode"] == "side":
        return side
    return up if np.linalg.norm(up) <= np.linalg.norm(side) else side


def motion_perp(A, j, center_xy, p_xy):
    """Unit horizontal vector perpendicular to the motion at step j (fallback: away from the box)."""
    v = A[j, :2] * GAIN[:2]
    if np.linalg.norm(v) < 2e-3:
        v = (A[:, :2] * GAIN[:2]).sum(0)
    if np.linalg.norm(v) < 5e-3:
        away = p_xy - center_xy
        n = np.linalg.norm(away)
        return np.array([away[0], away[1], 0.0]) / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
    v = v / np.linalg.norm(v)
    return np.array([-v[1], v[0], 0.0])


def correct_chunk(A, p0, offs, rad, boxes, cfg):
    """A: (H,7) env-unit chunk. Returns (A', shift profile (H,3))."""
    H = len(A)
    path = p0 + np.cumsum(A[:, :3] * GAIN, 0)
    need = np.zeros((H, 3))
    for j in range(H):
        for (lo, hi) in boxes:
            perp = motion_perp(A, j, (lo[:2] + hi[:2]) / 2, path[j][:2])
            for o, r in zip(offs, rad):
                s = required_shift(path[j] + o, r, lo, hi, cfg, perp)
                # combine per axis keeping the largest magnitude
                for ax in range(3):
                    if abs(s[ax]) > abs(need[j, ax]):
                        need[j, ax] = s[ax]
    if not need.any():
        return A.copy(), need
    # smooth into a bump: the shift needed at step k ramps in linearly over enough earlier steps
    # for the clipped action (|a| <= 1, i.e. GAIN m/step) to reach it, and ramps out over `ramp` steps
    prof = np.zeros_like(need)
    R = cfg["ramp"]
    for k in range(H):
        if not need[k].any():
            continue
        n_in = max(R, int(np.ceil(np.max(np.abs(need[k]) / (0.8 * GAIN)))))
        for j in range(max(0, k - n_in), min(H, k + R + 1)):
            w = 1.0 - (k - j) / (n_in + 1) if j <= k else 1.0 - (j - k) / (R + 1)
            for ax in range(3):
                v = need[k, ax] * w
                if abs(v) > abs(prof[j, ax]):
                    prof[j, ax] = v
    prof = np.clip(prof, -cfg["max_shift"], cfg["max_shift"])
    deltas = np.diff(np.vstack([np.zeros(3), prof]), axis=0) / GAIN
    A2 = A.copy()
    A2[:, :3] = np.clip(A[:, :3] + deltas, -1, 1)
    return A2, prof


class Oracle:
    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULT, **(cfg or {}))

    def reset(self, env, named):
        """named: list of object names the sentence asks to avoid (may be absent from the scene)."""
        self.named = [n for n in named if n in sl_env.OBSTACLES]
        self.boxes = None

    def boxes_now(self, env, obs):
        present = [n for n in self.named if n + "_pos" in obs and sl_env.on_table(obs[n + "_pos"])]
        boxes = []
        for n in present:
            boxes.append(sl_env.object_aabb(env.sim, n))
            sup = sl_env.support_of(env.sim, obs, n)  # the level-II platform moves the object if hit
            if sup:
                boxes.append(sl_env.object_aabb(env.sim, sup))
        return boxes

    def correct(self, env, obs, A):
        boxes = self.boxes_now(env, obs)
        if not boxes:
            return A.copy(), np.zeros((len(A), 3))
        p0 = obs["robot0_eef_pos"].copy()
        offs, rad = checkpoints(env.sim, p0, self.cfg)
        return correct_chunk(A, p0, offs, rad, boxes, self.cfg)


class OracleController:
    """Base policy + oracle correction for the named objects, executed closed loop.
    cfg['oracle_cfg'] is a JSON string overriding DEFAULT. Use --text_mode true for the real obstacle."""

    def __init__(self, cfg):
        from controllers import BaseRunner
        self.r = BaseRunner(cfg["ckpt"], cfg["device"], cfg["flow_steps"])
        self.k = cfg["exec_steps"]
        self.o = Oracle(json.loads(cfg["oracle_cfg"]) if cfg.get("oracle_cfg") else None)

    def reset(self, env, obs, lang, named, text):
        self.env, self.tid = env, self.r.task_id(lang)
        self.o.reset(env, named)
        self.n_chunks = self.n_corr = 0
        self.shift = 0.0

    def act_chunk(self, env, obs):
        x, _ = self.r.chunk(obs, self.tid)
        A = self.r.unnorm(x.cpu().numpy())
        A2, prof = self.o.correct(env, obs, A)
        self.n_chunks += 1
        if np.abs(prof).max() > 0:
            self.n_corr += 1; self.shift = max(self.shift, float(np.abs(prof).max()))
        return A2[: self.k]

    def episode_stats(self):
        return dict(n_chunks=self.n_chunks, n_corrected=self.n_corr, max_shift=round(self.shift, 3))
