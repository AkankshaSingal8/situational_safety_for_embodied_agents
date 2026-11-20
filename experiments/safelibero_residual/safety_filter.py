"""Task-aware safety filter used to produce residual labels (v2 data).

Two layers, both acting on the xyz part of the action only (rotation and gripper are never changed):

1. Predictive detour (per chunk). The geometric detour from oracle.py: predict the end-effector path of
   the base chunk with the demo-fitted gain, find where the robot checkpoints would enter the named
   obstacle's inflated box, and bend the path up or sideways around it. This starts the avoidance early,
   so the arm goes over or around instead of stopping in front of the obstacle.
2. Discrete-time CBF (per step). For every robot checkpoint sphere i and every named-obstacle box,
   h_i = signed distance(sphere centre, box) - radius - margin, and the executed action u must satisfy
   h_i(p + G u) >= (1 - gamma) h_i(p), linearised as (grad h_i * G) . u >= -gamma h_i. The action is the
   closest one to the planned action in a weighted norm (z weighted less, so the QP prefers lifting
   over slowing down), solved exactly by Hildreth's dual coordinate ascent. This catches what the
   open-loop prediction misses (arm links sweeping, drift from the predicted path).

Task awareness. Inside a protected zone the predictive detour is switched off, so the filter never pulls
the gripper away from what the task needs: around the manipulated object (first BDDL obj_of_interest)
until it is grasped, and around the target object/region while the object is carried. The CBF stays on
everywhere, so the named obstacle is still protected inside the zones.
"""
import json

import numpy as np

import oracle
import sl_env

DEFAULT = dict(
    detour=True, mode="auto", margin=0.02,          # predictive detour (oracle.py) and its box margin (m)
    cbf=True, cbf_margin=0.015, gamma=0.2,          # CBF box margin (m) and decay rate
    w_z=0.3, cbf_active=0.15,                       # z weight in the QP metric; constraints within this distance
    zone_r=0.06,                                    # protected-zone radius around manipulated object / target (m)
    r_link=0.05, link_pts=4,                        # arm links as chains of spheres between joint origins
)
ARM_CHAIN = ["robot0_link3", "robot0_link4", "robot0_link5", "robot0_link6", "robot0_link7", "robot0_right_hand"]


def box_sd(q, lo, hi):
    """Signed distance from point q to box [lo, hi] and its gradient (unit vector)."""
    c, h = (lo + hi) / 2, (hi - lo) / 2
    d = np.abs(q - c) - h
    out = np.maximum(d, 0.0)
    n = np.linalg.norm(out)
    if n > 0:
        return n, np.sign(q - c) * out / n
    ax = int(np.argmax(d))
    g = np.zeros(3)
    g[ax] = 1.0 if q[ax] >= c[ax] else -1.0
    return float(d[ax]), g


def qp_halfspaces(a, A, b, w, iters=60):
    """argmin_u (u-a)^T diag(w) (u-a)  s.t.  A u >= b  (Hildreth's method; exact for halfspaces)."""
    winv = 1.0 / w
    u, lam = a.copy(), np.zeros(len(b))
    for _ in range(iters):
        moved = 0.0
        for i in range(len(b)):
            ai = A[i]
            den = ai @ (winv * ai)
            if den < 1e-12:
                continue
            dl = max(-lam[i], (b[i] - ai @ u) / den)
            if dl != 0.0:
                lam[i] += dl
                u = u + dl * winv * ai
                moved = max(moved, abs(dl))
        if moved < 1e-7:
            break
    return u


class SafetyFilter:
    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULT, **(cfg or {}))
        c = self.cfg
        self.ocfg = dict(oracle.DEFAULT, m_xy=c["margin"], m_z=c["margin"], mode=c["mode"])

    # ---------- episode state ----------
    def reset(self, env, obs, named):
        self.named = [n for n in named if n in sl_env.OBSTACLES_ALL]
        ooi = list(getattr(env, "ooi", []))
        self.manip = ooi[0] if ooi else None
        self.targets = ooi[1:]
        self.manip_z0 = obs[self.manip + "_pos"][2] if self.manip and self.manip + "_pos" in obs else None
        self.grip_cmd = -1.0
        self.stats = dict(n_steps=0, n_cbf=0, cbf_max=0.0, n_chunks=0, n_detour=0, n_zone=0, detour_max=0.0)

    def boxes(self, env, obs):
        out = []
        for n in self.named:
            sn = sl_env.scene_name(obs, n)
            if sn and sl_env.on_table(obs[sn + "_pos"]):
                out.append(sl_env.object_aabb(env.sim, sn))
                sup = sl_env.support_of(env.sim, obs, sn)
                if sup:
                    out.append(sl_env.object_aabb(env.sim, sup))
        return out

    def _anchor(self, env, obs, name):
        if name + "_pos" in obs:
            return obs[name + "_pos"]
        for cand in (name, name.rsplit("_", 2)[0], name.rsplit("_", 1)[0]):
            ids = sl_env.body_tree_ids(env.sim, cand)
            if ids:
                return env.sim.data.body_xpos[ids[0]].copy()
        return None

    def robot_points(self, env, obs, p0):
        """(K,3) offsets from the eef and (K,) radii: fingertips + ring (oracle.py), spheres along the arm links
        (link3 -> hand), and the carried object while it is held."""
        offs, rad = oracle.checkpoints(env.sim, p0, dict(self.ocfg, use_arm=False))
        offs, rad = list(offs), list(rad)
        pts = [oracle.body_pos(env.sim, b) for b in ARM_CHAIN]
        for j, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            for s in np.linspace(0.0, 1.0, self.cfg["link_pts"])[(1 if j else 0):]:
                offs.append(a + (b - a) * s - p0); rad.append(self.cfg["r_link"])
        if self.holding(obs):
            lo, hi = sl_env.object_aabb(env.sim, self.manip)
            offs.append((lo + hi) / 2 - p0); rad.append(0.5 * float(np.max(hi - lo)) + 0.005)
        return np.array(offs), np.array(rad)

    def holding(self, obs):
        if self.manip_z0 is None or self.grip_cmd <= 0:
            return False
        p = obs[self.manip + "_pos"]
        return np.linalg.norm(p - obs["robot0_eef_pos"]) < 0.10 and p[2] > self.manip_z0 + 0.01

    def in_zone(self, env, obs):
        eef = obs["robot0_eef_pos"]
        if self.holding(obs):
            pts = [self._anchor(env, obs, t) for t in self.targets]
            d = min([np.linalg.norm((eef - p)[:2]) for p in pts if p is not None], default=np.inf)
        else:
            p = self._anchor(env, obs, self.manip) if self.manip else None
            d = np.linalg.norm(eef - p) if p is not None else np.inf
        return d < self.cfg["zone_r"]

    # ---------- layer 1: per chunk ----------
    def plan_chunk(self, env, obs, A):
        """A: (H,7) env-unit base chunk -> planned chunk (detoured unless in a protected zone)."""
        self.stats["n_chunks"] += 1
        if not self.cfg["detour"]:
            return A.copy()
        if self.in_zone(env, obs):
            self.stats["n_zone"] += 1
            return A.copy()
        boxes = self.boxes(env, obs)
        if not boxes:
            return A.copy()
        p0 = obs["robot0_eef_pos"].copy()
        offs, rad = self.robot_points(env, obs, p0)
        A2, prof = oracle.correct_chunk(A, p0, offs, rad, boxes, self.ocfg)
        if np.abs(prof).max() > 0:
            self.stats["n_detour"] += 1
            self.stats["detour_max"] = max(self.stats["detour_max"], float(np.abs(prof).max()))
        return A2

    # ---------- layer 2: per step ----------
    def step(self, env, obs, a):
        """a: (7,) planned action -> executed action."""
        self.stats["n_steps"] += 1
        a = np.asarray(a, np.float64).copy()
        self.grip_cmd = a[6]
        if not self.cfg["cbf"]:
            return a
        boxes = self.boxes(env, obs)
        if not boxes:
            return a
        c = self.cfg
        p0 = obs["robot0_eef_pos"].copy()
        offs, rad = self.robot_points(env, obs, p0)
        rows, rhs = [], []
        for lo, hi in boxes:
            for o, r in zip(offs, rad):
                sd, g = box_sd(p0 + o, lo, hi)
                h = sd - r - c["cbf_margin"]
                if h < c["cbf_active"]:
                    rows.append(g * oracle.GAIN); rhs.append(-c["gamma"] * h)
        if not rows:
            return a
        A, b = np.array(rows), np.array(rhs)
        u0 = a[:3]
        if np.all(A @ u0 >= b - 1e-9):
            return a
        u = np.clip(qp_halfspaces(u0, A, b, np.array([1.0, 1.0, c["w_z"]])), -1.0, 1.0)
        self.stats["n_cbf"] += 1
        self.stats["cbf_max"] = max(self.stats["cbf_max"], float(np.abs(u - u0).max()))
        a[:3] = u
        return a


class FilterController:
    """Base policy + SafetyFilter for the named obstacle (eval_sl --arm filter --oracle_cfg '<json>')."""

    def __init__(self, cfg):
        from controllers import make_runner
        self.r = make_runner(cfg)
        self.k = cfg["exec_steps"]
        self.f = SafetyFilter(json.loads(cfg["oracle_cfg"]) if cfg.get("oracle_cfg") else None)

    def reset(self, env, obs, lang, named, text):
        self.tid = self.r.task_id(lang)
        self.f.reset(env, obs, named)

    def act_chunk(self, env, obs):
        x, _ = self.r.chunk(obs, self.tid)
        A = self.r.unnorm(x.cpu().numpy())
        return self.f.plan_chunk(env, obs, A)[: self.k]

    def step_filter(self, env, obs, a):
        return self.f.step(env, obs, a)

    def episode_stats(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.f.stats.items()}
