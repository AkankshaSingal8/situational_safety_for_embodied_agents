"""Simulator lookahead shield for residual labels (v2 data).

At every chunk, candidate chunks are rolled forward in the real MuJoCo physics from a saved state, and the
first one that keeps the robot away from the named obstacle is executed. Exact geometry and dynamics, so
arm links, the gripper, the carried object and momentum are all accounted for.

Candidates, in order (the order is the preference, smallest change first):
  1. the base chunk itself              -> the base policy runs unchanged unless it would make contact
  2. geometric detours from oracle.py   (up / side, at increasing margins)
  3. escapes: a straight lift, and a rising retreat opposite to the base's heading
  4. hold: xyz deltas zeroed, rotation and gripper kept
A candidate fails if, at the end of any of its control steps, a robot body (robot0_*, gripper0_*) or the
manipulated object touches the named obstacle or the platform it stands on, or the obstacle has moved more
than 1 mm during the lookahead (the benchmark's collision test, counted from the lookahead's start so that
slow drift of a badly settled obstacle does not veto every candidate). The lookahead runs the whole predicted chunk (H steps), not
only the steps that will be executed. If every candidate fails, hold is executed and the chunk is counted.

Only xyz is ever changed. Episode-level verification (collect_v2.py keeps only safe successes) sits on top.
"""
import json

import numpy as np

import oracle
import sl_env

DEFAULT = dict(
    candidates=[["up", 0.02], ["side", 0.02], ["up", 0.05], ["side", 0.05], ["auto", 0.08], ["up", 0.12], ["side", 0.12]],
    escape=True,            # straight lift and retreat candidates (get out of a dead end instead of holding)
    hold=True, horizon=0,   # horizon 0 = the full chunk
)


class Lookahead:
    """Save / restore the robosuite env around a physics-only rollout (no rendering, no observables)."""

    def __init__(self, env):
        self.renv = getattr(env, "env", env)
        self.n_sub = int(round(self.renv.control_timestep / self.renv.model_timestep))

    def save(self):
        r = self.renv
        return (r.sim.get_state(), [rb.gripper.current_action.copy() for rb in r.robots], r.timestep, r.cur_time)

    def restore(self, s):
        r = self.renv
        r.sim.set_state(s[0]); r.sim.forward()
        for rb, ga in zip(r.robots, s[1]):
            rb.gripper.current_action = ga
        r.timestep, r.cur_time = s[2], s[3]

    def step(self, a):
        r, ps = self.renv, True
        for _ in range(self.n_sub):
            r.sim.forward()
            r._pre_action(np.asarray(a, np.float64), ps)
            r.sim.step()
            ps = False


class ShieldFilter:
    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULT, **(cfg or {}))

    def reset(self, env, obs, named):
        self.named = [sl_env.scene_name(obs, n) or n for n in named if n in sl_env.OBSTACLES_ALL]
        self.named = [n for n in self.named if n + "_pos" in obs and sl_env.on_table(obs[n + "_pos"])]
        self.la = Lookahead(env)
        sim, m = env.sim, env.sim.model
        ooi = list(getattr(env, "ooi", []))
        held = tuple(ooi[:1])
        self.robot_ids = {i for i in range(m.nbody) if (m.body_id2name(i) or "").startswith(("robot0", "gripper0") + held)}
        self.obs_ids, self.watch = set(), {}
        for n in self.named:
            self.obs_ids |= set(sl_env.body_tree_ids(sim, n))
            sup = sl_env.support_of(sim, obs, n)
            if sup:
                self.obs_ids |= set(sl_env.body_tree_ids(sim, sup))
            root = min(sl_env.body_tree_ids(sim, n))  # the object's root body (its _pos observable)
            self.watch[n] = (root, sim.data.body_xpos[root].copy())
        self.grip_cmd = -1.0
        self.stats = dict(n_chunks=0, n_shielded=0, n_hold=0, n_fail=0, choice={})

    # ---------- contact test ----------
    def mark(self, env):
        """Obstacle positions at the start of a lookahead: only motion caused by the candidate counts."""
        self.ref = {bid: env.sim.data.body_xpos[bid].copy() for bid, _ in self.watch.values()}

    def unsafe(self, env):
        d, m = env.sim.data, env.sim.model
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 in self.obs_ids and b2 in self.robot_ids) or (b2 in self.obs_ids and b1 in self.robot_ids):
                return True
        for bid, p0 in self.ref.items():
            if np.sum(np.abs(d.body_xpos[bid] - p0)) > 0.001:
                return True
        return False

    def rollout_ok(self, env, A):
        s = self.la.save()
        self.mark(env)
        ok = True
        try:
            H = self.cfg["horizon"] or len(A)
            for a in A[:H]:
                self.la.step(a)
                if self.unsafe(env):
                    ok = False
                    break
        finally:
            self.la.restore(s)
        return ok

    # ---------- chunk planning ----------
    def candidates(self, env, obs, A):
        yield "base", A
        if not self.named:
            return
        boxes = []
        for n in self.named:
            boxes.append(sl_env.object_aabb(env.sim, n))
            sup = sl_env.support_of(env.sim, obs, n)
            if sup:
                boxes.append(sl_env.object_aabb(env.sim, sup))
        p0 = obs["robot0_eef_pos"].copy()
        for mode, mg in self.cfg["candidates"]:
            ocfg = dict(oracle.DEFAULT, mode=mode, m_xy=mg, m_z=mg)
            offs, rad = oracle.checkpoints(env.sim, p0, ocfg)
            A2, prof = oracle.correct_chunk(A, p0, offs, rad, boxes, ocfg)
            if np.abs(prof).max() > 0:
                yield f"{mode}{int(mg * 100)}", A2
        if self.cfg["escape"]:
            L = A.copy()
            L[:, :2] = 0.0; L[:, 2] = 0.8                       # straight lift
            yield "lift", L
            v = A[:, :2].sum(0)
            n = np.linalg.norm(v)
            R = A.copy()
            R[:, :2] = (-v / n * 0.6) if n > 1e-6 else 0.0      # back away from where the base was heading, rising
            R[:, 2] = 0.4
            yield "retreat", R
        if self.cfg["hold"]:
            H = A.copy()
            H[:, :3] = 0.0
            yield "hold", H

    def plan_chunk(self, env, obs, A):
        self.stats["n_chunks"] += 1
        if not self.named:
            return A.copy()
        last = None
        for name, C in self.candidates(env, obs, A):
            last = (name, C)
            if self.rollout_ok(env, C):
                if name != "base":
                    self.stats["n_shielded"] += 1
                    self.stats["choice"][name] = self.stats["choice"].get(name, 0) + 1
                    self.stats["n_hold"] += name == "hold"
                return C.copy()
        self.stats["n_fail"] += 1
        return last[1].copy()

    def step(self, env, obs, a):
        a = np.asarray(a, np.float64).copy()
        self.grip_cmd = a[6]
        return a


def make_filter(cfg):
    cfg = dict(cfg or {})
    if cfg.pop("type", "filter") == "shield":
        return ShieldFilter(cfg)
    from safety_filter import SafetyFilter
    return SafetyFilter(cfg)


class ShieldController:
    """Base policy + lookahead shield (eval_sl --arm shield --oracle_cfg '<json>')."""

    def __init__(self, cfg):
        from controllers import make_runner
        self.r = make_runner(cfg)
        self.k = cfg["exec_steps"]
        self.f = ShieldFilter(json.loads(cfg["oracle_cfg"]) if cfg.get("oracle_cfg") else None)

    def reset(self, env, obs, lang, named, text):
        self.tid = self.r.task_id(lang)
        self.f.reset(env, obs, named)

    def act_chunk(self, env, obs):
        x, _ = self.r.chunk(obs, self.tid)
        A = self.r.unnorm(x.cpu().numpy())
        return self.f.plan_chunk(env, obs, A)[: self.k]

    def episode_stats(self):
        s = dict(self.f.stats)
        return s
