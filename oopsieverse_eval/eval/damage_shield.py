"""Simulator lookahead damage shield for GR00T on RoboCasa + DamageSim (OopsieVerse port of
experiments/safelibero_residual/shield.py).

At every chunk, candidate chunks are rolled forward in the real MuJoCo physics from a saved state, with
DamageSim's health update after every control step, and the first candidate that causes no damage event is
executed. A damage event = any tracked link (robot "agent" or the task's appliance) loses health.

Candidates, in order (smallest change first), xyz only as in shield.py:
  1. the base chunk itself
  2. slow-downs: xyz deltas scaled by 0.75, 0.5, 0.3
  3. lift: xyz replaced by a small +z
  4. hold: xyz zeroed (rotation, gripper, base unchanged)
The lookahead runs the whole predicted chunk (16 steps). If every candidate fails, hold is executed.
Physics, controller-visible state, gripper action, timestep and DamageSim health/evaluator state are all
restored after each lookahead, so the real episode is unaffected.
"""
import copy

import numpy as np

XYZ = "action.end_effector_position"
DEFAULT = dict(scales=[0.75, 0.5, 0.3], lift=0.3, hold=True, eps=1e-6, horizon=0, fallback="hold")
# v2 (OOPSIE_RESIDUAL.md deviation 1): horizon=8 (executed steps), fallback="min_damage"


def to_env_vector(genv, a):
    """GR00T action dict (one step, flat 'action.*' keys) -> robosuite action vector, exactly as
    GrootRoboCasaEnv.step + RoboCasaEnv.step do it (key_converter.unmap_action, then part split)."""
    from robosuite.controllers.composite.composite_controller import HybridMobileBase
    ad = genv.key_converter.unmap_action(dict(a))
    out = []
    for robot in genv.env.robots:
        cc = robot.composite_controller
        pf = robot.robot_model.naming_prefix
        v = np.zeros(cc.action_limits[0].shape)
        for part in cc.part_controllers:
            s, e = cc._action_split_indexes[part]
            v[s:e] = ad[f"{pf}{part}"]
        if isinstance(cc, HybridMobileBase):
            v[-1] = ad[f"{pf}base_mode"]
        out.append(v)
    return np.concatenate(out)


def _is_plain(v):
    if v is None or isinstance(v, (bool, int, float, str, np.ndarray, np.generic)):
        return True
    if isinstance(v, (list, tuple)):
        return all(_is_plain(x) for x in v)
    if isinstance(v, dict):
        return all(_is_plain(x) for x in v.values())
    return False


def _snap(obj):
    return {k: copy.deepcopy(v) for k, v in vars(obj).items() if _is_plain(v)}


def _restore_attrs(obj, snap):
    """Restore only attributes that changed, writing arrays back IN PLACE. Rebinding an attribute to a fresh
    copy breaks aliasing (robosuite shares arrays between robot / controllers / grippers and updates them in
    place), which corrupted the real episode even with a zero-step lookahead (diag variant B, 2026-11-24)."""
    for k, v in snap.items():
        cur = getattr(obj, k, None)
        if _same(cur, v):
            continue
        if isinstance(cur, np.ndarray) and isinstance(v, np.ndarray) and cur.shape == v.shape and cur.flags.writeable:
            cur[...] = v
        else:
            setattr(obj, k, copy.deepcopy(v))


def _controller_objs(r):
    """Every stateful robot-side object a control step can mutate: robots, composite controllers, part
    controllers (arm OSC, torso, base, gripper -- the GRIP controller accumulates its command), gripper models."""
    objs = []
    for rb in r.robots:
        objs.append(rb)
        cc = getattr(rb, "composite_controller", None)
        if cc is not None:
            objs.append(cc)
            objs.extend(getattr(cc, "part_controllers", {}).values())
        g = rb.gripper
        objs.extend(g.values() if isinstance(g, dict) else [g])
    return objs


def _same(a, b):
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return np.array_equal(np.asarray(a), np.asarray(b))
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
        return a == b
    except Exception:
        return repr(a) == repr(b)


def _fingerprint(r):
    """Debug: every plain attribute of the env, robots, controllers, fixtures, damageable objects + sim.data."""
    fp = {}
    d = r.sim.data
    for f in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat", "xfrc_applied", "qfrc_applied", "time"):
        fp[f"data.{f}"] = np.array(getattr(d, f), copy=True)
    groups = [("env", r)] + [(f"ctl{i}.{type(o).__name__}", o) for i, o in enumerate(_controller_objs(r))]
    groups += [(f"fx.{k}", f) for k, f in getattr(r, "fixtures", {}).items()]
    from damagesim.core.damageable_mixin import DamageableMixin
    groups += [(f"dmg.{o.name}", o) for o in r._get_all_objects() if isinstance(o, DamageableMixin)]
    for g, o in groups:
        for k, v in vars(o).items():
            if _is_plain(v):
                fp[f"{g}.{k}"] = copy.deepcopy(v)
    return fp


def _copy_mjdata_into(dst, src):
    """In-place mjData restore that works on old and new mujoco bindings (no mj_copyData needed):
    every writable ndarray field, then time."""
    for name in dir(src):
        if name.startswith("_"):
            continue
        try:
            v = getattr(src, name)
        except Exception:
            continue
        if isinstance(v, np.ndarray):
            try:
                tgt = getattr(dst, name)
                if isinstance(tgt, np.ndarray) and tgt.shape == v.shape and tgt.flags.writeable:
                    tgt[...] = v
            except Exception:
                pass
    dst.time = src.time


class DamageLookahead:
    def __init__(self, genv):
        self.genv = genv
        self.r = genv.env  # damageable robosuite env
        self.n_sub = int(round(self.r.control_timestep / self.r.model_timestep))

    def _objs(self):
        from damagesim.core.damageable_mixin import DamageableMixin
        return [o for o in self.r._get_all_objects() if isinstance(o, DamageableMixin) and o.track_damage]

    def save(self):
        r = self.r
        dmg = []
        for o in self._objs():
            evs = [{k: copy.deepcopy(v) for k, v in ev.__dict__.items() if k != "entity"} for ev in o.damage_evaluators]
            dmg.append((dict(o.link_healths), evs))
        ctl = [(o, _snap(o)) for o in _controller_objs(r)]
        # Full MuJoCo data copy: sim.get_state() omits ctrl, qacc_warmstart (contact-solver warm start), etc.,
        # which leaked from the lookahead into the real episode (shadow diagnostic, 2026-11-24).
        full = copy.copy(r.sim.data._data)  # MjData.__copy__: a full, independent copy of mjData
        return (r.sim.get_state(), [self._grips(rb, None) for rb in r.robots], r.timestep, r.cur_time, dmg, ctl, full)

    @staticmethod
    def _grips(rb, restore):
        """robosuite >= 1.5: robot.gripper is {arm: gripper}; older: a single gripper."""
        gs = rb.gripper if isinstance(rb.gripper, dict) else {"_": rb.gripper}
        if restore is None:
            return {k: (None if g.current_action is None else np.array(g.current_action, copy=True))
                    for k, g in gs.items() if hasattr(g, "current_action")}
        for k, ga in restore.items():
            gs[k].current_action = ga

    def restore(self, s):
        r = self.r
        _copy_mjdata_into(r.sim.data._data, s[6])
        for rb, ga in zip(r.robots, s[1]):
            self._grips(rb, ga)
        r.timestep, r.cur_time = s[2], s[3]
        for o, st in s[5]:
            _restore_attrs(o, st)
        for o, (lh, evs) in zip(self._objs(), s[4]):
            o.link_healths.clear(); o.link_healths.update(lh)
            for ev, st in zip(o.damage_evaluators, evs):
                _restore_attrs(ev, st)

    def total_health(self):
        return sum(sum(o.link_healths.values()) for o in self._objs())

    def step(self, v, health=True):
        r, ps = self.r, True
        for _ in range(self.n_sub):
            r.sim.forward()
            r._pre_action(np.asarray(v, np.float64), ps)
            r.sim.step()
            ps = False
        if health:
            r._update_all_health()


class CloneLookahead(DamageLookahead):
    """Lookahead on a second env instance with the same seeded scene. Before every candidate rollout the clone
    is synced from the live env (full mjData, gripper/controller plain state, DamageSim healths + evaluator
    state); the live env is never stepped or restored, so the lookahead cannot leak into the real episode
    (in-place restore on the live env still cut shadow TSR 28/30 -> 6/30, 2026-11-25)."""

    def __init__(self, live_genv, clone_genv):
        super().__init__(clone_genv)
        self.live = live_genv.env

    def _live_objs(self):
        from damagesim.core.damageable_mixin import DamageableMixin
        return [o for o in self.live._get_all_objects() if isinstance(o, DamageableMixin) and o.track_damage]

    def save(self):
        L, C = self.live, self.r
        _copy_mjdata_into(C.sim.data._data, L.sim.data._data)
        for lrb, crb in zip(L.robots, C.robots):
            self._grips(crb, self._grips(lrb, None))
        C.timestep, C.cur_time = L.timestep, L.cur_time
        for lo, co in zip(_controller_objs(L), _controller_objs(C)):
            _restore_attrs(co, _snap(lo))
        for lo, co in zip(self._live_objs(), self._objs()):
            co.link_healths.clear(); co.link_healths.update(lo.link_healths)
            for lev, cev in zip(lo.damage_evaluators, co.damage_evaluators):
                _restore_attrs(cev, {k: v for k, v in _snap(lev).items() if k != "entity"})
        return None

    def restore(self, s):
        self.save()


class DamageShield:
    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULT, **(cfg or {}))
        self.stats = dict(chunks=0, flagged=0, replaced=0, held=0, chosen={})

    def rollout_damage(self, la, genv, chunk):
        """Total health lost over the lookahead (0 = no damage event)."""
        s = la.save()
        h0 = la.total_health()
        H = self.cfg["horizon"] or len(chunk)
        if "la_steps" in self.cfg:  # diagnostic override (0 = save/restore only)
            H = self.cfg["la_steps"]
        try:
            for a in chunk[:H]:
                la.step(to_env_vector(la.genv, a), health=not self.cfg.get("no_health"))
            return max(0.0, h0 - la.total_health())
        finally:
            la.restore(s)

    def candidates(self, chunk, resample=None):
        yield "base", chunk
        if resample is not None:  # v3 (deviation 2): fresh policy samples for the same observation
            for k in range(self.cfg.get("resamples", 0)):
                rs = resample()  # xyz-only, like every other candidate: base chunk with the resample's xyz
                yield f"resample{k}", [dict(a, **{XYZ: np.asarray(b[XYZ])}) for a, b in zip(chunk, rs)]
        for sc in self.cfg["scales"]:
            yield f"scale{sc}", [dict(a, **{XYZ: np.asarray(a[XYZ]) * sc}) for a in chunk]
        if self.cfg["lift"]:
            yield "lift", [dict(a, **{XYZ: np.array([0.0, 0.0, self.cfg["lift"]])}) for a in chunk]
        if self.cfg["hold"]:
            yield "hold", [dict(a, **{XYZ: np.zeros(3)}) for a in chunk]

    def plan_worker(self, genv, chunk, worker, resample=None):
        """Out-of-process lookahead (lookahead_worker.py): the live env is only read."""
        H = self.cfg["horizon"] or len(chunk)
        eps = self.cfg["eps"]
        self.stats["chunks"] += 1
        d0 = worker.damage(genv, [chunk], H)[0]
        if self.cfg.get("shadow"):
            k = "shadow_dmg" if d0 > eps else "shadow_ok"
            self.stats["chosen"][k] = self.stats["chosen"].get(k, 0) + 1
            return "base", chunk
        if d0 <= eps:
            self.stats["chosen"]["base"] = self.stats["chosen"].get("base", 0) + 1
            return "base", chunk
        cands = [(n, c) for n, c in self.candidates(chunk, resample) if n != "base"]
        ds = worker.damage(genv, [c for _, c in cands], H, stop_first=True, eps=eps)
        self.stats["flagged"] += 1
        for (n, c), d in zip(cands, ds):
            if d <= eps:
                self.stats["replaced"] += 1
                self.stats["chosen"][n] = self.stats["chosen"].get(n, 0) + 1
                return n, c
        if self.cfg["fallback"] == "min_damage":
            allc = [("base", chunk, d0)] + [(n, c, d) for (n, c), d in zip(cands, ds)]
            n, c, _ = min(allc, key=lambda x: x[2])
            self.stats["chosen"][f"min_{n}"] = self.stats["chosen"].get(f"min_{n}", 0) + 1
            return f"min_{n}", c
        self.stats["held"] += 1
        self.stats["chosen"]["hold_fail"] = self.stats["chosen"].get("hold_fail", 0) + 1
        return "hold_fail", cands[-1][1]

    def plan(self, genv, chunk, resample=None, clone=None):
        """chunk: list of per-step GR00T action dicts (full predicted horizon). Returns (name, chunk)."""
        if self.cfg.get("noop"):  # diagnostic: shield code path, no lookahead object at all
            return "base", chunk
        probe = self.cfg.get("probe")  # diagnostic: one read-only access to the live env, then the base chunk
        if probe:
            r = genv.env
            if probe == "mj":
                from lookahead_worker import _mjdata_arrays
                _mjdata_arrays(r.sim.data._data)
            elif probe == "ctl":
                [_snap(o) for o in _controller_objs(r)]
            elif probe == "grip":
                [DamageLookahead._grips(rb, None) for rb in r.robots]
            elif probe == "dmg_ev":
                la0 = DamageLookahead.__new__(DamageLookahead); la0.r = r
                [[_snap(ev) for ev in o.damage_evaluators] for o in la0._objs()]
            elif probe == "dmg_lh":
                la0 = DamageLookahead.__new__(DamageLookahead); la0.r = r
                [dict(o.link_healths) for o in la0._objs()]
            elif probe == "dmg":
                la0 = DamageLookahead.__new__(DamageLookahead); la0.r = r
                [(dict(o.link_healths), [_snap(ev) for ev in o.damage_evaluators]) for o in la0._objs()]
            elif probe == "ser":
                from lookahead_worker import serialize_live
                serialize_live(genv)
            return "base", chunk
        la = CloneLookahead(genv, clone) if clone is not None else DamageLookahead(genv)
        self.stats["chunks"] += 1
        if self.cfg.get("shadow"):  # diagnostic: run the lookahead, always execute the base chunk
            fp0 = _fingerprint(la.r) if self.cfg.get("debug") else None
            d = self.rollout_damage(la, genv, chunk)
            if fp0 is not None:
                fp1 = _fingerprint(la.r)
                bad = sorted(k for k in set(fp0) | set(fp1) if not _same(fp0.get(k), fp1.get(k)))
                if bad:
                    self.stats.setdefault("leaks", {})
                    for k in bad:
                        self.stats["leaks"][k] = self.stats["leaks"].get(k, 0) + 1
            self.stats["chosen"]["shadow_dmg" if d > self.cfg["eps"] else "shadow_ok"] = \
                self.stats["chosen"].get("shadow_dmg" if d > self.cfg["eps"] else "shadow_ok", 0) + 1
            return "base", chunk
        last, best = None, None
        for name, cand in self.candidates(chunk, resample):
            last = (name, cand)
            d = self.rollout_damage(la, genv, cand)
            if d <= self.cfg["eps"]:
                if name != "base":
                    self.stats["flagged"] += 1; self.stats["replaced"] += 1
                self.stats["chosen"][name] = self.stats["chosen"].get(name, 0) + 1
                return name, cand
            if best is None or d < best[0] - self.cfg["eps"]:
                best = (d, name, cand)
        self.stats["flagged"] += 1
        if self.cfg["fallback"] == "min_damage":
            key = f"min_{best[1]}"
            self.stats["chosen"][key] = self.stats["chosen"].get(key, 0) + 1
            return key, best[2]
        self.stats["held"] += 1
        self.stats["chosen"]["hold_fail"] = self.stats["chosen"].get("hold_fail", 0) + 1
        return "hold_fail", last[1]
