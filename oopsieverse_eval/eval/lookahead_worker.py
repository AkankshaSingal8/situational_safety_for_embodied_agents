"""Out-of-process DamageSim lookahead (OOPSIE_RESIDUAL.md, 2026-11-25).

Why a separate process: an in-place lookahead on the live env still changed the real episode (shadow TSR 28/30 ->
6/30 even with full mjData/controller restore), and an in-process clone env blacked out the live env's offscreen
camera images (EGL). Here the live env is only READ: its state is serialized and sent to a worker process that
owns its own copy of the seeded scene (no rendering), rolls candidates out there, and returns predicted damage.

Main side: LookaheadClient(task, seed).damage(live_genv, candidates, horizon) -> list[float] (stops at the first
zero-damage candidate unless all=True).
"""
import multiprocessing as mp
import os
import sys
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _mjdata_arrays(d):
    out = {}
    for name in dir(d):
        if name.startswith("_"):
            continue
        try:
            v = getattr(d, name)
        except Exception:
            continue
        if isinstance(v, np.ndarray) and v.flags.writeable:
            out[name] = np.array(v, copy=True)
    out["__time__"] = float(d.time)
    return out


def serialize_live(genv):
    """Read-only snapshot of everything a control step depends on (no writes to the live env)."""
    from damage_shield import DamageLookahead, _controller_objs, _snap
    r = genv.env
    la = DamageLookahead.__new__(DamageLookahead)
    la.r = r
    dmg = []
    for o in la._objs():
        # Health values only. Reading the live evaluators' state (_snap(ev)) changed the live episode
        # (probe "dmg": TSR 13/15 -> 3/15, 2026-11-25); the worker resets its own evaluator tracking instead.
        dmg.append((o.name, dict(o.link_healths), None))
    return dict(
        mj=_mjdata_arrays(r.sim.data._data),
        grips=[DamageLookahead._grips(rb, None) for rb in r.robots],
        timestep=r.timestep, cur_time=r.cur_time,
        ctl=[_snap(o) for o in _controller_objs(r)],
        dmg=dmg,
    )


def _apply(genv, st):
    from damage_shield import DamageLookahead, _controller_objs, _restore_attrs
    r = genv.env
    d = r.sim.data._data
    for k, v in st["mj"].items():
        if k == "__time__":
            continue
        try:
            tgt = getattr(d, k)
            if isinstance(tgt, np.ndarray) and tgt.shape == v.shape and tgt.flags.writeable:
                tgt[...] = v
        except Exception:
            pass
    d.time = st["mj"]["__time__"]
    for rb, ga in zip(r.robots, st["grips"]):
        DamageLookahead._grips(rb, ga)
    r.timestep, r.cur_time = st["timestep"], st["cur_time"]
    for o, sn in zip(_controller_objs(r), st["ctl"]):
        _restore_attrs(o, sn)
    la = DamageLookahead.__new__(DamageLookahead)
    la.r = r
    objs = {o.name: o for o in la._objs()}
    for name, lh, evs in st["dmg"]:
        o = objs.get(name)
        if o is None:
            continue
        o.link_healths.clear(); o.link_healths.update(lh)
        if evs is None:
            o.reset_damage_evaluators()
        else:
            for ev, sn in zip(o.damage_evaluators, evs):
                _restore_attrs(ev, sn)


def _worker(conn, task, seed):
    sys.path.insert(0, _HERE)
    os.environ["MUJOCO_GL"] = "egl"
    try:
        import gymnasium as gym
        from robocasa.utils.gym_utils.gymnasium_groot import create_grootrobocasa_env_class
        from run_groot_stock_damagesim import make_damageable_stock_env
        from damage_shield import DamageLookahead, to_env_vector
        cls = make_damageable_stock_env(task)
        create_grootrobocasa_env_class(cls, "PandaOmron", "panda_omron")
        env = gym.make(f"robocasa_panda_omron/{cls}_PandaOmron_Env", enable_render=False)
        env.unwrapped.env.rng = np.random.default_rng(seed)
        env.reset(seed=seed)
        genv = env.unwrapped
        m = genv.env.sim.model
        conn.send(("ready", dict(nq=int(m.nq), nbody=int(m.nbody), body_pos_sum=float(np.asarray(m.body_pos).sum()))))
        la = DamageLookahead(genv)
        while True:
            msg = conn.recv()
            if msg[0] == "close":
                break
            _, st, cands, H, stop_first, eps = msg
            out = []
            for chunk in cands:
                _apply(genv, st)
                h0 = la.total_health()
                for a in chunk[:H]:
                    v = to_env_vector(genv, a)
                    la.step(v)
                    genv.env._post_action(v)  # fixture state (stove/microwave on) as in a real env step; throwaway env
                d = max(0.0, h0 - la.total_health())
                out.append(d)
                if stop_first and d <= eps:
                    break
            conn.send(("ok", out))
        env.close()
    except Exception:
        conn.send(("error", traceback.format_exc()))


class LookaheadClient:
    def __init__(self, task, seed):
        ctx = mp.get_context("spawn")  # fresh interpreter: no inherited GL / MuJoCo state
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(child, task, seed), daemon=True)
        self.proc.start()
        tag, info = self.conn.recv()
        if tag != "ready":
            raise RuntimeError(f"lookahead worker failed:\n{info}")
        self.info = info

    def matches(self, genv):
        m = genv.env.sim.model
        return (self.info["nq"] == int(m.nq) and self.info["nbody"] == int(m.nbody)
                and abs(self.info["body_pos_sum"] - float(np.asarray(m.body_pos).sum())) < 1e-6)

    def damage(self, genv, cands, H, stop_first=True, eps=1e-6):
        st = serialize_live(genv)
        cands = [[{k: np.asarray(v) for k, v in a.items()} for a in c] for c in cands]
        self.conn.send(("plan", st, cands, H, stop_first, eps))
        tag, out = self.conn.recv()
        if tag != "ok":
            raise RuntimeError(f"lookahead worker error:\n{out}")
        return out

    def close(self):
        try:
            self.conn.send(("close",))
        except Exception:
            pass
        self.proc.join(timeout=30)
        if self.proc.is_alive():
            self.proc.kill()
