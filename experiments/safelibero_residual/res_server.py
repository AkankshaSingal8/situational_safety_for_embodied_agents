"""Serve the residual action expert (train_expert.py checkpoint) to the eval controller (openpi env).

    python res_server.py --port 8200 --ckpt runs_v2/expert/ex1/best.pkl
Request:  {"agent": (224,224,3) uint8, "wrist": (224,224,3) uint8 (raw env orientation, INTER_AREA 224 as collected),
           "prompt": str ("<instruction>. <sentence>" or the instruction), "state": (8,), "base": (H,7) env units}
Response: {"delta": (H,3) xyz residual in env units}
"""
import argparse
import logging
import pickle

import numpy as np

import res_expert as RE


class ResidualPolicy:
    def __init__(self, ckpt, steps=10, n=1, seed=0, gate_thr=0.0, gate_ckpt=""):
        self.gate_thr = gate_thr
        import jax
        from flax import nnx
        from train_expert import to_model_inputs, unflat
        ck = pickle.load(open(ckpt, "rb"))
        self.stats = ck["stats"]
        self.motion = bool(ck.get("args", {}).get("motion"))
        graphdef, frozen, train_state, rest = RE.build(load_bf16=True, lora=bool(ck.get("args", {}).get("lora")),
                                                       cond_dim=RE.COND + (RE.MOTION if self.motion else 0))
        # trained weights back in fp32 as in training (unflat casts to the template dtype)
        train_state = unflat(jax.tree.map(lambda v: v.astype("float32"), train_state), ck["train_state"])
        self.tok = RE.Tokenizer()
        self.to_inputs = to_model_inputs
        self.rng = jax.random.key(seed)

        def fwd(ts, fz, rs, x, rng):
            m = nnx.merge(graphdef, fz, ts, rs)
            return RE.sample(m, rng, x["agent"], x["wrist"], x["toks"], x["tmask"], x["cond"], steps=steps, n=n, with_gate=True)

        self._fwd = jax.jit(fwd)
        self._args = (train_state, frozen, rest)
        self.jax = jax
        self._gate = None
        if gate_ckpt:   # gate from a second checkpoint; the frozen backbone is shared, only the expert differs
            ck2 = pickle.load(open(gate_ckpt, "rb"))
            self.stats2 = ck2["stats"]
            ts2 = unflat(jax.tree.map(lambda v: v.astype("float32"), train_state), ck2["train_state"])
            ts2 = jax.tree.map(lambda v: v.astype("bfloat16"), ts2)   # bf16 to fit next to pi0.5 on one GPU

            def gfwd(ts, fz, rs, x):
                m = nnx.merge(graphdef, fz, ts, rs)
                kv, pm = RE.prefix_kv(m, x["agent"], x["wrist"], x["toks"], x["tmask"])
                return jax.nn.sigmoid(RE.gate_logit(m, kv, pm, x["cond"]))

            self._gate = jax.jit(gfwd)
            self._gargs = (ts2, frozen, rest)

    def infer(self, obs):
        b = dict(agent=obs["agent"][None], wrist=obs["wrist"][None], state=np.asarray(obs["state"], np.float32)[None],
                 base=np.asarray(obs["base"], np.float32)[None], tgt=np.zeros((1, RE.H, 3), np.float32),
                 prompt=[str(obs["prompt"])])
        if self.motion:   # state change since the previous chunk (the client sends prev_state; 0 at the first chunk)
            b["dstate"] = b["state"] - np.asarray(obs.get("prev_state", obs["state"]), np.float32)[None]
        x = self.to_inputs(b, self.stats, self.tok)
        x.pop("tgt")
        self.rng, k = self.jax.random.split(self.rng)
        d, g = self._fwd(*self._args, x, k)
        d, g = np.asarray(d)[0] / float(self.stats["scale"]), float(np.asarray(g)[0])
        if self._gate is not None:
            x2 = self.to_inputs(b, self.stats2, self.tok)
            x2.pop("tgt")
            g = float(np.asarray(self._gate(*self._gargs, x2))[0])
        if g < self.gate_thr:   # gate says no correction needed: run pi0.5's chunk unchanged
            d = np.zeros_like(d)
        return {"delta": d.astype(np.float32), "gate": g}

    def reset(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--gate_thr", type=float, default=0.0, help="zero the residual when the correction-needed gate "
                    "probability is below this (needs a checkpoint trained with train_expert --gate_w; 0 = off)")
    ap.add_argument("--gate_ckpt", default="", help="take the gate probability from this checkpoint instead of --ckpt")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    pol = ResidualPolicy(a.ckpt, a.steps, a.n, gate_thr=a.gate_thr, gate_ckpt=a.gate_ckpt)
    z = np.zeros((224, 224, 3), np.uint8)
    pol.infer({"agent": z, "wrist": z, "prompt": "warm up", "state": np.zeros(8), "base": np.zeros((RE.H, 7))})
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(pol, host="0.0.0.0", port=a.port).serve_forever()


if __name__ == "__main__":
    main()
