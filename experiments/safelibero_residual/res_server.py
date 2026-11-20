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
    def __init__(self, ckpt, steps=10, n=1, seed=0):
        import jax
        from flax import nnx
        from train_expert import to_model_inputs, unflat
        ck = pickle.load(open(ckpt, "rb"))
        self.stats = ck["stats"]
        graphdef, frozen, train_state, rest = RE.build(load_bf16=True)
        # trained weights back in fp32 as in training (unflat casts to the template dtype)
        train_state = unflat(jax.tree.map(lambda v: v.astype("float32"), train_state), ck["train_state"])
        self.tok = RE.Tokenizer()
        self.to_inputs = to_model_inputs
        self.rng = jax.random.key(seed)

        def fwd(ts, fz, rs, x, rng):
            m = nnx.merge(graphdef, fz, ts, rs)
            return RE.sample(m, rng, x["agent"], x["wrist"], x["toks"], x["tmask"], x["cond"], steps=steps, n=n)

        self._fwd = jax.jit(fwd)
        self._args = (train_state, frozen, rest)
        self.jax = jax

    def infer(self, obs):
        b = dict(agent=obs["agent"][None], wrist=obs["wrist"][None], state=np.asarray(obs["state"], np.float32)[None],
                 base=np.asarray(obs["base"], np.float32)[None], tgt=np.zeros((1, RE.H, 3), np.float32),
                 prompt=[str(obs["prompt"])])
        x = self.to_inputs(b, self.stats, self.tok)
        x.pop("tgt")
        self.rng, k = self.jax.random.split(self.rng)
        d = np.asarray(self._fwd(*self._args, x, k))[0] / float(self.stats["scale"])
        return {"delta": d.astype(np.float32)}

    def reset(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--n", type=int, default=1)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    pol = ResidualPolicy(a.ckpt, a.steps, a.n)
    z = np.zeros((224, 224, 3), np.uint8)
    pol.infer({"agent": z, "wrist": z, "prompt": "warm up", "state": np.zeros(8), "base": np.zeros((RE.H, 7))})
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(pol, host="0.0.0.0", port=a.port).serve_forever()


if __name__ == "__main__":
    main()
