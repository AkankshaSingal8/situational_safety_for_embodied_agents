"""Serve frozen pi05_base PaliGemma features to the eval controller (openpi env), same path as pg_feats.py.

    python pg_server.py --port 8100 --pca data_v2/feats_pg/pca.npz
Request:  {"agent": (224,224,3) uint8, "wrist": (224,224,3) uint8, "prompt": str}   (raw env orientation,
          INTER_AREA-resized to 224 exactly as collect_v2 stores frames)
Response: {"pg": (128,K) fp16, "pg_mask": (48,) bool, "sig": (80,K) fp16}
"""
import argparse
import logging

import numpy as np

from pg_backbone import Backbone


class FeaturePolicy:
    def __init__(self, pca):
        self.bb = Backbone(pca=pca)

    def infer(self, obs):
        pg, sig, m = self.bb(obs["agent"][None], obs["wrist"][None], [str(obs["prompt"])])
        return {"pg": pg[0], "pg_mask": m[0], "sig": sig[0]}

    def reset(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--pca", default="data_v2/feats_pg/pca.npz")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    pol = FeaturePolicy(a.pca)
    z = np.zeros((224, 224, 3), np.uint8)
    pol.infer({"agent": z, "wrist": z, "prompt": "warm up"})  # compile before accepting connections
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(pol, host="0.0.0.0", port=a.port).serve_forever()


if __name__ == "__main__":
    main()
