"""Pre-training control check (POC spec): on every test scene the base trajectory
must violate both A and B, oracle-A must fix A only, oracle-B must fix B only,
and oracle-A+B must fix both while completing the task.
    python oracle_check.py --out runs/oracle_check.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.sim import PandaScene  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="runs/oracle_check.json"); a = ap.parse_args()
TE = np.load("data/test.npz", allow_pickle=True)
agg = {}
for name, key in [("base", None), ("oracle_A", "dA"), ("oracle_B", "dB"), ("oracle_AB", "dAB")]:
    rs = []
    for i, sc in enumerate(TE["scenes"]):
        wp = TE["base"][i].astype(float) + (0 if key is None else TE[key][i].astype(float))
        rs.append(PandaScene(sc).rollout(wp))
    agg[name] = {k: float(np.mean([r[k] for r in rs])) for k in ("viol_A", "viol_B", "success", "max_tilt_deg", "max_track_err")}
    print(name, agg[name], flush=True)
json.dump(agg, open(a.out, "w"), indent=1)
