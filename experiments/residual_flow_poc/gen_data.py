"""Generate scenes, base trajectories and oracle residual targets.

Train and test scenes come from disjoint seed streams. The combined A+B
residual is stored for the test split only (oracle reference for Exp. 3);
it is never written to the training file.
    python gen_data.py --n_train 4000 --n_test 100
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.traj import sample_scene, sample_scene_v2, base_trajectory, safe_A, safe_B, safe_AB, scene_features  # noqa: E402

TRAIN_SEED, TEST_SEED = 0, 1_000_000


def _arrays(sc):
    b = base_trajectory(sc)
    return (b.astype(np.float32), (safe_A(sc, b) - b).astype(np.float32),
            (safe_B(sc, b) - b).astype(np.float32), (safe_AB(sc, b) - b).astype(np.float32))


def sim_control(args):
    """The POC's control, executed in MuJoCo with the same float32 arrays evaluation uses:
    base breaks A and B, oracle A fixes only A, oracle B fixes only B, oracle A+B fixes both,
    all four place the mug, and tracking stays within 12 cm. Returns (ok, details)."""
    from src.sim import PandaScene
    sc, b, dA, dB, dAB = args
    sim = PandaScene(sc)
    want = {"base": (0, True, True), "oracle_A": (dA, False, True), "oracle_B": (dB, True, False),
            "oracle_AB": (dAB, False, False)}
    det = {}
    ok = True
    for k, (d, vA, vB) in want.items():
        r = sim.rollout(b.astype(np.float64) + (d.astype(np.float64) if k != "base" else 0.0))
        det[k] = r
        ok &= r["viol_A"] == vA and r["viol_B"] == vB and r["success"] and r["max_track_err"] < 0.12
    return bool(ok), det


def build(n, seed, with_ab, sampler=sample_scene, sim_filter=False, workers=5, log=None):
    """n scenes in seed order. With sim_filter, candidate scenes are drawn in seed order
    and kept only if sim_control passes (the kept set is deterministic given the seed)."""
    rng = np.random.default_rng(seed)
    keep, n_cand = [], 0
    pool = None
    if sim_filter:
        from multiprocessing import Pool
        pool = Pool(workers)
    while len(keep) < n:
        batch = [sampler(rng) for _ in range(max(50, (n - len(keep)) * 5 // 4) if sim_filter else n - len(keep))]
        n_cand += len(batch)
        arrs = [_arrays(sc) for sc in batch]
        if sim_filter:
            res = pool.map(sim_control, [(sc, *a) for sc, a in zip(batch, arrs)], chunksize=4)
            for sc, a, (ok, _) in zip(batch, arrs, res):
                if ok and len(keep) < n:
                    keep.append((sc, a))
        else:
            keep += list(zip(batch, arrs))
    if pool:
        pool.close()
    if log is not None:
        log.append(dict(seed=seed, kept=n, candidates=n_cand))
    scenes = [sc for sc, _ in keep]
    out = dict(scenes=np.array(scenes, dtype=object), feats=np.stack([scene_features(sc) for sc in scenes]),
               base=np.stack([a[0] for _, a in keep]), dA=np.stack([a[1] for _, a in keep]),
               dB=np.stack([a[2] for _, a in keep]))
    if with_ab:
        out["dAB"] = np.stack([a[3] for _, a in keep])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--out_dir", default=None, help="default: data (v1) or data_v2 (v2)")
    ap.add_argument("--sim_filter", action="store_true",
                    help="keep only scenes whose base/oracle rollouts pass the control check in MuJoCo")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--profile", choices=["v1", "v2"], default="v1",
                    help="v2: wider scene ranges + stochastic base habits; also stores dAB for train "
                         "(read only by the main_ab variant)")
    a = ap.parse_args()
    a.out_dir = a.out_dir or ("data_v2" if a.profile == "v2" else "data")
    if a.profile == "v2" and os.path.abspath(a.out_dir) == os.path.abspath("data"):
        raise SystemExit("refusing to write v2 data over the v1 data/ directory")
    os.makedirs(a.out_dir, exist_ok=True)
    sampler = sample_scene_v2 if a.profile == "v2" else sample_scene
    log = []
    tr = build(a.n_train, TRAIN_SEED, with_ab=a.profile == "v2", sampler=sampler, sim_filter=a.sim_filter,
               workers=a.workers, log=log)
    te = build(a.n_test, TEST_SEED, with_ab=True, sampler=sampler, sim_filter=a.sim_filter,
               workers=a.workers, log=log)
    import json
    json.dump(dict(profile=a.profile, sim_filter=a.sim_filter, splits=dict(zip(["train", "test"], log))),
              open(os.path.join(a.out_dir, "gen_log.json"), "w"), indent=1)
    print("candidates per split:", log)
    np.savez(os.path.join(a.out_dir, "train.npz"), **tr)
    np.savez(os.path.join(a.out_dir, "test.npz"), **te)
    for name, d in [("train", tr), ("test", te)]:
        pa = np.linalg.norm(d["dA"][..., :2], axis=-1).max(1)
        rb = np.linalg.norm(d["dB"][..., 3:], axis=-1).max(1)
        side = np.sign(d["dA"][:, 20, 1])
        print(f"{name}: n={len(d['base'])} |dA|max mean={pa.mean():.3f}m "
              f"[{pa.min():.3f},{pa.max():.3f}] |dB|max mean={np.degrees(rb.mean()):.1f}deg "
              f"detour side +y frac={np.mean(side > 0):.2f}")
