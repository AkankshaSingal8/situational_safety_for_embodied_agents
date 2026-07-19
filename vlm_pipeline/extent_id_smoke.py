"""G2 smoke: perception-derived object extents as the size signal in
symbolic identification (offline, saved SafeLIBERO captures).

Per episode: back-project every instance-seg region (agentview depth) into a
world point cloud, compute per-region centroid + axis extents (p5-p95),
match regions to object names by nearest GT centroid (GT used ONLY for this
validation matching — at runtime the mask itself is already name-keyed), then
re-score identification with size-aware variants.

Score variant under test:
    score = prior(name) * exp(-d_path^2 / 2 sigma^2) * (1 - 0.8 mf)
            * (V_xy(name) / median_V_xy) ** spow
where V_xy = footprint area (ex * ey) or volume proxy from percep extents —
collision likelihood scales with physical cross-section, which is exactly the
signal a name-only VLM property rating cannot carry.
"""

import glob
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import symbolic_identity as si  # noqa: E402
import vlm_slot_bench as vb  # noqa: E402

Z_TABLE = 0.81


def region_cloud(depth, mask, K, T):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    ok = (zs > 0.1) & (zs < 5.0)
    if ok.sum() < 30:
        return None
    ys, xs, zs = ys[ok], xs[ok], zs[ok]
    z_med = np.median(zs)
    keep = np.abs(zs - z_med) < 0.25
    ys, xs, zs = ys[keep], xs[keep], zs[keep]
    pix = np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float64)
    rays = np.linalg.inv(K) @ pix
    cam = rays * zs[None, :]
    world = (T @ np.vstack([cam, np.ones((1, cam.shape[1]))]))[:3].T
    return world


def episode_extents(ep_dir):
    ep = pathlib.Path(ep_dir)
    md = json.load(open(ep / "metadata.json"))
    cp = json.load(open(ep / "camera_params.json"))["agentview"]
    K = np.array(cp["intrinsic"])
    T = np.array(cp["extrinsic"])
    # agentview was saved 180°-rotated (policy preprocessing); undo before
    # back-projecting with the standard robosuite K/T.
    depth = np.load(ep / "agentview_depth.npy").squeeze()[::-1, ::-1]
    seg = np.load(ep / "agentview_seg.npy").squeeze()[::-1, ::-1]
    gt_pos = {n: np.array(o["position"]) for n, o in md["objects"].items()}
    ob = json.load(open(ep / "obstacle.json"))["active_obstacle"]
    if ob:
        gt_pos[ob["name"]] = np.array(ob["position"])
    out = {}
    for sid in np.unique(seg):
        if sid <= 0:
            continue
        cloud = region_cloud(depth, seg == sid, K, T)
        if cloud is None or len(cloud) < 30:
            continue
        cen = np.median(cloud, axis=0)
        if not (0.6 < cen[2] < 1.3):  # drop walls/floor/robot base
            continue
        ext = np.percentile(cloud, 95, axis=0) - np.percentile(cloud, 5, axis=0)
        # nearest GT object (xy) — validation-only name assignment
        name, d = min(((n, np.linalg.norm(cen[:2] - p[:2])) for n, p in gt_pos.items()),
                      key=lambda t: t[1])
        if d > 0.10:
            continue
        if name not in out or out[name]["npix"] < len(cloud):
            out[name] = {"extent": ext, "center": cen, "npix": len(cloud)}
    return out


def identify(e, wfn, extents, spow=0.0, head=True, vol="xy"):
    task = e["task"].lower()

    def mf(k):
        toks = [w for w in si._clean_object_name(k).lower().split()
                if len(w) >= 3 and w != "obstacle"]
        if not toks:
            return 0.0
        f = sum(w in task for w in toks) / len(toks)
        if head and toks[-1] in task:
            f = 1.0
        return f

    fr = {k: mf(k) for k in e["cands"]}
    part = [k for k, f in fr.items() if f < 1.0]
    if not part:
        return None
    goals = [k for k, f in fr.items() if f >= 0.5]
    spos = np.asarray(e["eef"])[:2]
    segs = [(spos, np.asarray(e["cands"][g])[:2]) for g in goals] or \
           [(spos, np.array([0.0, 0.15]))]

    def dp(k):
        p = np.asarray(e["cands"][k])[:2]
        best = np.inf
        for s, g in segs:
            v = g - s
            l2 = float(v @ v)
            t = 0.0 if l2 < 1e-9 else float(np.clip((p - s) @ v / l2, 0.0, 1.0))
            best = min(best, float(np.linalg.norm(p - (s + t * v))))
        return best

    sizes = {}
    for k in part:
        x = extents.get(k)
        if x is None:
            sizes[k] = None
            continue
        ex = np.clip(x["extent"], 0.01, 0.5)
        sizes[k] = float(ex[0] * ex[1]) if vol == "xy" else float(ex[0] * ex[1] * ex[2])
    known = [v for v in sizes.values() if v]
    med = np.median(known) if known else 1.0

    def sc(k):
        s = wfn(k) * np.exp(-dp(k) ** 2 / (2 * 0.25 ** 2)) * (1.0 - 0.8 * fr[k])
        if spow and sizes.get(k):
            s *= (sizes[k] / med) ** spow
        return s

    return max(part, key=sc)


def main():
    eps = vb.load_episodes(["safelibero_spatial"], 5)
    ext_cache = {e["ep"]: episode_extents(e["ep"]) for e in eps}
    # report extent quality on the obstacle
    for e in eps[:3]:
        x = ext_cache[e["ep"]].get(e["gt"])
        print("extent sample:", e["gt"],
              None if x is None else np.round(x["extent"], 3).tolist())
    cov = sum(e["gt"] in ext_cache[e["ep"]] for e in eps)
    print(f"obstacle extent coverage: {cov}/{len(eps)}")

    recs = [json.loads(l) for l in
            open(pathlib.Path(__file__).parents[1] / "results_tables/vlm_bench_s1_anthropic_claudesonnet5.jsonl")]
    vw = {r["name"]: r["weight"] for r in recs if r["kind"] == "s1_weight"}

    arms = [
        ("table (ref, no size)", si._hazard_weight, 0.0, "xy"),
        ("sonnet, no size", lambda n: vw.get(n, 0.4), 0.0, "xy"),
        ("sonnet + xy-area^0.5", lambda n: vw.get(n, 0.4), 0.5, "xy"),
        ("sonnet + xy-area^1", lambda n: vw.get(n, 0.4), 1.0, "xy"),
        ("sonnet + vol^0.5", lambda n: vw.get(n, 0.4), 0.5, "vol"),
        ("uniform + xy-area^1", lambda n: 0.5, 1.0, "xy"),
        ("uniform + vol^1", lambda n: 0.5, 1.0, "vol"),
    ]
    for label, wfn, spow, vol in arms:
        ok, fails = 0, []
        for e in eps:
            pick = identify(e, wfn, ext_cache[e["ep"]], spow=spow, vol=vol)
            if pick == e["gt"]:
                ok += 1
            else:
                fails.append((e["gt"], pick))
        print(f"{label}: {ok}/{len(eps)}", fails[:3] if fails else "")


if __name__ == "__main__":
    main()
