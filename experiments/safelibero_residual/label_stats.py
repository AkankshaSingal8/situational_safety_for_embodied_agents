"""What the residual labels look like: shield choices, correction sizes, and how often a correction is needed.
    python label_stats.py N_PER_SOURCE"""
import glob, json, sys, collections
import numpy as np
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SRC = {"safelibero r0": "data_v2/*/r0/ep_*.npz", "safelibero r1": "data_v2/*/r1_*/ep_*.npz",
       "dagger 1": "data_v2/*/dag1_*/ep_*.npz", "gso": "data_gso/*/ep_*.npz", "gso r2": "data_gso_r2/*/ep_*.npz",
       "gso r3": "data_gso_r3/*/ep_*.npz"}
rng = np.random.default_rng(0)
for name, pat in SRC.items():
    fs = sorted(glob.glob(pat))
    fs = [fs[i] for i in rng.choice(len(fs), min(N, len(fs)), replace=False)]
    ch, mags, nz_frac, corr_eps, n_ch, n_ep, steps_nz, hold = collections.Counter(), [], [], 0, 0, 0, [], 0
    for f in fs:
        with np.load(f, allow_pickle=True) as z:
            m = json.loads(str(z["meta"])); t = z["drive_tgt"]
        n_ep += 1
        if not m.get("drive"):
            continue
        k = np.abs(t).sum((1, 2)) > 1e-6
        n_ch += len(t); corr_eps += k.any(); nz_frac.append(k.mean())
        for c in t[k]:
            mags.append(np.abs(c[:5]).sum(0).max())      # largest summed |dx|,|dy|,|dz| over the executed 5 steps
        for kk, v in (m.get("filter", {}).get("choice") or {}).items():
            ch[kk] += v
    mags = np.array(mags) if mags else np.zeros(1)
    tot = sum(ch.values()) or 1
    print(f"\n## {name}: {len(glob.glob(pat))} files, sampled {n_ep}")
    print(f"  episodes with >=1 corrected chunk: {corr_eps}/{n_ep}; corrected chunks per driven episode: "
          f"{100*np.mean(nz_frac) if nz_frac else 0:.1f}%")
    print(f"  correction size (env units, 5 steps): p10 {np.percentile(mags,10):.2f} median {np.median(mags):.2f} "
          f"p90 {np.percentile(mags,90):.2f}")
    print("  shield choice share:", {k: f"{100*v/tot:.0f}%" for k, v in ch.most_common()})
