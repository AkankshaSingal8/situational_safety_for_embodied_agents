import sys, collections, numpy as np
sys.argv=["x"]
import train_expert as T
files=T.episode_list("data_gso,data_gso_r2","r0,r0_official,r1_*")
files=files[::8]
metas=[T.read_meta(f) for f in files]
pool,idx=T.noun_pool()
mix={"corr":0.35,"safe":0.30,"wrong":0.15,"none":0.20}
rng=np.random.default_rng(0); pools=T.mix_pools(metas)
C=collections.Counter(); nzrows=0; rows=0; ex=[]
for k in range(30):
    sel,kinds=T.pick_mix(rng,metas,pools,mix,16)
    b=T.load_batch(([files[i] for i in sel],[metas[i] for i in sel],k,2,pool,idx,0.5,kinds))
    C.update(b["kind"]); rows+=len(b["kind"])
    for t,kd,pr in zip(b["tgt"],b["kind"],b["prompt"]):
        nz=np.abs(t).sum()>1e-6; nzrows+=nz
        if len(ex)<12 and (k%3==0): ex.append((kd, round(float(np.abs(t).sum(-1).max()),3), pr[-70:]))
print("episodes",len(files),"corrected",len(pools["corr"]),"driven",len(pools["driven"]))
print("rows by kind",{k:round(v/rows,3) for k,v in C.items()},"nonzero-target rows",round(nzrows/rows,3))
bad=[e for e in ex]
for e in ex: print(e)
