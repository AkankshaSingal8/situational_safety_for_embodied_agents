import json, sys, collections
sys.path.insert(0, ".")
from train_expert import episode_list, is_val_object, read_meta
files = episode_list("data_gso,data_gso_r2,data_gso_retry,data_gso_r2_retry,data_gso_dag1,data_gso_r3", "r0,r0_official,r1_*,r0_retry,dag1_*")
nv = nt = 0
for f in files:
    if "data_gso" not in f:
        continue
    m = read_meta(f)
    if not m or not m.get("gso"):
        continue
    v = is_val_object(m["gso"])
    if (v and nv < 3) or (not v and nt < 2):
        print("VAL" if v else "TRAIN", f)
        print({k: m.get(k) for k in ("gso", "drive", "success", "named_hit", "safe_success", "hits", "lang", "suite")})
        nv += v; nt += not v
    if nv >= 3 and nt >= 2:
        break
