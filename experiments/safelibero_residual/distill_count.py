"""Count candidate distillation demos: shield-driven episodes that ended in safe success, per suite and data root.
    python distill_count.py   (reads the same episode list as train_expert.py; writes distill_count.json)"""
import collections
import json
from concurrent.futures import ThreadPoolExecutor

from train_expert import episode_list, is_val_object, read_meta

ROOTS = "data_gso,data_gso_r2,data_gso_retry,data_gso_r2_retry,data_gso_dag1,data_gso_r3"
ROUNDS = "r0,r0_official,r1_*,r0_retry,dag1_*"
files = episode_list(ROOTS, ROUNDS)
with ThreadPoolExecutor(32) as ex:
    metas = list(ex.map(read_meta, files))
c = collections.Counter()
for f, m in zip(files, metas):
    if m is None:
        continue
    split = "val" if m.get("gso") and is_val_object(m["gso"]) else "train"
    su = m.get("suite", "?").replace("safelibero_", "")
    drive = bool(m.get("drive"))
    corr = drive and (m.get("filter") or {}).get("n_shielded", 0) > 0
    for key in ("all", "drive", "drive_corr"):
        if key == "drive" and not drive or key == "drive_corr" and not corr:
            continue
        c[(split, su, key, "n")] += 1
        c[(split, su, key, "success")] += bool(m.get("success"))
        c[(split, su, key, "safe_success")] += bool(m.get("safe_success"))
rows = {"/".join(k): v for k, v in sorted(c.items())}
json.dump(rows, open("distill_count.json", "w"), indent=1)
for k, v in rows.items():
    print(k, v)
