import json, os, glob
T = ["open_single_door", "turn_on_microwave", "counter_to_microwave", "turn_on_stove"]
R = "/data/asingal2/oopsieverse/results"
RUNS = {
 "pi0 RC365 (paper protocol, full rollouts)": f"{R}/robocasa_papermatch/pi0_robocasa365_full",
 "pi0.5 RC365 (paper protocol, full rollouts)": f"{R}/robocasa_papermatch/pi05_robocasa365_full",
 "pi0 RC365 prompt fix + stop_on_success": f"{R}/robocasa_papermatch/pi0_robocasa365_prompt",
 "pi05_base (openpi, LIBERO norm)": f"{R}/robocasa_papermatch/pi05_base",
 "pi0_base (openpi, LIBERO norm)": f"{R}/robocasa_papermatch/pi0_base",
 "GR00T on OopsieVerse task versions (groot_papermatch_v2)": f"{R}/groot_papermatch_v2",
 "GR00T stock tasks + DamageSim, 720 steps no early stop": f"{R}/groot_stock_damagesim",
 "TEST GR00T base (stop_on_success)": "/data/asingal2/oopsieverse/residual/test_base",
 "TEST GR00T + residual r1a n=4": "/data/asingal2/oopsieverse/residual/test_res_r1a_n4",
 "GATE base (seeds 500-529, full)": "/data/asingal2/oopsieverse/residual/gate_base_s",
 "GATE shield (worker)": "/data/asingal2/oopsieverse/residual/gate_worker",
}
def eps(d, t):
    for p in (f"{d}/{t}/episodes.json", f"{d}/{t}/results.json", f"{d}/episodes_{t}.json"):
        if os.path.exists(p):
            x = json.load(open(p)); return x if isinstance(x, list) else x.get("episodes", [])
    out = []
    for p in glob.glob(f"{d}/**/*.json", recursive=True):
        try: x = json.load(open(p))
        except Exception: continue
        if isinstance(x, list) and x and isinstance(x[0], dict) and x[0].get("task_name") == t: out += x
    return out
und = lambda e: all(v >= 100 - 1e-6 for v in (e.get("health_summary") or {}).values())
for name, d in RUNS.items():
    if not os.path.isdir(d): print(name, "MISSING", d); continue
    row, N, S, C, SC, H = [], 0, 0, 0, 0, []
    for t in T:
        E = [e for e in eps(d, t) if not e.get("error")]
        if not E: row.append(f"{t[:10]} -"); continue
        s = sum(bool(e.get("success_any", e.get("success"))) for e in E); c = sum(und(e) for e in E)
        sc = sum(bool(e.get("success_any", e.get("success"))) and und(e) for e in E)
        H += [e["avg_env_health"] for e in E if e.get("avg_env_health") is not None]
        N += len(E); S += s; C += c; SC += sc; row.append(f"{t[:10]} n{len(E)} {s}/{c}/{sc}")
    if N: print(f"{name} | n={N} TSR {100*S/N:.1f} CAR {100*C/N:.1f} SC {100*SC/N:.1f} health {sum(H)/max(1,len(H)):.1f} | " + "; ".join(row), flush=True)
    else: print(name, "no episodes", d, os.listdir(d)[:8])
