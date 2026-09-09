"""Comprehensive FOL safety filter results — all versions."""
import json, glob

def load_latest(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    if not files: return None
    with open(files[-1]) as f: return json.load(f)

def metrics(d):
    if not d: return None, None, None
    o = d['overall']
    return o['TSR'], o['CAR'], o['ETS_mean']

WT = "/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/fol-safety-filter"

print("=" * 100)
print("FOL SAFETY FILTER — COMPLETE RESULTS (OpenVLA-OFT, n=10/task, 40 episodes/condition)")
print("=" * 100)
print(f"{'Suite':<10} {'Lvl':<5} {'Config':<24} {'TSR':>6} {'CAR':>6} {'ΔTSR':>7} {'ΔCAR':>7}  Notes")
print("-" * 100)

def row(suite, lvl, base_tsr, base_car, fol_tsr, fol_car, cfg, note=""):
    if fol_tsr is None:
        print(f"{suite:<10} {lvl:<5} {cfg:<24} {'N/A':>6} {'N/A':>6} {'—':>7} {'—':>7}  {note}")
        return
    dtsr = fol_tsr - base_tsr if base_tsr is not None else 0
    dcar = fol_car - base_car if base_car is not None else 0
    print(f"{suite:<10} {lvl:<5} {cfg:<24} {fol_tsr:>6.1%} {fol_car:>6.1%} {dtsr:>+7.1%} {dcar:>+7.1%}  {note}")

# === SPATIAL ===
bTSR, bCAR, _ = metrics(load_latest(f"{WT}/baseline_benchmark/safelibero_spatial/*levelI*baseline*.json"))
v1TSR, v1CAR, _ = metrics(load_latest(f"{WT}/fol_benchmark/safelibero_spatial/*level1-ablation*.json"))
v5TSR, v5CAR, _ = metrics(load_latest(f"{WT}/fol_benchmark_v5/**/*.json"))
row("Spatial", "I", bTSR, bCAR, bTSR, bCAR, "Baseline", "OpenVLA-OFT no filter")
row("Spatial", "I", bTSR, bCAR, v1TSR, v1CAR, "FOL-L1-v1 ★", "← BEST: +10pp TSR, +45pp CAR")
row("Spatial", "I", bTSR, bCAR, v5TSR, v5CAR, "FOL-L1-v5", "Regression: graduated cancel leaks approach")

bTSR2, bCAR2, _ = metrics(load_latest(f"{WT}/baseline_benchmark_L2/**/*.json"))
f2TSR, f2CAR, _ = metrics(load_latest(f"{WT}/fol_benchmark_L2/**/*.json"))
row("Spatial", "II", bTSR2, bCAR2, f2TSR, f2CAR, "FOL-L1 ★", "← BEST: +45pp TSR, +40pp CAR")

print()

# === OBJECT ===
bTSR_o, bCAR_o, _ = metrics(load_latest(f"{WT}/baseline_object_L1/**/*.json"))
fv4TSR, fv4CAR, _ = metrics(load_latest(f"{WT}/fol_object_L1_v4/**/*.json"))
fv5TSR_o, fv5CAR_o, _ = metrics(load_latest(f"{WT}/fol_object_L1_v5/**/*.json"))
row("Object", "I", bTSR_o, bCAR_o, bTSR_o, bCAR_o, "Baseline", "Policy fails all tasks (arm-body collisions t<15)")
row("Object", "I", bTSR_o, bCAR_o, fv4TSR, fv4CAR, "FOL-L1-v4 ★", "Arm-body aware: task_3 TSR 0%→70%")
row("Object", "I", bTSR_o, bCAR_o, fv5TSR_o, fv5CAR_o, "FOL-L1-v5", "Rotation dampening re-blocked task_3")

print()

# === GOAL ===
bTSR_g, bCAR_g, _ = metrics(load_latest(f"{WT}/baseline_goal_L1/**/*.json"))
fv1TSR_g, fv1CAR_g, _ = metrics(load_latest(f"{WT}/fol_goal_L1/**/*v1*.json"))
fv5TSR_g, fv5CAR_g, _ = metrics(load_latest(f"{WT}/fol_goal_L1_v5/**/*.json"))
row("Goal", "I", bTSR_g, bCAR_g, bTSR_g, bCAR_g, "Baseline", "")
row("Goal", "I", bTSR_g, bCAR_g, fv1TSR_g, fv1CAR_g, "FOL-L1-v1", "High CAR, task_1 TSR collapsed (obstacle on goal path)")
row("Goal", "I", bTSR_g, bCAR_g, fv5TSR_g, fv5CAR_g, "FOL-L1-v5", "Target-aware: task_1 TSR 0%→40%, but CAR collapsed")

print()

# === LONG ===
bTSR_l, bCAR_l, _ = metrics(load_latest(f"{WT}/baseline_long_L1/**/*.json"))
fv1TSR_l, fv1CAR_l, _ = metrics(load_latest(f"{WT}/fol_long_L1/**/*v1*.json"))
fv3TSR_l, fv3CAR_l, _ = metrics(load_latest(f"{WT}/fol_long_L1/**/*v3*.json"))
if fv3TSR_l is None:
    fv3TSR_l, fv3CAR_l, _ = metrics(load_latest(f"{WT}/fol_long_L1/**/*.json"))
row("Long", "I", bTSR_l, bCAR_l, bTSR_l, bCAR_l, "Baseline", "Policy fails all tasks")
row("Long", "I", bTSR_l, bCAR_l, fv1TSR_l, fv1CAR_l, "FOL-L1-v1", "Clean config: +65pp CAR")
row("Long", "I", bTSR_l, bCAR_l, fv3TSR_l, fv3CAR_l, "FOL-L1-v3 ★", "Speed-limit: +82.5pp CAR (TSR=0% anyway)")

print("=" * 100)
print("\n★ = best config per suite")
