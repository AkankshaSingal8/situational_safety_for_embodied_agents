"""Render MuJoCo videos of scored v2 rollouts for the artifact.

Replays the exact residual that evaluate_v2.py scored (waypoints.npz) and asserts
the replayed viol_A / viol_B / success match the scored record, so every clip shows
the run that was counted. Clips are chosen by a fixed rule (first N scenes, flow
sample 0), not by outcome; every clip is labelled with its simulated outcome.

    MUJOCO_GL=egl python render_v2.py --evals runs_v2/eval_train runs_v2/eval_test --out artifact_v2
    python render_v2.py --preview 0          # one still of test scene 0, for framing the cameras
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--evals", nargs="*", default=[])
ap.add_argument("--data_dir", default="data_v2")
ap.add_argument("--out", default="artifact_v2")
ap.add_argument("--n", type=int, default=6, help="scenes per group")
ap.add_argument("--every", type=int, default=20, help="physics steps per video frame")
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--preview", type=int, default=-1)
args = ap.parse_args()

import mujoco  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from src.sim import PandaScene  # noqa: E402

W, H = 432, 324           # per view; two views side by side -> 864 x 324 (+ caption)
CAP_H = 44


def cameras():
    oblique = mujoco.MjvCamera()
    oblique.type = mujoco.mjtCamera.mjCAMERA_FREE
    oblique.lookat[:] = [-0.05, 0.0, 0.10]
    oblique.distance, oblique.azimuth, oblique.elevation = 1.35, 90.0, -38.0
    top = mujoco.MjvCamera()
    top.type = mujoco.mjtCamera.mjCAMERA_FREE
    top.lookat[:] = [0.0, 0.0, 0.0]
    top.distance, top.azimuth, top.elevation = 0.95, 90.0, -89.9
    return oblique, top


try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 13)
    FONT_B = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
except OSError:
    FONT = FONT_B = ImageFont.load_default()


def compose(views, line1, line2):
    img = Image.new("RGB", (W * len(views), H + CAP_H), (250, 250, 248))
    for i, v in enumerate(views):
        img.paste(Image.fromarray(v), (i * W, CAP_H))
    d = ImageDraw.Draw(img)
    d.text((10, 6), line1, fill=(20, 20, 20), font=FONT_B)
    d.text((10, 24), line2, fill=(70, 70, 70), font=FONT)
    return np.asarray(img)


def outcome_str(r):
    a = "laptop avoided" if not r["viol_A"] else "passed over laptop"
    b = "mug upright" if not r["viol_B"] else f"tilted {r['max_tilt_deg']:.0f}°"
    s = "placed" if r["success"] else "task failed"
    return f"{a} · {b} · {s}"


def render(scene, wp, line1):
    sim = PandaScene(scene)
    sim.m.vis.headlight.diffuse[:] = 0.35    # visual only
    sim.m.vis.headlight.ambient[:] = 0.25
    rend = mujoco.Renderer(sim.m, H, W)
    cams = cameras()
    opt = mujoco.MjvOption()
    frames, trail = [], []
    base_id = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, "base")
    arm = np.where(sim.m.body_rootid[sim.m.geom_bodyid] == base_id)[0]
    rgba_arm = sim.m.geom_rgba[arm].copy()
    ghost = rgba_arm.copy(); ghost[:, 3] = 0.18

    def add_trail():
        sc = rend.scene
        for p in trail:
            if sc.ngeom >= sc.maxgeom:
                break
            mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.006, 0, 0]),
                                p, np.eye(3).ravel(), np.array([0.16, 0.47, 0.84, 0.9], dtype=np.float32))
            sc.ngeom += 1

    def grab(s, k):
        if k % args.every == 0:
            trail.append(s.d.xpos[s.mug].copy())
            views = []
            for j, c in enumerate(cams):
                sim.m.geom_rgba[arm] = ghost if j == 1 else rgba_arm   # visual only; no effect on contacts
                rend.update_scene(s.d, camera=c, scene_option=opt)
                add_trail()
                views.append(rend.render().copy())
            sim.m.geom_rgba[arm] = rgba_arm
            frames.append(views)

    r = sim.rollout(wp, on_step=grab)
    line2 = "outcome (simulated): " + outcome_str(r)
    out = [compose(v, line1, line2) for v in frames]
    out += [out[-1]] * args.fps            # hold the last frame for one second
    rend.close()
    return r, out


def write_clip(out, path):
    import imageio.v2 as imageio
    imageio.mimwrite(path, out, fps=args.fps, codec="libx264", quality=7,
                     pixelformat="yuv420p", macro_block_size=8)
    Image.fromarray(out[len(out) // 3]).save(path.replace(".mp4", ".jpg"), quality=85)


if args.preview >= 0:
    TE = np.load(os.path.join(args.data_dir, "test.npz"), allow_pickle=True)
    i = args.preview
    _, out = render(TE["scenes"][i], TE["base"][i].astype(np.float64), f"preview test scene {i} (base)")
    os.makedirs(args.out, exist_ok=True)
    for f in (0, len(out) // 4, len(out) // 2):
        Image.fromarray(out[f]).save(os.path.join(args.out, f"preview_{f}.png"))
    print("preview frames:", len(out))
    sys.exit(0)

# ---------------- clip selection (fixed rule) ----------------
from src.texts import COMBINED, COMBINED_FAR  # noqa: E402

GROUPS = [  # (group id, label, cond pattern, splits)
    ("base", "Base trajectory (no instruction)", lambda i: "base", ("train", "test")),
    ("A", "Constraint A text (training phrasing)", lambda i: "main|A", ("train", "test")),
    ("B", "Constraint B text (training phrasing)", lambda i: "main|B", ("train", "test")),
    ("ab_direct", "Combined sentence, no combined training (main)", lambda i: f"main|combined{i % len(COMBINED)}", ("test",)),
    ("ab_stack", "Combined, no training: clauses applied in turn A→B (main)", lambda i: f"stack_AB|combined{i % len(COMBINED)}", ("test",)),
    ("ab_trained", "Combined sentence, with combined training (main_ab)", lambda i: f"main_ab|combined{i % len(COMBINED)}", ("test",)),
    ("ab_trained_far", "Far-worded combined sentence, with combined training (main_ab)", lambda i: f"main_ab|far{i % len(COMBINED_FAR)}", ("test",)),
]
os.makedirs(os.path.join(args.out, "clips"), exist_ok=True)
EV = {}
for ev in args.evals:
    recs = [json.loads(l) for l in open(os.path.join(ev, "rollouts.jsonl"))]
    Wz = np.load(os.path.join(ev, "waypoints.npz"), allow_pickle=True)
    metas = [json.loads(m) for m in Wz["meta"]]
    split = recs[0]["split"]
    idx = {(m["cond"], m["scene"], m["sample"]): k for k, m in enumerate(metas)}
    rec = {(r["cond"], r["scene"], r["sample"]): r for r in recs}
    EV[split] = (np.load(os.path.join(args.data_dir, f"{split}.npz"), allow_pickle=True), Wz["residual"], idx, rec)

manifest = []
for gid, glabel, condf, splits in GROUPS:
    for split in splits:
        if split not in EV:
            continue
        D, RES, idx, rec = EV[split]
        for i in range(args.n):
            cond = condf(i)
            key = (cond, i, 0)
            if key not in idx:
                print("missing", split, key); continue
            r0 = rec[key]
            wp = D["base"][i].astype(np.float64) + RES[idx[key]].astype(np.float64)
            text = r0.get("text") or ""
            line1 = f"{split.upper()} scene {i} · {glabel}" + (f' · "{text}"' if text else "")
            if len(line1) > 118:
                line1 = line1[:115] + "…"
            name = f"{gid}_{split}_{i}.mp4"
            r, frames = render(D["scenes"][i], wp, line1)
            for k in ("viol_A", "viol_B", "success", "max_tilt_deg", "final_dist", "min_laptop_clearance"):
                assert r[k] == r0[k], (split, cond, i, k, r[k], r0[k])     # bit-exact replay of the scored run
            write_clip(frames, os.path.join(args.out, "clips", name))
            sc = D["scenes"][i]
            manifest.append(dict(
                id=name[:-4], group=gid, group_label=glabel, split=split, scene=i, cond=cond, text=text,
                video=f"clips/{name}", poster=f"clips/{name[:-4]}.jpg",
                viol_A=r["viol_A"], viol_B=r["viol_B"], success=r["success"], max_tilt_deg=round(r["max_tilt_deg"], 1),
                path=r0["exec_grip"], tilt=r0["exec_tilt"],
                laptop=dict(center=sc["laptop_center"], size=sc["laptop_size"], yaw=sc["laptop_yaw"]),
                mug_start=sc["mug_start"], target=sc["target"]))
            print("rendered", name, outcome_str(r), flush=True)
json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"))
print("clips:", len(manifest))
