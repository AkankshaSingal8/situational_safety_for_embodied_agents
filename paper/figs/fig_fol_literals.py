"""Paper figure: what each FOL literal elects, and the one barrier knob it turns.

    HAZARD(x) := PROTECTED(x) | MOVING(x) | (NEAR_PATH(x) & ~MENTIONED(x) & ~TARGET(x))

Top row (REAL): one cached LIBERO-Safety frame per literal, the deployed
election (`symbolic_identity.fol_obstacle_id`, top-2) drawn on it, and only
that literal's evidence highlighted:
  (a) PROTECTED      -- oah scene: bottle-in-hand elected by class, mug exempt.
  (b) NEAR_PATH      -- oa scene: eef->target segment; per-candidate segment
                        distance; nearest unmentioned candidate elected.
  (c) ~MENTIONED/~TARGET -- hs scene: head-noun-mentioned objects exempt.
MOVING has no evidence in the cached logs (no hand displaces >1 cm in any
logged episode), so it is not drawn -- stated in the caption, not faked.

Bottom row (SCHEMATIC on the same scene's real xy positions): the barrier
knob that literal turns.
  (a) class envelope r_obs: hand 0.10 vs default 0.065 (OBSTACLE_RADII in
      run_guided_libero_safety_eval.py) -> r_eff = r_obs + eef 0.09 + d_safe 0.01.
  (b) selection only: the barrier at the elected entity is the default one;
      plus the corridor: cylinder r=0.07 around eef->target, margins x0.4
      inside (corridor_relax 0.6, guided_policy.py / _corridor_scale).
  (c) exempt entities get NO barrier; the remaining candidates keep theirs.

CPU-only, no sim, no API. Output: paper/figs/fig_fol_literals.png/.pdf
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vlm_pipeline"))
import symbolic_identity as si  # noqa: E402
import visual_conditioning as vc  # noqa: E402

FIGDIR = pathlib.Path(__file__).resolve().parent
CACHE = FIGDIR / "scene_cache"

# --- deployed constants (sources in module docstring) ---------------------
OBSTACLE_RADII = {"hand": 0.10, "car": 0.05, "train": 0.05, "ball": 0.04}
DEFAULT_OBSTACLE_RADIUS = 0.065
EEF_R, D_SAFE = 0.09, 0.01
CORRIDOR_R, CORRIDOR_RELAX = 0.07, 0.6
GUARD_TOPK = 2

FS = 8
CRIMSON, GREEN, DGREEN, GRAY, BLUE = "crimson", "#1faa4b", "#0d6b30", "0.45", "#2a78d6"
BG = "#f3ede2"


def r_obs(name):
    label = name.lower()
    return next((r for k, r in OBSTACLE_RADII.items() if k in label),
                DEFAULT_OBSTACLE_RADIUS)


def load(scene):
    d = CACHE / scene
    sc = json.load(open(d / "scene.json"))
    cam = json.load(open(d / "camera_params.json"))
    K = np.asarray(cam[sc.get("camera", "agentview")]["intrinsic"])
    ext = np.asarray(cam[sc.get("camera", "agentview")]["extrinsic"])
    raw = plt.imread(d / f"{sc.get('camera', 'agentview')}_rgb.png")
    if raw.dtype != np.uint8:
        raw = (raw[..., :3] * 255).astype(np.uint8)
    img = np.ascontiguousarray(raw[::-1, ::-1])
    cands = {k: np.asarray(v, float) for k, v in sc["obj_pos"].items()}
    eef = np.asarray(sc["eef_pos"], float)
    desc = sc["language"]
    tgt = si.parse_target_heuristic(desc, list(cands))
    ranked = si.fol_obstacle_id(desc, cands, eef)
    guard = ranked[:GUARD_TOPK]
    print(f"{scene}: target={tgt} guard={guard}")
    return dict(K=K, ext=ext, img=img, cands=cands, eef=eef, desc=desc,
                tgt=tgt, guard=guard)


def px(S, p):
    rc, z = vc._project(S["K"], S["ext"], [p])
    return np.array([rc[0][1], rc[0][0]], float), float(z[0])


def prad(S, p, r):
    return vc._pixel_radius(S["K"], S["ext"], p, r)


def img_ax(ax, S):
    ax.imshow(S["img"])
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.6")


def disk(ax, c, r, color=CRIMSON, layered=True, lw=1.6, z=3):
    if layered:
        for f, a in ((1.0, 0.20), (0.7, 0.16), (0.4, 0.16)):
            ax.add_patch(Circle(c, r * f, facecolor=color, alpha=a,
                                edgecolor="none", zorder=z))
    ax.add_patch(Circle(c, r, facecolor="none", edgecolor=color, lw=lw,
                        zorder=z))


def chip(ax, xy, text, color, dy=-1, fs=FS):
    ax.annotate(text, xy, xytext=(0, 12 * dy), textcoords="offset points",
                ha="center", va="bottom" if dy > 0 else "top", fontsize=fs,
                color="white", fontweight="bold", zorder=9,
                bbox=dict(facecolor=color, edgecolor="none", pad=2.0))


def head_mentioned(name, desc):
    import re
    base = re.sub(r"(__\d+)?(_\d+)?$", "", name)
    return base.split("_")[-1].lower() in desc.lower()


# --------------------------------------------------------------------------
# top row: real frames
# --------------------------------------------------------------------------

def top_protected(ax, S):
    img_ax(ax, S)
    for k in S["guard"]:
        if any(w in k.lower() for w in si._PROTECTED_CLASSES):
            c, z = px(S, S["cands"][k])
            r = prad(S, S["cands"][k], r_obs(k))
            disk(ax, c, r)
            chip(ax, c + [0, -r], "protected", CRIMSON, dy=1)
    c, _ = px(S, S["cands"][S["tgt"]])
    r = prad(S, S["cands"][S["tgt"]], 0.075)
    ax.add_patch(Circle(c, r, facecolor=GREEN, alpha=0.18, edgecolor=GREEN,
                        lw=1.8, zorder=3))


def top_nearpath(ax, S):
    img_ax(ax, S)
    eef, tgt = S["eef"], S["cands"][S["tgt"]]
    a, _ = px(S, eef)
    b, _ = px(S, tgt)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                 color="0.15", lw=1.8, ls=(0, (4, 3)),
                                 zorder=5))
    ax.plot(*a, marker="^", ms=8, mfc="k", mec="k", zorder=6)
    r = prad(S, tgt, 0.06)
    ax.add_patch(Circle(b, r, facecolor=GREEN, alpha=0.18, edgecolor=GREEN,
                        lw=1.8, zorder=3))
    for k, p in S["cands"].items():
        if k == S["tgt"] or head_mentioned(k, S["desc"]):
            continue
        d = si.segment_distance(p, eef, tgt)
        ab = tgt - eef
        t = np.clip(np.dot(p - eef, ab) / np.dot(ab, ab), 0, 1)
        q = eef + t * ab
        c, _ = px(S, p)
        qc, _ = px(S, q)
        elected = k in S["guard"][:1]
        col = CRIMSON if elected else GRAY
        ax.plot([c[0], qc[0]], [c[1], qc[1]], color=col, lw=1.2,
                ls=(0, (2, 2)), zorder=4)
        mid = 0.35 * c + 0.65 * qc
        ax.text(mid[0], mid[1], f"{d:.2f} m", fontsize=FS - 1, color=col,
                ha="center", va="center", zorder=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                          alpha=0.85))
        if elected:
            disk(ax, c, prad(S, p, r_obs(k)))
            chip(ax, c + [0, -prad(S, p, r_obs(k))], "nearest", CRIMSON, dy=1)


def top_exempt(ax, S):
    img_ax(ax, S)
    for k, p in S["cands"].items():
        c, _ = px(S, p)
        if k == S["tgt"] or head_mentioned(k, S["desc"]):
            r = prad(S, p, 0.06)
            ax.add_patch(Circle(c, r, facecolor=GREEN, alpha=0.18,
                                edgecolor=GREEN, lw=1.8, zorder=3))
            # strike-through: exempt from election
            ax.plot([c[0] - r * 0.7, c[0] + r * 0.7],
                    [c[1] + r * 0.7, c[1] - r * 0.7], color=GREEN, lw=2.0,
                    zorder=4)
        elif k in S["guard"]:
            disk(ax, c, prad(S, p, r_obs(k)))
    chip(ax, px(S, S["cands"]["cookies_1"])[0] + [0, -30], "mentioned",
         GREEN, dy=1)


# --------------------------------------------------------------------------
# bottom row: barrier knobs, top-down, real xy
# --------------------------------------------------------------------------

def plan_ax(ax, S, extra=()):
    pts = [p[:2] for p in S["cands"].values()] + [S["eef"][:2]] + list(extra)
    for k in S["guard"]:
        p, re = S["cands"][k][:2], r_obs(k) + EEF_R + D_SAFE
        pts += [p + [re, 0], p - [re, 0], p + [0, re], p - [0, re]]
    xs, ys = zip(*pts)
    pad = 0.07
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("0.6")
    for k, p in S["cands"].items():
        ax.add_patch(Circle(p[:2], 0.022, facecolor="0.55", edgecolor="k",
                            lw=0.6, zorder=5))
    ax.plot(*S["eef"][:2], marker="^", ms=7, mfc="k", mec="k", zorder=6)


def label(ax, xy, text, color="0.2", dx=0.0, dy=0.0, fs=FS - 1.5):
    ax.text(xy[0] + dx, xy[1] + dy, text, fontsize=fs, color=color,
            ha="center", va="center", zorder=8,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.8))


def bot_protected(ax, S):
    plan_ax(ax, S)
    k = S["guard"][0]
    p = S["cands"][k][:2]
    ro, re = r_obs(k), r_obs(k) + EEF_R + D_SAFE
    ax.add_patch(Circle(p, DEFAULT_OBSTACLE_RADIUS, facecolor="none",
                        edgecolor="0.4", lw=1.0, ls=(0, (2, 2)), zorder=3))
    ax.add_patch(Circle(p, ro, facecolor=CRIMSON, alpha=0.30,
                        edgecolor=CRIMSON, lw=1.4, zorder=3))
    ax.add_patch(Circle(p, re, facecolor=CRIMSON, alpha=0.10,
                        edgecolor=CRIMSON, lw=1.0, ls=(0, (4, 2)), zorder=2))
    label(ax, p, f"class envelope\n{ro:.2f} m (hand)\nvs {DEFAULT_OBSTACLE_RADIUS:.3f} default",
          CRIMSON, dy=-ro - 0.045)
    label(ax, p, f"$r_\\mathrm{{eff}}$ = {re:.2f} m", CRIMSON, dx=-re * 0.55, dy=re * 0.75)
    t = S["cands"][S["tgt"]][:2]
    ax.add_patch(Circle(t, 0.05, facecolor=GREEN, alpha=0.2, edgecolor=GREEN,
                        lw=1.4, zorder=3))


def bot_nearpath(ax, S):
    plan_ax(ax, S)
    eef, tgt = S["eef"], S["cands"][S["tgt"]]
    a, b = eef[:2], tgt[:2]
    # corridor cylinder (top-down band) around eef->target, valid t>0.15
    ab = b - a
    n = np.array([-ab[1], ab[0]]) / np.linalg.norm(ab)
    a15 = a + 0.15 * ab
    band = np.array([a15 + n * CORRIDOR_R, b + n * CORRIDOR_R,
                     b - n * CORRIDOR_R, a15 - n * CORRIDOR_R])
    ax.add_patch(Polygon(band, closed=True, facecolor=GREEN, alpha=0.15,
                         edgecolor=GREEN, lw=0.8, ls=(0, (3, 2)), zorder=2))
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11,
                                 color="0.15", lw=1.4, ls=(0, (4, 3)),
                                 zorder=4))
    ax.add_patch(Circle(b, 0.05, facecolor=GREEN, alpha=0.2, edgecolor=GREEN,
                        lw=1.4, zorder=3))
    k = S["guard"][0]
    p = S["cands"][k]
    re = r_obs(k) + EEF_R + D_SAFE
    ax.add_patch(Circle(p[:2], re, facecolor=CRIMSON, alpha=0.12,
                        edgecolor=CRIMSON, lw=1.0, ls=(0, (4, 2)), zorder=2))
    ax.add_patch(Circle(p[:2], re * (1 - CORRIDOR_RELAX), facecolor=CRIMSON,
                        alpha=0.30, edgecolor=CRIMSON, lw=1.4, zorder=3))
    label(ax, p[:2], f"$r_\\mathrm{{eff}}$ {re:.3f} m", CRIMSON, dy=-re + 0.02)
    label(ax, p[:2], f"×{1 - CORRIDOR_RELAX:.1f} inside corridor", CRIMSON,
          dy=-re * (1 - CORRIDOR_RELAX) - 0.03)
    label(ax, 0.5 * (a15 + b) + n * (CORRIDOR_R + 0.03),
          f"corridor r={CORRIDOR_R:.2f} m", DGREEN)


def bot_exempt(ax, S):
    plan_ax(ax, S)
    for k, p in S["cands"].items():
        if k == S["tgt"] or head_mentioned(k, S["desc"]):
            ax.add_patch(Circle(p[:2], 0.05, facecolor=GREEN, alpha=0.2,
                                edgecolor=GREEN, lw=1.4, zorder=3))
            ax.plot([p[0] - 0.035, p[0] + 0.035], [p[1] + 0.035, p[1] - 0.035],
                    color=GREEN, lw=1.8, zorder=4)
        elif k in S["guard"]:
            re = r_obs(k) + EEF_R + D_SAFE
            ax.add_patch(Circle(p[:2], re, facecolor=CRIMSON, alpha=0.22,
                                edgecolor=CRIMSON, lw=1.2, zorder=3))
    label(ax, S["cands"]["cookies_1"][:2], "no barrier", DGREEN, dy=-0.075)
    hk = S["guard"][0]
    label(ax, S["cands"][hk][:2], "barrier", CRIMSON,
          dy=-(r_obs(hk) + EEF_R + D_SAFE) * 0.6)


def main():
    S = {"oah": load("oah_L2_t14_ep7"), "oa": load("oa_L2_t11_ep1"),
         "hs": load("hs_L0_cookies_ep0")}
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.9), dpi=300,
                             gridspec_kw=dict(height_ratios=[1, 1]))
    titles = [r"$\textsc{Protected}(x)$", r"$\textsc{NearestToPath}(x)$",
              r"$\lnot\textsc{Mentioned}(x)\wedge\lnot\textsc{Target}(x)$"]
    titles = ["Protected(x)", "NearestToPath(x)", "¬Mentioned(x) ∧ ¬Target(x)"]
    top_protected(axes[0, 0], S["oah"])
    top_nearpath(axes[0, 1], S["oa"])
    top_exempt(axes[0, 2], S["hs"])
    bot_protected(axes[1, 0], S["oah"])
    bot_nearpath(axes[1, 1], S["oa"])
    bot_exempt(axes[1, 2], S["hs"])
    for ax, t, key in zip(axes[0], titles, ("oah", "oa", "hs")):
        ax.set_title(t, fontsize=FS + 2.5, pad=14)
        ax.text(0.5, 1.01, "“" + S[key]["desc"] + "”", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=FS - 1.5, style="italic",
                color="0.35")
    axes[0, 0].set_ylabel("election\n(real frame)", fontsize=FS + 1)
    axes[1, 0].set_ylabel("barrier knob\n(schematic, real positions)",
                          fontsize=FS + 1)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.90, bottom=0.01,
                        wspace=0.04, hspace=0.06)
    for e in ("png", "pdf"):
        p = FIGDIR / f"fig_fol_literals.{e}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03)
        print("wrote", p)


if __name__ == "__main__":
    main()
