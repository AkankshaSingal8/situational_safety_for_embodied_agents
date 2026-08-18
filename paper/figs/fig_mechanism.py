"""Paper method figure: the FOL -> guard -> noise-space enforcement pipeline.

Four panels, left to right, on ONE captured LIBERO-Safety scene
(paper/figs/scene_cache/oah_L2_t14_ep7, obstacle_avoidance_human L2
"put the white yellow mug in the microwave and close it" -- a hand-attached
hazard: bottle_of_alcohol_1_with_hand_1):

  (a) Observation      -- raw agentview RGB (policy view) + the instruction.
  (b) FOL election     -- the HAZARD(x) rule + a predicate truth table for
                          this scene's REAL objects, every value computed by
                          the real code in vlm_pipeline/symbolic_identity.py
                          (fol arm, top-2, the frozen incumbent config).
  (c) Guard set        -- elected guard region projected onto the RGB frame
                          (r=0.09 m, visual_conditioning.py conventions).
  (d) Noise-space enforcement (schematic) -- K=8 flow-denoised candidate
                          chunks fanning from the eef in a top-down view of
                          the same scene geometry; green = certified prefix
                          safe, red = rejected, selected bold. Trajectories
                          are ILLUSTRATIVE (panel is labeled schematic);
                          object/eef positions are the scene's real ones.

Style matches fig_fol_ablation.py (fonts, (a)-(d) letters, dpi=300 PNG+PDF).
CPU-only, no sim, no GPU, no API calls.
Run: python3 paper/figs/fig_mechanism.py
"""

import json
import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vlm_pipeline"))
import symbolic_identity as si  # noqa: E402  (real election code)
import visual_conditioning as vc  # noqa: E402  (validated projection)

FIGDIR = pathlib.Path(__file__).resolve().parent
SCENE_DIR = FIGDIR / "scene_cache" / "oah_L2_t14_ep7"

GUARD_TOPK = 2   # frozen incumbent config
GUARD_R = 0.09   # visual_conditioning.py guard-circle radius convention (m)
FS = 8           # base font size (matches fig_fol_ablation.py)

PRETTY = {
    "white_yellow_mug_1": "white-yellow mug",
    "bottle_of_alcohol_1_with_hand_1": "alcohol bottle (in hand)",
}


def pretty(name):
    return PRETTY.get(name, re.sub(r"(_\d+)+$", "", name).replace("_", " "))


def load_scene():
    scene = json.load(open(SCENE_DIR / "scene.json"))
    cam = json.load(open(SCENE_DIR / "camera_params.json"))[scene["camera"]]
    raw = plt.imread(SCENE_DIR / f"{scene['camera']}_rgb.png")
    if raw.dtype != np.uint8:
        raw = (raw[..., :3] * 255).astype(np.uint8)
    img = np.ascontiguousarray(raw[::-1, ::-1])  # policy-view rotation; after
    # this, vc._project's (row, col) applies DIRECTLY (verified on the hs_L0
    # capture in fig_fol_ablation.py -- capture_scene_rgb.py saves raw obs).
    return scene, np.asarray(cam["intrinsic"]), np.asarray(cam["extrinsic"]), img


def election_truth_table(desc, cands, eef, moving=None):
    """All predicate values via the REAL module helpers; the HAZARD column is
    membership of the actual `fol_obstacle_id` top-k guard set."""
    tgt = si.parse_target_heuristic(desc, list(cands))
    tpos = cands.get(tgt, eef)
    guard = si.fol_obstacle_id(desc, cands, eef, moving=moving)[:GUARD_TOPK]
    rows = []
    for k, p in cands.items():
        rows.append({
            "name": k,
            "PROT": si._is_protected_or_moving(k, None),
            "MOV": bool(moving and k in moving),
            "NEAR": si.near_path(p, eef, tpos),
            "MENT": si._head_mentioned(k, desc),
            "TGT": k == tgt,
            "HAZARD": k in guard,
        })
    return rows, guard, tgt, tpos


# --------------------------------------------------------------------------

def panel_a(ax, img, desc):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a) Observation", fontsize=FS + 2, pad=6)
    # Two centered lines so the caption stays within panel (a)'s width.
    words = desc.split()
    half = (len(words) + 1) // 2
    cap = "“" + " ".join(words[:half]) + "\n" + " ".join(words[half:]) + "”"
    ax.text(0.5, -0.045, cap, transform=ax.transAxes, ha="center",
            va="top", fontsize=FS - 0.5, style="italic", linespacing=1.25)


def panel_b(ax, rows):
    ax.set_axis_off()
    ax.set_title("(b) FOL hazard election", fontsize=FS + 2, pad=6)
    # Plain text + unicode logic symbols: mathtext spaces ':' and '=' as two
    # relations (": ="), and \coloneqq is unsupported -- this keeps a clean
    # ":=" (verified in DejaVu Sans, which covers U+2228/2227/00AC).
    ax.text(0.5, 0.97,
            "HAZARD(x) := PROT(x) ∨ MOV(x) ∨\n"
            "(NEAR(x) ∧ ¬MENT(x) ∧ ¬TGT(x))",
            transform=ax.transAxes, ha="center", va="top", fontsize=FS,
            linespacing=1.5)

    cols = ["PROT", "MOV", "NEAR", "MENT", "TGT"]
    x_name, x0, dx = 0.02, 0.46, 0.096
    x_haz = x0 + len(cols) * dx + 0.015
    y0, dy = 0.60, 0.145

    ax.text(x_name, y0, "object", fontsize=FS - 1, style="italic",
            transform=ax.transAxes, va="center")
    for j, c in enumerate(cols):
        ax.text(x0 + j * dx, y0, c, fontsize=FS - 1.5, ha="center",
                va="center", transform=ax.transAxes)
    ax.text(x_haz, y0, r"$\rightarrow$HAZ", fontsize=FS - 1.5, ha="center",
            va="center", transform=ax.transAxes)
    ax.plot([0.01, 0.99], [y0 - 0.55 * dy] * 2, color="0.55", lw=0.7,
            transform=ax.transAxes, clip_on=False)

    for i, r in enumerate(rows):
        y = y0 - (i + 1) * dy
        exempt = (not r["HAZARD"]) and (r["MENT"] or r["TGT"])
        tint = ("#f6c9c9" if r["HAZARD"]
                else "#cdeccd" if exempt else "none")
        if tint != "none":
            ax.add_patch(plt.Rectangle((0.005, y - 0.48 * dy), 0.99,
                                       0.96 * dy, transform=ax.transAxes,
                                       facecolor=tint, edgecolor="none",
                                       zorder=0))
        ax.text(x_name, y, pretty(r["name"]), fontsize=FS - 1,
                transform=ax.transAxes, va="center")
        for j, c in enumerate(cols):
            ax.text(x0 + j * dx, y, "✓" if r[c] else "–", fontsize=FS - 1,
                    ha="center", va="center", transform=ax.transAxes)
        ax.text(x_haz, y, "✓" if r["HAZARD"] else "–", fontsize=FS - 1,
                ha="center", va="center", transform=ax.transAxes,
                fontweight="bold",
                color="crimson" if r["HAZARD"] else "#1faa4b")

    y_leg = y0 - (len(rows) + 1.15) * dy
    ax.text(0.02, y_leg, "elected guard", fontsize=FS - 1.5, va="center",
            transform=ax.transAxes,
            bbox=dict(facecolor="#f6c9c9", edgecolor="none", pad=2.2))
    ax.text(0.44, y_leg, "exempt (task language)", fontsize=FS - 1.5,
            va="center", transform=ax.transAxes,
            bbox=dict(facecolor="#cdeccd", edgecolor="none", pad=2.2))


def panel_c(ax, img, K, ext, cands, guard, tgt):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(c) Guard set", fontsize=FS + 2, pad=6)

    def to_px(pos_w):
        rc, z = vc._project(K, ext, [pos_w])
        return (rc[0][1], rc[0][0]), float(z[0])

    for k in guard:
        p = cands[k]
        (cx, cy), z = to_px(p)
        if z <= 0.05:
            continue
        r_px = vc._pixel_radius(K, ext, p, GUARD_R)
        ax.add_patch(Circle((cx, cy), r_px, facecolor="crimson", alpha=0.30,
                            edgecolor="crimson", lw=1.4))
        ax.annotate("guard\n" + r"$h(x)=0$", (cx + r_px * 0.72, cy - r_px * 0.72),
                    (cx + r_px * 1.55, cy - r_px * 1.4), fontsize=FS - 1.5,
                    color="crimson", ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color="crimson", lw=0.8))
    if tgt is not None:
        (cx, cy), _ = to_px(cands[tgt])
        r_t = vc._pixel_radius(K, ext, cands[tgt], 0.055)
        ax.add_patch(Circle((cx, cy), r_t, facecolor="none",
                            edgecolor="#1faa4b", lw=2.0))
        ax.text(cx, cy + r_t * 1.45, "target", fontsize=FS - 1.5,
                color="#1faa4b", ha="center", va="top", fontweight="bold")


def panel_d(ax, cands, eef, guard, tgt):
    """Illustrative K=8 candidate chunks in the scene's REAL geometry
    (top-down xy); safe/rejected classification against the real guard
    circle. Labeled schematic."""
    ax.set_title("(d) Noise-space enforcement\n(schematic)",
                 fontsize=FS + 2, pad=6, linespacing=1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_color("0.6")

    pts = [p[:2] for p in cands.values()] + [eef[:2]]
    xs, ys = zip(*pts)
    pad = 0.14
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_facecolor("#f3ede2")

    gpos = cands[guard[0]][:2]
    tpos = cands[tgt][:2] if tgt is not None else eef[:2]
    ax.add_patch(Circle(gpos, GUARD_R, facecolor="crimson", alpha=0.25,
                        edgecolor="crimson", lw=1.2, zorder=2))
    for k, p in cands.items():
        ax.add_patch(Circle(p[:2], 0.026, facecolor="0.55", edgecolor="k",
                            lw=0.6, zorder=3))
    ax.add_patch(Circle(tpos, 0.045, facecolor="none", edgecolor="#1faa4b",
                        lw=1.6, zorder=4))
    ax.plot(*eef[:2], marker="^", ms=6, mfc="k", mec="k", zorder=4)

    # K=8 quadratic-bezier candidates eef -> target: a deterministic lateral
    # fan, biased toward the guard side so the panel shows BOTH outcomes
    # (some chunks cross the guard region and are rejected).
    a, b = np.asarray(eef[:2]), np.asarray(tpos)
    mid = 0.5 * (a + b)
    lat = np.array([-(b - a)[1], (b - a)[0]])
    lat /= max(np.linalg.norm(lat), 1e-9)
    if np.dot(gpos - mid, lat) < 0:
        lat = -lat  # +offset now points toward the guard
    t = np.linspace(0, 1, 60)[:, None]
    offsets = np.linspace(-0.14, 0.46, 8)
    trajs, safe = [], []
    for off in offsets:
        ctrl = mid + off * lat
        traj = (1 - t) ** 2 * a + 2 * (1 - t) * t * ctrl + t ** 2 * b
        trajs.append(traj)
        prefix = traj[:36]  # checked prefix of the chunk
        safe.append(bool(np.min(np.linalg.norm(prefix - gpos, axis=1))
                         > GUARD_R))
    print("panel (d): %d/%d candidates certified safe" % (sum(safe), len(safe)))
    assert 0 < sum(safe) < len(safe), "fan must show both outcomes"
    sel = None
    for i in np.argsort(np.abs(offsets)):  # least-deviating safe candidate
        if safe[i]:
            sel = i
            break
    for i, traj in enumerate(trajs):
        if i == sel:
            continue
        ax.plot(traj[:, 0], traj[:, 1],
                color="#1faa4b" if safe[i] else "crimson",
                lw=1.0, alpha=0.65, zorder=5)
    st = trajs[sel]
    ax.plot(st[:, 0], st[:, 1], color="#1faa4b", lw=2.4, zorder=6)
    ax.plot(st[:36, 0], st[:36, 1], color="#0d6b30", lw=3.2, zorder=6)

    # Short leaders (offset-points textcoords keeps every leader ~a few mm),
    # each label adjacent to its anchor: fan origin, bold midpoint, prefix.
    ax.annotate("K candidates", trajs[0][6], xytext=(-6, 14),
                textcoords="offset points", fontsize=FS - 1.5, ha="left",
                va="bottom",
                arrowprops=dict(arrowstyle="-", color="0.3", lw=0.7))
    ax.annotate(r"$h \geq 0$ checked prefix", st[24], xytext=(-14, -16),
                textcoords="offset points", fontsize=FS - 1.5, ha="right",
                va="top",
                arrowprops=dict(arrowstyle="-", color="0.3", lw=0.7))
    ax.annotate("selected", st[44], xytext=(16, 4),
                textcoords="offset points", fontsize=FS - 1.5, ha="left",
                va="bottom",
                arrowprops=dict(arrowstyle="-", color="0.3", lw=0.7))


def main():
    scene, K, ext, img = load_scene()
    desc = scene["language"]
    cands = {k: np.asarray(v, float) for k, v in scene["obj_pos"].items()}
    eef = np.asarray(scene["eef_pos"], float)

    rows, guard, tgt, _ = election_truth_table(desc, cands, eef)
    print("target:", tgt)
    print("guard (fol, top-%d):" % GUARD_TOPK, guard)
    for r in rows:
        print("  ", {k: (v if k == "name" else int(v)) for k, v in r.items()})
    assert guard, "degenerate election on this scene -- use oa_L2_t11_ep1"

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6), dpi=300)
    panel_a(axes[0], img, desc)
    panel_b(axes[1], rows)
    panel_c(axes[2], img, K, ext, cands, guard, tgt)
    panel_d(axes[3], cands, eef, guard, tgt)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.10,
                        wspace=0.10)
    # inter-panel arrows in the gaps (figure coords)
    for ax_l, ax_r in zip(axes[:-1], axes[1:]):
        x0 = ax_l.get_position().x1
        x1 = ax_r.get_position().x0
        fig.add_artist(FancyArrowPatch((x0 + 0.12 * (x1 - x0), 0.5),
                                       (x1 - 0.12 * (x1 - x0), 0.5),
                                       transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=13,
                                       color="0.35", lw=1.2))
    for ext_ in ("png", "pdf"):
        p = FIGDIR / f"fig_mechanism.{ext_}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03)
        print("wrote", p)


if __name__ == "__main__":
    main()
