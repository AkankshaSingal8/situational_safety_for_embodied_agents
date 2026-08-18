"""Paper figure: what each FOL conjunct buys (4-panel ablation on one scene).

HAZARD(x) := PROTECTED(x) | MOVING(x) | (PRED(x) & ~MENTIONED(x) & ~TARGET(x))

Panels (all on ONE cached LIBERO-Safety scene, guard sets computed by calling
the REAL election functions `symbolic_identity.ablated_obstacle_id`, top-k=2,
the frozen production config):
  (a) Full FOL            -- protected hand elected first; task bowls exempt.
  (b) - logic (geom only) -- nearest-to-path top-k: guards the task's own
                             bowl and MISSES the hand.
  (c) - exemptions        -- a task-mentioned bowl is guarded (blocks task).
  (d) - class predicates  -- the human hand is left UNGUARDED.

Scene: human_safety L0 "pick the akita black bowl next to the cookies box and
bring it for me" (results_tables/libero_safety_scenes.jsonl) -- chosen because
it is the scene where all four guard sets differ pairwise (verified offline).

Two rendering modes (--mode, default auto):
  rgb       -- overlays on the captured agentview frame (needs a scene dir
               from vlm_pipeline/capture_scene_rgb.py: agentview_rgb.png +
               camera_params.json + scene.json). Projection reuses
               vlm_pipeline/visual_conditioning.py's validated conventions
               (`_project`/`_pixel_radius`, client rotation [::-1, ::-1]).
  schematic -- top-down workspace plan view from the cached jsonl geometry
               (no camera needed). Fallback when no capture exists.
`auto` picks rgb when --scene_dir has the three files, else schematic.

Guard footprints use r=0.09 m, the guard-circle radius convention from
vlm_pipeline/visual_conditioning.py.

CPU-only, no sim, no GPU, no API calls.
Output: paper/figs/fig_fol_ablation.png + .pdf (dpi=300)
Run:    python3 paper/figs/fig_fol_ablation.py [--mode rgb|schematic|auto]
"""

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vlm_pipeline"))
import symbolic_identity as si  # noqa: E402  (real election functions)
import visual_conditioning as vc  # noqa: E402  (validated projection)

SCENES = ROOT / "results_tables" / "libero_safety_scenes.jsonl"
FIGDIR = pathlib.Path(__file__).resolve().parent
DEFAULT_SCENE_DIR = FIGDIR / "scene_cache" / "hs_L0_cookies_ep0"

# Frozen production config (task-2-brief / fol_ablation_binding.py).
GUARD_TOPK = 2
GUARD_R = 0.09  # visual_conditioning.py guard-circle radius convention (m)

# The one cached scene where all four panels' guard sets differ pairwise
# (verified by running ablated_obstacle_id over the full 60-scene cache).
SCENE_KEY = dict(
    suite="human_safety", level=0,
    task="pick_the_akita_black_bowl_next_to_the_cookies_box_and_bring_it_for_me",
)

# Ground-truth semantic roles for annotation (this scene only): the task
# target is the bowl next to the cookies box; the true hazard is the hand.
TRUE_TARGET = "akita_black_bowl_1"
TRUE_HAZARD = "left_hand_1"

PANELS = [
    ("(a) Full FOL", "none", "hand guarded; task bowls exempt"),
    (r"(b) $-$ logic (geom only)", "geom_only", "guards task bowl, misses hand"),
    (r"(c) $-$ exemptions", "no_exempt", "task-mentioned bowl guarded"),
    (r"(d) $-$ class predicates", "no_class", "human hand left unguarded"),
]

SHORT = {
    "akita_black_bowl_1": "bowl 1\n(target)",
    "akita_black_bowl_2": "bowl 2",
    "cookies_1": "cookies",
    "glazed_rim_porcelain_ramekin_1": "ramekin",
    "left_hand_1": "hand",
}

FS = 8  # base font size (paper 8pt-equivalent)


def load_cached_scene():
    for line in open(SCENES):
        r = json.loads(line)
        if all(r.get(k) == v for k, v in SCENE_KEY.items()):
            return r
    raise SystemExit(f"scene {SCENE_KEY} not found in {SCENES}")


def compute_guards(desc, cands, eef):
    """Four guard sets from the REAL election dispatcher (id_source='fol',
    the frozen production routing for the OA/OAH/HS family)."""
    guards = {}
    for _, ablation, _ in PANELS:
        ranked = si.ablated_obstacle_id("fol", ablation, desc, cands, eef,
                                        moving=None)
        guards[ablation] = ranked[:GUARD_TOPK]
        print(f"guard[{ablation:9s}] = {guards[ablation]}")
    distinct = {tuple(sorted(g)) for g in guards.values()}
    assert len(distinct) == 4, (
        f"panels no longer all differ on this scene: {guards}")
    return guards


def compute_mistakes(guards):
    """Per-panel annotation: (wrongly guarded, hazard missed)."""
    full = set(guards["none"])
    mistakes = {"none": ([], [])}
    for ab in ("geom_only", "no_exempt", "no_class"):
        g = set(guards[ab])
        wrong = sorted((g - full) | ({TRUE_TARGET} & g))
        missed = sorted({TRUE_HAZARD} - g)
        mistakes[ab] = (wrong, missed)
    return mistakes


def finish(fig, axes, desc, mode):
    handles = [
        mlines.Line2D([], [], marker="o", ls="none", ms=10, mfc="crimson",
                      mec="crimson", alpha=0.4, label="guarded (top-2)"),
        mlines.Line2D([], [], marker="o", ls="none", ms=9, mfc="none",
                      mec="#1faa4b", mew=1.8, label="task target"),
        mlines.Line2D([], [], marker="o", ls="none", ms=9, mfc="none",
                      mec="darkorange", mew=1.8, label="wrongly guarded"),
        mlines.Line2D([], [], marker="o", ls="none", ms=9, mfc="none",
                      mec="darkorange", mew=1.8,
                      label="hazard unguarded (dashed)"),
    ]
    if mode == "schematic":
        handles.append(mlines.Line2D([], [], marker="o", ls="none", ms=8,
                                     mfc="#7b5cd6", mec="k",
                                     label="human hand"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=FS - 0.5, frameon=False,
               bbox_to_anchor=(0.5, -0.005), handletextpad=0.4,
               columnspacing=1.2)
    fig.suptitle(f'“{desc}”', fontsize=FS + 1.5, y=1.0)
    for ext in ("png", "pdf"):
        p = FIGDIR / f"fig_fol_ablation.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03)
        print("wrote", p)


# --------------------------------------------------------------------------
# RGB overlay mode
# --------------------------------------------------------------------------

def render_rgb(scene_dir):
    scene = json.load(open(scene_dir / "scene.json"))
    cam = json.load(open(scene_dir / "camera_params.json"))
    camera = scene.get("camera", "agentview")
    K = np.asarray(cam[camera]["intrinsic"])
    ext = np.asarray(cam[camera]["extrinsic"])
    raw = plt.imread(scene_dir / f"{camera}_rgb.png")  # float [0,1] or uint8
    if raw.dtype != np.uint8:
        raw = (raw[..., :3] * 255).astype(np.uint8)
    img = np.ascontiguousarray(raw[::-1, ::-1])  # client policy-input rotation
    h, w = img.shape[:2]

    desc = scene["language"]
    cands = {k: np.asarray(v, float) for k, v in scene["obj_pos"].items()}
    eef = np.asarray(scene["eef_pos"], float)

    # Cross-check the capture against the cached jsonl record the offline
    # binding study used (same init state => positions within ~1 cm).
    cached = load_cached_scene()
    worst = 0.0
    for k, v in cached["obj_pos"].items():
        if k in cands:
            worst = max(worst, float(np.linalg.norm(cands[k] - np.asarray(v))))
    print(f"capture-vs-cache max position delta: {worst * 100:.2f} cm")
    if worst > 0.02:
        print("WARNING: capture does not match the cached scene record "
              "(different init state?) -- guard sets recomputed on the "
              "capture's own geometry")

    guards = compute_guards(desc, cands, eef)
    mistakes = compute_mistakes(guards)

    def to_px(pos_w):
        """World -> (col,row) in the ROTATED (policy-view) image.

        capture_scene_rgb.py saves the RAW obs render (mujoco frame, 180 deg
        from the policy view), unlike the old vlm_inputs captures which were
        saved pre-rotated; after this file's `[::-1, ::-1]` rotation,
        `vc._project`'s (row, col) applies DIRECTLY -- no further flip.
        Verified per-object on the hs_L0 capture (markers land on the hand /
        ramekin / cookies / both bowls)."""
        rc, z = vc._project(K, ext, [pos_w])
        r, c = rc[0]
        return (c, r), float(z[0])

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.75), dpi=300)
    for ax, (title, ablation, subtitle) in zip(axes, PANELS):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.6")

        # translucent red guard footprints
        for k in guards[ablation]:
            p = cands[k]
            (cx, cy), z = to_px(p)
            if z <= 0.05:
                continue
            r_px = vc._pixel_radius(K, ext, p, GUARD_R)
            ax.add_patch(Circle((cx, cy), r_px, facecolor="crimson",
                                alpha=0.30, edgecolor="crimson", lw=1.4))

        # green ring: task target
        (cx, cy), _ = to_px(cands[TRUE_TARGET])
        r_t = vc._pixel_radius(K, ext, cands[TRUE_TARGET], 0.055)
        ax.add_patch(Circle((cx, cy), r_t, facecolor="none",
                            edgecolor="#1faa4b", lw=2.0))

        # orange mistake rings
        wrong, missed = mistakes[ablation]
        for k in wrong:
            (cx, cy), _ = to_px(cands[k])
            r_o = vc._pixel_radius(K, ext, cands[k], GUARD_R + 0.03)
            ax.add_patch(Circle((cx, cy), r_o, facecolor="none",
                                edgecolor="darkorange", lw=2.0))
        for k in missed:
            (cx, cy), _ = to_px(cands[k])
            r_o = vc._pixel_radius(K, ext, cands[k], 0.065)
            ax.add_patch(Circle((cx, cy), r_o, facecolor="none",
                                edgecolor="darkorange", lw=2.0,
                                linestyle=(0, (2.2, 2.2))))

        ax.set_title(title, fontsize=FS + 2, pad=15)
        ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=FS - 1, color="0.35")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.82, bottom=0.09,
                        wspace=0.03)
    finish(fig, axes, desc, "rgb")


# --------------------------------------------------------------------------
# Schematic (plan view) fallback mode
# --------------------------------------------------------------------------

def render_schematic():
    rec = load_cached_scene()
    desc = rec["language"]
    cands = {k: np.asarray(v, float) for k, v in rec["obj_pos"].items()}
    eef = np.asarray(rec["eef_pos"], float)
    guards = compute_guards(desc, cands, eef)
    mistakes = compute_mistakes(guards)

    xs = [p[0] for p in cands.values()] + [eef[0]]
    ys = [p[1] for p in cands.values()] + [eef[1]]
    pad = 0.16
    xlim = (min(xs) - pad, max(xs) + pad)
    ylim = (min(ys) - pad, max(ys) + pad)

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.75), dpi=300)
    for ax, (title, ablation, subtitle) in zip(axes, PANELS):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.6")
        ax.add_patch(Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0],
                               ylim[1] - ylim[0], facecolor="#f3ede2",
                               edgecolor="none", zorder=0))
        tp = cands[TRUE_TARGET]
        ax.add_patch(FancyArrowPatch((eef[0], eef[1]), (tp[0], tp[1]),
                                     arrowstyle="-|>", mutation_scale=11,
                                     linestyle=(0, (3, 3)), color="0.45",
                                     lw=1.0, zorder=2))
        for k in guards[ablation]:
            p = cands[k]
            ax.add_patch(Circle((p[0], p[1]), GUARD_R, facecolor="crimson",
                                alpha=0.28, edgecolor="crimson", lw=1.4,
                                zorder=3))
        for k, p in cands.items():
            hand = k == TRUE_HAZARD
            ax.add_patch(Circle((p[0], p[1]), 0.036 if hand else 0.030,
                                facecolor="#7b5cd6" if hand else "0.55",
                                edgecolor="k", lw=0.7, zorder=4))
            dy = 0.045 if p[1] < (ylim[0] + ylim[1]) / 2 else -0.045
            va = "bottom" if dy > 0 else "top"
            ax.text(p[0], p[1] + dy * (1.6 if hand else 1.0),
                    SHORT.get(k, k), ha="center", va=va, fontsize=FS - 1.5,
                    zorder=6, linespacing=0.95)
        ax.plot(eef[0], eef[1], marker="^", ms=7, mfc="k", mec="k", zorder=5)
        ax.text(eef[0], eef[1] - 0.045, "eef", ha="center", va="top",
                fontsize=FS - 1.5)
        ax.add_patch(Circle((tp[0], tp[1]), 0.052, facecolor="none",
                            edgecolor="#1faa4b", lw=1.8, zorder=5))
        wrong, missed = mistakes[ablation]
        for k in wrong:
            p = cands[k]
            ax.add_patch(Circle((p[0], p[1]), 0.115, facecolor="none",
                                edgecolor="darkorange", lw=1.8, zorder=5))
        for k in missed:
            p = cands[k]
            ax.add_patch(Circle((p[0], p[1]), 0.062, facecolor="none",
                                edgecolor="darkorange", lw=1.8,
                                linestyle=(0, (2.2, 2.2)), zorder=5))
        ax.set_title(title, fontsize=FS + 2, pad=15)
        ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=FS - 1, color="0.35")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.82, bottom=0.09,
                        wspace=0.03)
    finish(fig, axes, desc, "schematic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto", "rgb", "schematic"],
                    default="auto")
    ap.add_argument("--scene_dir", default=str(DEFAULT_SCENE_DIR))
    args = ap.parse_args()
    scene_dir = pathlib.Path(args.scene_dir)
    have_capture = all((scene_dir / f).exists() for f in
                       ("scene.json", "camera_params.json"))
    if have_capture:
        cam = json.load(open(scene_dir / "camera_params.json"))
        camname = next(iter(cam))
        have_capture = (scene_dir / f"{camname}_rgb.png").exists()
    mode = args.mode
    if mode == "auto":
        mode = "rgb" if have_capture else "schematic"
        print(f"auto mode -> {mode}")
    if mode == "rgb":
        if not have_capture:
            raise SystemExit(f"no capture in {scene_dir}; run "
                             "vlm_pipeline/capture_scene_rgb.py first "
                             "(see paper/figs/capture_scenes.sh)")
        render_rgb(scene_dir)
    else:
        render_schematic()


if __name__ == "__main__":
    main()
