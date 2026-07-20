"""Sim-name-free object inventory study (19 saved obstacle scenes).

Question: can the candidate NAME LIST come from pixels instead of the
simulator's `*_pos` keys without hurting identification?

Arms:
  ref  sim names (current tier)                 — reference 17/19 argmax, 19/19 top-2
  som  Set-of-Marks: seg regions (class-agnostic proposal stand-in for SAM)
       + ONE haiku call per scene names every numbered region; candidates =
       {vlm_name: percep region centroid}. Novel names rated once with the
       anchored property prompt and merged into the prior dict.
  inv  whole-image inventory (one haiku call per TASK, amortized): recall
       check only — does the free-form list mention the true obstacle?

Metrics: argmax hit + guard-set top-2 hit, where a hit = picked candidate's
world centroid within 10 cm of the GT obstacle center (names never compared
to GT strings — position is the ground truth).

Credits: ~19 + 4 + (novel x 3) haiku calls, one-time.
"""

import base64
import io
import json
import os
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import extent_id_smoke as eis  # noqa: E402
import guard_set as gs  # noqa: E402
import symbolic_identity as si  # noqa: E402
import vlm_slot_bench as vb  # noqa: E402

MODEL = os.environ.get("INV_MODEL", "claude-haiku-4-5-20251001")
MODEL_TAG = MODEL.split("-")[1]
SCRATCH = pathlib.Path(os.environ.get(
    "SOM_SCRATCH",
    "/ocean/projects/cis250185p/asingal/tmp/claude-101024/-ocean-projects-cis250185p-asingal-situational-safety-for-embodied-agents/b45c2fca-41cc-4fbc-8f56-007885d7e3ef/scratchpad/som"))
SCRATCH.mkdir(parents=True, exist_ok=True)
PRIORS_PATH = pathlib.Path(__file__).parents[1] / "results_tables/vlm_hazard_priors.json"


def client():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def ask(cl, prompt, image_path, max_tokens=400):
    img = base64.standard_b64encode(open(image_path, "rb").read()).decode()
    r = cl.messages.create(model=MODEL, max_tokens=max_tokens, messages=[{
        "role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
            {"type": "text", "text": prompt}]}])
    return next(b.text for b in r.content if getattr(b, "type", "") == "text")


def scene_regions(ep_dir):
    """Class-agnostic region proposals: (pixel centroid, world centroid)."""
    ep = pathlib.Path(ep_dir)
    cp = json.load(open(ep / "camera_params.json"))["agentview"]
    K = np.array(cp["intrinsic"]); T = np.array(cp["extrinsic"])
    depth = np.load(ep / "agentview_depth.npy").squeeze()[::-1, ::-1]
    seg = np.load(ep / "agentview_seg.npy").squeeze()[::-1, ::-1]
    regions = []
    for sid in np.unique(seg):
        if sid <= 0:
            continue
        mask = seg == sid
        cloud = eis.region_cloud(depth, mask, K, T)
        if cloud is None or len(cloud) < 60:
            continue
        cen = np.median(cloud, axis=0)
        if not (0.75 < cen[2] < 1.3):
            continue
        if not (-0.5 < cen[0] < 0.5 and -0.5 < cen[1] < 0.5):
            continue
        ys, xs = np.nonzero(mask)
        # seg/depth arrays above are flipped copies; PNG on disk matches the
        # UNflipped npy orientation, so flip pixel coords back for drawing.
        H, W = seg.shape
        regions.append({"px": (W - 1 - int(np.median(xs)), H - 1 - int(np.median(ys))),
                        "world": cen, "npix": int(mask.sum()), "sid": int(sid)})
    return regions


def crop_montage(ep_dir, regions, out_png):
    """Numbered tile grid of per-region crops — much easier to name than
    markers on the cluttered full scene (haiku called the moka pot a shirt)."""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as _np
    ep = pathlib.Path(ep_dir)
    seg = _np.load(ep / "agentview_seg.npy").squeeze()[::-1, ::-1]
    img = Image.open(ep / "agentview_rgb.png").convert("RGB")
    W, H = img.size
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    tiles = []
    for i, r in enumerate(regions, 1):
        sid = r["sid"]
        ys, xs = _np.nonzero(seg == sid)
        # flip mask coords into PNG orientation
        ys, xs = H - 1 - ys, W - 1 - xs
        x0, x1 = max(int(xs.min()) - 12, 0), min(int(xs.max()) + 12, W)
        y0, y1 = max(int(ys.min()) - 12, 0), min(int(ys.max()) + 12, H)
        tile = img.crop((x0, y0, x1, y1)).resize((140, 140))
        dr = ImageDraw.Draw(tile)
        dr.rectangle([0, 0, 34, 24], fill=(0, 0, 0))
        dr.text((6, 2), str(i), fill=(255, 255, 0), font=font)
        tiles.append(tile)
    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    board = Image.new("RGB", (cols * 146, rows * 146), (255, 255, 255))
    for j, t in enumerate(tiles):
        board.paste(t, ((j % cols) * 146 + 3, (j // cols) * 146 + 3))
    board.save(out_png)


def mark_image(ep_dir, regions, out_png):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(pathlib.Path(ep_dir) / "agentview_rgb.png").convert("RGB")
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    for i, r in enumerate(regions, 1):
        x, y = r["px"]
        dr.ellipse([x - 14, y - 14, x + 14, y + 14], outline=(255, 0, 0), width=3)
        dr.text((x - 7, y - 12), str(i), fill=(255, 255, 0), font=font)
    img.save(out_png)


SOM_PROMPT = (
    "This is a robot workspace photo. Red numbered circles mark objects on the "
    "table. For each number give a short generic object name (1-3 lowercase "
    "words, e.g. 'moka pot', 'black bowl'). Reply with ONLY a JSON object "
    "mapping number to name, like {\"1\": \"plate\", \"2\": \"wine bottle\"}.")

CROP_PROMPT = (
    "Each numbered tile shows one object cropped from a robot tabletop "
    "workspace photo (rendered simulation). For each tile give a short "
    "generic name (1-3 lowercase words) and whether it is a movable tabletop "
    "object (true) or a fixture/appliance/furniture/robot part (false). "
    "Reply with ONLY a JSON object mapping tile number to [name, movable], "
    "like {\"1\": [\"plate\", true], \"2\": [\"stove\", false]}.")

INV_PROMPT = (
    "List every distinct object on the table in this robot workspace photo. "
    "Short generic names, one per line, no commentary.")


def main():
    cl = client()
    eps_list = vb.load_episodes(["safelibero_spatial"], 5)
    base_priors = json.load(open(PRIORS_PATH))
    rated = dict(base_priors)
    backend = vb.make_backend("anthropic:claude-haiku-4-5-20251001")  # ratings stay haiku (production prior model)

    ref_top1 = ref_top2 = som_top1 = som_top2 = n = 0
    inv_seen = {}
    inv_recall = []
    rows = []
    for e in eps_list:
        gt_pos = np.array(json.load(open(pathlib.Path(e["ep"]) / "obstacle.json"))["active_obstacle"]["position"])

        def hit(name, cands):
            return name is not None and np.linalg.norm(np.asarray(cands[name])[:2] - gt_pos[:2]) < 0.10

        # ---- ref arm (sim names)
        pick = si.scored_obstacle_id(e["task"], e["cands"], e["eef"])
        top2 = gs.guard_set(e["task"], e["cands"], e["eef"], k=2)
        r1 = bool(hit(pick, e["cands"])); r2 = bool(any(hit(t, e["cands"]) for t in top2))

        # ---- som arm
        regions = scene_regions(e["ep"])
        tag = e["ep"].replace("/", "_")
        png = SCRATCH / f"{tag}_crops.png"
        cache = SCRATCH / f"{tag}_cropnames_mv_{MODEL_TAG}.json"
        if cache.exists():
            names = json.load(open(cache))
        else:
            crop_montage(e["ep"], regions, png)
            txt = ask(cl, CROP_PROMPT, png)
            m = re.search(r"\{.*\}", txt, re.S)
            names = json.loads(m.group(0)) if m else {}
            json.dump(names, open(cache, "w"))
        cands = {}
        ROBOT_WORDS = ("robot", "arm", "gripper", "base", "manipulator", "claw", "hand")
        for i, r in enumerate(regions, 1):
            ent = names.get(str(i))
            if not ent:
                continue
            if isinstance(ent, list):
                nm, movable = ent[0], bool(ent[1])
            else:
                nm, movable = ent, True
            if not movable:
                continue
            # A region the VLM names as robot anatomy is the AGENT, not an
            # obstacle candidate — sim-name arm never sees these because its
            # inventory is objects-only. Name-based exclusion is legitimate
            # here: the name came from pixels, not the simulator.
            if any(w in nm.lower().split() for w in ROBOT_WORDS):
                continue
            key = nm if nm not in cands else f"{nm} {i}"
            cands[key] = r["world"]
        # rate novel names once (anchored prompt, 3 votes)
        for nm in cands:
            cn = si._clean_object_name(nm).lower()
            if cn not in rated:
                rc = SCRATCH / "rating_cache.json"
                cache_r = json.load(open(rc)) if rc.exists() else {}
                if cn in cache_r:
                    rated[cn] = cache_r[cn]
                else:
                    w, _detail = vb.rate_name(backend, cn, 3, "anchored")
                    rated[cn] = 0.4 if w is None else float(w)
                    cache_r[cn] = rated[cn]
                    json.dump(cache_r, open(rc, "w"))
        si._VLM_PRIORS = {k: max(v, 0.5) for k, v in rated.items()}
        pick_s = si.scored_obstacle_id(e["task"], cands, e["eef"]) if cands else None
        top2_s = gs.guard_set(e["task"], cands, e["eef"], k=2) if cands else []
        s1 = bool(hit(pick_s, cands)) if pick_s else False
        s2 = bool(any(hit(t, cands) for t in top2_s))

        # ---- inv arm (once per task dir)
        task_dir = str(pathlib.Path(e["ep"]).parent)
        if task_dir not in inv_seen:
            ic = SCRATCH / f"inv_{pathlib.Path(task_dir).name}_{MODEL_TAG}.txt"
            if ic.exists():
                inv_seen[task_dir] = ic.read_text()
            else:
                inv_seen[task_dir] = ask(cl, INV_PROMPT, pathlib.Path(e["ep"]) / "agentview_rgb.png", 300).lower()
                ic.write_text(inv_seen[task_dir])
        gt_tokens = [w for w in si._clean_object_name(
            json.load(open(pathlib.Path(e["ep"]) / "obstacle.json"))["active_obstacle"]["name"]).lower().split()
            if len(w) >= 3 and w != "obstacle"]
        inv_recall.append(bool(any(t in inv_seen[task_dir] for t in gt_tokens)))

        n += 1
        ref_top1 += r1; ref_top2 += r2; som_top1 += s1; som_top2 += s2
        rows.append({"ep": tag, "ref": [r1, r2], "som": [s1, s2],
                     "som_pick": pick_s, "n_regions": len(regions),
                     "n_named": len(cands)})

    for r in rows:
        print(json.dumps(r))
    print(json.dumps({
        "n": n,
        "ref": {"top1": ref_top1, "top2": ref_top2},
        "som": {"top1": som_top1, "top2": som_top2},
        "inventory_recall": f"{sum(inv_recall)}/{len(inv_recall)}",
        "novel_names_rated": len(rated) - len(base_priors),
    }, indent=1))
    json.dump(rated, open(SCRATCH / "merged_priors.json", "w"), indent=1)


if __name__ == "__main__":
    main()
