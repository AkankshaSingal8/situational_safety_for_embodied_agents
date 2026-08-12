"""Phase 3 (spec 2026-08-11) learned hazard localizer: model, training, and
offline G-L1 eval.

Motivation: the no-GT tier's GroundingDINO-based detector localizes hazards
with a structural bias, costing ~8.3 TSR points on obstacle_avoidance. This
module trains a small supervised conv net -- RGB in, 3D hazard position out
-- on frames + FREE sim-GT labels captured by
`run_guided_libero_safety_eval._build_frame_record` (see that docstring for
the exact npz keys: `agentview_rgb`, optional depth, per-camera
`camera_K_<cam>` / `camera_T_<cam>`, `entity_names` / `entity_pos`,
`eef_pos`, `t`, `guard`). Trained on ALL entities per frame (not just
guards) for more supervision from one net.

Architecture (kept simple, matched to runtime use -- NOT a per-entity model
zoo): a shared small CNN trunk (no pretrained-weight downloads -- offline
HPC nodes) predicts, from a single RGB frame:
  - a 2D heatmap over the image (where is this entity?)
  - a per-heatmap-cell depth value (how far is it, in camera frame?)
conditioned on a lightweight, open-vocabulary entity identity via a
hashed-token embedding table + FiLM modulation of the trunk features (no
text encoder, no fixed entity vocabulary -- an unseen entity name still gets
a deterministic embedding from its token hashes).

At inference: soft-argmax the heatmap -> pixel location; read the depth at
that cell; backproject with the SAME K/T (via `percep_obstacle._camera_transform`)
and the SAME validated pixel formula as `percep_obstacle._backproject`
(empirically confirmed at med 0.067 m on the live sim -- see the "Camera
geometry" comment block below for the one row-flip needed to reach that
validated orientation from our unflipped stored frames) -> 3D world
position.

Model input is RGB ONLY. Depth channels in the npz records (when present)
are never read by this module -- data may be RGB-only (no `camera_depths`
in the env build) and the model must work either way.

CLI:
    python vlm_pipeline/hazard_localizer.py verify --data DIR    # run FIRST on real data
    python vlm_pipeline/hazard_localizer.py train  --data DIR --out DIR
    python vlm_pipeline/hazard_localizer.py eval   --data DIR --model DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re
import statistics
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover - hard requirement, documented
    raise ImportError(
        "hazard_localizer.py requires torch (CPU is fine for tests; use "
        "the openvla_libero_merged conda env, which has torch 2.2.0)."
    ) from e


CAMERA_NAME = "agentview"
INPUT_SIZE = 256          # network input resolution (square)
STRIDE = 8                # trunk downsampling factor -> heatmap res = INPUT_SIZE // STRIDE
HEATMAP_SIZE = INPUT_SIZE // STRIDE
DEFAULT_HELD_OUT_TASKS = (2, 7, 12)  # deterministic default split-by-task
FRAME_RE = re.compile(r"task(?P<task>\d+)_ep(?P<ep>\d+)_replan(?P<replan>\d+)\.npz$")


# ---------------------------------------------------------------------------
# Camera geometry.
#
# K, T are obtained via the SAME `percep_obstacle._camera_transform` call
# (`get_camera_intrinsic_matrix` / `get_camera_extrinsic_matrix`) the runtime
# percep tier uses -- that part is unambiguous and shared verbatim (see
# `run_guided_libero_safety_eval.py`'s frame-logging call site, which
# literally imports `_camera_transform` from percep_obstacle).
#
# PIXEL FORMULA: the exact algebraic inverse of `percep_obstacle._backproject`
# (`p_cam = z * inv(K) @ [u, v, 1]`, naive K, NO column negation). That
# formula is not a guess -- percep_obstacle's module docstring states it is
# empirically validated at runtime on the LIVE sim, med 0.067 m / p90 0.152 m
# (ledger 2026-07-16), and explicitly says "the [saved-array orientation]
# bug lives only in the offline captures" (i.e. NOT on this live-sim path).
# `visual_conditioning.py`'s independently-derived x-negation (calibrated on
# saved PNG captures, a different code path with its own history of
# orientation bugs) was a false lead for this module and has been removed --
# stacking a second module's fix onto a formula already validated on the
# path we care about double-corrects a problem that doesn't exist here.
#
# ROW ORIENTATION: percep_obstacle's `estimate_obstacle_pos` builds its pixel
# regions from `sim.render(...)[::-1]` (one row-flip) before calling
# `_backproject` -- i.e. the *validated* convention pairs the naive formula
# with a row-0-at-top array. `_build_frame_record` stores
# `obs["agentview_image"]` UNFLIPPED. Checked directly against the robosuite
# fork actually on PYTHONPATH at runtime -- this module's own docstring says
# "PYTHONPATH must put LIBERO-Safety BEFORE SafeLIBERO", i.e.
# `LIBERO-Safety/third_party/robosuite-1.4/robosuite/`, not whatever
# robosuite happens to be pip-installed in the conda env -- its
# `environments/robot_env.py` `camera_rgb` sensor does
# `return img[::convention]` with `macros.IMAGE_CONVENTION = "opengl"` ->
# `IMAGE_CONVENTION_MAPPING["opengl"] == 1` (identity slice, i.e. NO flip),
# and no `macros_private.py` override exists anywhere under that tree. So
# `obs["agentview_image"]` is byte-identical to raw `sim.render(...)` output
# -- the SAME raw, row-0-at-bottom array percep_obstacle itself flips before
# it is safe to use with K/T (also consistent with `save_vlm_inputs.py`'s
# "robosuite returns images upside-down; flip vertically" comment on that
# same raw obs array). So our stored `agentview_rgb` needs the identical
# one-row-flip percep_obstacle applies to reach the validated orientation;
# `img_h` (always known at index/predict time) parameterizes that mirror.
# `backproject` is the exact algebraic inverse of `project_world_to_pixel`,
# which the round-trip test checks; the `verify` CLI subcommand is the
# empirical backstop once real frames exist (in-bounds fraction + depth-map
# cross-check) -- run FIRST in slurm/localizer_train.slurm, hard-aborting
# before any GPU training time is spent if this derivation is wrong.
# NOTE: the in-bounds fraction is provably INVARIANT to a pure row-mirror
# (a mirror maps in-bounds pixels to in-bounds pixels), so `verify` alone
# cannot discriminate mirror-vs-no-mirror. That question was settled
# empirically by visual QC on real collected frames (2026-08-12): GT entity
# positions projected under THIS convention land exactly on their objects
# in the stored images across sampled oa/oah frames at all levels; the
# no-mirror variant scatters them into empty space.
# ---------------------------------------------------------------------------

def project_world_to_pixel(p_w, K, T, img_h):
    """World 3D -> (u, v, z) in OUR STORED (unflipped) image's pixel frame.
    z is camera-frame depth (behind-camera points return None or z <= 0)."""
    p_w = np.asarray(p_w, dtype=np.float64)
    p_cam_h = np.linalg.inv(T) @ np.array([*p_w, 1.0])
    cam_x, cam_y, z = p_cam_h[0], p_cam_h[1], p_cam_h[2]
    if z == 0.0:
        return None
    uv1 = (K @ np.array([cam_x, cam_y, z])) / z              # percep_obstacle's validated naive formula
    col, row_validated = float(uv1[0]), float(uv1[1])        # row-0-at-top (the validated orientation)
    row_stored = (img_h - 1) - row_validated                  # our stored image is row-flipped relative to it
    return col, row_stored, float(z)


def backproject(u, v, z, K, T, img_h):
    """Pixel (u, v) in OUR STORED (unflipped) image's pixel frame + depth z
    -> world 3D. Exact algebraic inverse of `project_world_to_pixel`."""
    row_validated = (img_h - 1) - v
    p_cam = z * (np.linalg.inv(K) @ np.array([u, row_validated, 1.0]))
    p_w = T @ np.array([*p_cam, 1.0])
    return p_w[:3]


# ---------------------------------------------------------------------------
# Frame indexing (path layout mirrors `_percep_frame_path`:
# DIR/<suite>/L<level>/task<T>_ep<E>_replan<K>.npz)
# ---------------------------------------------------------------------------

def index_frame_files(root):
    """Return list of dicts: path, suite, level, task_id, ep, replan.
    Pure filesystem walk + filename parsing (no npz I/O)."""
    root = pathlib.Path(root)
    out = []
    for path in sorted(root.rglob("*.npz")):
        m = FRAME_RE.search(path.name)
        if not m:
            continue
        level_dir = path.parent.name  # "L<level>"
        suite = path.parent.parent.name
        level = level_dir[1:] if level_dir.startswith("L") else level_dir
        out.append({
            "path": str(path),
            "suite": suite,
            "level": level,
            "task_id": int(m.group("task")),
            "ep": int(m.group("ep")),
            "replan": int(m.group("replan")),
        })
    return out


def index_entity_samples(frame_files, camera_name=CAMERA_NAME, max_oob_frac=None):
    """For each indexed frame file, open the npz once, project every scene
    entity into the named camera, and emit one sample dict per entity whose
    projection lands inside the image and in front of the camera. Skips
    frames missing the RGB, camera K/T, or entity arrays entirely (these are
    already recorded in the frame's `skipped` list at collection time).

    `max_oob_frac` (None = off): training-time tripwire. Among entities that
    project in FRONT of the camera (z > 0), if more than this fraction land
    OUTSIDE the image bounds, raise -- a high out-of-bounds rate on live
    data is the signature of a camera/pixel-convention mismatch (row flip,
    column order, wrong camera), not of scenes genuinely having most
    objects off-screen. See the "Camera geometry" comment above and the
    `verify` CLI subcommand, which runs this same check standalone before
    a GPU training job starts."""
    samples = []
    n_forward = 0
    n_oob = 0
    for f in frame_files:
        try:
            with np.load(f["path"], allow_pickle=False) as z:
                if "agentview_rgb" not in z or f"camera_K_{camera_name}" not in z \
                        or f"camera_T_{camera_name}" not in z or "entity_names" not in z:
                    continue
                rgb_shape = z["agentview_rgb"].shape
                K = z[f"camera_K_{camera_name}"]
                T = z[f"camera_T_{camera_name}"]
                names = [str(n) for n in z["entity_names"]]
                positions = z["entity_pos"]
                guard = set(str(g) for g in z["guard"]) if "guard" in z else set()
        except Exception:
            continue

        h_orig, w_orig = rgb_shape[0], rgb_shape[1]
        for name, pos in zip(names, positions):
            proj = project_world_to_pixel(pos, K, T, h_orig)
            if proj is None:
                continue
            u, v, depth = proj
            if depth <= 0.0:
                continue
            n_forward += 1
            if not (0.0 <= u < w_orig and 0.0 <= v < h_orig):
                n_oob += 1
                continue
            samples.append({
                "path": f["path"],
                "suite": f["suite"],
                "level": f["level"],
                "task_id": f["task_id"],
                "ep": f["ep"],
                "replan": f["replan"],
                "entity_name": name,
                "entity_pos": [float(x) for x in pos],
                "u": u, "v": v, "depth": depth,
                "w_orig": w_orig, "h_orig": h_orig,
                "K": K.tolist(), "T": T.tolist(),
                "is_guard": name in guard,
            })

    if max_oob_frac is not None and n_forward > 0:
        oob_frac = n_oob / n_forward
        if oob_frac > max_oob_frac:
            raise ValueError(
                f"index_entity_samples: {oob_frac:.1%} ({n_oob}/{n_forward}) "
                f"of forward-facing (z>0) projected entity targets landed "
                f"OUTSIDE the image bounds, exceeding the {max_oob_frac:.0%} "
                f"tripwire. This is the signature of a camera/pixel "
                f"convention mismatch, not of scenes genuinely having most "
                f"objects off-screen -- see the 'Camera geometry' comment "
                f"in hazard_localizer.py and run the `verify` subcommand.")
    return samples


# ---------------------------------------------------------------------------
# Split by task (never by episode)
# ---------------------------------------------------------------------------

def split_by_task(samples, held_out_task_ids=DEFAULT_HELD_OUT_TASKS):
    held = set(held_out_task_ids)
    train = [s for s in samples if s["task_id"] not in held]
    val = [s for s in samples if s["task_id"] in held]
    return train, val


def build_split_summary(train, val, held_out_task_ids):
    def per_suite(samples):
        d = {}
        for s in samples:
            d[s["suite"]] = d.get(s["suite"], 0) + 1
        return d

    return {
        "held_out_task_ids": sorted(int(t) for t in held_out_task_ids),
        "n_train": len(train),
        "n_val": len(val),
        "train_per_suite": per_suite(train),
        "val_per_suite": per_suite(val),
        "train_tasks": sorted({s["task_id"] for s in train}),
        "val_tasks": sorted({s["task_id"] for s in val}),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _hash_token(token, num_buckets):
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % num_buckets


def tokenize_entity_name(name):
    toks = [t for t in re.split(r"[_\-\s]+", name.lower()) if t]
    return toks or [name.lower() or "unk"]


class HazardFrameEntityDataset(torch.utils.data.Dataset):
    """One sample = one (frame, entity) pair. `samples` is the list produced
    by `index_entity_samples` (already split train/val by task)."""

    def __init__(self, samples, input_size=INPUT_SIZE):
        self.samples = samples
        self.input_size = input_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        with np.load(s["path"], allow_pickle=False) as z:
            rgb = z["agentview_rgb"]
        img = torch.from_numpy(np.ascontiguousarray(rgb)).float() / 255.0
        img = img.permute(2, 0, 1)  # C,H,W
        if img.shape[0] == 4:
            img = img[:3]
        img = F.interpolate(img.unsqueeze(0), size=(self.input_size, self.input_size),
                             mode="bilinear", align_corners=False).squeeze(0)

        u_resized = s["u"] * (self.input_size / s["w_orig"])
        v_resized = s["v"] * (self.input_size / s["h_orig"])
        u_hm = u_resized / STRIDE
        v_hm = v_resized / STRIDE

        return {
            "image": img,
            "entity_name": s["entity_name"],
            "u_hm": float(u_hm),
            "v_hm": float(v_hm),
            "depth": float(s["depth"]),
            "u_orig": float(s["u"]),
            "v_orig": float(s["v"]),
            "w_orig": float(s["w_orig"]),
            "h_orig": float(s["h_orig"]),
            "K": torch.tensor(s["K"], dtype=torch.float64),
            "T": torch.tensor(s["T"], dtype=torch.float64),
            "entity_pos": torch.tensor(s["entity_pos"], dtype=torch.float64),
            "suite": s["suite"],
            "is_guard": bool(s["is_guard"]),
        }


# ---------------------------------------------------------------------------
# Model: shared CNN trunk + hashed entity-token embedding + FiLM
# ---------------------------------------------------------------------------

class EntityEmbedding(nn.Module):
    def __init__(self, dim=32, num_buckets=4096):
        super().__init__()
        self.dim = dim
        self.num_buckets = num_buckets
        self.emb = nn.Embedding(num_buckets, dim)

    def forward(self, names, device):
        vecs = []
        for name in names:
            ids = [_hash_token(t, self.num_buckets) for t in tokenize_entity_name(name)]
            idx = torch.tensor(ids, dtype=torch.long, device=device)
            vecs.append(self.emb(idx).mean(dim=0))
        return torch.stack(vecs, dim=0)


class HazardLocalizerNet(nn.Module):
    """Small conv trunk (6 conv layers, no pretrained weights) -> heatmap +
    per-cell depth head, conditioned on entity identity via FiLM."""

    def __init__(self, entity_dim=32, num_buckets=4096, base_ch=32):
        super().__init__()
        self.entity_embed = EntityEmbedding(dim=entity_dim, num_buckets=num_buckets)

        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 3
        self.conv1 = nn.Conv2d(3, c1, 3, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(min(8, c1), c1)
        self.conv2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(min(8, c2), c2)
        self.conv3 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(min(8, c3), c3)

        self.film = nn.Linear(entity_dim, c3 * 2)

        self.conv4 = nn.Conv2d(c3, c3, 3, stride=1, padding=1)
        self.gn4 = nn.GroupNorm(min(8, c3), c3)
        self.conv5 = nn.Conv2d(c3, c2, 3, stride=1, padding=1)
        self.gn5 = nn.GroupNorm(min(8, c2), c2)

        self.heatmap_head = nn.Conv2d(c2, 1, kernel_size=1)
        self.depth_head = nn.Conv2d(c2, 1, kernel_size=1)

    def forward(self, image, entity_names):
        x = F.relu(self.gn1(self.conv1(image)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))

        ent = self.entity_embed(entity_names, image.device)
        gamma_beta = self.film(ent)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x = x * (1.0 + gamma) + beta

        x = F.relu(self.gn4(self.conv4(x)))
        x = F.relu(self.gn5(self.conv5(x)))

        heat_logits = self.heatmap_head(x).squeeze(1)          # B, Hh, Wh
        # depth head predicts NORMALIZED depth ((z - mean) / std, computed
        # over the train split and stored in norm_stats.json / the
        # checkpoint) -- linear output, no softplus: normalization already
        # keeps the raw metric depth (denormalized at decode time in
        # `predict_position`) positive for any physically-sensible target.
        depth = self.depth_head(x).squeeze(1)                   # B, Hh, Wh
        return heat_logits, depth


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def _gaussian_heatmap_target(u_hm, v_hm, size, sigma=1.0, device=None):
    """u_hm/v_hm: (B,) float tensors of target center in heatmap coords.
    Returns (B, size, size) gaussian targets."""
    ys = torch.arange(size, device=device, dtype=torch.float32).view(1, size, 1)
    xs = torch.arange(size, device=device, dtype=torch.float32).view(1, 1, size)
    v = v_hm.view(-1, 1, 1)
    u = u_hm.view(-1, 1, 1)
    g = torch.exp(-((xs - u) ** 2 + (ys - v) ** 2) / (2 * sigma ** 2))
    return g


def _sample_depth_at(depth_map, u_hm, v_hm):
    """Bilinearly sample `depth_map` (B,Hh,Wh) at float heatmap coords
    (u_hm, v_hm) via grid_sample."""
    b, h, w = depth_map.shape
    gx = (u_hm / (w - 1)) * 2 - 1
    gy = (v_hm / (h - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).view(b, 1, 1, 2)
    sampled = F.grid_sample(depth_map.unsqueeze(1), grid, align_corners=True,
                             mode="bilinear", padding_mode="border")
    return sampled.view(b)


def compute_loss(heat_logits, depth_map, u_hm, v_hm, depth_gt, depth_mean, depth_std, sigma=1.0):
    """`depth_map` predicts NORMALIZED depth ((z - depth_mean) / depth_std);
    `depth_gt` is raw metric depth and is normalized here before the L1
    term, so the depth head regresses a roughly unit-scale target."""
    target = _gaussian_heatmap_target(u_hm, v_hm, heat_logits.shape[-1], sigma,
                                       device=heat_logits.device)
    heat_pred = torch.sigmoid(heat_logits)
    # A sigma=1 gaussian on a 32x32 grid covers ~0.6% of cells -- unweighted
    # MSE lets the all-zeros heatmap be a strong local optimum (which would
    # make soft-argmax always return the image center, independent of the
    # projection/geometry chain being correct). Upweight the positive
    # region so the loss can't be driven low by predicting nothing.
    weight = 1.0 + 5.0 * target
    heat_loss = (weight * (heat_pred - target) ** 2).mean()

    depth_pred_at_gt = _sample_depth_at(depth_map, u_hm.clamp(0, depth_map.shape[-1] - 1),
                                         v_hm.clamp(0, depth_map.shape[-2] - 1))
    depth_gt_norm = (depth_gt - depth_mean) / depth_std
    depth_loss = F.l1_loss(depth_pred_at_gt, depth_gt_norm)
    return heat_loss + depth_loss, heat_loss.item(), depth_loss.item()


# ---------------------------------------------------------------------------
# Inference: predicted 3D position from a single sample's model outputs
# ---------------------------------------------------------------------------

def _soft_argmax_2d(heat, temperature=1.0):
    """Continuous (sub-cell) peak location via softmax-weighted centroid --
    cheap fix for the quantization error a hard argmax leaves at STRIDE=8
    (each cell spans 8 px in the 256-space, i.e. ~1-1.5 cm of lateral error
    at typical agentview scale/depth -- a large slice of the 2.5 cm gate).
    Returns (row, col) as continuous cell-index coordinates, matching the
    (v_hm, u_hm) convention used at training time (no +0.5 offset either
    side, so encode/decode round-trip through the same cell-index origin)."""
    h, w = heat.shape
    flat = heat.reshape(-1).astype(np.float64)
    flat = flat - flat.max()
    weights = np.exp(flat / temperature)
    weights /= weights.sum()
    ys, xs = np.mgrid[0:h, 0:w]
    row = float((weights * ys.reshape(-1)).sum())
    col = float((weights * xs.reshape(-1)).sum())
    return row, col


def _bilinear_sample_2d(arr, x, y):
    h, w = arr.shape
    x = min(max(x, 0.0), w - 1)
    y = min(max(y, 0.0), h - 1)
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    top = arr[y0, x0] * (1 - dx) + arr[y0, x1] * dx
    bot = arr[y1, x0] * (1 - dx) + arr[y1, x1] * dx
    return float(top * (1 - dy) + bot * dy)


def predict_position(heat_logits, depth_map, w_orig, h_orig, K, T, input_size,
                      depth_mean, depth_std, temperature=1.0):
    """heat_logits/depth_map: (Hh, Wh) single-sample arrays. `depth_map`
    holds NORMALIZED depth predictions; denormalized with `depth_mean`/
    `depth_std` before backprojection. Returns predicted 3D world
    position, decoded in the SAME (row-mirrored) pixel convention
    `project_world_to_pixel`/`backproject` use."""
    heat = np.asarray(heat_logits)
    depth = np.asarray(depth_map)
    iy, ix = _soft_argmax_2d(heat, temperature=temperature)
    depth_norm = _bilinear_sample_2d(depth, ix, iy)
    z_pred = depth_norm * depth_std + depth_mean

    u_resized = ix * STRIDE
    v_resized = iy * STRIDE
    u_orig = u_resized * (w_orig / input_size)
    v_orig = v_resized * (h_orig / input_size)

    return backproject(u_orig, v_orig, z_pred, np.asarray(K), np.asarray(T), h_orig)


# ---------------------------------------------------------------------------
# Collate (default_collate handles str lists fine, but be explicit/robust)
# ---------------------------------------------------------------------------

def collate_batch(items):
    out = {}
    for key in items[0]:
        vals = [it[key] for it in items]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], str):
            out[key] = vals
        else:
            out[key] = torch.tensor(vals)
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _eval_median_error(model, dataset, device, input_size, depth_mean, depth_std,
                        batch_size=32, num_workers=0):
    """Always returns a 4-tuple (median, errors, per_entity, per_suite) --
    including on the empty-dataset path -- so callers never need to special
    case an empty val/held-out set; `per_entity`/`per_suite` map name ->
    list[float] errors, and each also carries a parallel `*_guard` dict
    restricted to samples flagged `is_guard` (the quantity the G-L1 gate is
    actually meant to track -- hazard localization, not arbitrary-entity
    localization)."""
    if len(dataset) == 0:
        return None, [], {}, {}, {}, {}
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_batch, num_workers=num_workers)
    errors = []
    guard_errors = []
    per_entity = {}
    per_suite = {}
    per_suite_guard = {}
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            heat_logits, depth_map = model(img, batch["entity_name"])
            heat_np = heat_logits.cpu().numpy()
            depth_np = depth_map.cpu().numpy()
            for i in range(img.shape[0]):
                pred = predict_position(
                    heat_np[i], depth_np[i],
                    float(batch["w_orig"][i]), float(batch["h_orig"][i]),
                    batch["K"][i].numpy(), batch["T"][i].numpy(),
                    input_size, depth_mean, depth_std)
                gt = batch["entity_pos"][i].numpy()
                err = float(np.linalg.norm(pred - gt))
                errors.append(err)
                ent = batch["entity_name"][i]
                per_entity.setdefault(ent, []).append(err)
                suite = batch["suite"][i]
                per_suite.setdefault(suite, []).append(err)
                if bool(batch["is_guard"][i]):
                    guard_errors.append(err)
                    per_suite_guard.setdefault(suite, []).append(err)
    model.train()
    median = statistics.median(errors) if errors else None
    return median, errors, per_entity, per_suite, guard_errors, per_suite_guard


def train_main(args):
    set_seed(args.seed)
    device = torch.device("cpu")

    max_oob_frac = getattr(args, "max_oob_frac", 0.05)
    if max_oob_frac is not None and max_oob_frac < 0:
        max_oob_frac = None

    frame_files = index_frame_files(args.data)
    samples = index_entity_samples(frame_files, camera_name=args.camera,
                                    max_oob_frac=max_oob_frac)
    if not samples:
        raise SystemExit(f"No usable (frame, entity) samples found under {args.data}")

    held_out = tuple(int(t) for t in args.held_out_tasks.split(",")) if args.held_out_tasks \
        else DEFAULT_HELD_OUT_TASKS
    present_tasks = {s["task_id"] for s in samples}
    held_out = tuple(sorted(t for t in held_out if t in present_tasks)) or \
        tuple(sorted(present_tasks))[:1]  # fallback: hold out at least one task

    train_samples, val_samples = split_by_task(samples, held_out)
    if not train_samples:
        raise SystemExit("Split-by-task left an empty train set; check --held_out_tasks.")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_summary = build_split_summary(train_samples, val_samples, held_out)
    with open(out_dir / "split.json", "w") as f:
        json.dump(split_summary, f, indent=2)

    train_depths = np.array([s["depth"] for s in train_samples], dtype=np.float64)
    depth_mean = float(train_depths.mean())
    depth_std = float(train_depths.std())
    if depth_std < 1e-6:
        depth_std = 1.0

    norm_stats = {
        "depth_mean_m": depth_mean,
        "depth_std_m": depth_std,
        "image_normalization": "none (raw [0,1] RGB, no pretrained mean/std)",
        "entity_dim": args.entity_dim,
        "num_buckets": args.num_buckets,
        "base_ch": args.base_ch,
        "input_size": args.input_size,
        "stride": STRIDE,
        "camera": args.camera,
    }
    with open(out_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    train_ds = HazardFrameEntityDataset(train_samples, input_size=args.input_size)
    val_ds = HazardFrameEntityDataset(val_samples, input_size=args.input_size)

    model = HazardLocalizerNet(entity_dim=args.entity_dim, num_buckets=args.num_buckets,
                                base_ch=args.base_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                          collate_fn=collate_batch, num_workers=args.num_workers)

    history = []
    best_val = float("inf")
    best_state = None
    epochs_since_improve = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for batch in loader:
            img = batch["image"].to(device)
            heat_logits, depth_map = model(img, batch["entity_name"])
            loss, heat_l, depth_l = compute_loss(
                heat_logits, depth_map,
                batch["u_hm"].to(device).float(), batch["v_hm"].to(device).float(),
                batch["depth"].to(device).float(), depth_mean, depth_std,
                sigma=args.heatmap_sigma)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        val_median = _eval_median_error(
            model, val_ds, device, args.input_size, depth_mean, depth_std,
            batch_size=args.batch_size, num_workers=args.num_workers)[0]
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_median_error_m": val_median})

        improved = val_median is not None and val_median < best_val
        if improved:
            best_val = val_median
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if args.patience is not None and epochs_since_improve >= args.patience and epoch >= 1:
            break

    final_state = best_state if best_state is not None else model.state_dict()
    torch.save({
        "state_dict": final_state,
        "entity_dim": args.entity_dim,
        "num_buckets": args.num_buckets,
        "base_ch": args.base_ch,
        "input_size": args.input_size,
        "camera": args.camera,
        "depth_mean_m": depth_mean,
        "depth_std_m": depth_std,
    }, out_dir / "model.pt")

    with open(out_dir / "train_history.json", "w") as f:
        json.dump({"history": history, "best_val_median_error_m": best_val
                    if best_val != float("inf") else None}, f, indent=2)

    print(f"[hazard_localizer] wrote {out_dir}/model.pt "
          f"(best val median err = {best_val if best_val != float('inf') else 'n/a'})")
    return out_dir


# ---------------------------------------------------------------------------
# G-L1 offline eval
# ---------------------------------------------------------------------------

GATE_MEDIAN_M = 0.025


def load_model(model_dir):
    model_dir = pathlib.Path(model_dir)
    ckpt = torch.load(model_dir / "model.pt", map_location="cpu")
    model = HazardLocalizerNet(entity_dim=ckpt["entity_dim"], num_buckets=ckpt["num_buckets"],
                                base_ch=ckpt["base_ch"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def eval_main(args):
    model_dir = pathlib.Path(args.model)
    model, ckpt = load_model(model_dir)
    input_size = ckpt.get("input_size", INPUT_SIZE)
    camera = ckpt.get("camera", CAMERA_NAME)
    depth_mean = ckpt.get("depth_mean_m", 0.0)
    depth_std = ckpt.get("depth_std_m", 1.0)

    with open(model_dir / "split.json") as f:
        split_summary = json.load(f)
    held_out = tuple(split_summary["held_out_task_ids"])

    frame_files = index_frame_files(args.data)
    samples = index_entity_samples(frame_files, camera_name=camera)
    _, val_samples = split_by_task(samples, held_out)

    val_ds = HazardFrameEntityDataset(val_samples, input_size=input_size)
    device = torch.device("cpu")
    (median_err, errors, per_entity, per_suite,
     guard_errors, per_suite_guard) = _eval_median_error(
        model, val_ds, device, input_size, depth_mean, depth_std,
        batch_size=args.batch_size, num_workers=args.num_workers)

    def _stats(errs):
        if not errs:
            return {"median_m": None, "p90_m": None, "n": 0}
        s = sorted(errs)
        p90_idx = min(len(s) - 1, int(round(0.9 * (len(s) - 1))))
        return {"median_m": statistics.median(s), "p90_m": s[p90_idx], "n": len(s)}

    report = {
        "gate_median_m": GATE_MEDIAN_M,
        "overall": _stats(errors),
        "per_suite": {k: _stats(v) for k, v in per_suite.items()},
        "per_entity": {k: _stats(v) for k, v in per_entity.items()},
        # Guard-only subset: the quantity the G-L1 gate is meant to track
        # (hazard localization specifically, not arbitrary-entity
        # localization) -- `is_guard` comes straight from each frame's
        # `guard` field, never from a `_obstacle_` name substring.
        "guard_only": _stats(guard_errors),
        "guard_only_per_suite": {k: _stats(v) for k, v in per_suite_guard.items()},
        "held_out_task_ids": list(held_out),
        "current_detector": None,  # OPTIONAL per brief; skipped (needs GPU detector weights)
    }
    report["gate_pass"] = (report["overall"]["median_m"] is not None
                            and report["overall"]["median_m"] < GATE_MEDIAN_M)

    with open(model_dir / "g_l1_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# `verify`: standalone empirical convention check on REAL data, meant to run
# BEFORE a GPU training job (wired as the first step in
# slurm/localizer_train.slurm, which aborts before `train` if this fails).
# Samples N frames, projects every GT entity with the SAME
# `project_world_to_pixel` training/eval use, and reports the in-bounds
# fraction (should be near 1.0 for a correct camera/pixel convention -- a
# depressed fraction is the empirical signature of exactly the kind of
# row/column mismatch this module's derivation is trying to avoid) plus,
# where a per-camera depth map is present in the frame, a cross-check
# between the depth map's value AT the projected pixel and the projected
# camera-frame z (large disagreement on otherwise-in-bounds points also
# indicates a convention mismatch, even when the in-bounds fraction looks
# fine).
# ---------------------------------------------------------------------------

VERIFY_MIN_IN_BOUNDS_FRAC = 0.95


def _discover_cameras(paths):
    cams = set()
    for p in paths:
        try:
            with np.load(p, allow_pickle=False) as z:
                for k in z.files:
                    if k.startswith("camera_K_"):
                        cams.add(k[len("camera_K_"):])
        except Exception:
            continue
    return sorted(cams)


def verify_main(args):
    frame_files = index_frame_files(args.data)
    if not frame_files:
        print(f"[hazard_localizer verify] HARD FAIL: no frame files found under {args.data}")
        sys.exit(1)

    rng = random.Random(args.seed)
    sampled = frame_files if len(frame_files) <= args.n else rng.sample(frame_files, args.n)

    cams = _discover_cameras([f["path"] for f in sampled]) or [args.camera]

    per_cam = {}
    for cam in cams:
        n_entities = 0
        n_in_bounds = 0
        depth_diffs = []
        for f in sampled:
            try:
                with np.load(f["path"], allow_pickle=False) as z:
                    if f"camera_K_{cam}" not in z or f"camera_T_{cam}" not in z \
                            or "entity_names" not in z or "agentview_rgb" not in z:
                        continue
                    rgb_shape = z["agentview_rgb"].shape
                    K = z[f"camera_K_{cam}"]
                    T = z[f"camera_T_{cam}"]
                    names = [str(n) for n in z["entity_names"]]
                    positions = z["entity_pos"]
                    depth_key = f"{cam}_depth" if cam != "agentview" else "agentview_depth"
                    depth_map = np.asarray(z[depth_key]) if depth_key in z else None
            except Exception:
                continue

            if depth_map is not None and depth_map.ndim == 3:
                depth_map = depth_map[..., 0]
            h_orig, w_orig = rgb_shape[0], rgb_shape[1]
            for name, pos in zip(names, positions):
                proj = project_world_to_pixel(pos, K, T, h_orig)
                if proj is None:
                    continue
                u, v, z_proj = proj
                if z_proj <= 0.0:
                    continue
                n_entities += 1
                in_bounds = (0.0 <= u < w_orig and 0.0 <= v < h_orig)
                if in_bounds:
                    n_in_bounds += 1
                    if depth_map is not None:
                        z_pixel = _bilinear_sample_2d(depth_map, u, v)
                        depth_diffs.append(abs(float(z_pixel) - z_proj))

        frac = (n_in_bounds / n_entities) if n_entities else None
        per_cam[cam] = {
            "n_entities": n_entities,
            "n_in_bounds": n_in_bounds,
            "in_bounds_fraction": frac,
            "depth_agreement_median_m": (statistics.median(depth_diffs) if depth_diffs else None),
            "depth_agreement_n": len(depth_diffs),
        }

    report = {
        "n_frames_available": len(frame_files),
        "n_frames_sampled": len(sampled),
        "min_in_bounds_fraction_gate": VERIFY_MIN_IN_BOUNDS_FRAC,
        "primary_camera": args.camera,
        "cameras": per_cam,
    }
    print(json.dumps(report, indent=2))

    primary = per_cam.get(args.camera)
    frac = primary["in_bounds_fraction"] if primary else None
    if frac is None or frac < VERIFY_MIN_IN_BOUNDS_FRAC:
        print(f"[hazard_localizer verify] HARD FAIL: '{args.camera}' in-bounds "
              f"fraction = {frac} (< {VERIFY_MIN_IN_BOUNDS_FRAC}). This is the "
              f"signature of a camera/pixel convention mismatch -- see the "
              f"'Camera geometry' comment in hazard_localizer.py. Aborting "
              f"before training.")
        sys.exit(1)

    print(f"[hazard_localizer verify] OK: '{args.camera}' in-bounds fraction = {frac:.3f}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    tp = sub.add_parser("train")
    tp.add_argument("--data", required=True)
    tp.add_argument("--out", required=True)
    tp.add_argument("--camera", default=CAMERA_NAME)
    tp.add_argument("--held_out_tasks", default=",".join(str(t) for t in DEFAULT_HELD_OUT_TASKS))
    tp.add_argument("--epochs", type=int, default=60)
    tp.add_argument("--patience", type=int, default=8)
    tp.add_argument("--batch_size", type=int, default=32)
    tp.add_argument("--lr", type=float, default=1e-3)
    tp.add_argument("--seed", type=int, default=0)
    tp.add_argument("--input_size", type=int, default=INPUT_SIZE)
    tp.add_argument("--entity_dim", type=int, default=32)
    tp.add_argument("--num_buckets", type=int, default=4096)
    tp.add_argument("--base_ch", type=int, default=32)
    tp.add_argument("--heatmap_sigma", type=float, default=1.0)
    tp.add_argument("--num_workers", type=int, default=4)
    tp.add_argument("--max_oob_frac", type=float, default=0.05,
                     help="Training-time tripwire: raise if more than this "
                          "fraction of forward-facing projected entity "
                          "targets land outside the image bounds. Pass a "
                          "negative value (e.g. -1) to disable.")

    ep = sub.add_parser("eval")
    ep.add_argument("--data", required=True)
    ep.add_argument("--model", required=True)
    ep.add_argument("--batch_size", type=int, default=32)
    ep.add_argument("--num_workers", type=int, default=4)

    vp = sub.add_parser("verify")
    vp.add_argument("--data", required=True)
    vp.add_argument("--camera", default=CAMERA_NAME)
    vp.add_argument("--n", type=int, default=50)
    vp.add_argument("--seed", type=int, default=0)

    return ap


def main(argv=None):
    ap = build_argparser()
    args = ap.parse_args(argv)
    if args.cmd == "train":
        train_main(args)
    elif args.cmd == "eval":
        eval_main(args)
    elif args.cmd == "verify":
        verify_main(args)
    else:  # pragma: no cover
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
