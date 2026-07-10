"""
GT-free visual obstacle grounding (v18).

Identity AND position from perception — no simulator object state is read:
  1. Qwen dense detection (agentview + wrist views, union of votes)
  2. ¬MENTIONED_IN_TASK filter on DETECTED names
  3. Cross-view association by minimal ray gap (epipolar consistency)
  4. Symbolic FOL on vision-estimated 3D positions:
       OBSTACLE(x) := ¬MENTIONED(x) ∧ ¬SUPPORTS(x,·) ∧ ¬FURNITURE_SCALE(x)
                      ∧ NEAREST_TO_PATH(x)
  5. z-band fallback ray when the wrist view loses the obstacle (low confidence)

Camera model (validated against rendered pixels, 2026-07-05):
  in-front: pc[2] > 0;  u = cx + fx*px/pz;  v = cy + fy*py/pz
  Images from env obs must be rotated 180° (obs[::-1, ::-1]) to match.
  Agentview extrinsic is a fixed rig constant; wrist extrinsic =
  T_eef(proprioception, xyzw quat) @ T_HANDEYE (fixed hand-eye calibration).

Offline validation (21 saved episodes, GT masks → perfect-detector bound):
  triangulation median 6.3cm (xy 1.3cm).  With real Qwen-3B detection (harness
  v5): 20/21 localized, xy median 12.6cm — consumed downstream via
  uncertainty-inflated CBF radii.
"""

from __future__ import annotations

import json
import logging
import re
import re as _re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calibration constants (rig-fixed; derived from camera_params.json which is
# constant across all episodes — equivalent to one-time camera calibration).
# Values are for 512px renders; intrinsics are rescaled to the actual image.
# ---------------------------------------------------------------------------

AGENTVIEW_K_512 = np.array([
    [618.0386719675123, 0.0, 256.0],
    [0.0, 618.0386719675123, 256.0],
    [0.0, 0.0, 1.0],
])
AGENTVIEW_E = np.array([
    [-7.268184498698815e-08, 0.6282664504107425, -0.7779982437565529, 0.6586131746834771],
    [0.9999999999999858, -7.268184498698815e-08, -1.521152659944569e-07, 0.0],
    [-1.521152659944569e-07, -0.7779982437565529, -0.6282664504107283, 1.6103500240372424],
    [0.0, 0.0, 0.0, 1.0],
])
EIH_K_512 = np.array([
    [333.6256954473487, 0.0, 256.0],
    [0.0, 333.6256954473487, 256.0],
    [0.0, 0.0, 1.0],
])
# T_handeye = inv(T_eef) @ E_eih, constant across episodes (xyzw quat), verified
# to 1e-5 over 8 episodes.
T_HANDEYE = None  # set below from the reference episode values


def _set_handeye():
    global T_HANDEYE
    eef_pos = np.array([-0.21128578443338253, -0.011001144956700267, 1.1741991418805875])
    eef_quat = np.array([0.9996059100423984, -0.00018294069521890277,
                         -0.028071089297739164, -7.132073859967072e-05])  # xyzw
    E_eih = np.array([
        [-0.0003685489554273724, -0.9984239439873566, -0.05612033717286558, -0.15592098141933983],
        [-0.9999999234392745, 0.0003605881147050738, 0.0001519791220409121, -0.011033554857756768],
        [-0.00013150326785404953, 0.05612038888799861, -0.9984240004416697, 1.2682401860329686],
        [0.0, 0.0, 0.0, 1.0],
    ])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(eef_quat).as_matrix()
    T[:3, 3] = eef_pos
    T_HANDEYE = np.linalg.inv(T) @ E_eih


_set_handeye()

WORKSPACE = dict(x=(-0.6, 0.6), y=(-0.5, 0.8), z=(0.7, 1.5))
Z_BAND_CENTER = 0.95        # workspace obstacle height band (fallback ray)
FURNITURE_SIZE_M = 0.35     # metric size gate: larger = workspace structure
RAY_GAP_MAX = 0.08          # epipolar association gate
REFINE_JUMP_MAX = 0.15      # max estimate shift accepted per refinement
_CROP_PROMPT = (
    'Outline each distinct object in this image patch and output '
    'all coordinates in JSON format:\n'
    '[{"bbox_2d": [x1, y1, x2, y2], "label": "<object name>"}]'
)
UNCERT_TRIANGULATED = 0.03  # radius inflation, two-view estimate
UNCERT_FALLBACK = 0.08      # radius inflation, z-band fallback estimate

DETECT_PROMPT = (
    'Outline the position of each object on the table (dishes, containers, '
    'bottles, boxes, food items, appliances, books — every distinct item, '
    'but NOT the robot arm, table, or walls) and output all the coordinates '
    'in JSON format:\n'
    '[{"bbox_2d": [x1, y1, x2, y2], "label": "<object name>"}]'
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _project(p, K, Einv):
    pc = Einv @ np.append(p, 1.0)
    if pc[2] <= 1e-6:
        return None
    return K[0, 2] + K[0, 0] * pc[0] / pc[2], K[1, 2] + K[1, 1] * pc[1] / pc[2]


def _make_ray(u, v, K, E):
    d_cam = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    d = E[:3, :3] @ d_cam
    return E[:3, 3], d / np.linalg.norm(d)


def _lsq_rays(origins, dirs):
    """Least-squares point closest to all rays: min_p sum ||(I-ddT)(p-c)||^2.
    Returns (p, rms_gap) or (None, inf) when the ray bundle is degenerate."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(origins, dirs):
        d = np.asarray(d, dtype=np.float64)
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ np.asarray(c, dtype=np.float64)
    if np.linalg.matrix_rank(A, tol=1e-9) < 3:
        return None, np.inf
    p = np.linalg.solve(A, b)
    sq = []
    for c, d in zip(origins, dirs):
        d = np.asarray(d, dtype=np.float64)
        d = d / np.linalg.norm(d)
        r = p - np.asarray(c, dtype=np.float64)
        sq.append(float(r @ r - (d @ r) ** 2))
    return p, float(np.sqrt(max(0.0, np.mean(sq))))


def _closest_point(c1, d1, c2, d2):
    w0 = c1 - c2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d_, e_ = d1 @ w0, d2 @ w0
    den = a * c - b * b
    if abs(den) < 1e-9:
        return None, None
    t1 = (b * e_ - c * d_) / den
    t2 = (a * e_ - b * d_) / den
    p1, p2 = c1 + t1 * d1, c2 + t2 * d2
    return (p1 + p2) / 2, float(np.linalg.norm(p1 - p2))


def _obstacle_frame(dda, ddb):
    """Orthonormal obstacle frame: axis0 = xy-projection of the viewing-ray
    bisector (horizontal depth direction, where triangulation error
    concentrates), axis2 = world-z, axis1 completes the right-handed basis.
    The projection keeps the inflated axis horizontal — a raw 3D bisector
    from two downward-looking cameras is near-vertical and would build a
    tall barrier that blocks grasp descent instead of covering depth error."""
    a = np.asarray(dda, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = np.asarray(ddb, dtype=np.float64)
    b = b / np.linalg.norm(b)
    bis = a + b
    if np.linalg.norm(bis) < 1e-6:
        bis = a
    ax0 = np.array([bis[0], bis[1], 0.0])
    n0 = np.linalg.norm(ax0)
    ax0 = np.array([1.0, 0.0, 0.0]) if n0 < 1e-6 else ax0 / n0
    ax2 = np.array([0.0, 0.0, 1.0])
    ax1 = np.cross(ax2, ax0)
    return np.column_stack([ax0, ax1, ax2])


def _bbox_extents(bb, f, rng):
    """Metric half-extents from an image bbox at estimated range: xy
    isotropic from pixel width (yaw unknown), z from pixel height."""
    w = (bb[2] - bb[0]) / f * rng
    h = (bb[3] - bb[1]) / f * rng
    return np.array([min(w / 2, 0.12), min(w / 2, 0.12), min(h / 2, 0.12)])


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _in_workspace(p):
    return (WORKSPACE['x'][0] < p[0] < WORKSPACE['x'][1]
            and WORKSPACE['y'][0] < p[1] < WORKSPACE['y'][1]
            and WORKSPACE['z'][0] < p[2] < WORKSPACE['z'][1])


def _parse_detections(raw: str, img_wh: int) -> List[Dict]:
    m = re.search(r'\[[\s\S]*\]', raw or "")
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        bb = it.get("bbox_2d") or it.get("bbox")
        nm = it.get("label") or it.get("name")
        if not (isinstance(bb, list) and len(bb) == 4 and nm):
            continue
        bb = [float(x) for x in bb]
        mx = max(bb)
        # per-box regime: coords are native unless they exceed the image size,
        # in which case that box is 0-1000 normalized (mixed regimes observed).
        if mx > img_wh * 1.05:
            bb = [x * img_wh / 1000.0 for x in bb] if mx <= 1000 else \
                 [x * img_wh / mx for x in bb]
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if w < 0.03 * img_wh or h < 0.03 * img_wh:
            continue  # degenerate sliver (hallucinated tiling)
        out.append({"name": str(nm).lower().strip(), "bbox": bb})
    # drop hallucinated tilings: >2 same-label boxes of near-identical size
    from collections import Counter
    lab_counts = Counter(d["name"] for d in out)
    out = [d for d in out if lab_counts[d["name"]] <= 2]
    return out


def _mentioned(name: str, task: str) -> bool:
    words = [w for w in name.lower().split() if len(w) > 3]
    return any(w in task.lower() for w in words)


# ---------------------------------------------------------------------------
# Main grounder
# ---------------------------------------------------------------------------

class VisualObstacleGrounder:
    """Grounds the obstacle (name + 3D position + uncertainty) from images,
    task text and proprioception only."""

    def __init__(self, vlm_client):
        self.client = vlm_client
        self._cache: Dict[str, Dict] = {}
        self._refine_state: Optional[Dict] = None

    @property
    def has_state(self) -> bool:
        return self._refine_state is not None

    def _crop_detect(self, img: np.ndarray, pts_px) -> tuple:
        """Crop around pixel points, upscale, dense-detect.  Returns
        (locs, x1, y1, scale); locs are in upscaled-crop coordinates."""
        wh = img.shape[0]
        us = [p[0] for p in pts_px]
        vs = [p[1] for p in pts_px]
        pad = max(40, int(0.18 * wh))
        x1 = max(0, int(min(us)) - pad)
        x2 = min(wh, int(max(us)) + pad)
        y1 = max(0, int(min(vs)) - pad)
        y2 = min(wh, int(max(vs)) + pad)
        if x2 - x1 < 24 or y2 - y1 < 24:
            return [], 0, 0, 1
        crop = img[y1:y2, x1:x2]
        scale = max(1, int(round(320 / max(crop.shape[:2]))))
        if scale > 1:
            from PIL import Image as _Image
            crop = np.array(_Image.fromarray(crop).resize(
                (crop.shape[1] * scale, crop.shape[0] * scale)))
        raw = self.client.infer(_CROP_PROMPT, [crop], max_tokens=512)
        return _parse_detections(raw, max(crop.shape[:2])), x1, y1, scale

    def refine(
        self,
        eih_img_raw: np.ndarray,
        eef_pos: np.ndarray,
        eef_quat_xyzw: np.ndarray,
    ) -> Optional[Dict]:
        """Add one wrist-camera ray to the episode's obstacle estimate and
        re-solve over all rays.  Returns {pos, uncertainty_m} on an accepted
        update, else None.  Uses only the wrist image + proprioception."""
        st = self._refine_state
        if st is None:
            return None

        eih_img = np.ascontiguousarray(eih_img_raw[::-1, ::-1])
        wh = eih_img.shape[0]
        Ke = EIH_K_512 * (wh / 512.0)
        Ke[2, 2] = 1.0
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(eef_quat_xyzw).as_matrix()
        T[:3, 3] = np.asarray(eef_pos, dtype=np.float64)
        Ee = T @ T_HANDEYE

        uv = _project(st["pos"], Ke, np.linalg.inv(Ee))
        if uv is None or not (0 <= uv[0] < wh and 0 <= uv[1] < wh):
            return None

        locs, x1, y1, scale = self._crop_detect(eih_img, [uv])
        if not locs:
            logger.info("[VisualGrounder] refine: crop-detect empty")
            return None
        uc, vc = (uv[0] - x1) * scale, (uv[1] - y1) * scale
        b = min((l["bbox"] for l in locs),
                key=lambda b: ((b[0] + b[2]) / 2 - uc) ** 2
                + ((b[1] + b[3]) / 2 - vc) ** 2)
        u_e = x1 + (b[0] + b[2]) / 2 / scale
        v_e = y1 + (b[1] + b[3]) / 2 / scale
        cb, ddb = _make_ray(u_e, v_e, Ke, Ee)

        rays = st["rays"] + [(cb, ddb)]
        p, gap = _lsq_rays([r[0] for r in rays], [r[1] for r in rays])
        if (p is None or gap > RAY_GAP_MAX or not _in_workspace(p)
                or float(np.linalg.norm(p - st["pos"])) > REFINE_JUMP_MAX):
            logger.info(
                f"[VisualGrounder] refine rejected: gap={gap} p={p}")
            return None

        st["rays"] = rays
        st["pos"] = np.asarray(p, dtype=np.float64)
        n_rays = len(rays)
        uncert = max(0.015, UNCERT_TRIANGULATED * 2.0 / n_rays)
        logger.info(
            f"[VisualGrounder] refine accepted ({n_rays} rays): "
            f"pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) ± {uncert:.3f}m")
        return {"pos": st["pos"].copy(), "uncertainty_m": uncert}

    def _detect(self, img: np.ndarray, votes: int) -> List[Dict]:
        wh = img.shape[0]
        merged: List[Dict] = []
        for attempt in range(votes + 1):
            raw = self.client.infer(DETECT_PROMPT, [img], max_tokens=1024)
            for det in _parse_detections(raw, wh):
                if all(_iou(det["bbox"], m["bbox"]) < 0.5 for m in merged):
                    merged.append(det)
            if attempt >= votes - 1 and merged:
                break
        return merged

    def ground(
        self,
        agent_img_raw: np.ndarray,
        eih_img_raw: np.ndarray,
        task_description: str,
        eef_pos: np.ndarray,
        eef_quat_xyzw: np.ndarray,
        cache_key: Optional[str] = None,
    ) -> Optional[Dict]:
        """Returns {name, pos(3,), uncertainty_m, method} or None."""
        self._refine_state = None
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]

        # match the validated pixel convention (saved images are raw[::-1, ::-1])
        agent_img = np.ascontiguousarray(agent_img_raw[::-1, ::-1])
        eih_img = np.ascontiguousarray(eih_img_raw[::-1, ::-1])


        wh_a, wh_e = agent_img.shape[0], eih_img.shape[0]
        Ka = AGENTVIEW_K_512 * (wh_a / 512.0); Ka[2, 2] = 1.0
        Ke = EIH_K_512 * (wh_e / 512.0); Ke[2, 2] = 1.0
        Ea = AGENTVIEW_E
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(eef_quat_xyzw).as_matrix()
        T[:3, 3] = np.asarray(eef_pos, dtype=np.float64)
        Ee = T @ T_HANDEYE

        det_a = self._detect(agent_img, votes=3)
        logger.info(f"[VisualGrounder] agentview detections: {len(det_a)}")

        ra = [(d, _make_ray((d["bbox"][0]+d["bbox"][2])/2,
                            (d["bbox"][1]+d["bbox"][3])/2, Ka, Ea)) for d in det_a]

        # Epipolar-guided crop localization in the wrist view: the agentview
        # ray, swept over the workspace height band, projects to a short pixel
        # segment in the wrist image.  Crop around it (upscaled) and ask the
        # VLM to locate the NAMED object inside the crop — geometry reduces
        # the top-down search problem to a trivial patch query.
        cands = []
        wh_e_img = eih_img.shape[0]
        Einv_e = np.linalg.inv(Ee)
        for da, (ca, dda) in ra:
            seg_px = []
            for z_hyp in np.arange(0.80, 1.31, 0.05):
                if abs(dda[2]) < 1e-6:
                    continue
                t = (z_hyp - ca[2]) / dda[2]
                if t <= 0:
                    continue
                uv = _project(ca + t * dda, Ke, Einv_e)
                if uv and 0 <= uv[0] < wh_e_img and 0 <= uv[1] < wh_e_img:
                    seg_px.append(uv)
            if not seg_px:
                continue
            # Name-agnostic: dense-detect the crop, keep the detection whose
            # center is nearest the projected epipolar segment (crop coords).
            # Cross-view naming fails for top-down appearances; geometry doesn't.
            locs, x1, y1, scale = self._crop_detect(eih_img, seg_px)
            if not locs:
                logger.info(f"[VisualGrounder] crop-detect empty for '{da['name']}'")
                continue
            seg_c = [((u - x1) * scale, (v - y1) * scale) for u, v in seg_px]
            def d_seg(b):
                bu, bv = (b[0]+b[2])/2, (b[1]+b[3])/2
                return min((bu-su)**2 + (bv-sv)**2 for su, sv in seg_c)
            b = min((l["bbox"] for l in locs), key=d_seg)
            u_e = x1 + (b[0] + b[2]) / 2 / scale
            v_e = y1 + (b[1] + b[3]) / 2 / scale
            cb, ddb = _make_ray(u_e, v_e, Ke, Ee)
            p, gap = _closest_point(ca, dda, cb, ddb)
            if p is None or gap > RAY_GAP_MAX or not _in_workspace(p):
                logger.info(f"[VisualGrounder] crop-triangulation rejected for "
                            f"'{da['name']}': gap={gap} p={p}")
                continue
            bb = da["bbox"]
            rng = float(np.linalg.norm(p - Ea[:3, 3]))
            size_m = max(bb[2]-bb[0], bb[3]-bb[1]) / Ka[0, 0] * rng
            cands.append({"name": da["name"], "pos": p, "gap": gap,
                          "size_m": size_m, "dda": dda, "ddb": ddb,
                          "ca": ca, "cb": cb,
                          "bb": bb, "f": Ka[0, 0], "rng": rng})

        # ¬SUPPORTS on estimated positions
        drop = set()
        for a in cands:
            for b in cands:
                if a is b:
                    continue
                if (np.linalg.norm(a["pos"][:2] - b["pos"][:2]) < 0.10
                        and a["pos"][2] > b["pos"][2] + 0.04):
                    drop.add(id(b))
        cands = [c for c in cands if id(c) not in drop] or cands

        # ¬FURNITURE_SCALE
        cands = [c for c in cands if c["size_m"] <= FURNITURE_SIZE_M] or cands

        # obstacle height band: designated hazards sit above the table plane
        # (hover/pedestal band); table-level flats are not the safety obstacle.
        cands = [c for c in cands if 0.85 <= c["pos"][2] <= 1.30] or cands

        # MENTIONED(x, task) is a vision-language predicate under open-vocab
        # labels: the task may say "stove" while detection says "appliance".
        # String match is only a fast-positive path; string-negatives are
        # verified with a per-candidate crop query (task-role binary).
        unmen, mentioned_c = [], []
        for c in cands:
            if _mentioned(c["name"], task_description):
                mentioned_c.append(c)
                continue
            bb = next((d["bbox"] for d, _ in ra if d["name"] == c["name"]), None)
            is_task_obj = False
            if bb is not None:
                x1c = max(0, int(bb[0]) - 8); y1c = max(0, int(bb[1]) - 8)
                x2c = min(agent_img.shape[1], int(bb[2]) + 8)
                y2c = min(agent_img.shape[0], int(bb[3]) + 8)
                if x2c - x1c > 16 and y2c - y1c > 16:
                    crop_a = agent_img[y1c:y2c, x1c:x2c]
                    q = ('What is the single main object in this image? '
                         'Answer with 1-3 words only.')
                    try:
                        # VLM names the crop; MENTIONED is decided symbolically
                        # against the fresh name (naming is the easy VLM task;
                        # the reasoning stays in the FOL layer).
                        ans = (self.client.infer(q, [crop_a], max_tokens=12) or "").lower()
                        is_task_obj = _mentioned(ans, task_description)
                    except Exception:
                        is_task_obj = False
            if is_task_obj:
                logger.info(f"[VisualGrounder] '{c['name']}' is a task object "
                            f"per crop query — excluded from obstacles")
                mentioned_c.append(c)
            else:
                unmen.append(c)

        # NEAREST_TO_PATH (eef start → detected target, else workspace center)
        spos = np.asarray(eef_pos)[:2]
        if mentioned_c:
            tgt = min(mentioned_c,
                      key=lambda c: float(np.linalg.norm(c["pos"][:2] - [0.0, 0.15])))
            tpos = np.array(tgt["pos"][:2])
        else:
            tpos = np.array([0.0, 0.15])
        seg = tpos - spos
        L2 = float(seg @ seg)

        def dpath(p):
            q = np.asarray(p)[:2]
            t = float(np.clip((q - spos) @ seg / L2, 0, 1)) if L2 > 1e-9 else 0.0
            return float(np.linalg.norm(q - (spos + t * seg)))

        target_est = np.asarray(tpos) if mentioned_c else None
        result = None
        if unmen:
            pick = min(unmen, key=lambda c: dpath(c["pos"]))
            u = UNCERT_TRIANGULATED
            self._refine_state = {
                "name": pick["name"],
                "pos": np.asarray(pick["pos"], dtype=np.float64),
                "rays": [(pick["ca"], pick["dda"]), (pick["cb"], pick["ddb"])],
            }
            result = {"name": pick["name"], "pos": np.asarray(pick["pos"]),
                      "uncertainty_m": u, "method": "visual-triangulated",
                      "target_xy": target_est,
                      "frame_R": _obstacle_frame(pick["dda"], pick["ddb"]),
                      "u_axes": np.array([2.5 * u, u, u]),
                      "half_extents": _bbox_extents(pick["bb"], pick["f"],
                                                    pick["rng"])}
        else:
            # z-band fallback: best unmentioned agentview ray ∩ z = Z_BAND_CENTER
            best = None
            for d, (c0, dd) in ra:
                if _mentioned(d["name"], task_description):
                    continue
                bb = d["bbox"]
                # same VLM-grounded MENTIONED verification as the main path
                x1c = max(0, int(bb[0]) - 8); y1c = max(0, int(bb[1]) - 8)
                x2c = min(agent_img.shape[1], int(bb[2]) + 8)
                y2c = min(agent_img.shape[0], int(bb[3]) + 8)
                if x2c - x1c > 16 and y2c - y1c > 16:
                    q = ('What is the single main object in this image? '
                         'Answer with 1-3 words only.')
                    try:
                        ans = (self.client.infer(q, [agent_img[y1c:y2c, x1c:x2c]],
                                                 max_tokens=12) or "").lower()
                        if _mentioned(ans, task_description):
                            logger.info(f"[VisualGrounder] fallback: '{d['name']}' "
                                        f"is a task object — skipped")
                            continue
                    except Exception:
                        pass
                if max(bb[2]-bb[0], bb[3]-bb[1]) > 0.43 * wh_a:
                    continue  # furniture-scale in pixels
                if abs(dd[2]) < 1e-6:
                    continue
                t = (Z_BAND_CENTER - c0[2]) / dd[2]
                p = c0 + t * dd
                if not _in_workspace(p):
                    continue
                if best is None or dpath(p) < dpath(best["pos"]):
                    best = {"name": d["name"], "pos": p}
            if best is not None:
                result = {"name": best["name"], "pos": np.asarray(best["pos"]),
                          "uncertainty_m": UNCERT_FALLBACK, "method": "visual-zband",
                          "target_xy": target_est}

        if result is not None:
            logger.info(
                f"[VisualGrounder] obstacle '{result['name']}' at "
                f"({result['pos'][0]:.3f},{result['pos'][1]:.3f},{result['pos'][2]:.3f}) "
                f"± {result['uncertainty_m']:.2f}m ({result['method']})"
            )
        else:
            logger.warning("[VisualGrounder] NO OBSTACLE GROUNDED from vision")
        if cache_key:
            self._cache[cache_key] = result
        return result
