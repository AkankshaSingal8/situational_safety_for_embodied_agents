"""
RGB-D CBF Pipeline: 3D Semantic Safety from RGB-D Input
=======================================================

Implements the Brunke et al. (IEEE RA-L 2025) pipeline adapted for
robosuite + MuJoCo simulation environments.

Pipeline:
  RGB-D frame
    ├── VLM (multi-prompt, majority vote)  →  object names + semantic properties
    ├── Grounding DINO + SAM               →  segmentation masks
    │       + depth map + camera K         →  3D object positions (camera frame)
    │       + T_cam_to_world               →  3D object positions (world frame)
    └── VLM (multi-prompt)                 →  spatial / behavioral safety constraints
                                                 ↓
                                      3D superquadric CBFs
                                                 ↓
                                      CBF-QP safety filter
                                      (certified velocity commands)

Key design choices (matching Brunke et al.):
  - VLM does NOT output metric 3D coordinates (avoids hallucination)
  - Depth image provides all metric geometry
  - Camera intrinsics K + extrinsic T_cam_to_world give consistent world frame
  - Segmentation masks used instead of bboxes for cleaner depth sampling

References:
  Brunke et al., "Semantically Safe Robot Manipulation", IEEE RA-L 2025
  perceive_semantix: https://github.com/utiasDSL/perceive_semantix_release
"""

import json
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ObjectInfo3D:
    name:          str
    position:      np.ndarray   # (3,) in world frame, metres
    dimensions:    np.ndarray   # (3,) [w, h, d] approximate, metres
    semantic_label: str
    properties:    Dict = field(default_factory=dict)
    pixel_bbox:    Optional[np.ndarray] = None   # [u0, v0, u1, v1] pixels
    mask:          Optional[np.ndarray] = None   # (H, W) bool segmentation mask


@dataclass
class SemanticConstraint3D:
    constraint_type: str        # "spatial" | "behavioral"
    source_object:   str        # manipulated object
    target_object:   str        # scene object creating the constraint
    relationship:    str        # "above" | "below" | "around" | "near"
    parameters:      Dict = field(default_factory=dict)


@dataclass
class SafetyContext3D:
    objects:                List[ObjectInfo3D]
    spatial_constraints:    List[SemanticConstraint3D]
    behavioral_constraints: List[SemanticConstraint3D]
    pose_constraint:        str   # "constrained_rotation" | "free_rotation"
    manipulated_object:     str
    reasoning:              str


# ============================================================================
# FREE / LOCAL VLM BACKENDS
# ============================================================================

class HuggingFaceVLMBackend:
    """
    Local VLM backend using HuggingFace transformers.  No API key required.
    Runs entirely on local GPU (V100-32GB is sufficient for 7B models).

    Recommended models:
      "Qwen/Qwen2-VL-7B-Instruct"           best quality,  ~14 GB VRAM
      "Qwen/Qwen2-VL-2B-Instruct"           fast,           ~4 GB VRAM
      "llava-hf/llava-v1.6-mistral-7b-hf"   good quality,  ~14 GB VRAM

    Install:
      pip install transformers accelerate pillow
    """

    DEFAULT_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"
    FALLBACK_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(self, model_id: Optional[str] = None,
                 device: str = "cuda",
                 max_new_tokens: int = 768):
        self.model_id       = model_id or self.DEFAULT_MODEL
        self.device         = device
        self.max_new_tokens = max_new_tokens
        self._model         = None
        self._processor     = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor

            dtype = torch.bfloat16
            candidates = [self.model_id]
            # LLaVA often breaks when transformers/processor APIs drift.
            # Fall back to a stable local VLM so the pipeline still runs.
            if "llava" in self.model_id.lower():
                candidates.append(self.FALLBACK_MODEL)

            last_error = None
            for idx, model_id in enumerate(candidates):
                try:
                    print(f"[LocalVLM] Loading {model_id} (this may take a minute)...")
                    processor = AutoProcessor.from_pretrained(
                        model_id, trust_remote_code=True)

                    m = model_id.lower()
                    if "qwen2" in m and "vl" in m:
                        from transformers import Qwen2VLForConditionalGeneration
                        model = Qwen2VLForConditionalGeneration.from_pretrained(
                            model_id, torch_dtype=dtype, device_map="auto",
                            trust_remote_code=True)
                    elif "llava" in m:
                        from transformers import LlavaNextForConditionalGeneration
                        model = LlavaNextForConditionalGeneration.from_pretrained(
                            model_id, torch_dtype=dtype, device_map="auto")
                    else:
                        from transformers import AutoModelForVision2Seq
                        model = AutoModelForVision2Seq.from_pretrained(
                            model_id, torch_dtype=dtype, device_map="auto",
                            trust_remote_code=True)

                    model.eval()
                    self.model_id = model_id
                    self._processor = processor
                    self._model = model
                    if idx > 0:
                        print(f"[LocalVLM] Fallback succeeded with {model_id}")
                    print(f"[LocalVLM] {model_id} ready")
                    return
                except Exception as e:
                    last_error = e
                    print(f"[LocalVLM] Could not load {model_id}: {e}")

            if last_error is not None:
                print(f"[LocalVLM] No local VLM could be loaded (last error: {last_error})")
        except Exception as e:
            print(f"[LocalVLM] Could not load {self.model_id}: {e}")

    def call(self, system_prompt: str, user_text: str,
             rgb: Optional[np.ndarray] = None) -> Optional[dict]:
        if self._model is None or self._processor is None:
            return None
        try:
            import torch
            from PIL import Image

            full_prompt = f"{system_prompt}\n\n{user_text}"
            content: list = []
            pil_img = None
            if rgb is not None:
                pil_img = Image.fromarray(rgb.astype(np.uint8))
                content.append({"type": "image", "image": pil_img})
            content.append({"type": "text", "text": full_prompt})

            messages = [{"role": "user", "content": content}]
            text_input = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            images = [pil_img] if pil_img is not None else None
            inputs = self._processor(
                text=[text_input], images=images,
                return_tensors="pt", padding=True,
            ).to(self._model.device)

            with torch.no_grad():
                out_ids = self._model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

            new_tokens = out_ids[0][inputs.input_ids.shape[1]:]
            raw = self._processor.decode(new_tokens, skip_special_tokens=True).strip()
            return _parse_json_from_text(raw)
        except Exception as e:
            print(f"[LocalVLM] call error ({type(e).__name__}): {e}")
            return None


class GeminiFreeBackend:
    """
    Google Gemini free-tier backend (google-genai SDK).  No GPU required.

    Free quota: 15 req/min for gemini-2.0-flash.
    Get a free API key at: https://ai.google.dev/

    Install:  pip install google-genai pillow
    """

    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(self, api_key: str, model_id: Optional[str] = None):
        self.api_key  = api_key
        self.model_id = model_id or self.DEFAULT_MODEL
        self._client  = None
        self._types   = None
        self._load()

    def _load(self) -> None:
        try:
            from google import genai
            from google.genai import types
            self._client = genai.Client(api_key=self.api_key)
            self._types  = types
            print(f"[GeminiBackend] Ready (model={self.model_id})")
        except ImportError:
            print("[GeminiBackend] Install: pip install google-genai")

    def call(self, system_prompt: str, user_text: str,
             rgb: Optional[np.ndarray] = None,
             max_retries: int = 3) -> Optional[dict]:
        if self._client is None:
            return None

        import io, time, re
        from PIL import Image

        contents: list = []
        if rgb is not None:
            pil_img = Image.fromarray(rgb.astype(np.uint8))
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            contents.append(
                self._types.Part.from_bytes(
                    data=buf.getvalue(), mime_type="image/jpeg"))
        contents.append(user_text)

        for attempt in range(max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=self._types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=768,
                    ),
                )
                return _parse_json_from_text(resp.text.strip())
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    m = re.search(r"retryDelay.*?(\d+)s", err_str)
                    wait = int(m.group(1)) + 2 if m else 65
                    print(f"[GeminiBackend] rate-limited (attempt {attempt+1}/{max_retries})"
                          f" — sleeping {wait}s ...")
                    time.sleep(wait)
                else:
                    print(f"[GeminiBackend] call error ({type(e).__name__}): {e}")
                    return None
        print(f"[GeminiBackend] gave up after {max_retries} retries")
        return None


def _parse_json_from_text(raw: str) -> Optional[dict]:
    """Extract the first JSON object from text that may contain markdown fences."""
    if not raw:
        return None
    if "```" in raw:
        inner = raw.split("```")[1]
        if inner.startswith("json"):
            inner = inner[4:]
        raw = inner.strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start >= 0 and end > start:
        raw = raw[start:end]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ============================================================================
# ROBOSUITE CAMERA  —  depth conversion + unprojection
# ============================================================================

class RobosuiteCamera:
    """
    Camera utilities for robosuite + MuJoCo environments.

    Handles:
      - NDC → metric depth conversion  (MuJoCo specific)
      - Pinhole intrinsics from sim
      - Camera extrinsics (T_cam_to_world)
      - Mask-based and bbox-based 3D unprojection
    """

    def __init__(self, sim, camera_name: str = "agentview",
                 img_height: int = 256, img_width: int = 256):
        self.sim         = sim
        self.camera_name = camera_name
        self.img_height  = img_height
        self.img_width   = img_width

    def depth_ndc_to_metric(self, depth_ndc: np.ndarray) -> np.ndarray:
        """Convert MuJoCo NDC depth buffer to metric depth (metres)."""
        near = self.sim.model.vis.map.znear * self.sim.model.stat.extent
        far  = self.sim.model.vis.map.zfar  * self.sim.model.stat.extent
        return near * far / (far - depth_ndc * (far - near))

    def get_intrinsics(self) -> np.ndarray:
        """Return (3,3) pinhole K matrix from sim camera parameters."""
        cam_id = self.sim.model.camera_name2id(self.camera_name)
        fovy   = self.sim.model.cam_fovy[cam_id]
        f      = (self.img_height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
        cx, cy = self.img_width / 2.0, self.img_height / 2.0
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    def get_camera_pose_world(self) -> np.ndarray:
        """Return (4,4) SE3 transform T_cam_to_world."""
        cam_id = self.sim.model.camera_name2id(self.camera_name)
        pos    = self.sim.data.cam_xpos[cam_id]
        R_mat  = self.sim.data.cam_xmat[cam_id].reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3,  3] = pos
        return T

    def unproject_pixel(self, u: float, v: float, depth: float,
                        K: np.ndarray) -> np.ndarray:
        x = (u - K[0, 2]) * depth / K[0, 0]
        y = (v - K[1, 2]) * depth / K[1, 1]
        return np.array([x, y, depth])

    def unproject_mask_to_3d(self, mask: np.ndarray, depth_metric: np.ndarray,
                              K: np.ndarray, T_cam_to_world: np.ndarray,
                              n_samples: int = 300) -> Tuple[np.ndarray, np.ndarray]:
        """Unproject segmentation mask pixels + depth → 3D center + dims (world frame)."""
        vs, us = np.where(mask)
        if len(vs) == 0:
            return np.array([0.0, 0.0, 1.0]), np.array([0.08, 0.08, 0.05])

        depths = depth_metric[vs, us]
        valid  = (depths > 0.01) & (depths < 5.0)
        vs, us, depths = vs[valid], us[valid], depths[valid]

        if len(vs) == 0:
            cy_px = int(np.mean(np.where(mask)[0]))
            cx_px = int(np.mean(np.where(mask)[1]))
            d = float(np.median(depth_metric[mask]))
            d = d if 0.01 < d < 5.0 else 0.8
            xs = np.array([(cx_px - K[0, 2]) * d / K[0, 0]])
            ys = np.array([(cy_px - K[1, 2]) * d / K[1, 1]])
            pts_cam = np.column_stack([xs, ys, [d]])
        else:
            if len(vs) > n_samples:
                idx = np.random.choice(len(vs), n_samples, replace=False)
                vs, us, depths = vs[idx], us[idx], depths[idx]
            xs = (us - K[0, 2]) * depths / K[0, 0]
            ys = (vs - K[1, 2]) * depths / K[1, 1]
            pts_cam = np.column_stack([xs, ys, depths])

        pts_h     = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
        pts_world = (T_cam_to_world @ pts_h.T).T[:, :3]
        position  = np.median(pts_world, axis=0)
        dims = (np.maximum(pts_world.max(axis=0) - pts_world.min(axis=0), 0.02)
                if len(pts_world) > 2 else np.array([0.08, 0.08, 0.05]))
        return position, dims

    def unproject_bbox_to_3d(self, bbox: np.ndarray, depth_metric: np.ndarray,
                              K: np.ndarray, T_cam_to_world: np.ndarray,
                              n_samples: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback bbox-based 3D unprojection."""
        u_min, v_min, u_max, v_max = bbox.astype(int)
        u_min = max(0, u_min); v_min = max(0, v_min)
        u_max = min(depth_metric.shape[1] - 1, u_max)
        v_max = min(depth_metric.shape[0] - 1, v_max)

        roi   = depth_metric[v_min:v_max + 1, u_min:u_max + 1]
        valid = (roi > 0.01) & (roi < 5.0)

        if valid.sum() < 3:
            u_c, v_c = (u_min + u_max) / 2.0, (v_min + v_max) / 2.0
            d = float(depth_metric[int(v_c), int(u_c)])
            d = d if 0.01 < d < 5.0 else 0.8
            pts_cam = np.array([[(u_c - K[0,2])*d/K[0,0],
                                  (v_c - K[1,2])*d/K[1,1], d]])
        else:
            d_thresh = np.percentile(roi[valid], 30) * 1.2
            fg       = (roi <= d_thresh) & valid
            vl, ul   = np.where(fg)
            if len(vl) > n_samples:
                idx = np.random.choice(len(vl), n_samples, replace=False)
                vl, ul = vl[idx], ul[idx]
            xs = (ul + u_min - K[0,2]) * roi[vl, ul] / K[0,0]
            ys = (vl + v_min - K[1,2]) * roi[vl, ul] / K[1,1]
            pts_cam = np.column_stack([xs, ys, roi[vl, ul]])

        pts_h     = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
        pts_world = (T_cam_to_world @ pts_h.T).T[:, :3]
        position  = np.median(pts_world, axis=0)
        dims = (np.maximum(pts_world.max(axis=0) - pts_world.min(axis=0), 0.02)
                if len(pts_world) > 2 else np.array([0.08, 0.08, 0.05]))
        return position, dims


# ============================================================================
# GROUNDED SAM LOCALIZER
# ============================================================================

class GroundedSAMLocalizer:
    """
    Open-vocabulary object localizer: Grounding DINO (detection) + SAM (masks).

    Falls back to a mock implementation when weights are not installed.

    Install:
      pip install groundingdino-py
      pip install git+https://github.com/facebookresearch/segment-anything.git
    """

    # Resolve defaults relative to this file's parent directory so the weights
    # are found regardless of the working directory the script is launched from.
    _HERE = str(__import__("pathlib").Path(__file__).resolve().parent)
    GDINO_CONFIG = str(__import__("pathlib").Path(_HERE).parent / "GroundingDINO_SwinT_OGC.py")
    GDINO_CKPT   = str(__import__("pathlib").Path(_HERE).parent / "groundingdino_swint_ogc.pth")
    SAM_CKPT     = str(__import__("pathlib").Path(_HERE).parent / "sam_vit_h_4b8939.pth")
    SAM_TYPE     = "vit_h"
    BOX_THRESH   = 0.30
    TEXT_THRESH  = 0.25

    def __init__(self, gdino_config: Optional[str] = None,
                 gdino_ckpt:   Optional[str] = None,
                 sam_ckpt:     Optional[str] = None,
                 device:       str = "cuda"):
        self.device      = device
        self._gdino      = None
        self._sam        = None
        self._predictor  = None
        self._loaded     = False
        self._load(gdino_config or self.GDINO_CONFIG,
                   gdino_ckpt   or self.GDINO_CKPT,
                   sam_ckpt     or self.SAM_CKPT)

    def _load(self, gdino_config, gdino_ckpt, sam_ckpt) -> None:
        try:
            import torch
            from groundingdino.util.inference import load_model
            from segment_anything import sam_model_registry, SamPredictor

            self._gdino = load_model(gdino_config, gdino_ckpt, device=self.device)
            sam_model   = sam_model_registry[self.SAM_TYPE](checkpoint=sam_ckpt)
            sam_model   = sam_model.to(self.device)
            self._predictor = SamPredictor(sam_model)
            self._loaded = True
            print("[GroundedSAM] Models loaded")
        except Exception as e:
            print(f"[GroundedSAM] Models not loaded ({type(e).__name__}: {e}) — using mock localizer")

    def localize(self, rgb: np.ndarray,
                 object_names: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """
        Detect and segment objects by name.

        Returns:
            dict mapping name → boolean mask (H, W) or None if not found
        """
        if self._loaded:
            try:
                return self._real_localize(rgb, object_names)
            except Exception as e:
                print(f"[GroundedSAM] Runtime localisation failed ({type(e).__name__}: {e})"
                      " — falling back to mock localizer")
                return self._mock_localize(rgb, object_names)
        return self._mock_localize(rgb, object_names)

    MAX_BOX_AREA_RATIO = 0.50

    def _real_localize(self, rgb: np.ndarray,
                       object_names: List[str]) -> Dict[str, Optional[np.ndarray]]:
        import torch
        from groundingdino.util.inference import predict
        from groundingdino.util import box_ops
        from PIL import Image
        import torchvision.transforms as T

        H, W = rgb.shape[:2]
        img_area = float(H * W)
        transform = T.Compose([
            T.Resize(800), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        pil_img = Image.fromarray(rgb)
        img_t   = transform(pil_img).to(self.device)

        prompt  = " . ".join(object_names) + " ."
        boxes, logits, phrases = predict(self._gdino, img_t, prompt,
                                          self.BOX_THRESH, self.TEXT_THRESH,
                                          device=self.device)

        self._predictor.set_image(rgb)
        results: Dict[str, Optional[np.ndarray]] = {n: None for n in object_names}

        if boxes.shape[0] == 0:
            print(f"  [GroundedSAM] No detections for prompt: '{prompt}'")
            return results

        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([W, H, W, H])

        print(f"  [GroundedSAM] {boxes_xyxy.shape[0]} raw detections:")
        for i in range(boxes_xyxy.shape[0]):
            b = boxes_xyxy[i].cpu().numpy()
            area_ratio = (b[2] - b[0]) * (b[3] - b[1]) / img_area
            print(f"    det[{i}] phrase='{phrases[i]}'  score={float(logits[i]):.3f}"
                  f"  box=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]"
                  f"  area={area_ratio*100:.1f}%")

        for name in object_names:
            name_lower = name.lower().split()
            best_box = None
            best_score = -1.0
            for i, phrase in enumerate(phrases):
                phrase_lower = phrase.lower().strip()
                if not phrase_lower:
                    continue
                phrase_words = phrase_lower.split()
                matched = (any(nw in phrase_words for nw in name_lower)
                           or any(pw in name_lower for pw in phrase_words))
                if not matched:
                    continue

                b = boxes_xyxy[i].cpu().numpy()
                box_area = (b[2] - b[0]) * (b[3] - b[1])
                if box_area / img_area > self.MAX_BOX_AREA_RATIO:
                    print(f"  [GroundedSAM] Skipping det[{i}] for '{name}': "
                          f"box covers {box_area/img_area*100:.1f}% of image (>{self.MAX_BOX_AREA_RATIO*100:.0f}%)")
                    continue

                score = float(logits[i])
                if score > best_score:
                    best_score = score
                    best_box   = b.astype(int)

            if best_box is not None:
                print(f"  [GroundedSAM] '{name}' → box=[{best_box[0]},{best_box[1]},"
                      f"{best_box[2]},{best_box[3]}]  score={best_score:.3f}")
                masks, scores, _ = self._predictor.predict(
                    box=best_box, multimask_output=True)
                best_mask_idx = int(np.argmax(scores))
                mask = masks[best_mask_idx].astype(bool)
                mask_ratio = mask.sum() / img_area
                print(f"  [GroundedSAM] '{name}' mask: {int(mask.sum())} pixels "
                      f"({mask_ratio*100:.1f}% of image), SAM score={float(scores[best_mask_idx]):.3f}")
                results[name] = mask
            else:
                print(f"  [GroundedSAM] '{name}' → no valid detection found")

        return results

    def _mock_localize(self, rgb: np.ndarray,
                       object_names: List[str]) -> Dict[str, Optional[np.ndarray]]:
        H, W = rgb.shape[:2]
        n    = max(len(object_names), 1)
        results = {}
        for i, name in enumerate(object_names):
            cx = int(W * (i + 1) / (n + 1))
            cy = int(H * 0.60)
            rw, rh = int(W * 0.16), int(H * 0.18)
            mask = np.zeros((H, W), dtype=bool)
            y0, y1 = max(0, cy - rh), min(H, cy + rh)
            x0, x1 = max(0, cx - rw), min(W, cx + rw)
            mask[y0:y1, x0:x1] = True
            print(f"  [GroundedSAM mock] {name:20s}  "
                  f"center=({cx},{cy})  mask_pixels={mask.sum()}")
            results[name] = mask
        return results


# ============================================================================
# VLM 3D SCENE ANALYZER  (Brunke et al. multi-prompt strategy)
# ============================================================================

class VLM3DSceneAnalyzer:
    """
    VLM scene analyzer following Brunke et al. Sec V-B multi-prompt strategy.

    STRICT SEPARATION OF CONCERNS:
      VLM output  → object names + pixel bboxes + semantic safety judgments
      Depth image → actual 3D metric positions (via camera)
      VLM never outputs metric coordinates — avoids hallucinated positions.

    Multi-prompt approach (higher recall than single prompt):
      For each (held_object, scene_object, relationship):
        → query VLM: "is this safe?" (n_votes times, majority vote)
    """

    OBJECT_DETECTION_PROMPT = """You are a robot workspace safety analyzer.
Identify ALL objects on the table/workspace surface that could be affected by robot manipulation.
Output ONLY valid JSON (no markdown):
{
  "objects": [
    {
      "name": "object_name",
      "semantic_label": "category",
      "properties": {
        "fragile": true/false,
        "water_sensitive": true/false,
        "flammable": true/false,
        "electronic": true/false
      }
    }
  ]
}
Only list distinct physical objects. Do not include the robot arm or table surface."""

    SPATIAL_SAFETY_PROMPT = """You are a robot safety reasoning system.
Answer ONLY with valid JSON (no markdown): {"safe": true/false, "reason": "one sentence"}
true  = this action is safe
false = this action is UNSAFE and must be prevented"""

    BEHAVIORAL_CAUTION_PROMPT = """You are a robot safety system.
Answer ONLY with valid JSON (no markdown):
{"caution_needed": true/false, "caution_level": 0.0 to 1.0, "max_velocity": 0.05 to 0.3,
 "reason": "one sentence"}"""

    POSE_CONSTRAINT_PROMPT = (
        "Determine if the held object must stay upright during manipulation. "
        "Output ONLY JSON: {\"constrained\": true/false, \"reason\": \"one sentence\"}"
    )

    SPATIAL_RELATIONSHIPS = ["above", "below", "around", "near"]

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "claude-3-5-sonnet-20241022",
                 n_votes: int = 1,
                 backend=None):
        """
        Args:
            api_key:  Anthropic API key (used only when backend=None)
            model:    Anthropic model ID (used only when backend=None)
            n_votes:  Majority-vote repetitions per spatial safety query (1 = no voting)
            backend:  Optional free/local backend — takes priority over Anthropic:
                        HuggingFaceVLMBackend(...)    local GPU inference
                        GeminiFreeBackend(api_key=...) Google Gemini free tier
        """
        self.api_key  = api_key
        self.model    = model
        self.n_votes  = n_votes
        self._backend = backend
        self._client  = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                return None
        return self._client

    def _encode_image(self, rgb: np.ndarray) -> Tuple[str, str]:
        try:
            from PIL import Image
            import base64, io
            buf = io.BytesIO()
            Image.fromarray(rgb.astype(np.uint8)).save(buf, format="PNG")
            return base64.standard_b64encode(buf.getvalue()).decode(), "image/png"
        except ImportError:
            raise ImportError("Install pillow: pip install pillow")

    def _call_vlm(self, system: str, user_text: str,
                  rgb: Optional[np.ndarray] = None,
                  max_tokens: int = 200) -> Optional[dict]:
        """Single VLM call. Routes to free backend first, then Anthropic."""
        if self._backend is not None:
            return self._backend.call(system, user_text, rgb)

        client = self._get_client()
        if client is None:
            return None
        content = []
        if rgb is not None:
            b64, mt = self._encode_image(rgb)
            content.append({"type": "image",
                             "source": {"type": "base64", "media_type": mt, "data": b64}})
        content.append({"type": "text", "text": user_text})
        try:
            resp = self._client.messages.create(
                model=self.model, max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": content}]
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(raw)
        except Exception as e:
            print(f"[VLM] _call_vlm error ({type(e).__name__}): {e}")
            return None

    # ------------------------------------------------------------------
    # Step 1 — object detection
    # ------------------------------------------------------------------

    def detect_objects(self, rgb: np.ndarray) -> List[dict]:
        result = self._call_vlm(
            system=self.OBJECT_DETECTION_PROMPT,
            user_text="Identify all objects in this robot workspace.",
            rgb=rgb, max_tokens=2000
        )
        if result and "objects" in result:
            return result["objects"]
        print("[VLM] Object detection failed or no API, using mock objects")
        return self._mock_objects()

    def _mock_objects(self) -> List[dict]:
        return [
            {"name": "can",         "semantic_label": "container",
             "properties": {"fragile": False, "water_sensitive": True,
                            "flammable": False, "electronic": False}},
            {"name": "milk_carton", "semantic_label": "container",
             "properties": {"fragile": False, "water_sensitive": True,
                            "flammable": False, "electronic": False}},
            {"name": "cereal_box",  "semantic_label": "container",
             "properties": {"fragile": False, "water_sensitive": True,
                            "flammable": False, "electronic": False}},
        ]

    # ------------------------------------------------------------------
    # Step 2 — spatial safety (multi-prompt majority vote)
    # ------------------------------------------------------------------

    def query_spatial_safety(self, rgb: np.ndarray, held: str,
                              scene_obj: str, relationship: str,
                              obj_properties: Optional[dict] = None) -> dict:
        props_hint = ""
        if obj_properties:
            flags = [k for k, v in obj_properties.items() if v is True]
            if flags:
                props_hint = f" (properties: {', '.join(flags)})"
        prompt = (
            f"The robot is holding '{held}' and is about to move it "
            f"'{relationship}' the '{scene_obj}'{props_hint}.\n"
            f"Could this cause damage, spills, fire, or other hazards?\n"
            f"Reply with ONLY this JSON and nothing else:\n"
            f"{{\"safe\": true}} if safe, or {{\"safe\": false, \"reason\": \"one sentence\"}} if UNSAFE."
        )
        votes_unsafe = 0
        reasons      = []
        for _ in range(self.n_votes):
            r = self._call_vlm(self.SPATIAL_SAFETY_PROMPT, prompt, rgb=rgb)
            if r is not None:
                if not r.get("safe", True):
                    votes_unsafe += 1
                reasons.append(r.get("reason", ""))

        total = max(self.n_votes, 1)
        unsafe_pct = int(100 * votes_unsafe / total)
        unsafe = votes_unsafe > total / 2
        reason = reasons[0] if reasons else f"Liquid may spill onto {scene_obj}"
        return {"unsafe": unsafe, "unsafe_pct": unsafe_pct, "reason": reason}

    # ------------------------------------------------------------------
    # Step 3 — behavioral caution
    # ------------------------------------------------------------------

    def query_behavioral_caution(self, rgb: np.ndarray, held: str,
                                  scene_obj: str) -> dict:
        prompt = (f"The robot is holding '{held}' and moving near '{scene_obj}'. "
                  f"Should the robot exercise extra caution?")
        r = self._call_vlm(self.BEHAVIORAL_CAUTION_PROMPT, prompt, rgb=rgb)
        if r is not None:
            return r
        return {"caution_needed": False, "caution_level": 0.0,
                "max_velocity": 0.2, "reason": ""}

    # ------------------------------------------------------------------
    # Step 4 — pose constraint
    # ------------------------------------------------------------------

    def query_pose_constraint(self, rgb: np.ndarray, held: str) -> bool:
        prompt = f"The robot is holding '{held}'."
        r = self._call_vlm(self.POSE_CONSTRAINT_PROMPT, prompt, rgb=rgb)
        if r is not None:
            return bool(r.get("constrained", False))
        held_l = held.lower()
        return any(w in held_l for w in
                   ["water", "cup", "coffee", "liquid", "soup", "glass"])

    # ------------------------------------------------------------------
    # Full scene analysis
    # ------------------------------------------------------------------

    def analyze_scene(self, rgb: np.ndarray, depth_metric: np.ndarray,
                      camera, K: np.ndarray, T_cam_to_world: np.ndarray,
                      held_object: str,
                      localizer: Optional["GroundedSAMLocalizer"] = None
                      ) -> SafetyContext3D:
        print("  [Step 1] VLM: object names + semantic properties...")
        raw_objects = self.detect_objects(rgb)
        print(f"           {len(raw_objects)} objects: {[o['name'] for o in raw_objects]}")

        print("  [Step 2] GroundedSAM localisation + depth → 3D positions...")
        objects_3d: List[ObjectInfo3D] = []

        if localizer is not None:
            masks = localizer.localize(rgb, [o["name"] for o in raw_objects])
        else:
            masks = {o["name"]: None for o in raw_objects}

        for obj_data in raw_objects:
            name = obj_data["name"]
            mask = masks.get(name)

            if mask is not None and mask.any():
                pos, dims = camera.unproject_mask_to_3d(mask, depth_metric, K, T_cam_to_world)
                method = "mask"
            else:
                H, W = rgb.shape[:2]
                bbox = np.array([W * 0.1, H * 0.1, W * 0.9, H * 0.9])
                pos, dims = camera.unproject_bbox_to_3d(bbox, depth_metric, K, T_cam_to_world)
                method = "bbox-fallback"

            print(f"           [{method}] {name:20s}  "
                  f"pos={np.array2string(pos, precision=3)}  "
                  f"dims={np.array2string(dims, precision=3)}")
            objects_3d.append(ObjectInfo3D(
                name=name,
                position=pos,
                dimensions=dims,
                semantic_label=obj_data.get("semantic_label", "object"),
                properties=obj_data.get("properties", {}),
                mask=mask,
            ))

        print(f"  [Step 3] Spatial constraints for '{held_object}' (multi-prompt)...")
        spatial_constraints: List[SemanticConstraint3D] = []
        n_unsafe = 0
        for obj in objects_3d:
            for rel in self.SPATIAL_RELATIONSHIPS:
                result = self.query_spatial_safety(
                    rgb, held_object, obj.name, rel, obj.properties)
                if result["unsafe"]:
                    n_unsafe += 1
                    margin = 0.10
                    print(f"           UNSAFE: {held_object} {rel} {obj.name} "
                          f"({result['unsafe_pct']}%) — {result['reason']}")
                    spatial_constraints.append(SemanticConstraint3D(
                        "spatial", held_object, obj.name, rel,
                        {"safety_margin": margin, "reason": result["reason"],
                         "unsafe_pct": result["unsafe_pct"]}))
        print(f"           {n_unsafe} unsafe pairs found")

        # Fallback: when the VLM produced zero unsafe pairs (e.g. small local models
        # like LLaVA-7B are often unreliable at returning structured safety JSON),
        # apply rule-based constraints derived from object semantic properties.
        if n_unsafe == 0 and objects_3d:
            held_l   = held_object.lower()
            is_liquid = any(w in held_l for w in
                            ["water", "cup", "coffee", "liquid", "soup", "glass", "juice"])
            is_fire   = any(w in held_l for w in
                            ["candle", "flame", "lighter", "torch"])
            print("           [fallback] VLM returned 0 unsafe pairs — "
                  "applying property-based heuristic constraints")
            for obj in objects_3d:
                props = obj.properties
                if is_liquid and props.get("water_sensitive", False):
                    for rel, margin, reason in [
                        ("above",  0.10, f"Liquid above {obj.name} risks spillage damage"),
                        ("around", 0.08, f"Splash radius around {obj.name}"),
                    ]:
                        print(f"           [fallback] UNSAFE: {held_object} {rel} {obj.name}"
                              f" — {reason}")
                        spatial_constraints.append(SemanticConstraint3D(
                            "spatial", held_object, obj.name, rel,
                            {"safety_margin": margin, "reason": reason,
                             "unsafe_pct": 100, "source": "property_fallback"}))
                        n_unsafe += 1
                if is_liquid and props.get("electronic", False) and not props.get("water_sensitive", False):
                    reason = f"Electronics near liquid are a damage hazard"
                    print(f"           [fallback] UNSAFE: {held_object} near {obj.name}"
                          f" — {reason}")
                    spatial_constraints.append(SemanticConstraint3D(
                        "spatial", held_object, obj.name, "around",
                        {"safety_margin": 0.10, "reason": reason,
                         "unsafe_pct": 100, "source": "property_fallback"}))
                    n_unsafe += 1
                if is_fire and props.get("flammable", False):
                    reason = f"Open flame near flammable {obj.name} is a fire hazard"
                    print(f"           [fallback] UNSAFE: {held_object} near {obj.name}"
                          f" — {reason}")
                    spatial_constraints.append(SemanticConstraint3D(
                        "spatial", held_object, obj.name, "around",
                        {"safety_margin": 0.20, "reason": reason,
                         "unsafe_pct": 100, "source": "property_fallback"}))
                    n_unsafe += 1
            print(f"           [fallback] added {n_unsafe} property-based constraints")

        print("  [Step 4] Behavioral caution queries...")
        behavioral_constraints: List[SemanticConstraint3D] = []
        for obj in objects_3d:
            result = self.query_behavioral_caution(rgb, held_object, obj.name)
            if result.get("caution_needed", False):
                level = result.get("caution_level", 0.5)
                print(f"           CAUTION near {obj.name}: level={level:.2f}")
                behavioral_constraints.append(SemanticConstraint3D(
                    "behavioral", held_object, obj.name, "near",
                    {"caution_level": level,
                     "max_approach_velocity": result.get("max_velocity", 0.1),
                     "reason": result.get("reason", "")}))

        print("  [Step 5] Pose constraint query...")
        constrained = self.query_pose_constraint(rgb, held_object)
        pose = "constrained_rotation" if constrained else "free_rotation"
        print(f"           {pose} — "
              f"{'Liquid container must stay upright' if constrained else 'No rotation constraint'}")

        return SafetyContext3D(
            objects=objects_3d,
            spatial_constraints=spatial_constraints,
            behavioral_constraints=behavioral_constraints,
            pose_constraint=pose,
            manipulated_object=held_object,
            reasoning="Multi-prompt VLM analysis",
        )


# ============================================================================
# 3D CBF CONSTRUCTOR
# ============================================================================

class CBFConstructor3D:
    """
    Builds 3D superquadric CBFs from a SafetyContext3D.

    h(x) = [(dx/ax)^(2/e) + (dy/ay)^(2/e) + (dz/az)^(2/e)]^e - 1
    h(x) > 0  →  safe (outside superquadric)
    h(x) < 0  →  unsafe (inside superquadric)
    """

    def build_cbfs(self, ctx: SafetyContext3D) -> dict:
        obj_lookup = {o.name: o for o in ctx.objects}
        spatial_cbfs = []

        for sc in ctx.spatial_constraints:
            if sc.target_object not in obj_lookup:
                warnings.warn(f"Object '{sc.target_object}' not in scene, skipping")
                continue
            obj    = obj_lookup[sc.target_object]
            margin = sc.parameters.get("safety_margin", 0.10)
            cbf    = self._build_spatial_cbf(obj, sc.relationship, margin)
            if cbf is not None:
                spatial_cbfs.append(cbf)

        behavioral_params = {}
        for bc in ctx.behavioral_constraints:
            caution  = bc.parameters.get("caution_level", 0.5)
            max_vel  = bc.parameters.get("max_approach_velocity", 0.1)
            behavioral_params[bc.target_object] = {
                "alpha_scale": 1.0 - 0.9 * caution,
                "max_approach_velocity": max_vel,
                "caution_level": caution,
            }

        pose_params = {
            "constrained": ctx.pose_constraint == "constrained_rotation",
            "max_angular_velocity": 0.1 if ctx.pose_constraint == "constrained_rotation"
                                         else float("inf"),
        }

        return {"spatial_cbfs": spatial_cbfs,
                "behavioral_params": behavioral_params,
                "pose_params": pose_params}

    def _build_spatial_cbf(self, obj: ObjectInfo3D,
                            relationship: str, margin: float):
        cx, cy, cz = obj.position
        w, h, d    = obj.dimensions

        if relationship == "above":
            # Local directional "above" constraint (slab/extruded).
            # In camera-frame-as-world (SingleImageCamera, T=I):
            #   X = right,  Y = down,  Z = depth (into scene)
            # "above object" = smaller Y than object top.
            # Unsafe when BOTH:
            #   1) EE is within object's horizontal/depth footprint (X and Z, with margin)
            #   2) EE.y < object_top_y - margin  (above the object)
            half_x = w / 2 + margin
            half_z = d / 2 + margin
            y_limit = cy - h / 2 - margin
            sharpness = 12.0
            params = {
                "center": np.array([cx, cy, cz]),
                "half_x": half_x,
                "half_z": half_z,
                "y_limit": y_limit,
                "sharpness": sharpness,
                "object_name": obj.name,
                "relationship": relationship,
            }
            name = f"no_{relationship}_{obj.name}"

            def h_func(x_ee, p=params):
                dx = (x_ee[0] - p["center"][0]) / p["half_x"]
                dz = (x_ee[2] - p["center"][2]) / p["half_z"]
                h_x = dx * dx - 1.0                # <0 inside horizontal extent
                h_z = dz * dz - 1.0                # <0 inside depth extent
                h_y = x_ee[1] - p["y_limit"]       # <0 when above (smaller Y)
                k = p["sharpness"]
                m = max(h_x, h_y, h_z)
                return m + np.log(
                    np.exp(k * (h_x - m)) + np.exp(k * (h_y - m)) + np.exp(k * (h_z - m))
                ) / k

            def grad_h_func(x_ee, p=params):
                dx = (x_ee[0] - p["center"][0]) / p["half_x"]
                dz = (x_ee[2] - p["center"][2]) / p["half_z"]
                h_x = dx * dx - 1.0
                h_z = dz * dz - 1.0
                h_y = x_ee[1] - p["y_limit"]
                k = p["sharpness"]
                m = max(h_x, h_y, h_z)
                e_x = np.exp(k * (h_x - m))
                e_y = np.exp(k * (h_y - m))
                e_z = np.exp(k * (h_z - m))
                den = e_x + e_y + e_z + 1e-12
                w_x = e_x / den
                w_y = e_y / den
                w_z = e_z / den
                grad_x = np.array([
                    2.0 * (x_ee[0] - p["center"][0]) / (p["half_x"] ** 2),
                    0.0,
                    0.0,
                ], dtype=float)
                grad_y = np.array([0.0, 1.0, 0.0], dtype=float)
                grad_z = np.array([
                    0.0,
                    0.0,
                    2.0 * (x_ee[2] - p["center"][2]) / (p["half_z"] ** 2),
                ], dtype=float)
                return w_x * grad_x + w_y * grad_y + w_z * grad_z

            return (h_func, grad_h_func, name, params)
        elif relationship == "around":
            ax, ay, az = w/2 + margin, h/2 + margin, d/2 + margin
            epsilon    = 0.8
        elif relationship in ("near", "below"):
            ax, ay, az = w/2 + margin*0.8, h/2 + margin*0.8, d/2 + margin*0.8
            epsilon    = 0.5
        else:
            ax, ay, az = w/2 + margin, h/2 + margin, d/2 + margin
            epsilon    = 0.5

        params = {
            "center":    np.array([cx, cy, cz]),
            "semi_axes": np.array([ax, ay, az]),
            "epsilon":   epsilon,
            "object_name": obj.name,
            "relationship": relationship,
        }
        name = f"no_{relationship}_{obj.name}"

        def h_func(x_ee, p=params):
            dx = (x_ee[0] - p["center"][0]) / p["semi_axes"][0]
            dy = (x_ee[1] - p["center"][1]) / p["semi_axes"][1]
            dz = (x_ee[2] - p["center"][2]) / p["semi_axes"][2]
            e  = 2.0 / p["epsilon"]
            g  = (np.abs(dx)**e + np.abs(dy)**e + np.abs(dz)**e) ** p["epsilon"]
            return g - 1.0

        def grad_h_func(x_ee, p=params):
            dx  = (x_ee[0] - p["center"][0]) / p["semi_axes"][0]
            dy  = (x_ee[1] - p["center"][1]) / p["semi_axes"][1]
            dz  = (x_ee[2] - p["center"][2]) / p["semi_axes"][2]
            eps = 1e-8
            e   = 2.0 / p["epsilon"]
            adx = np.abs(dx) + eps; ady = np.abs(dy) + eps; adz = np.abs(dz) + eps
            inner   = adx**e + ady**e + adz**e + eps
            outer_e = p["epsilon"] - 1.0
            common  = p["epsilon"] * (inner ** outer_e)
            g = np.zeros(3)
            g[0] = common * e * adx**(e-1) * np.sign(dx) / p["semi_axes"][0]
            g[1] = common * e * ady**(e-1) * np.sign(dy) / p["semi_axes"][1]
            g[2] = common * e * adz**(e-1) * np.sign(dz) / p["semi_axes"][2]
            return g

        return (h_func, grad_h_func, name, params)


# ============================================================================
# 3D CBF SAFETY FILTER (QP)
# ============================================================================

class CBFSafetyFilter3D:
    """
    QP-based CBF safety filter for 3D velocity commands.

    Solves at each timestep:
        u_cert = argmin ||u - u_cmd||^2
        s.t.   ∇h_i · u >= -α_i(h_i(x))   for all CBFs
    """

    def __init__(self, cbf_data: dict, dt: float = 0.02, u_max: float = 0.5):
        self.spatial_cbfs     = cbf_data["spatial_cbfs"]
        self.behavioral_params = cbf_data.get("behavioral_params", {})
        self.pose_params      = cbf_data.get("pose_params", {})
        self.dt    = dt
        self.u_max = u_max

    def certify(self, x_ee: np.ndarray, u_cmd: np.ndarray
                ) -> Tuple[np.ndarray, dict]:
        if not self.spatial_cbfs:
            u_cert = np.clip(u_cmd, -self.u_max, self.u_max)
            return u_cert, {"cbf_values": [], "active_constraints": [],
                            "modified": False,
                            "u_cmd": u_cmd.copy(), "u_cert": u_cert.copy()}

        h_vals, grads, alphas, names = [], [], [], []
        for (h_func, grad_h_func, name, params) in self.spatial_cbfs:
            h  = h_func(x_ee)
            gh = grad_h_func(x_ee)
            h_vals.append(h); grads.append(gh); names.append(name)
            obj_name    = params.get("object_name", "")
            alpha_scale = self.behavioral_params.get(obj_name, {}).get("alpha_scale", 1.0)
            alphas.append(alpha_scale * max(h, 0.0))

        A = np.array(grads)
        b = np.array([-a for a in alphas])
        u_cert = self._solve_qp(u_cmd, A, b)
        u_cert = np.clip(u_cert, -self.u_max, self.u_max)

        active   = [names[i] for i in range(len(names)) if h_vals[i] < 0.3]
        modified = np.linalg.norm(u_cert - u_cmd) > 1e-4
        return u_cert, {
            "cbf_values": list(zip(names, h_vals)),
            "active_constraints": active,
            "modified": modified,
            "u_cmd": u_cmd.copy(),
            "u_cert": u_cert.copy(),
        }

    def _solve_qp(self, u_cmd: np.ndarray, A: np.ndarray, b: np.ndarray,
                  max_iter: int = 150) -> np.ndarray:
        if A.shape[0] == 0:
            return u_cmd.copy()
        if np.all(A @ u_cmd - b >= -1e-8):
            return u_cmd.copy()
        u = u_cmd.copy()
        for _ in range(max_iter):
            all_ok = True
            for i in range(A.shape[0]):
                margin = A[i] @ u - b[i]
                if margin < -1e-8:
                    all_ok = False
                    a_norm = np.dot(A[i], A[i]) + 1e-10
                    u += (-margin / a_norm) * A[i]
            if all_ok:
                break
        return u


# ============================================================================
# RGBD CBF PIPELINE  (robosuite top-level wrapper)
# ============================================================================

class RGBDCBFPipeline:
    """
    Top-level pipeline for robosuite + MuJoCo experiments.
    """

    def __init__(self, sim, camera_name: str = "agentview",
                 img_height: int = 256, img_width: int = 256,
                 held_object: str = "cup of water",
                 n_votes: int = 1,
                 api_key: Optional[str] = None,
                 backend=None,
                 localizer: Optional[GroundedSAMLocalizer] = None):
        self.camera  = RobosuiteCamera(sim, camera_name, img_height, img_width)
        self.vlm     = VLM3DSceneAnalyzer(api_key=api_key, n_votes=n_votes,
                                           backend=backend)
        self.constructor = CBFConstructor3D()
        self.held_object = held_object
        self.localizer   = localizer
        self.safety_ctx  = None
        self.cbf_data    = None
        self.safety_filter = None

    def update_scene(self, rgb: np.ndarray, depth_ndc: np.ndarray) -> None:
        depth_metric = self.camera.depth_ndc_to_metric(depth_ndc)
        K            = self.camera.get_intrinsics()
        T            = self.camera.get_camera_pose_world()
        self.safety_ctx = self.vlm.analyze_scene(
            rgb, depth_metric, self.camera, K, T,
            self.held_object, localizer=self.localizer)
        self.cbf_data      = self.constructor.build_cbfs(self.safety_ctx)
        self.safety_filter = CBFSafetyFilter3D(self.cbf_data)

    def certify_action(self, x_ee: np.ndarray,
                       u_cmd: np.ndarray) -> Tuple[np.ndarray, dict]:
        if self.safety_filter is None:
            return u_cmd, {"cbf_values": [], "modified": False,
                           "u_cmd": u_cmd.copy(), "u_cert": u_cmd.copy()}
        return self.safety_filter.certify(x_ee, u_cmd)
