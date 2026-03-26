"""
Single-Image CBF Demo
=====================
Runs the full semantic CBF pipeline on a single RGB input image.

Depth is estimated by Depth Anything V2 (metric indoor model).
VLM backend is selectable: Gemini (free API), local LLaVA/Qwen2-VL (GPU),
or Anthropic Claude. Default is Gemini when GEMINI_API_KEY/GOOGLE_API_KEY
is available.

Usage:
    # Gemini (default if GEMINI_API_KEY or GOOGLE_API_KEY is set):
    python single_image_cbf_demo.py --image single_laptop.jpeg

    # Gemini with explicit key:
    python single_image_cbf_demo.py --image single_laptop.jpeg \
        --gemini-key AIza...

    # local LLaVA (explicit opt-in):
    python single_image_cbf_demo.py --image single_laptop.jpeg

    # local Qwen2-VL:
    python single_image_cbf_demo.py --image single_laptop.jpeg \\
        --local-vlm-model Qwen/Qwen2-VL-7B-Instruct

    # Anthropic Claude:
    python single_image_cbf_demo.py --image single_laptop.jpeg \\
        --api-key sk-ant-...

    # fully offline (mock everything):
    python single_image_cbf_demo.py --image single_laptop.jpeg --mock

Output (saved to --out-dir):
    <stem>_cbf_2d.png   four-panel 2D figure
    <stem>_cbf_3d.png   3D point-cloud + CBF surfaces + trajectory

Pipeline:
    RGB  ──► Depth Anything V2  ──────────────► metric depth (m)
         ──► VLM                ──────────────► object names + properties
         ──► Grounding DINO + SAM ────────────► segmentation masks
                + depth + K + T  ────────────► 3D positions (camera frame)
    ──► multi-prompt VLM ────────────────────► safety constraints
                                                     ↓
                                           3D superquadric CBFs
                                                     ↓
                                           CBF-QP safety filter
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from rgbd_cbf_pipeline import (
    ObjectInfo3D,
    SemanticConstraint3D,
    SafetyContext3D,
    GroundedSAMLocalizer,
    VLM3DSceneAnalyzer,
    HuggingFaceVLMBackend,
    GeminiFreeBackend,
    CBFConstructor3D,
    CBFSafetyFilter3D,
)


# ============================================================================
# MONOCULAR DEPTH ESTIMATOR  (Depth Anything V2 metric indoor)
# ============================================================================

class MonocularDepthEstimator:
    """
    Wraps Depth Anything V2 (metric indoor) for single-image depth estimation.
    Falls back to a synthetic depth map when the model is unavailable.

    Model: depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
    Output: depth map (H, W) float32 in metres.
    Install: pip install transformers accelerate
    """

    MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

    def __init__(self, device: str = "cuda", force_mock: bool = False):
        self.device  = device
        self._model  = None
        self._proc   = None
        self._loaded = False
        if not force_mock:
            self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            self._proc  = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self._model = AutoModelForDepthEstimation.from_pretrained(self.MODEL_ID)
            self._model = self._model.to(self.device)
            self._model.eval()
            self._loaded = True
            print(f"[DepthEstimator] Depth Anything V2 loaded on {self.device}")
        except Exception as e:
            print(f"[DepthEstimator] Could not load model ({e}), using mock depth")

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        if self._loaded:
            return self._run_model(rgb)
        return self._mock_depth(rgb)

    def _run_model(self, rgb: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        from PIL import Image
        H, W   = rgb.shape[:2]
        pil    = Image.fromarray(rgb)
        with torch.no_grad():
            inputs  = self._proc(images=pil, return_tensors="pt")
            inputs  = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            pred    = outputs.predicted_depth
        pred = F.interpolate(pred.unsqueeze(1).float(), size=(H, W),
                             mode="bilinear", align_corners=False)
        return pred.squeeze().cpu().numpy().astype(np.float32)

    def _mock_depth(self, rgb: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        H, W = rgb.shape[:2]
        print("[DepthEstimator] Using synthetic mock depth (no model)")
        vert  = np.linspace(2.2, 0.85, H)[:, None] * np.ones((1, W))
        gray  = gaussian_filter(rgb.mean(axis=2).astype(np.float32) / 255.0, sigma=12)
        depth = vert - 0.35 * gray
        cy, cx = int(H * 0.55), int(W * 0.45)
        Y, X   = np.ogrid[:H, :W]
        depth -= 0.25 * np.exp(-((X-cx)**2/(W*0.12)**2 + (Y-cy)**2/(H*0.15)**2))
        return np.clip(depth, 0.3, 3.0).astype(np.float32)


# ============================================================================
# SINGLE IMAGE CAMERA  (no sim required)
# ============================================================================

class SingleImageCamera:
    """
    Camera model for a static single RGB image.

    Same unproject interface as RobosuiteCamera, but intrinsics are
    assumed from image dimensions + a horizontal FOV estimate.
    T_cam_to_world defaults to identity (camera frame = world frame).
    """

    def __init__(self, img_height: int, img_width: int,
                 hfov_deg: float = 65.0,
                 T_cam_to_world: Optional[np.ndarray] = None):
        self.img_height = img_height
        self.img_width  = img_width
        self.hfov_deg   = hfov_deg
        self.T          = T_cam_to_world if T_cam_to_world is not None else np.eye(4)
        self._K         = self._build_K()

    def _build_K(self) -> np.ndarray:
        hfov = np.deg2rad(self.hfov_deg)
        fx   = (self.img_width / 2.0) / np.tan(hfov / 2.0)
        return np.array([[fx, 0,  self.img_width  / 2.0],
                         [0,  fx, self.img_height / 2.0],
                         [0,  0,  1.0]], dtype=np.float64)

    @property
    def K(self) -> np.ndarray:
        return self._K

    def unproject_mask_to_3d(self, mask: np.ndarray, depth_metric: np.ndarray,
                              K: np.ndarray, T_cam_to_world: np.ndarray,
                              n_samples: int = 300) -> Tuple[np.ndarray, np.ndarray]:
        vs, us = np.where(mask)
        if len(vs) == 0:
            return np.array([0.0, 0.0, 1.0]), np.array([0.1, 0.1, 0.05])
        depths = depth_metric[vs, us]
        valid  = (depths > 0.05) & (depths < 6.0)
        vs, us, depths = vs[valid], us[valid], depths[valid]
        if len(vs) == 0:
            cy_px, cx_px = int(np.mean(np.where(mask)[0])), int(np.mean(np.where(mask)[1]))
            d = float(np.median(depth_metric[mask]))
            d = d if 0.05 < d < 6.0 else 0.8
            pts_cam = np.array([[(cx_px-K[0,2])*d/K[0,0], (cy_px-K[1,2])*d/K[1,1], d]])
        else:
            if len(vs) > n_samples:
                idx = np.random.choice(len(vs), n_samples, replace=False)
                vs, us, depths = vs[idx], us[idx], depths[idx]
            xs = (us - K[0,2]) * depths / K[0,0]
            ys = (vs - K[1,2]) * depths / K[1,1]
            pts_cam = np.column_stack([xs, ys, depths])
        pts_h     = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
        pts_world = (T_cam_to_world @ pts_h.T).T[:, :3]
        position  = np.median(pts_world, axis=0)
        dims = (np.maximum(pts_world.max(axis=0) - pts_world.min(axis=0), 0.02)
                if len(pts_world) > 2 else np.array([0.1, 0.1, 0.05]))
        return position, dims

    def unproject_bbox_to_3d(self, bbox: np.ndarray, depth_metric: np.ndarray,
                              K: np.ndarray, T_cam_to_world: np.ndarray,
                              n_samples: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        u_min, v_min, u_max, v_max = bbox.astype(int)
        u_min = max(0, u_min); v_min = max(0, v_min)
        u_max = min(depth_metric.shape[1]-1, u_max)
        v_max = min(depth_metric.shape[0]-1, v_max)
        roi   = depth_metric[v_min:v_max+1, u_min:u_max+1]
        valid = (roi > 0.05) & (roi < 6.0)
        if valid.sum() < 3:
            uc, vc = (u_min+u_max)/2.0, (v_min+v_max)/2.0
            d = float(depth_metric[int(vc), int(uc)])
            d = d if 0.05 < d < 6.0 else 0.8
            pts_cam = np.array([[(uc-K[0,2])*d/K[0,0], (vc-K[1,2])*d/K[1,1], d]])
        else:
            d_thresh = np.percentile(roi[valid], 30) * 1.2
            fg       = (roi <= d_thresh) & valid
            vl, ul   = np.where(fg)
            if len(vl) > n_samples:
                idx = np.random.choice(len(vl), n_samples, replace=False)
                vl, ul = vl[idx], ul[idx]
            xs = (ul+u_min-K[0,2]) * roi[vl,ul] / K[0,0]
            ys = (vl+v_min-K[1,2]) * roi[vl,ul] / K[1,1]
            pts_cam = np.column_stack([xs, ys, roi[vl,ul]])
        pts_h     = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
        pts_world = (T_cam_to_world @ pts_h.T).T[:, :3]
        position  = np.median(pts_world, axis=0)
        dims = (np.maximum(pts_world.max(axis=0) - pts_world.min(axis=0), 0.02)
                if len(pts_world) > 2 else np.array([0.1, 0.1, 0.05]))
        return position, dims

    def project_to_image(self, pts_world: np.ndarray,
                         T_cam_to_world: np.ndarray) -> np.ndarray:
        T_world_to_cam = np.linalg.inv(T_cam_to_world)
        pts_h   = np.hstack([pts_world, np.ones((len(pts_world), 1))])
        pts_cam = (T_world_to_cam @ pts_h.T).T[:, :3]
        in_front = pts_cam[:, 2] > 0.01
        uvw = (self._K @ pts_cam.T).T
        pixels = np.zeros((len(pts_world), 2))
        pixels[in_front, 0] = uvw[in_front, 0] / uvw[in_front, 2]
        pixels[in_front, 1] = uvw[in_front, 1] / uvw[in_front, 2]
        pixels[~in_front] = -1
        return pixels


# ============================================================================
# MOCK SCENE BUILDER  (offline fallback for the laptop image)
# ============================================================================

def build_mock_scene(rgb: np.ndarray, depth_metric: np.ndarray,
                     camera: SingleImageCamera,
                     held_object: str = "cup of water") -> SafetyContext3D:
    H, W = rgb.shape[:2]
    K, T = camera.K, camera.T

    laptop_bbox = np.array([W*0.12, H*0.25, W*0.82, H*0.82], dtype=float)
    desk_bbox   = np.array([W*0.00, H*0.60, W*1.00, H*1.00], dtype=float)

    cy_l, cx_l = int(H*0.53), int(W*0.47)
    Y, X = np.ogrid[:H, :W]
    laptop_mask = ((X-cx_l)/int(W*0.30))**2 + ((Y-cy_l)/int(H*0.26))**2 <= 1.0

    pos_laptop, dims_laptop = camera.unproject_mask_to_3d(laptop_mask, depth_metric, K, T)
    pos_desk,   dims_desk   = camera.unproject_bbox_to_3d(desk_bbox,   depth_metric, K, T)

    objects = [
        ObjectInfo3D("laptop", pos_laptop, dims_laptop, "electronics",
                     {"fragile": True, "water_sensitive": True,
                      "flammable": False, "electronic": True},
                     pixel_bbox=laptop_bbox, mask=laptop_mask),
        ObjectInfo3D("desk", pos_desk, dims_desk, "furniture",
                     {"fragile": False, "water_sensitive": False,
                      "flammable": False, "electronic": False},
                     pixel_bbox=desk_bbox, mask=None),
    ]

    held_l   = held_object.lower()
    is_liquid = any(w in held_l for w in ["water","cup","coffee","liquid","soup"])
    is_fire   = any(w in held_l for w in ["candle","flame","lighter"])

    spatial, behavioral = [], []
    if is_liquid:
        spatial += [
            SemanticConstraint3D("spatial", held_object, "laptop", "above",
                                 {"safety_margin": 0.10,
                                  "reason": "Water spillage would damage the laptop"}),
            SemanticConstraint3D("spatial", held_object, "laptop", "around",
                                 {"safety_margin": 0.08,
                                  "reason": "Splash radius around the laptop"}),
        ]
        behavioral.append(
            SemanticConstraint3D("behavioral", held_object, "laptop", "near",
                                 {"caution_level": 0.85,
                                  "max_approach_velocity": 0.05,
                                  "reason": "Slow down near electronics when carrying liquid"}))
        pose_constraint = "constrained_rotation"
    elif is_fire:
        spatial.append(
            SemanticConstraint3D("spatial", held_object, "laptop", "near",
                                 {"safety_margin": 0.20,
                                  "reason": "Open flame near electronics is a fire hazard"}))
        pose_constraint = "constrained_rotation"
    else:
        pose_constraint = "free_rotation"

    return SafetyContext3D(objects=objects, spatial_constraints=spatial,
                            behavioral_constraints=behavioral,
                            pose_constraint=pose_constraint,
                            manipulated_object=held_object,
                            reasoning="Mock scene: laptop on desk")


# ============================================================================
# PIXEL-LEVEL SAFETY MAP
# ============================================================================

def compute_safety_map(depth_metric: np.ndarray, K: np.ndarray,
                       T_cam_to_world: np.ndarray,
                       cbf_data: dict, downsample: int = 6) -> np.ndarray:
    H, W = depth_metric.shape
    H_ds, W_ds = H // downsample, W // downsample
    us = np.arange(W_ds) * downsample + downsample // 2
    vs = np.arange(H_ds) * downsample + downsample // 2
    UU, VV = np.meshgrid(us, vs)
    depths_ds = depth_metric[VV, UU]
    valid = (depths_ds > 0.05) & (depths_ds < 6.0)
    xs = (UU - K[0,2]) * depths_ds / K[0,0]
    ys = (VV - K[1,2]) * depths_ds / K[1,1]
    pts_cam  = np.stack([xs, ys, depths_ds], axis=-1)
    pts_h    = np.concatenate([pts_cam, np.ones((H_ds, W_ds, 1))], axis=-1)
    pts_world = (T_cam_to_world @ pts_h.reshape(-1, 4).T).T[:, :3].reshape(H_ds, W_ds, 3)

    safety_map = np.full((H_ds, W_ds), 999.0, dtype=np.float32)
    for h_func, _, _, _ in cbf_data["spatial_cbfs"]:
        h_vals = np.array([
            [h_func(pts_world[r, c]) if valid[r, c] else 999.0
             for c in range(W_ds)]
            for r in range(H_ds)
        ], dtype=np.float32)
        safety_map = np.minimum(safety_map, h_vals)
    return safety_map


# ============================================================================
# 2D VISUALIZATION
# ============================================================================

def visualize_2d(rgb, depth_metric, safety_ctx, cbf_data,
                 camera, safety_map, save_path="/tmp/cbf_2d.png"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap

    H, W = rgb.shape[:2]
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    # Panel 1: RGB + bboxes
    ax = axes[0]
    ax.imshow(rgb); ax.axis("off")
    ax.set_title("RGB + Detected Objects", fontsize=12, fontweight="bold")
    colors_obj = plt.cm.Set2(np.linspace(0, 1, max(len(safety_ctx.objects), 1)))
    for i, obj in enumerate(safety_ctx.objects):
        if obj.pixel_bbox is not None:
            u0, v0, u1, v1 = obj.pixel_bbox
            ax.add_patch(mpatches.Rectangle(
                (u0, v0), u1-u0, v1-v0, linewidth=2.5,
                edgecolor=colors_obj[i], facecolor="none"))
            ax.text(u0+4, v0-6, obj.name, color=colors_obj[i],
                    fontsize=10, fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.5, pad=2, boxstyle="round"))

    # Panel 2: Depth map
    ax = axes[1]
    dm = ax.imshow(depth_metric, cmap="jet_r", vmin=0.3, vmax=2.5)
    ax.set_title("Depth Anything V2\n(metric, metres)", fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.colorbar(dm, ax=ax, fraction=0.046, pad=0.04, label="depth (m)")

    # Panel 3: Segmentation masks
    ax = axes[2]
    overlay = rgb.copy().astype(np.float32)
    for i, obj in enumerate(safety_ctx.objects):
        if obj.mask is not None and obj.mask.any():
            color = np.array(plt.cm.Set1(i % 9)[:3]) * 255
            overlay[obj.mask] = overlay[obj.mask]*0.45 + color*0.55
    ax.imshow(overlay.astype(np.uint8)); ax.axis("off")
    ax.set_title("Segmentation Masks\n(Grounding DINO + SAM)",
                 fontsize=12, fontweight="bold")
    patches = [mpatches.Patch(color=plt.cm.Set1(i % 9), label=o.name)
               for i, o in enumerate(safety_ctx.objects) if o.mask is not None]
    if patches:
        ax.legend(handles=patches, loc="lower right", fontsize=9)

    # Panel 4: Safety map
    ax = axes[3]
    ds = H // safety_map.shape[0]
    cmap = LinearSegmentedColormap.from_list(
        "safety", [(1,0,0), (1,0.85,0), (0.1,0.75,0.1)], N=256)
    sm_display = np.clip(safety_map, -0.5, 2.0)
    im = ax.imshow(sm_display, cmap=cmap, vmin=-0.5, vmax=2.0,
                   extent=[0, W, H, 0], aspect="auto")
    ax.imshow(rgb, alpha=0.35, extent=[0, W, H, 0], aspect="auto")
    ax.contour(np.arange(safety_map.shape[1])*ds + ds//2,
               np.arange(safety_map.shape[0])*ds + ds//2,
               safety_map, levels=[0.0], colors="white", linewidths=2.0)
    ax.set_title("Pixel-Level CBF Safety Map\n(red=unsafe, white=h=0 boundary)",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="min h(x)")

    constraints_str = "  ".join(
        f"{c.source_object} {c.relationship} {c.target_object}"
        for c in safety_ctx.spatial_constraints)
    fig.text(0.5, -0.02,
             f"Held: '{safety_ctx.manipulated_object}'  |  "
             f"Pose: {safety_ctx.pose_constraint}  |  "
             f"Constraints: {constraints_str}",
             ha="center", fontsize=10, style="italic",
             bbox=dict(facecolor="lightyellow", alpha=0.8, boxstyle="round"))
    plt.suptitle("Single-Image Semantic CBF Pipeline  |  2D Visualization",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[VIZ 2D] Saved to {save_path}")
    plt.close(fig)


# ============================================================================
# 3D VISUALIZATION
# ============================================================================

def _superquadric_wireframe(center, semi_axes, epsilon, n=18):
    theta = np.linspace(-np.pi/2, np.pi/2, n)
    phi   = np.linspace(-np.pi,   np.pi,   n)
    TH, PH = np.meshgrid(theta, phi)
    def _sp(v, e): return np.sign(v) * np.abs(v)**e
    X = center[0] + semi_axes[0] * _sp(np.cos(TH), epsilon) * _sp(np.cos(PH), epsilon)
    Y = center[1] + semi_axes[1] * _sp(np.cos(TH), epsilon) * _sp(np.sin(PH), epsilon)
    Z = center[2] + semi_axes[2] * _sp(np.sin(TH), epsilon)
    return X, Y, Z

def _draw_box_3d(ax, center, half_dims, color="blue", alpha=0.6, lw=1.5):
    c, d = center, half_dims
    edges = [
        ([c[0]-d[0],c[1]-d[1],c[2]-d[2]], [c[0]+d[0],c[1]-d[1],c[2]-d[2]]),
        ([c[0]-d[0],c[1]+d[1],c[2]-d[2]], [c[0]+d[0],c[1]+d[1],c[2]-d[2]]),
        ([c[0]-d[0],c[1]-d[1],c[2]+d[2]], [c[0]+d[0],c[1]-d[1],c[2]+d[2]]),
        ([c[0]-d[0],c[1]+d[1],c[2]+d[2]], [c[0]+d[0],c[1]+d[1],c[2]+d[2]]),
        ([c[0]-d[0],c[1]-d[1],c[2]-d[2]], [c[0]-d[0],c[1]+d[1],c[2]-d[2]]),
        ([c[0]+d[0],c[1]-d[1],c[2]-d[2]], [c[0]+d[0],c[1]+d[1],c[2]-d[2]]),
        ([c[0]-d[0],c[1]-d[1],c[2]+d[2]], [c[0]-d[0],c[1]+d[1],c[2]+d[2]]),
        ([c[0]+d[0],c[1]-d[1],c[2]+d[2]], [c[0]+d[0],c[1]+d[1],c[2]+d[2]]),
        ([c[0]-d[0],c[1]-d[1],c[2]-d[2]], [c[0]-d[0],c[1]-d[1],c[2]+d[2]]),
        ([c[0]+d[0],c[1]-d[1],c[2]-d[2]], [c[0]+d[0],c[1]-d[1],c[2]+d[2]]),
        ([c[0]-d[0],c[1]+d[1],c[2]-d[2]], [c[0]-d[0],c[1]+d[1],c[2]+d[2]]),
        ([c[0]+d[0],c[1]+d[1],c[2]-d[2]], [c[0]+d[0],c[1]+d[1],c[2]+d[2]]),
    ]
    for s, e in edges:
        ax.plot3D(*zip(s, e), color=color, alpha=alpha, linewidth=lw)


def visualize_3d(rgb, depth_metric, safety_ctx, cbf_data,
                 camera, trajectory, info_history, save_path="/tmp/cbf_3d.png"):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    H, W = rgb.shape[:2]
    K, T = camera.K, camera.T

    # Denser point cloud for clearer visualization
    ds = 4
    rs = np.arange(0, H, ds); cs_ = np.arange(0, W, ds)
    RR, CC = np.meshgrid(rs, cs_, indexing="ij")
    d_sub  = depth_metric[RR, CC]
    valid  = (d_sub > 0.05) & (d_sub < 4.0)
    xs = (CC[valid] - K[0,2]) * d_sub[valid] / K[0,0]
    ys = (RR[valid] - K[1,2]) * d_sub[valid] / K[1,1]
    pts_cam  = np.column_stack([xs, ys, d_sub[valid]])
    pts_h    = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
    pts_w    = (T @ pts_h.T).T[:, :3]
    rgb_sub  = rgb[RR[valid], CC[valid]] / 255.0
    if len(pts_w) > 0:
        z_vis_min = float(np.percentile(pts_w[:, 2], 5))
        z_vis_max = float(np.percentile(pts_w[:, 2], 95))
    else:
        z_vis_min, z_vis_max = 0.5, 1.5

    fig = plt.figure(figsize=(14, 10))

    # ---- Single 3D scene panel ----
    ax1 = fig.add_subplot(111, projection="3d")
    ax1.set_title("3D Scene: Point Cloud + Objects + CBF Zones + EE Trajectory",
                  fontsize=13, fontweight="bold")
    ax1.scatter(pts_w[:,0], pts_w[:,1], pts_w[:,2],
                c=rgb_sub, s=1.4, alpha=0.55, depthshade=True)

    for obj in safety_ctx.objects:
        _draw_box_3d(ax1, obj.position, obj.dimensions/2, color="royalblue", lw=1.5)
        ax1.text(*obj.position + np.array([0, 0, obj.dimensions[2]/2 + 0.03]),
                 obj.name, ha="center", fontsize=8,
                 color="royalblue", fontweight="bold")

    for h_func, _, name, params in cbf_data["spatial_cbfs"]:
        try:
            # Standard superquadric CBFs
            if "semi_axes" in params and "epsilon" in params:
                Xq, Yq, Zq = _superquadric_wireframe(
                    params["center"], params["semi_axes"], params["epsilon"], n=14)
                ax1.plot_wireframe(Xq, Yq, Zq, color="red",
                                   alpha=0.12, linewidth=0.5, rstride=2, cstride=2)
                for i_lat in [2, 7, 11]:
                    ax1.plot(Xq[i_lat], Yq[i_lat], Zq[i_lat], "r--", alpha=0.45, linewidth=1.0)
            # Custom local "above" CBF boundary: y = y_limit, x in [cx-half_x, cx+half_x]
            elif params.get("relationship") == "above" and "half_x" in params and "y_limit" in params:
                cx, cy, cz = params["center"]
                half_x = float(params["half_x"])
                half_z = float(params.get("half_z", 0.0))
                y_lim = float(params["y_limit"])
                z0 = (cz - half_z) if half_z > 0 else z_vis_min
                z1 = (cz + half_z) if half_z > 0 else z_vis_max
                Xp = np.array([[cx - half_x, cx + half_x],
                               [cx - half_x, cx + half_x]])
                Yp = np.full((2, 2), y_lim)
                Zp = np.array([[z0, z0],
                               [z1, z1]])
                ax1.plot_surface(Xp, Yp, Zp, color="orangered", alpha=0.18, linewidth=0)
                # boundary rectangle
                ax1.plot([cx-half_x, cx+half_x], [y_lim, y_lim], [z0, z0],
                         color="orangered", linestyle="--", linewidth=1.2, alpha=0.7)
                ax1.plot([cx-half_x, cx+half_x], [y_lim, y_lim], [z1, z1],
                         color="orangered", linestyle="--", linewidth=1.2, alpha=0.7)
                ax1.plot([cx-half_x, cx-half_x], [y_lim, y_lim], [z0, z1],
                         color="orangered", linestyle="--", linewidth=1.2, alpha=0.7)
                ax1.plot([cx+half_x, cx+half_x], [y_lim, y_lim], [z0, z1],
                         color="orangered", linestyle="--", linewidth=1.2, alpha=0.7)
        except Exception:
            pass

    traj = np.array(trajectory)
    if len(traj) > 1:
        n_t = len(traj)
        for i in range(n_t - 1):
            ax1.plot3D(traj[i:i+2,0], traj[i:i+2,1], traj[i:i+2,2],
                       "-", color=plt.cm.plasma(i / max(n_t-1, 1)),
                       linewidth=2.5, alpha=0.9)
        ax1.scatter(*traj[0],  color="lime",   s=80, zorder=6)
        ax1.scatter(*traj[-1], color="yellow", s=80, marker="s", zorder=6)

    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)"); ax1.set_zlabel("z (m)")
    ax1.view_init(elev=18, azim=-58)

    # Legend via proxy artists (avoid add_artist on 3D axes — crashes matplotlib)
    import matplotlib.patches as mpatches
    proxy = [
        mpatches.Patch(color="royalblue", alpha=0.8, label="Object boxes"),
        mpatches.Patch(color="red",       alpha=0.5, label="CBF unsafe zone"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="lime",
                   markersize=8, label="EE start"),
        plt.Line2D([0],[0], marker="s", color="w", markerfacecolor="yellow",
                   markersize=8, label="EE end"),
    ]
    ax1.legend(handles=proxy, fontsize=8, loc="upper left")

    plt.suptitle(
        f"Single-Image Semantic CBF Pipeline  |  3D Visualization\n"
        f"Held: '{safety_ctx.manipulated_object}'  |  "
        f"{len(cbf_data['spatial_cbfs'])} spatial CBFs  |  "
        f"Pose: {safety_ctx.pose_constraint}",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[VIZ 3D] Saved to {save_path}")
    plt.close(fig)


def visualize_3d_interactive(
    rgb,
    depth_metric,
    safety_ctx,
    cbf_data,
    camera,
    trajectory,
    save_path="/tmp/cbf_3d_interactive.html",
):
    """Interactive 3D visualization (drag/zoom/rotate) exported as HTML."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[VIZ 3D interactive] Plotly not installed. Run: pip install plotly")
        return

    H, W = rgb.shape[:2]
    K, T = camera.K, camera.T

    ds = 4
    rs = np.arange(0, H, ds)
    cs_ = np.arange(0, W, ds)
    RR, CC = np.meshgrid(rs, cs_, indexing="ij")
    d_sub = depth_metric[RR, CC]
    valid = (d_sub > 0.05) & (d_sub < 4.0)

    xs = (CC[valid] - K[0, 2]) * d_sub[valid] / K[0, 0]
    ys = (RR[valid] - K[1, 2]) * d_sub[valid] / K[1, 1]
    pts_cam = np.column_stack([xs, ys, d_sub[valid]])
    pts_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
    pts_w = (T @ pts_h.T).T[:, :3]
    rgb_sub = rgb[RR[valid], CC[valid]]
    if len(pts_w) > 0:
        z_vis_min = float(np.percentile(pts_w[:, 2], 5))
        z_vis_max = float(np.percentile(pts_w[:, 2], 95))
    else:
        z_vis_min, z_vis_max = 0.5, 1.5

    # Keep HTML responsive.
    max_pts = 70000
    if len(pts_w) > max_pts:
        step = max(1, len(pts_w) // max_pts)
        pts_w = pts_w[::step]
        rgb_sub = rgb_sub[::step]

    colors = [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in rgb_sub]
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=pts_w[:, 0], y=pts_w[:, 1], z=pts_w[:, 2],
        mode="markers",
        marker=dict(size=1.7, color=colors, opacity=0.65),
        name="Point cloud",
    ))

    # Object boxes (12 edges each)
    for obj in safety_ctx.objects:
        c = obj.position
        d = obj.dimensions / 2.0
        corners = np.array([
            [c[0]-d[0], c[1]-d[1], c[2]-d[2]],
            [c[0]+d[0], c[1]-d[1], c[2]-d[2]],
            [c[0]+d[0], c[1]+d[1], c[2]-d[2]],
            [c[0]-d[0], c[1]+d[1], c[2]-d[2]],
            [c[0]-d[0], c[1]-d[1], c[2]+d[2]],
            [c[0]+d[0], c[1]-d[1], c[2]+d[2]],
            [c[0]+d[0], c[1]+d[1], c[2]+d[2]],
            [c[0]-d[0], c[1]+d[1], c[2]+d[2]],
        ])
        edge_ids = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i0, i1 in edge_ids:
            fig.add_trace(go.Scatter3d(
                x=[corners[i0, 0], corners[i1, 0]],
                y=[corners[i0, 1], corners[i1, 1]],
                z=[corners[i0, 2], corners[i1, 2]],
                mode="lines",
                line=dict(color="royalblue", width=4),
                showlegend=False,
                hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter3d(
            x=[c[0]], y=[c[1]], z=[c[2]],
            mode="text",
            text=[obj.name],
            textposition="top center",
            textfont=dict(color="royalblue", size=12),
            showlegend=False,
            hoverinfo="skip",
        ))

    # CBF geometry
    for _, _, name, params in cbf_data["spatial_cbfs"]:
        try:
            # Superquadric CBF wireframes
            if "semi_axes" in params and "epsilon" in params:
                Xq, Yq, Zq = _superquadric_wireframe(
                    params["center"], params["semi_axes"], params["epsilon"], n=16
                )
                for k in range(0, Xq.shape[0], 2):
                    fig.add_trace(go.Scatter3d(
                        x=Xq[k, :], y=Yq[k, :], z=Zq[k, :],
                        mode="lines",
                        line=dict(color="red", width=2),
                        opacity=0.35,
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                for k in range(0, Xq.shape[1], 2):
                    fig.add_trace(go.Scatter3d(
                        x=Xq[:, k], y=Yq[:, k], z=Zq[:, k],
                        mode="lines",
                        line=dict(color="red", width=2),
                        opacity=0.35,
                        showlegend=False,
                        hoverinfo="skip",
                    ))
            # Custom local "above" CBF boundary plane
            elif params.get("relationship") == "above" and "half_x" in params and "y_limit" in params:
                cx, cy, cz = params["center"]
                hx = float(params["half_x"])
                hz = float(params.get("half_z", 0.0))
                yl = float(params["y_limit"])
                z0 = (cz - hz) if hz > 0 else z_vis_min
                z1 = (cz + hz) if hz > 0 else z_vis_max
                xv = [cx - hx, cx + hx, cx + hx, cx - hx]
                yv = [yl, yl, yl, yl]
                zv = [z0, z0, z1, z1]
                fig.add_trace(go.Mesh3d(
                    x=xv, y=yv, z=zv,
                    i=[0, 0], j=[1, 2], k=[2, 3],
                    color="orangered", opacity=0.22, name=f"{name} boundary",
                    hoverinfo="skip",
                    showscale=False,
                ))
        except Exception:
            pass

    traj = np.array(trajectory)
    if len(traj) > 1:
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode="lines",
            line=dict(color="orange", width=6),
            name="EE trajectory",
        ))
        fig.add_trace(go.Scatter3d(
            x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]],
            mode="markers", marker=dict(color="lime", size=6), name="EE start"
        ))
        fig.add_trace(go.Scatter3d(
            x=[traj[-1, 0]], y=[traj[-1, 1]], z=[traj[-1, 2]],
            mode="markers", marker=dict(color="yellow", size=6, symbol="square"),
            name="EE end"
        ))

    fig.update_layout(
        title=(
            "Interactive 3D Scene (drag to rotate, scroll to zoom)<br>"
            f"Held: {safety_ctx.manipulated_object} | "
            f"{len(cbf_data['spatial_cbfs'])} spatial CBFs | "
            f"Pose: {safety_ctx.pose_constraint}"
        ),
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
            aspectmode="data",
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=70, b=0),
        dragmode="turntable",
    )
    fig.write_html(save_path, include_plotlyjs="cdn", full_html=True)
    print(f"[VIZ 3D interactive] Saved to {save_path}")


# ============================================================================
# SIMULATED TRAJECTORY
# ============================================================================

def simulate_trajectory(safety_ctx, cbf_data, dt=0.02):
    filt = CBFSafetyFilter3D(cbf_data, dt=dt, u_max=0.4)
    if safety_ctx.objects:
        lp    = safety_ctx.objects[0].position
        start = lp + np.array([0.35, -0.30, 0.40])
        targets = [lp + np.array([0.00, 0.00, 0.05]),
                   lp + np.array([-0.10,-0.05, 0.05]),
                   lp + np.array([0.45,  0.30, 0.30]),
                   lp + np.array([-0.40,-0.20, 0.35])]
    else:
        start   = np.array([0.30, -0.25, 0.75])
        targets = [np.array([0.0, 0.0, 0.5]), np.array([0.4, 0.3, 0.8])]

    ee, traj, hist = start.copy(), [start.copy()], []
    for target in targets:
        for _ in range(180):
            diff = target - ee
            if np.linalg.norm(diff) < 0.015:
                break
            u_cmd = diff / np.linalg.norm(diff) * 0.10
            u_cert, info = filt.certify(ee, u_cmd)
            ee = ee + u_cert * dt
            traj.append(ee.copy()); hist.append(info)
    return traj, hist


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_demo(image_path: str,
             held_object: str = "cup of water",
             api_key: Optional[str] = None,
             use_mock: bool = False,
             hfov_deg: float = 65.0,
             out_dir: str = "/tmp",
             force_cpu: bool = False,
             local_vlm: bool = False,
             local_vlm_model: Optional[str] = None,
             gemini_key: Optional[str] = None) -> None:
    from PIL import Image as PILImage

    print("=" * 72)
    print("Single-Image Semantic CBF Demo")
    print("=" * 72)

    pil_img = PILImage.open(image_path).convert("RGB")
    rgb     = np.array(pil_img, dtype=np.uint8)
    H, W    = rgb.shape[:2]
    print(f"[Input]  image: {image_path}  ({W}x{H})  |  held: '{held_object}'")

    device    = "cpu" if force_cpu else "cuda"
    depth_est = MonocularDepthEstimator(device=device, force_mock=use_mock)
    print("\n[Step 1] Estimating depth (Depth Anything V2 Metric Indoor)...")
    depth_metric = depth_est.estimate(rgb)
    print(f"         depth range: {depth_metric.min():.2f} – {depth_metric.max():.2f} m  "
          f"(median {np.median(depth_metric):.2f} m)")

    camera = SingleImageCamera(H, W, hfov_deg=hfov_deg)
    K, T   = camera.K, camera.T
    print(f"         hfov={hfov_deg}°  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  "
          f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    # Build VLM backend
    vlm_backend = None
    if not use_mock:
        if gemini_key:
            vlm_backend = GeminiFreeBackend(api_key=gemini_key)
            backend_name = "Gemini free-tier"
        elif local_vlm:
            model_id     = local_vlm_model or HuggingFaceVLMBackend.DEFAULT_MODEL
            vlm_backend  = HuggingFaceVLMBackend(model_id=model_id, device=device)
            backend_name = f"local:{model_id}"
        elif api_key:
            backend_name = "Anthropic Claude"
        else:
            raise ValueError(
                "No VLM backend configured. Provide --gemini-key (or GEMINI_API_KEY/"
                "GOOGLE_API_KEY), or pass --local-vlm, or provide --api-key, or use --mock."
            )

    if use_mock:
        print("\n[Step 2] Building mock scene (offline mode)...")
        safety_ctx = build_mock_scene(rgb, depth_metric, camera, held_object)
    else:
        print(f"\n[Step 2] VLM ({backend_name}) + GroundedSAM scene analysis...")
        localizer  = GroundedSAMLocalizer()
        vlm        = VLM3DSceneAnalyzer(api_key=api_key, n_votes=1,
                                        backend=vlm_backend)
        safety_ctx = vlm.analyze_scene(rgb, depth_metric, camera, K, T,
                                        held_object, localizer=localizer)

    print(f"\n         Objects: {[o.name for o in safety_ctx.objects]}")
    for o in safety_ctx.objects:
        print(f"           {o.name:15s}  pos={np.array2string(o.position, precision=3)}  "
              f"dims={np.array2string(o.dimensions, precision=3)}")
    print(f"         Spatial constraints:    {len(safety_ctx.spatial_constraints)}")
    for c in safety_ctx.spatial_constraints:
        print(f"           UNSAFE: {c.source_object} {c.relationship} {c.target_object}"
              f" — {c.parameters.get('reason','')}")
    print(f"         Behavioral constraints: {len(safety_ctx.behavioral_constraints)}")
    print(f"         Pose:                   {safety_ctx.pose_constraint}")

    print("\n[Step 3] Building 3D superquadric CBFs...")
    cbf_data = CBFConstructor3D().build_cbfs(safety_ctx)
    print(f"         {len(cbf_data['spatial_cbfs'])} spatial CBFs:")
    for _, _, name, params in cbf_data["spatial_cbfs"]:
        center_s = np.array2string(params.get("center", np.zeros(3)), precision=3)
        if "semi_axes" in params:
            shape_s = f"axes={np.array2string(params['semi_axes'], precision=3)}"
        elif "half_x" in params and "y_limit" in params:
            shape_s = (f"above-local half_x={params['half_x']:.3f} "
                       f"half_z={params.get('half_z', np.nan):.3f} "
                       f"y_limit={params['y_limit']:.3f}")
        else:
            shape_s = "shape=custom"
        print(f"           {name:35s}  center={center_s}  {shape_s}")

    print("\n[Step 4] Simulating EE trajectory with CBF-QP filter...")
    trajectory, info_history = simulate_trajectory(safety_ctx, cbf_data)
    n_mod = sum(1 for i in info_history if i["modified"])
    viol  = sum(1 for i in info_history
                if any(h < -0.01 for _, h in i["cbf_values"]))
    print(f"         {len(info_history)} steps  |  "
          f"filter modified: {n_mod} ({100*n_mod/max(len(info_history),1):.1f}%)  |  "
          f"violations: {viol}")

    print("\n[Step 5] Computing pixel-level safety map...")
    safety_map = compute_safety_map(depth_metric, K, T, cbf_data, downsample=6)
    n_unsafe = (safety_map < 0).sum()
    print(f"         {n_unsafe} downsampled pixels inside unsafe zone "
          f"({100*n_unsafe/safety_map.size:.1f}%)")

    stem   = Path(image_path).stem
    out_2d = os.path.join(out_dir, f"{stem}_cbf_2d.png")
    out_3d = os.path.join(out_dir, f"{stem}_cbf_3d.png")

    print(f"\n[Step 6] Generating 2D visualization → {out_2d}")
    visualize_2d(rgb, depth_metric, safety_ctx, cbf_data,
                 camera, safety_map, save_path=out_2d)

    print(f"[Step 7] Generating 3D visualization → {out_3d}")
    visualize_3d(rgb, depth_metric, safety_ctx, cbf_data,
                 camera, trajectory, info_history, save_path=out_3d)

    print("\n" + "=" * 72)
    print(f"  2D result : {out_2d}")
    print(f"  3D result : {out_3d}")
    print("=" * 72)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-Image Semantic CBF Demo",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--image", default="single_laptop.jpeg",
                        help="Path to input RGB image")
    parser.add_argument("--held-object", default="cup of water",
                        help="Object held by the robot")
    parser.add_argument("--mock", action="store_true",
                        help="Fully offline — no API / no GPU model")

    vlm_group = parser.add_mutually_exclusive_group()
    vlm_group.add_argument("--api-key", default=None,
                            help="Anthropic API key")
    vlm_group.add_argument("--gemini-key", default=None,
                            help="Google Gemini free-tier API key\n"
                                 "  defaults to GEMINI_API_KEY / GOOGLE_API_KEY\n"
                                 "  pip install google-genai")
    vlm_group.add_argument("--local-vlm", action="store_true", default=False,
                            help="Use local HuggingFace VLM (explicit opt-in)\n"
                                 "  default local model: llava-hf/llava-v1.6-mistral-7b-hf\n"
                                 "  pip install transformers accelerate")

    parser.add_argument("--local-vlm-model", default=None,
                        help="HuggingFace model ID for --local-vlm\n"
                             "  default: llava-hf/llava-v1.6-mistral-7b-hf\n"
                             "  other:   Qwen/Qwen2-VL-7B-Instruct\n"
                             "           Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--hfov", type=float, default=65.0,
                        help="Camera horizontal FOV in degrees (default: 65)")
    parser.add_argument("--out-dir", default=".",
                        help="Directory for output PNG files (default: current dir)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU inference (no GPU)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    gemini_key = (
        args.gemini_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )

    image_path = args.image
    if not os.path.isabs(image_path):
        image_path = str(_HERE / image_path)

    # Local VLM is now explicit opt-in. Default is Gemini (when key available).
    use_local = args.local_vlm

    run_demo(
        image_path      = image_path,
        held_object     = args.held_object,
        api_key         = api_key,
        use_mock        = args.mock,
        hfov_deg        = args.hfov,
        out_dir         = args.out_dir,
        force_cpu       = args.cpu,
        local_vlm       = use_local,
        local_vlm_model = args.local_vlm_model,
        gemini_key      = gemini_key,
    )
