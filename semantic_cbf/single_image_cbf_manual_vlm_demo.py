"""
Single-Image Semantic CBF Demo (Manual VLM Mode)
=================================================

Purpose
-------
Run the same single-image CBF pipeline WITHOUT calling any online/local VLM API.
You manually query a VLM (e.g., GPT-4o) and paste its JSON output into a file.

This script will:
  1) load your image
  2) estimate depth (Depth Anything V2)
  3) use GroundedSAM + depth to get 3D object geometry
  4) use your manual JSON for semantic constraints
  5) build CBFs, simulate trajectory, and save 2D/3D visualizations

No API keys are required by this script.

Quick Start
-----------
1) Create a template JSON file:
   python single_image_cbf_manual_vlm_demo.py --write-template manual_vlm_output.json

2) Ask your VLM with the prompt below and fill that JSON file.

3) Run:
   python single_image_cbf_manual_vlm_demo.py \
       --image single_laptop.jpeg \
       --manual-vlm-json manual_vlm_output.json


Prompt to ask GPT (copy-paste)
------------------------------
You are a robot workspace safety analyzer. Given the input image, output ONLY valid JSON
matching the exact schema below (no markdown, no extra text).

{
  "manipulated_object": "cup of water",
  "objects": [
    {
      "name": "object_name",
      "semantic_label": "category",
      "properties": {
        "fragile": true,
        "water_sensitive": false,
        "flammable": false,
        "electronic": false
      }
    }
  ],
  "spatial_constraints": [
    {
      "source_object": "cup of water",
      "target_object": "object_name",
      "relationship": "above",
      "safety_margin": 0.10,
      "reason": "one short sentence"
    }
  ],
  "behavioral_constraints": [
    {
      "target_object": "object_name",
      "caution_level": 0.7,
      "max_velocity": 0.08,
      "reason": "one short sentence"
    }
  ],
  "pose_constraint": "constrained_rotation"
}

Rules:
- Use only relationship in {"above","below","around","near"}.
- Keep safety_margin in meters (typically 0.05~0.25).
- Keep caution_level in [0,1], max_velocity in [0.05,0.3].
- "objects" should include only physical scene objects (not robot arm, not table).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from rgbd_cbf_pipeline import (
    ObjectInfo3D,
    SemanticConstraint3D,
    SafetyContext3D,
    GroundedSAMLocalizer,
    CBFConstructor3D,
)
from single_image_cbf_demo import (
    MonocularDepthEstimator,
    SingleImageCamera,
    compute_safety_map,
    visualize_2d,
    visualize_3d,
    visualize_3d_interactive,
    simulate_trajectory,
)


ALLOWED_REL = {"above", "below", "around", "near"}


def write_template(path: str) -> None:
    template = {
        "manipulated_object": "cup of water",
        "objects": [
            {
                "name": "laptop",
                "semantic_label": "electronics",
                "properties": {
                    "fragile": True,
                    "water_sensitive": True,
                    "flammable": False,
                    "electronic": True,
                },
            }
        ],
        "spatial_constraints": [
            {
                "source_object": "cup of water",
                "target_object": "laptop",
                "relationship": "above",
                "safety_margin": 0.10,
                "reason": "Spillage can damage electronics",
            },
            {
                "source_object": "cup of water",
                "target_object": "laptop",
                "relationship": "around",
                "safety_margin": 0.08,
                "reason": "Splash radius around electronics",
            },
        ],
        "behavioral_constraints": [
            {
                "target_object": "laptop",
                "caution_level": 0.85,
                "max_velocity": 0.05,
                "reason": "Move slowly near electronics with liquid",
            }
        ],
        "pose_constraint": "constrained_rotation",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    print(f"[Template] Wrote manual VLM template to: {path}")


def _load_manual_vlm_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "objects" not in data or not isinstance(data["objects"], list):
        raise ValueError("manual VLM JSON must contain a list field 'objects'")
    return data


def _build_safety_context_from_manual(
    manual: Dict,
    rgb: np.ndarray,
    depth_metric: np.ndarray,
    camera: SingleImageCamera,
    K: np.ndarray,
    T_cam_to_world: np.ndarray,
    held_object: str,
) -> SafetyContext3D:
    object_specs: List[Dict] = manual.get("objects", [])
    object_names = [o.get("name", "").strip() for o in object_specs if o.get("name")]
    if not object_names:
        raise ValueError("manual JSON has no valid object names in 'objects'")

    localizer = GroundedSAMLocalizer()
    masks = localizer.localize(rgb, object_names)

    objects_3d: List[ObjectInfo3D] = []
    H, W = rgb.shape[:2]
    for spec in object_specs:
        name = spec.get("name", "").strip()
        if not name:
            continue
        mask = masks.get(name)
        if mask is not None and mask.any():
            pos, dims = camera.unproject_mask_to_3d(mask, depth_metric, K, T_cam_to_world)
            method = "mask"
        else:
            bbox = np.array([W * 0.1, H * 0.1, W * 0.9, H * 0.9], dtype=float)
            pos, dims = camera.unproject_bbox_to_3d(bbox, depth_metric, K, T_cam_to_world)
            method = "bbox-fallback"
        print(f"           [{method}] {name:20s}  "
              f"pos={np.array2string(pos, precision=3)}  "
              f"dims={np.array2string(dims, precision=3)}")
        objects_3d.append(
            ObjectInfo3D(
                name=name,
                position=pos,
                dimensions=dims,
                semantic_label=spec.get("semantic_label", "object"),
                properties=spec.get("properties", {}),
                mask=mask,
            )
        )

    spatial_constraints: List[SemanticConstraint3D] = []
    for sc in manual.get("spatial_constraints", []):
        rel = str(sc.get("relationship", "around")).strip().lower()
        if rel not in ALLOWED_REL:
            print(f"[Manual JSON] Skip invalid relationship '{rel}'")
            continue
        spatial_constraints.append(
            SemanticConstraint3D(
                "spatial",
                sc.get("source_object", held_object),
                sc["target_object"],
                rel,
                {
                    "safety_margin": float(sc.get("safety_margin", 0.10)),
                    "reason": sc.get("reason", ""),
                    "unsafe_pct": 100,
                    "source": "manual_vlm",
                },
            )
        )

    behavioral_constraints: List[SemanticConstraint3D] = []
    for bc in manual.get("behavioral_constraints", []):
        behavioral_constraints.append(
            SemanticConstraint3D(
                "behavioral",
                held_object,
                bc["target_object"],
                "near",
                {
                    "caution_level": float(bc.get("caution_level", 0.5)),
                    "max_approach_velocity": float(bc.get("max_velocity", 0.1)),
                    "reason": bc.get("reason", ""),
                    "source": "manual_vlm",
                },
            )
        )

    pose = str(manual.get("pose_constraint", "free_rotation"))
    if pose not in {"constrained_rotation", "free_rotation"}:
        pose = "free_rotation"

    return SafetyContext3D(
        objects=objects_3d,
        spatial_constraints=spatial_constraints,
        behavioral_constraints=behavioral_constraints,
        pose_constraint=pose,
        manipulated_object=held_object,
        reasoning="Manual VLM JSON + GroundedSAM geometry",
    )


def run_demo(
    image_path: str,
    manual_vlm_json: str,
    held_object: Optional[str] = None,
    hfov_deg: float = 65.0,
    out_dir: str = ".",
    force_cpu: bool = False,
    use_mock_depth: bool = False,
    interactive_3d: bool = False,
) -> None:
    from PIL import Image as PILImage

    print("=" * 72)
    print("Single-Image Semantic CBF Demo (Manual VLM Mode)")
    print("=" * 72)

    manual = _load_manual_vlm_json(manual_vlm_json)
    held = held_object or manual.get("manipulated_object", "cup of water")

    pil_img = PILImage.open(image_path).convert("RGB")
    rgb = np.array(pil_img, dtype=np.uint8)
    H, W = rgb.shape[:2]
    print(f"[Input]  image: {image_path}  ({W}x{H})  |  held: '{held}'")
    print(f"[Input]  manual VLM JSON: {manual_vlm_json}")

    device = "cpu" if force_cpu else "cuda"
    depth_est = MonocularDepthEstimator(device=device, force_mock=use_mock_depth)
    print("\n[Step 1] Estimating depth (Depth Anything V2 Metric Indoor)...")
    depth_metric = depth_est.estimate(rgb)
    print(f"         depth range: {depth_metric.min():.2f} – {depth_metric.max():.2f} m  "
          f"(median {np.median(depth_metric):.2f} m)")

    camera = SingleImageCamera(H, W, hfov_deg=hfov_deg)
    K, T = camera.K, camera.T
    print(f"         hfov={hfov_deg}°  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  "
          f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    print("\n[Step 2] Manual VLM semantics + GroundedSAM 3D localisation...")
    safety_ctx = _build_safety_context_from_manual(
        manual, rgb, depth_metric, camera, K, T, held
    )

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
    viol = sum(1 for i in info_history if any(h < -0.01 for _, h in i["cbf_values"]))
    print(f"         {len(info_history)} steps  |  "
          f"filter modified: {n_mod} ({100*n_mod/max(len(info_history),1):.1f}%)  |  "
          f"violations: {viol}")

    print("\n[Step 5] Computing pixel-level safety map...")
    safety_map = compute_safety_map(depth_metric, K, T, cbf_data, downsample=6)
    n_unsafe = int((safety_map < 0).sum())
    print(f"         {n_unsafe} downsampled pixels inside unsafe zone "
          f"({100*n_unsafe/safety_map.size:.1f}%)")

    stem = Path(image_path).stem
    out_2d = os.path.join(out_dir, f"{stem}_manual_cbf_2d.png")
    out_3d = os.path.join(out_dir, f"{stem}_manual_cbf_3d.png")
    out_3d_html = os.path.join(out_dir, f"{stem}_manual_cbf_3d_interactive.html")

    print(f"\n[Step 6] Generating 2D visualization → {out_2d}")
    visualize_2d(rgb, depth_metric, safety_ctx, cbf_data, camera, safety_map, save_path=out_2d)

    print(f"[Step 7] Generating 3D visualization → {out_3d}")
    visualize_3d(rgb, depth_metric, safety_ctx, cbf_data, camera, trajectory, info_history, save_path=out_3d)

    if interactive_3d:
        print(f"[Step 8] Generating interactive 3D visualization → {out_3d_html}")
        visualize_3d_interactive(
            rgb, depth_metric, safety_ctx, cbf_data, camera, trajectory, save_path=out_3d_html
        )

    print("\n" + "=" * 72)
    print(f"  2D result : {out_2d}")
    print(f"  3D result : {out_3d}")
    if interactive_3d:
        print(f"  3D interactive : {out_3d_html}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-Image Semantic CBF Demo (Manual VLM Mode)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--image", default="single_laptop.jpeg", help="Path to input RGB image")
    parser.add_argument(
        "--manual-vlm-json",
        default="manual_vlm_output.json",
        help="Path to manual VLM JSON output file",
    )
    parser.add_argument(
        "--held-object",
        default=None,
        help="Optional override for manipulated object (defaults to JSON field)",
    )
    parser.add_argument("--hfov", type=float, default=65.0, help="Camera horizontal FOV in degrees")
    parser.add_argument("--out-dir", default=".", help="Directory for output PNG files")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument(
        "--interactive-3d",
        action="store_true",
        help="Also export an interactive 3D HTML (drag/zoom/rotate)",
    )
    parser.add_argument(
        "--mock-depth",
        action="store_true",
        help="Use synthetic mock depth instead of Depth Anything model",
    )
    parser.add_argument(
        "--write-template",
        default=None,
        help="Write a manual VLM JSON template to this path, then exit",
    )
    args = parser.parse_args()

    if args.write_template:
        write_template(args.write_template)
    else:
        image_path = args.image
        here = Path(__file__).parent
        if not os.path.isabs(image_path):
            image_path = str(here / image_path)
        run_demo(
            image_path=image_path,
            manual_vlm_json=args.manual_vlm_json,
            held_object=args.held_object,
            hfov_deg=args.hfov,
            out_dir=args.out_dir,
            force_cpu=args.cpu,
            use_mock_depth=args.mock_depth,
            interactive_3d=args.interactive_3d,
        )
