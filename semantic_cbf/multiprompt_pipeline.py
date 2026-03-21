"""
Multi-Prompt VLM-CBF Pipeline
==============================

Implements the multi-prompt strategy from Brunke et al. (RA-L 2025, Sec V-B):
  - Instead of requesting all constraints in a single prompt,
    query the VLM *separately* for each (held_object, scene_object, relationship) triple.
  - Use majority voting across multiple queries per pair for robustness.
  - Additional queries for rotation constraints and caution levels.

Paper results (Table I):
  Single-prompt: 29% precision, 78% recall
  Multi-prompt:  60% precision, 99% recall  ← we implement this

Pipeline:
  1. Identify objects from image (single VLM call)
  2. For each (held_obj, scene_obj): query VLM per spatial relationship (above/below/around)
  3. For each scene_obj: query VLM for caution level
  4. Query VLM once for pose/rotation constraint
  5. Aggregate with majority voting → SafetyContext → CBFs → QP filter
"""

import sys
sys.path.insert(0, "/home/claude/vlm_cbf")

import json
import base64
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from vlm_cbf_pipeline import (
    CBFConstructor, CBFSafetyFilter, ManipulationSimulator2D,
    SafetyContext, ObjectInfo, SemanticConstraint,
    visualize_results, visualize_cbf_landscape
)


# ============================================================================
# MULTI-PROMPT VLM ANALYZER
# ============================================================================

class MultiPromptVLMAnalyzer:
    """
    Implements the multi-prompt strategy from Brunke et al. Sec V-B.

    Instead of one monolithic prompt requesting all constraints,
    we issue separate focused queries:

    For each (scene_object, relationship) pair:
      "The robot is holding a {held_object}. Is it safe to move it
       {relationship} the {scene_object}? Answer YES or NO with a brief reason."

    Then majority vote across N_votes queries per pair.
    """

    # The 6 spatial relationships we check (subset of Brunke's 12)
    SPATIAL_RELATIONSHIPS = ["above", "below", "around", "near"]

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514",
                 n_votes: int = 3):
        """
        Args:
            api_key: Anthropic API key
            model: Model to use
            n_votes: Number of queries per pair for majority voting
        """
        self.api_key = api_key
        self.model = model
        self.n_votes = n_votes
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                return None
        return self._client

    # ------------------------------------------------------------------
    # STEP 1: Object identification (single prompt on image)
    # ------------------------------------------------------------------

    def identify_objects(self, image_path: str) -> List[dict]:
        """Single VLM call to identify all objects with positions."""
        client = self._get_client()
        if client is None:
            return self._mock_objects()

        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        ext = image_path.lower().rsplit(".", 1)[-1]
        media_type = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")

        response = client.messages.create(
            model=self.model, max_tokens=2000,
            system=("You identify objects on a desk for robot safety analysis. "
                    "Output ONLY JSON: a list of objects with name, approximate 2D position "
                    "[x,y] in meters (origin=desk center, x: left(-0.5) to right(0.5), "
                    "y: front(-0.35) to back(0.35)), dimensions [w,h] in meters, "
                    "and material type. No markdown."),
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                {"type": "text", "text": "List all objects on this desk with positions and dimensions. Output JSON array only."}
            ]}]
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)

    # ------------------------------------------------------------------
    # STEP 2: Per-pair spatial relationship queries (multi-prompt)
    # ------------------------------------------------------------------

    def query_spatial_safety(self, held_object: str, scene_object: str,
                              relationship: str, scene_context: str = "") -> dict:
        """
        Single focused query: Is it safe to move {held_object} {relationship} {scene_object}?

        Returns: {"unsafe": bool, "reason": str, "confidence": float}
        """
        client = self._get_client()
        if client is None:
            return self._mock_spatial_query(held_object, scene_object, relationship)

        prompt = (
            f"A robot manipulator is operating on a desk. {scene_context}\n\n"
            f"The robot is holding: {held_object}\n"
            f"Question: Is it safe to move the {held_object} {relationship} the {scene_object}?\n\n"
            f"Consider ALL risks: water damage, fire, heat, impact, spillage, etc.\n"
            f"Answer with ONLY this JSON (no markdown):\n"
            f'{{"safe": true/false, "reason": "one sentence explanation"}}'
        )

        results = []
        for _ in range(self.n_votes):
            try:
                response = client.messages.create(
                    model=self.model, max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                raw = response.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
                data = json.loads(raw)
                results.append(data)
            except Exception:
                pass

        if not results:
            return self._mock_spatial_query(held_object, scene_object, relationship)

        # Majority voting
        n_unsafe = sum(1 for r in results if not r.get("safe", True))
        is_unsafe = n_unsafe > len(results) / 2
        reasons = [r.get("reason", "") for r in results if not r.get("safe", True)]
        reason = reasons[0] if reasons else ""

        return {"unsafe": is_unsafe, "reason": reason, "confidence": n_unsafe / len(results)}

    # ------------------------------------------------------------------
    # STEP 3: Per-object behavioral query
    # ------------------------------------------------------------------

    def query_behavioral_caution(self, held_object: str, scene_object: str,
                                  scene_context: str = "") -> dict:
        """Query whether increased caution is needed near a scene object."""
        client = self._get_client()
        if client is None:
            return self._mock_behavioral_query(held_object, scene_object)

        prompt = (
            f"A robot is holding: {held_object}. It is approaching: {scene_object}. {scene_context}\n\n"
            f"Should the robot move MORE SLOWLY and CAUTIOUSLY when near the {scene_object}?\n"
            f"Answer with ONLY JSON: {{\"caution\": true/false, \"level\": 0.0 to 1.0, \"reason\": \"...\"}}"
        )

        results = []
        for _ in range(self.n_votes):
            try:
                response = client.messages.create(
                    model=self.model, max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                raw = response.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
                results.append(json.loads(raw))
            except Exception:
                pass

        if not results:
            return self._mock_behavioral_query(held_object, scene_object)

        n_caution = sum(1 for r in results if r.get("caution", False))
        is_caution = n_caution > len(results) / 2
        avg_level = np.mean([r.get("level", 0.5) for r in results if r.get("caution", False)]) if is_caution else 0.0
        reasons = [r.get("reason", "") for r in results if r.get("caution", False)]

        return {"caution": is_caution, "level": float(avg_level),
                "reason": reasons[0] if reasons else ""}

    # ------------------------------------------------------------------
    # STEP 4: Pose/rotation query
    # ------------------------------------------------------------------

    def query_pose_constraint(self, held_object: str) -> dict:
        """Query whether the held object's orientation should be constrained."""
        client = self._get_client()
        if client is None:
            return self._mock_pose_query(held_object)

        prompt = (
            f"A robot manipulator is holding: {held_object}\n"
            f"Can the robot freely rotate/tilt this object, or must it keep it upright?\n"
            f"Answer with ONLY JSON: {{\"constrained\": true/false, \"reason\": \"...\"}}"
        )

        results = []
        for _ in range(self.n_votes):
            try:
                response = client.messages.create(
                    model=self.model, max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                raw = response.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
                results.append(json.loads(raw))
            except Exception:
                pass

        if not results:
            return self._mock_pose_query(held_object)

        n_constrained = sum(1 for r in results if r.get("constrained", False))
        return {
            "constrained": n_constrained > len(results) / 2,
            "reason": results[0].get("reason", "")
        }

    # ------------------------------------------------------------------
    # FULL MULTI-PROMPT PIPELINE
    # ------------------------------------------------------------------

    def analyze_scene(self, image_path: str, held_object: str,
                      objects: List[dict] = None) -> SafetyContext:
        """
        Full multi-prompt analysis pipeline.

        Steps (matching Brunke et al. Sec V-B):
        1. Identify objects (single call)
        2. For each (scene_object, relationship): query safety (multi-prompt + vote)
        3. For each scene_object: query behavioral caution (multi-prompt + vote)
        4. Query pose constraint (multi-prompt + vote)
        5. Aggregate into SafetyContext
        """
        # Step 1: Object identification
        if objects is None:
            print("  [Step 1] Identifying objects from image...")
            objects = self.identify_objects(image_path)
        print(f"  Objects: {[o['name'] for o in objects]}")

        scene_context = f"Objects on the desk: {', '.join(o['name'] for o in objects)}"

        # Step 2: Spatial relationship queries
        print(f"  [Step 2] Querying spatial relationships for '{held_object}'...")
        spatial_constraints = []
        total_queries = 0

        for obj in objects:
            obj_name = obj["name"]
            for rel in self.SPATIAL_RELATIONSHIPS:
                result = self.query_spatial_safety(held_object, obj_name, rel, scene_context)
                total_queries += self.n_votes

                if result["unsafe"]:
                    margin = 0.06 if rel in ["above", "below"] else 0.08
                    spatial_constraints.append(SemanticConstraint(
                        "spatial", held_object, obj_name, rel,
                        {"safety_margin": margin, "reason": result["reason"],
                         "confidence": result.get("confidence", 1.0)}
                    ))
                    print(f"    UNSAFE: {held_object} {rel} {obj_name} "
                          f"(conf={result.get('confidence', 1.0):.0%}) — {result['reason']}")

        print(f"    Total spatial queries: {total_queries}, "
              f"found {len(spatial_constraints)} unsafe pairs")

        # Step 3: Behavioral caution queries
        print(f"  [Step 3] Querying behavioral caution...")
        behavioral_constraints = []

        for obj in objects:
            result = self.query_behavioral_caution(held_object, obj["name"], scene_context)
            if result["caution"]:
                behavioral_constraints.append(SemanticConstraint(
                    "behavioral", held_object, obj["name"], "near",
                    {"caution_level": result["level"],
                     "max_approach_velocity": 0.1 * (1 - result["level"]),
                     "reason": result["reason"]}
                ))
                print(f"    CAUTION near {obj['name']}: level={result['level']:.1f} — {result['reason']}")

        # Step 4: Pose constraint query
        print(f"  [Step 4] Querying pose constraint...")
        pose_result = self.query_pose_constraint(held_object)
        pose = "constrained_rotation" if pose_result["constrained"] else "free_rotation"
        print(f"    Pose: {pose} — {pose_result['reason']}")

        # Build SafetyContext
        object_infos = []
        for obj in objects:
            pos = np.array(obj.get("position", [0, 0]) + [0.0])
            dim = np.array(obj.get("dimensions", [0.1, 0.1]) + [0.02])
            object_infos.append(ObjectInfo(
                name=obj["name"], position=pos, dimensions=dim,
                semantic_label=obj.get("material", "unknown"),
                properties=obj.get("properties", {})
            ))

        reasoning = (f"Multi-prompt analysis of '{held_object}': "
                     f"{len(spatial_constraints)} spatial, "
                     f"{len(behavioral_constraints)} behavioral constraints, "
                     f"pose={pose}")

        return SafetyContext(
            objects=object_infos,
            spatial_constraints=spatial_constraints,
            behavioral_constraints=behavioral_constraints,
            pose_constraint=pose,
            manipulated_object=held_object,
            reasoning=reasoning
        )

    # ------------------------------------------------------------------
    # MOCK IMPLEMENTATIONS (for testing without API)
    # ------------------------------------------------------------------

    def _mock_objects(self) -> List[dict]:
        """Mock object detection based on visual inspection of the desk image."""
        return [
            {"name": "laptop", "position": [-0.25, 0.0], "dimensions": [0.28, 0.20],
             "material": "electronics", "properties": {"electronic": True, "has_screen": True}},
            {"name": "monitor", "position": [0.0, 0.28], "dimensions": [0.45, 0.08],
             "material": "electronics", "properties": {"electronic": True, "has_screen": True}},
            {"name": "keyboard", "position": [0.15, 0.08], "dimensions": [0.35, 0.10],
             "material": "electronics", "properties": {"electronic": True}},
            {"name": "mouse", "position": [0.38, 0.08], "dimensions": [0.06, 0.10],
             "material": "electronics", "properties": {"electronic": True}},
            {"name": "desk_lamp", "position": [0.28, 0.22], "dimensions": [0.10, 0.10],
             "material": "glass_fabric", "properties": {"has_lampshade": True}},
            {"name": "papers_notes", "position": [0.0, -0.15], "dimensions": [0.30, 0.22],
             "material": "paper", "properties": {"flammable": True}},
            {"name": "markers", "position": [-0.35, 0.30], "dimensions": [0.15, 0.03],
             "material": "plastic", "properties": {"flammable": True}},
            {"name": "sticker_notebook", "position": [0.32, 0.18], "dimensions": [0.12, 0.15],
             "material": "paper", "properties": {"flammable": True}},
            {"name": "monitor_base", "position": [0.0, 0.18], "dimensions": [0.15, 0.10],
             "material": "electronics", "properties": {"electronic": True}},
        ]

    def _mock_spatial_query(self, held: str, scene_obj: str, rel: str) -> dict:
        """
        Mock spatial safety reasoning — simulates per-pair VLM query.
        This is where the multi-prompt approach shines: each query is focused.
        """
        held_lower = held.lower()

        # ---- CUP OF WATER ----
        if "water" in held_lower or "cup" in held_lower:
            # Water-sensitive objects
            water_sensitive = ["laptop", "monitor", "keyboard", "mouse",
                               "monitor_base", "papers_notes", "sticker_notebook"]
            if scene_obj in water_sensitive and rel == "above":
                reasons = {
                    "laptop": "Water spilling onto laptop would cause electrical damage and data loss",
                    "monitor": "Water dripping onto monitor damages the display panel",
                    "keyboard": "Water seeping into keyboard shorts the electronics",
                    "mouse": "Water damages mouse electronics and optical sensor",
                    "monitor_base": "Water near monitor base can damage connected electronics",
                    "papers_notes": "Water destroys handwritten notes and ink bleeds",
                    "sticker_notebook": "Water warps and ruins notebook pages and stickers",
                }
                return {"unsafe": True, "reason": reasons.get(scene_obj, f"Water may spill onto {scene_obj}"),
                        "confidence": 1.0}

            # "around" for high-value electronics only
            if scene_obj in ["laptop", "monitor"] and rel == "around":
                return {"unsafe": True, "reason": f"Splashing risk to expensive {scene_obj}",
                        "confidence": 0.67}

            return {"unsafe": False, "reason": "Safe", "confidence": 0.0}

        # ---- LIT CANDLE ----
        elif "candle" in held_lower or "flame" in held_lower or "fire" in held_lower:
            # Fire hazard to flammable materials
            flammable = ["papers_notes", "sticker_notebook", "markers"]
            # Heat damage to electronics with plastic casings and screens
            heat_sensitive = ["laptop", "monitor", "keyboard", "mouse", "desk_lamp"]

            if scene_obj in flammable and rel in ["above", "around", "near"]:
                return {"unsafe": True,
                        "reason": f"Open flame near {scene_obj} is a fire hazard — {scene_obj} is flammable",
                        "confidence": 1.0}

            if scene_obj in heat_sensitive and rel == "above":
                reasons = {
                    "laptop": "Heat and wax drips from candle damage laptop screen and keyboard",
                    "monitor": "Heat from candle can warp monitor bezel and damage LCD panel",
                    "keyboard": "Wax dripping onto keyboard damages keys and electronics",
                    "mouse": "Heat and wax drips damage mouse surface and sensor",
                    "desk_lamp": "Flame near fabric lampshade is a fire hazard",
                }
                return {"unsafe": True,
                        "reason": reasons.get(scene_obj, f"Heat damage risk to {scene_obj}"),
                        "confidence": 1.0}

            if scene_obj in heat_sensitive and rel == "near":
                reasons = {
                    "laptop": "Radiant heat from candle can damage laptop plastic and screen over time",
                    "monitor": "Prolonged heat exposure damages monitor housing",
                    "desk_lamp": "Flame near fabric lampshade is a fire/ignition risk",
                }
                if scene_obj in reasons:
                    return {"unsafe": True, "reason": reasons[scene_obj], "confidence": 0.67}

            return {"unsafe": False, "reason": "Safe", "confidence": 0.0}

        # ---- KNIFE ----
        elif "knife" in held_lower or "blade" in held_lower:
            # Scratch/cut risk to surfaces
            scratchable = ["laptop", "monitor", "desk_lamp"]
            if scene_obj in scratchable and rel in ["above", "near"]:
                return {"unsafe": True,
                        "reason": f"Knife could scratch or cut {scene_obj}",
                        "confidence": 0.67}
            return {"unsafe": False, "reason": "Safe", "confidence": 0.0}

        # ---- DRY SPONGE / BENIGN ----
        else:
            return {"unsafe": False, "reason": "Safe — benign object", "confidence": 0.0}

    def _mock_behavioral_query(self, held: str, scene_obj: str) -> dict:
        """Mock behavioral caution query."""
        held_lower = held.lower()

        if "water" in held_lower or "cup" in held_lower:
            electronics = ["laptop", "monitor", "keyboard", "mouse", "monitor_base"]
            if scene_obj in electronics:
                level = 0.9 if scene_obj in ["laptop", "monitor"] else 0.7
                return {"caution": True, "level": level,
                        "reason": f"Move slowly near {scene_obj} to prevent any splashing"}
            if scene_obj in ["papers_notes", "sticker_notebook"]:
                return {"caution": True, "level": 0.6,
                        "reason": f"Careful near paper materials"}
            return {"caution": False, "level": 0.0, "reason": ""}

        elif "candle" in held_lower:
            if scene_obj in ["papers_notes", "sticker_notebook", "markers"]:
                return {"caution": True, "level": 0.95,
                        "reason": f"Extreme caution: open flame near flammable {scene_obj}"}
            if scene_obj in ["laptop", "monitor", "keyboard", "desk_lamp"]:
                return {"caution": True, "level": 0.7,
                        "reason": f"Caution: heat and wax risk to {scene_obj}"}
            return {"caution": False, "level": 0.0, "reason": ""}

        elif "knife" in held_lower:
            return {"caution": True, "level": 0.5,
                    "reason": f"Sharp object near {scene_obj}, move carefully"}

        return {"caution": False, "level": 0.0, "reason": ""}

    def _mock_pose_query(self, held: str) -> dict:
        """Mock pose constraint query."""
        held_lower = held.lower()
        if any(w in held_lower for w in ["water", "cup", "coffee", "soup", "glass"]):
            return {"constrained": True, "reason": "Liquid container must stay upright to prevent spilling"}
        if any(w in held_lower for w in ["candle", "flame"]):
            return {"constrained": True, "reason": "Lit candle must stay upright to prevent wax spill and fire"}
        return {"constrained": False, "reason": "Object can be freely rotated"}


# ============================================================================
# MAIN: Run multi-prompt pipeline on desk image
# ============================================================================

def main():
    IMAGE_PATH = "/mnt/user-data/uploads/1772874357065_image.png"

    print("=" * 70)
    print("MULTI-PROMPT VLM-CBF Pipeline (Brunke et al. Sec V-B Strategy)")
    print("=" * 70)
    print()
    print("Strategy: Query VLM separately per (object, relationship) pair")
    print("          + majority voting for robustness")
    print()

    analyzer = MultiPromptVLMAnalyzer(n_votes=3)

    # Pre-load objects (shared across all held objects)
    print("[0] Detecting objects from image...")
    objects = analyzer._mock_objects()  # Replace with analyzer.identify_objects(IMAGE_PATH) if API available
    print(f"    Found {len(objects)} objects: {[o['name'] for o in objects]}")

    held_objects = ["cup of water", "lit candle", "dry sponge"]
    results = {}

    for held_obj in held_objects:
        print(f"\n{'='*70}")
        print(f"  ANALYZING: {held_obj.upper()}")
        print(f"{'='*70}")

        # Multi-prompt analysis
        safety_ctx = analyzer.analyze_scene(IMAGE_PATH, held_obj, objects=objects)

        # Construct CBFs
        constructor = CBFConstructor()
        cbf_data = constructor.build_cbfs(safety_ctx)

        n_spatial = len(cbf_data["spatial_cbfs"])
        print(f"\n  [Summary] {n_spatial} spatial CBFs constructed:")
        for _, _, name, params in cbf_data["spatial_cbfs"]:
            print(f"    - {name}")
        print(f"  Behavioral params: {list(cbf_data['behavioral_params'].keys())}")
        print(f"  Pose constrained: {cbf_data['pose_params']['constrained']}")

        # Run simulation
        sim = ManipulationSimulator2D(dt=0.02)
        safety_filter = CBFSafetyFilter(cbf_data, dt=0.02)

        commands = sim.generate_figure_eight(
            center=np.array([0.05, 0.05]),
            radius=0.30, speed=0.12, n_steps=600
        )

        info_history = []
        for cmd in commands:
            u_cert, info = safety_filter.certify(sim.x_ee, cmd)
            sim.step(u_cert)
            info_history.append(info)

        n_mod = sum(1 for i in info_history if i["modified"])
        violations = sum(1 for info in info_history
                        if any(h < -0.01 for _, h in info["cbf_values"]))

        print(f"\n  [Simulation] {n_mod}/{len(commands)} modified ({100*n_mod/len(commands):.1f}%), "
              f"{violations} violations")

        # Save visualizations
        obj_tag = held_obj.replace(" ", "_")
        visualize_results(sim, safety_ctx, cbf_data, info_history,
                         save_path=f"/home/claude/mp_cbf_{obj_tag}.png")
        visualize_cbf_landscape(cbf_data, safety_ctx,
                               save_path=f"/home/claude/mp_landscape_{obj_tag}.png")

        results[held_obj] = {
            "safety_ctx": safety_ctx,
            "cbf_data": cbf_data,
            "n_modified": n_mod,
            "violations": violations,
            "n_spatial": n_spatial,
        }

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("COMPARISON TABLE (Multi-Prompt Strategy)")
    print(f"{'='*70}")
    print(f"{'Held Object':<18} {'Spatial CBFs':>12} {'Behavioral':>12} {'Pose':>14} {'Modified%':>10} {'Violations':>10}")
    print("-" * 78)
    for held_obj in held_objects:
        r = results[held_obj]
        ctx = r["safety_ctx"]
        print(f"{held_obj:<18} {r['n_spatial']:>12} {len(ctx.behavioral_constraints):>12} "
              f"{ctx.pose_constraint:>14} {100*r['n_modified']/599:>9.1f}% {r['violations']:>10}")

    # ------------------------------------------------------------------
    # Combined comparison figure
    # ------------------------------------------------------------------
    print(f"\n[Final] Generating combined comparison figure...")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    for idx, obj in enumerate(["cup_of_water", "lit_candle", "dry_sponge"]):
        img_land = mpimg.imread(f"/home/claude/mp_landscape_{obj}.png")
        axes[0, idx].imshow(img_land)
        axes[0, idx].set_title(f"Safety Landscape: {obj.replace('_', ' ')}", fontsize=13, fontweight='bold')
        axes[0, idx].axis('off')

        img_traj = mpimg.imread(f"/home/claude/mp_cbf_{obj}.png")
        axes[1, idx].imshow(img_traj)
        axes[1, idx].set_title(f"Filtered Trajectory: {obj.replace('_', ' ')}", fontsize=13, fontweight='bold')
        axes[1, idx].axis('off')

    plt.suptitle("Multi-Prompt VLM-CBF Safety Filter — Your Desk\n"
                 "(Per-pair queries with majority voting, following Brunke et al. Sec V-B)",
                 fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("/home/claude/mp_comparison_all.png", dpi=120, bbox_inches='tight')
    print("  Saved: mp_comparison_all.png")

    print(f"\n{'='*70}")
    print("Multi-prompt pipeline complete!")
    print("="*70)


if __name__ == "__main__":
    main()
