"""
VLM-to-CBF Pipeline: Constructing Control Barrier Functions from Vision-Language Models
========================================================================================

This prototype demonstrates how to bypass the traditional perception pipeline
(segmentation → point clouds → shape fitting → CBF) and instead use a VLM to
go directly from images to semantic CBF constraints.

Architecture:
  Image → VLM (Claude/GPT-4V) → Structured Safety Context → CBF Construction → QP Safety Filter

Three modes of operation:
  1. VLM mode: Uses Claude API to analyze scene images
  2. Mock mode: Uses predefined scene for testing without API
  3. Interactive mode: Visualizes safety filter in real-time

Author: Akanksha Singal (CMU Robotics Institute)
References:
  - Brunke et al., "Semantically Safe Robot Manipulation", IEEE RA-L 2025
  - Nakamura et al., "How to Train Your Latent CBF", arXiv 2511.18606
  - Ames et al., "Control Barrier Functions: Theory and Applications", ECC 2019
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import json
import warnings

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ObjectInfo:
    """Represents an object detected/described by the VLM."""
    name: str
    position: np.ndarray  # [x, y, z] in world frame
    dimensions: np.ndarray  # [width, height, depth] approximate
    semantic_label: str  # e.g., "laptop", "cup_of_water"
    properties: Dict = field(default_factory=dict)  # e.g., {"fragile": True, "liquid": False}


@dataclass
class SemanticConstraint:
    """A semantic safety constraint inferred by the VLM."""
    constraint_type: str  # "spatial", "behavioral", "pose"
    source_object: str  # The manipulated object
    target_object: str  # The object creating the constraint
    relationship: str  # e.g., "above", "around", "near"
    parameters: Dict = field(default_factory=dict)  # Additional params


@dataclass
class SafetyContext:
    """Complete safety context output by the VLM."""
    objects: List[ObjectInfo]
    spatial_constraints: List[SemanticConstraint]
    behavioral_constraints: List[SemanticConstraint]
    pose_constraint: str  # "constrained_rotation" or "free_rotation"
    manipulated_object: str
    reasoning: str  # VLM's chain-of-thought reasoning


# ============================================================================
# VLM SCENE ANALYZER
# ============================================================================

class VLMSceneAnalyzer:
    """
    Uses a Vision-Language Model to analyze a scene and produce
    structured safety constraints — the key module that replaces
    the traditional segmentation → point cloud → label pipeline.
    """

    SYSTEM_PROMPT = """You are a robot safety reasoning system. Given a scene description 
or image, you must identify ALL semantic safety constraints for a robot manipulator.

You must output ONLY valid JSON (no markdown, no explanation) with this exact schema:
{
  "objects": [
    {
      "name": "object_name",
      "position": [x, y, z],
      "dimensions": [w, h, d],
      "semantic_label": "category",
      "properties": {"key": "value"}
    }
  ],
  "spatial_constraints": [
    {
      "source_object": "manipulated_object",
      "target_object": "scene_object",
      "relationship": "above|below|around|near",
      "safety_margin": 0.15,
      "reason": "brief reason"
    }
  ],
  "behavioral_constraints": [
    {
      "target_object": "scene_object",
      "caution_level": 0.0 to 1.0,
      "max_approach_velocity": 0.1,
      "reason": "brief reason"
    }
  ],
  "pose_constraint": "constrained_rotation" or "free_rotation",
  "reasoning": "brief chain-of-thought"
}

Position coordinates are in meters, relative to the table center.
Dimensions are approximate bounding box sizes in meters.
Safety margin is the minimum clearance in meters.
Caution level 0.0 = no caution, 1.0 = maximum caution.

Think about:
- What could spill, break, or cause damage?
- What spatial relationships are dangerous given the held object?
- Should the robot move slowly near certain objects?
- Can the held object be rotated freely?"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")
        return self._client

    def analyze_scene_from_text(self, scene_description: str, manipulated_object: str) -> SafetyContext:
        """
        Analyze a text-described scene using the VLM.
        For real deployment, replace with image-based analysis.
        """
        user_prompt = f"""Scene description: {scene_description}
The robot is holding: {manipulated_object}

Analyze this scene and output the safety constraints as JSON.
Remember: positions in meters relative to table center, approximate is fine."""

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )
            raw = response.content[0].text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(raw)
            return self._parse_response(data, manipulated_object)
        except Exception as e:
            print(f"[VLM] API call failed ({e}), falling back to mock analysis")
            return self.mock_analyze(scene_description, manipulated_object)

    def analyze_scene_from_image(self, image_path: str, manipulated_object: str) -> SafetyContext:
        """
        Analyze a scene from an actual image using Claude's vision.
        """
        import base64

        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        # Determine media type
        ext = image_path.lower().rsplit(".", 1)[-1]
        media_type = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")

        user_content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": image_data}
            },
            {
                "type": "text",
                "text": f"The robot is holding: {manipulated_object}\n\n"
                        f"Analyze this scene image and output safety constraints as JSON. "
                        f"Estimate object positions in meters relative to the table center. "
                        f"Identify ALL semantic safety risks."
            }
        ]

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}]
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(raw)
            return self._parse_response(data, manipulated_object)
        except Exception as e:
            print(f"[VLM] Image analysis failed ({e}), falling back to mock")
            return self.mock_analyze("kitchen table with objects", manipulated_object)

    def mock_analyze(self, scene_description: str, manipulated_object: str) -> SafetyContext:
        """
        Mock analysis for testing without API access.
        Simulates the VLM output for common scenarios.
        """
        print("[VLM] Using mock scene analysis")

        # Default kitchen-table scene
        objects = [
            ObjectInfo("laptop", np.array([0.3, 0.0, 0.02]), np.array([0.35, 0.25, 0.02]),
                        "electronics", {"fragile": True, "water_sensitive": True}),
            ObjectInfo("book", np.array([-0.2, 0.15, 0.02]), np.array([0.2, 0.15, 0.03]),
                        "paper", {"fragile": False, "water_sensitive": True}),
            ObjectInfo("phone", np.array([0.0, -0.2, 0.01]), np.array([0.08, 0.15, 0.01]),
                        "electronics", {"fragile": True, "water_sensitive": True}),
            ObjectInfo("cutting_board", np.array([-0.3, -0.1, 0.01]), np.array([0.3, 0.2, 0.02]),
                        "kitchen", {"fragile": False, "water_sensitive": False}),
        ]

        spatial_constraints = []
        behavioral_constraints = []
        pose_constraint = "free_rotation"

        is_liquid = manipulated_object.lower() in ["cup of water", "cup_of_water", "coffee", "soup", "glass of water"]
        is_flame = manipulated_object.lower() in ["candle", "lit candle", "lighter"]
        is_sharp = manipulated_object.lower() in ["knife", "scissors", "blade"]

        for obj in objects:
            if is_liquid and obj.properties.get("water_sensitive", False):
                spatial_constraints.append(SemanticConstraint(
                    "spatial", manipulated_object, obj.name, "above",
                    {"safety_margin": 0.15, "reason": f"Water may spill onto {obj.name}"}
                ))
                behavioral_constraints.append(SemanticConstraint(
                    "behavioral", manipulated_object, obj.name, "near",
                    {"caution_level": 0.8, "max_approach_velocity": 0.05,
                     "reason": f"Move slowly near {obj.name} to prevent splashing"}
                ))

            if is_flame and obj.semantic_label == "paper":
                spatial_constraints.append(SemanticConstraint(
                    "spatial", manipulated_object, obj.name, "around",
                    {"safety_margin": 0.25, "reason": f"Fire hazard near {obj.name}"}
                ))

            if is_sharp:
                behavioral_constraints.append(SemanticConstraint(
                    "behavioral", manipulated_object, obj.name, "near",
                    {"caution_level": 0.6, "max_approach_velocity": 0.08,
                     "reason": f"Sharp object, approach {obj.name} carefully"}
                ))

        if is_liquid:
            pose_constraint = "constrained_rotation"

        reasoning = (f"Analyzing scene with {len(objects)} objects. "
                     f"Held object '{manipulated_object}' is "
                     f"{'liquid-containing' if is_liquid else 'flame' if is_flame else 'sharp' if is_sharp else 'benign'}. "
                     f"Generated {len(spatial_constraints)} spatial and "
                     f"{len(behavioral_constraints)} behavioral constraints.")

        return SafetyContext(
            objects=objects,
            spatial_constraints=spatial_constraints,
            behavioral_constraints=behavioral_constraints,
            pose_constraint=pose_constraint,
            manipulated_object=manipulated_object,
            reasoning=reasoning
        )

    def _parse_response(self, data: dict, manipulated_object: str) -> SafetyContext:
        """Parse VLM JSON response into SafetyContext."""
        objects = []
        for obj_data in data.get("objects", []):
            objects.append(ObjectInfo(
                name=obj_data["name"],
                position=np.array(obj_data["position"]),
                dimensions=np.array(obj_data["dimensions"]),
                semantic_label=obj_data.get("semantic_label", "unknown"),
                properties=obj_data.get("properties", {})
            ))

        spatial_constraints = []
        for sc in data.get("spatial_constraints", []):
            spatial_constraints.append(SemanticConstraint(
                "spatial", sc.get("source_object", manipulated_object),
                sc["target_object"], sc["relationship"],
                {k: v for k, v in sc.items() if k not in ["source_object", "target_object", "relationship"]}
            ))

        behavioral_constraints = []
        for bc in data.get("behavioral_constraints", []):
            behavioral_constraints.append(SemanticConstraint(
                "behavioral", manipulated_object, bc["target_object"], "near",
                {k: v for k, v in bc.items() if k != "target_object"}
            ))

        return SafetyContext(
            objects=objects,
            spatial_constraints=spatial_constraints,
            behavioral_constraints=behavioral_constraints,
            pose_constraint=data.get("pose_constraint", "free_rotation"),
            manipulated_object=manipulated_object,
            reasoning=data.get("reasoning", "")
        )


# ============================================================================
# CBF CONSTRUCTOR — Maps VLM output to differentiable barrier functions
# ============================================================================

class CBFConstructor:
    """
    Constructs Control Barrier Functions from the VLM's SafetyContext.

    This is the key bridging module: it takes semantic constraints
    (in natural language/structured form) and produces differentiable
    barrier functions h(x) that can be used in a QP-based safety filter.

    CBF types:
    - Spatial: Superquadric-based regions the EE must avoid
    - Behavioral: Velocity-dependent barriers near certain objects  
    - Pose: Orientation constraints on the end-effector
    """

    def __init__(self, workspace_bounds: np.ndarray = None):
        """
        Args:
            workspace_bounds: [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        """
        if workspace_bounds is None:
            self.workspace_bounds = np.array([[-0.6, -0.4, 0.0], [0.6, 0.4, 0.5]])
        else:
            self.workspace_bounds = workspace_bounds

    def build_cbfs(self, safety_context: SafetyContext) -> dict:
        """
        Convert SafetyContext into a set of CBF functions and their parameters.

        Returns dict with:
            - 'spatial_cbfs': List of (h_func, grad_h_func, name, params)
            - 'behavioral_params': Dict mapping object_name → {alpha_scale, max_vel}
            - 'pose_params': Dict with rotation constraint parameters
        """
        spatial_cbfs = []
        behavioral_params = {}

        # Build object lookup
        obj_lookup = {obj.name: obj for obj in safety_context.objects}

        # --- Spatial CBFs ---
        for sc in safety_context.spatial_constraints:
            if sc.target_object not in obj_lookup:
                warnings.warn(f"Object '{sc.target_object}' not found in scene, skipping constraint")
                continue

            obj = obj_lookup[sc.target_object]
            margin = sc.parameters.get("safety_margin", 0.1)

            # Build superquadric parameters based on relationship type
            cbf_info = self._build_spatial_cbf(obj, sc.relationship, margin)
            if cbf_info is not None:
                spatial_cbfs.append(cbf_info)

        # --- Behavioral parameters ---
        for bc in safety_context.behavioral_constraints:
            caution = bc.parameters.get("caution_level", 0.5)
            max_vel = bc.parameters.get("max_approach_velocity", 0.1)
            behavioral_params[bc.target_object] = {
                "alpha_scale": 1.0 - 0.9 * caution,  # Lower = more cautious
                "max_approach_velocity": max_vel,
                "caution_level": caution,
            }

        # --- Pose parameters ---
        pose_params = {
            "constrained": safety_context.pose_constraint == "constrained_rotation",
            "max_angular_velocity": 0.1 if safety_context.pose_constraint == "constrained_rotation" else float('inf'),
            "orientation_weight": 10.0 if safety_context.pose_constraint == "constrained_rotation" else 0.0,
        }

        return {
            "spatial_cbfs": spatial_cbfs,
            "behavioral_params": behavioral_params,
            "pose_params": pose_params,
        }

    def _build_spatial_cbf(self, obj: ObjectInfo, relationship: str, margin: float):
        """
        Build a superquadric-based CBF for a spatial constraint.

        The CBF h(x) > 0 defines the safe region.
        h(x) = g(x; θ) - 1, where g is the superquadric function.

        For 2D (top-down) simplification:
            g(x) = ((x - cx)/ax)^(2/ε) + ((y - cy)/ay)^(2/ε)
            h(x) = g(x) - 1  (positive outside the superquadric = safe)
        """
        cx, cy = obj.position[0], obj.position[1]
        w, h_dim = obj.dimensions[0], obj.dimensions[1]

        # Expand dimensions based on relationship
        if relationship == "above":
            # In 2D top-down view, "above" means directly over the object
            ax = (w / 2 + margin * 0.5)
            ay = (h_dim / 2 + margin * 0.5)
            epsilon = 0.5  # Rounder shape
            name = f"no_{relationship}_{obj.name}"

        elif relationship == "around":
            # Full surrounding exclusion zone
            ax = (w / 2 + margin * 1.0)
            ay = (h_dim / 2 + margin * 1.0)
            epsilon = 0.8  # More rectangular
            name = f"stay_away_{obj.name}"

        elif relationship == "near":
            ax = (w / 2 + margin * 0.8)
            ay = (h_dim / 2 + margin * 0.8)
            epsilon = 0.5
            name = f"caution_near_{obj.name}"

        else:
            # Default: treat as "around"
            ax = (w / 2 + margin)
            ay = (h_dim / 2 + margin)
            epsilon = 0.5
            name = f"constraint_{obj.name}"

        params = {
            "center": np.array([cx, cy]),
            "semi_axes": np.array([ax, ay]),
            "epsilon": epsilon,
            "object_name": obj.name,
            "relationship": relationship,
        }

        def h_func(x_ee, p=params):
            """CBF value: h(x) > 0 is safe."""
            dx = (x_ee[0] - p["center"][0]) / p["semi_axes"][0]
            dy = (x_ee[1] - p["center"][1]) / p["semi_axes"][1]
            exp = 2.0 / p["epsilon"]
            g = (np.abs(dx) ** exp + np.abs(dy) ** exp) ** p["epsilon"]
            return g - 1.0  # Positive outside (safe), negative inside (unsafe)

        def grad_h_func(x_ee, p=params):
            """Gradient of CBF: ∂h/∂x."""
            dx = (x_ee[0] - p["center"][0]) / p["semi_axes"][0]
            dy = (x_ee[1] - p["center"][1]) / p["semi_axes"][1]
            exp = 2.0 / p["epsilon"]

            # Numerical stability
            eps_val = 1e-8
            abs_dx = np.abs(dx) + eps_val
            abs_dy = np.abs(dy) + eps_val

            inner = abs_dx ** exp + abs_dy ** exp + eps_val
            outer_exp = p["epsilon"] - 1.0

            # Chain rule
            dg_dxee = np.zeros(2)
            # ∂g/∂x_ee = ε * (|dx|^e + |dy|^e)^(ε-1) * e * |dx|^(e-1) * sign(dx) / ax
            common = p["epsilon"] * (inner ** outer_exp)
            dg_dxee[0] = common * exp * (abs_dx ** (exp - 1)) * np.sign(dx) / p["semi_axes"][0]
            dg_dxee[1] = common * exp * (abs_dy ** (exp - 1)) * np.sign(dy) / p["semi_axes"][1]

            return dg_dxee  # ∂h/∂x = ∂g/∂x since h = g - 1

        return (h_func, grad_h_func, name, params)


# ============================================================================
# SAFETY FILTER — QP-based CBF safety filter
# ============================================================================

class CBFSafetyFilter:
    """
    Quadratic Program (QP) based safety filter using CBFs.

    Solves at each timestep:
        u_cert = argmin ||u - u_cmd||^2 + w_rot * L_rot
        s.t.  ḣ_i(x, u) >= -α_i(h_i(x))   for all CBFs

    For our 2D end-effector control:
        x_ee = [px, py], u = [vx, vy]
        ḣ = ∇h · u >= -α(h)
    """

    def __init__(self, cbf_data: dict, dt: float = 0.02):
        self.spatial_cbfs = cbf_data["spatial_cbfs"]
        self.behavioral_params = cbf_data["behavioral_params"]
        self.pose_params = cbf_data["pose_params"]
        self.dt = dt
        self.u_max = 0.5  # Max velocity (m/s)

    def certify(self, x_ee: np.ndarray, u_cmd: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Certify a control input using the CBF-QP safety filter.

        Args:
            x_ee: Current end-effector position [x, y]
            u_cmd: Desired velocity command [vx, vy]

        Returns:
            u_cert: Certified (safe) velocity command
            info: Dict with CBF values, constraint activity, etc.
        """
        n_u = len(u_cmd)
        n_constraints = len(self.spatial_cbfs)

        if n_constraints == 0:
            # No constraints — pass through
            u_cert = np.clip(u_cmd, -self.u_max, self.u_max)
            return u_cert, {"cbf_values": [], "active_constraints": [], "modified": False,
                            "u_cmd": u_cmd.copy(), "u_cert": u_cert.copy()}

        # Evaluate all CBFs and gradients
        h_values = []
        grad_h_values = []
        alpha_values = []
        names = []

        for (h_func, grad_h_func, name, params) in self.spatial_cbfs:
            h_val = h_func(x_ee)
            grad_h = grad_h_func(x_ee)
            h_values.append(h_val)
            grad_h_values.append(grad_h)
            names.append(name)

            # Class-K function α(h) — determines approach speed to boundary
            # Check if this object has behavioral constraints
            obj_name = params.get("object_name", "")
            if obj_name in self.behavioral_params:
                alpha_scale = self.behavioral_params[obj_name]["alpha_scale"]
            else:
                alpha_scale = 1.0

            # α(h) = scale * h (linear class-K∞ — more conservative than quadratic)
            alpha = alpha_scale * max(h_val, 0.0)
            alpha_values.append(alpha)

        # --- Solve QP ---
        # min ||u - u_cmd||^2
        # s.t. ∇h_i · u >= -α_i(h_i) for all i
        #      ||u||_inf <= u_max
        #
        # This is: min 0.5 * u^T I u - u_cmd^T u
        #          s.t. A u >= b

        # Build constraint matrix
        A = np.array(grad_h_values)  # (n_constraints, n_u)
        b = np.array([-alpha for alpha in alpha_values])  # (n_constraints,)

        # Solve via simple projected gradient / analytical for 2D
        u_cert = self._solve_qp_simple(u_cmd, A, b)

        # Clamp to velocity limits
        u_cert = np.clip(u_cert, -self.u_max, self.u_max)

        # Check which constraints are active (close to boundary)
        active = [names[i] for i in range(n_constraints) if h_values[i] < 0.3]
        modified = np.linalg.norm(u_cert - u_cmd) > 1e-4

        info = {
            "cbf_values": list(zip(names, h_values)),
            "active_constraints": active,
            "modified": modified,
            "u_cmd": u_cmd.copy(),
            "u_cert": u_cert.copy(),
        }

        return u_cert, info

    def _solve_qp_simple(self, u_cmd: np.ndarray, A: np.ndarray, b: np.ndarray,
                          max_iter: int = 100) -> np.ndarray:
        """
        Simple QP solver: iterative constraint projection (Dykstra's algorithm).
        For production, use cvxpy, qpsolvers, or OSQP.
        """
        n_u = len(u_cmd)
        n_c = A.shape[0]

        if n_c == 0:
            return u_cmd.copy()

        # Check if u_cmd already satisfies all constraints
        margins = A @ u_cmd - b
        if np.all(margins >= -1e-8):
            return u_cmd.copy()

        # Iterative projection onto half-spaces
        u = u_cmd.copy()
        for iteration in range(max_iter):
            all_satisfied = True
            for i in range(n_c):
                margin = A[i] @ u - b[i]
                if margin < -1e-8:
                    all_satisfied = False
                    # Project onto half-space: a^T u >= b_i
                    a = A[i]
                    a_norm_sq = np.dot(a, a) + 1e-10
                    correction = (-margin / a_norm_sq) * a
                    u = u + correction

            if all_satisfied:
                break

        return u


# ============================================================================
# 2D SIMULATION ENVIRONMENT
# ============================================================================

class ManipulationSimulator2D:
    """
    Simple 2D top-down manipulation environment for testing the CBF pipeline.
    The end-effector moves in the XY plane carrying an object.
    """

    def __init__(self, dt: float = 0.02):
        self.dt = dt
        self.x_ee = np.array([0.0, -0.3])  # Start position
        self.trajectory = [self.x_ee.copy()]
        self.commands = []
        self.certified = []

    def step(self, u: np.ndarray):
        """Integrate one timestep."""
        self.x_ee = self.x_ee + u * self.dt
        self.trajectory.append(self.x_ee.copy())

    def generate_sweep_trajectory(self, target: np.ndarray, speed: float = 0.15) -> List[np.ndarray]:
        """Generate velocity commands to sweep from current position to target."""
        commands = []
        pos = self.x_ee.copy()
        while np.linalg.norm(target - pos) > speed * self.dt:
            direction = (target - pos) / np.linalg.norm(target - pos)
            cmd = direction * speed
            commands.append(cmd)
            pos = pos + cmd * self.dt
        return commands

    def generate_figure_eight(self, center: np.ndarray, radius: float = 0.3,
                               speed: float = 0.15, n_steps: int = 500) -> List[np.ndarray]:
        """Generate velocity commands tracing a figure-8 (passes through multiple constraint zones)."""
        commands = []
        for i in range(n_steps):
            t = 2 * np.pi * i / n_steps
            # Lissajous figure
            target_x = center[0] + radius * np.sin(t)
            target_y = center[1] + radius * np.sin(2 * t) * 0.6
            if i == 0:
                pos = np.array([target_x, target_y])
                self.x_ee = pos.copy()
                self.trajectory = [pos.copy()]
                continue

            target = np.array([target_x, target_y])
            direction = target - self.x_ee
            cmd = direction / self.dt  # Velocity to reach target in one step
            # Clamp speed
            if np.linalg.norm(cmd) > speed:
                cmd = cmd / np.linalg.norm(cmd) * speed
            commands.append(cmd)
        return commands


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(sim: ManipulationSimulator2D, safety_context: SafetyContext,
                       cbf_data: dict, info_history: List[dict], save_path: str = None):
    """
    Visualize the safety filter results: trajectory, constraint regions, CBF values.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Plot 1: Trajectory with constraint regions ---
    ax1 = axes[0]
    ax1.set_title("Trajectory & Semantic Constraints", fontsize=12, fontweight='bold')
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Draw objects
    for obj in safety_context.objects:
        rect = patches.Rectangle(
            (obj.position[0] - obj.dimensions[0]/2, obj.position[1] - obj.dimensions[1]/2),
            obj.dimensions[0], obj.dimensions[1],
            linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.7, zorder=2
        )
        ax1.add_patch(rect)
        ax1.annotate(obj.name, (obj.position[0], obj.position[1]),
                     ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)

    # Draw CBF constraint boundaries (h=0 contour)
    x_range = np.linspace(-0.6, 0.6, 200)
    y_range = np.linspace(-0.4, 0.4, 200)
    X, Y = np.meshgrid(x_range, y_range)

    for (h_func, _, name, params) in cbf_data["spatial_cbfs"]:
        H = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                H[i, j] = h_func(np.array([X[i, j], Y[i, j]]))

        # Fill unsafe region
        ax1.contourf(X, Y, H, levels=[-10, 0], colors=['red'], alpha=0.15, zorder=1)
        # Draw boundary
        ax1.contour(X, Y, H, levels=[0], colors=['red'], linewidths=2,
                    linestyles='--', zorder=3)
        # Draw margin
        ax1.contour(X, Y, H, levels=[0.3], colors=['orange'], linewidths=1,
                    linestyles=':', alpha=0.6, zorder=3)

    # Draw trajectory
    traj = np.array(sim.trajectory)
    if len(traj) > 1:
        # Color by time
        for i in range(len(traj) - 1):
            t_frac = i / max(len(traj) - 1, 1)
            color = plt.cm.viridis(t_frac)
            ax1.plot(traj[i:i+2, 0], traj[i:i+2, 1], '-', color=color, linewidth=2, zorder=4)

        ax1.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, zorder=5, label='Start')
        ax1.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=10, zorder=5, label='End')
    ax1.legend(loc='upper right', fontsize=8)

    # --- Plot 2: CBF values over time ---
    ax2 = axes[1]
    ax2.set_title("CBF Values Over Time", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("h(x) — CBF value")
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Safety boundary')
    ax2.grid(True, alpha=0.3)

    if info_history:
        cbf_names = [name for name, _ in info_history[0]["cbf_values"]]
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(cbf_names), 1)))

        for idx, name in enumerate(cbf_names):
            values = [info["cbf_values"][idx][1] for info in info_history]
            ax2.plot(values, label=name, color=colors[idx], linewidth=1.5)

    ax2.legend(fontsize=7, loc='upper right')
    ax2.set_ylim(bottom=-0.5)

    # --- Plot 3: Control modification ---
    ax3 = axes[2]
    ax3.set_title("Safety Filter Intervention", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("||u_cmd - u_cert|| (m/s)")
    ax3.grid(True, alpha=0.3)

    if info_history:
        modifications = [np.linalg.norm(info["u_cmd"] - info["u_cert"]) for info in info_history]
        ax3.fill_between(range(len(modifications)), modifications, alpha=0.3, color='coral')
        ax3.plot(modifications, color='red', linewidth=1.5, label='Control modification')

        # Mark active constraints
        active_steps = [i for i, info in enumerate(info_history) if info["active_constraints"]]
        if active_steps:
            ax3.scatter(active_steps, [modifications[i] for i in active_steps],
                       color='red', s=5, zorder=5, label='Constraint active')
    ax3.legend(fontsize=8)

    plt.suptitle(f"VLM-CBF Safety Filter | Holding: {safety_context.manipulated_object}",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VIZ] Saved to {save_path}")
    else:
        plt.savefig("/tmp/vlm_cbf_result.png", dpi=150, bbox_inches='tight')
        print("[VIZ] Saved to /tmp/vlm_cbf_result.png")

    return fig


def visualize_cbf_landscape(cbf_data: dict, safety_context: SafetyContext, save_path: str = None):
    """Visualize the CBF value landscape as a heatmap."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    x_range = np.linspace(-0.6, 0.6, 300)
    y_range = np.linspace(-0.4, 0.4, 200)
    X, Y = np.meshgrid(x_range, y_range)

    # Compute minimum CBF value at each point (most restrictive constraint)
    H_min = np.full_like(X, np.inf)
    for (h_func, _, name, params) in cbf_data["spatial_cbfs"]:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                h_val = h_func(np.array([X[i, j], Y[i, j]]))
                H_min[i, j] = min(H_min[i, j], h_val)

    if np.all(np.isinf(H_min)):
        H_min = np.ones_like(X)

    # Plot heatmap
    im = ax.contourf(X, Y, H_min, levels=50, cmap='RdYlGn', vmin=-1, vmax=3)
    ax.contour(X, Y, H_min, levels=[0], colors='black', linewidths=3)
    plt.colorbar(im, ax=ax, label='min h(x) — CBF value')

    # Draw objects
    for obj in safety_context.objects:
        rect = patches.Rectangle(
            (obj.position[0] - obj.dimensions[0]/2, obj.position[1] - obj.dimensions[1]/2),
            obj.dimensions[0], obj.dimensions[1],
            linewidth=2, edgecolor='white', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        ax.annotate(obj.name, (obj.position[0], obj.position[1]),
                    ha='center', va='center', fontsize=9, color='white',
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.2',
                    facecolor='black', alpha=0.7))

    ax.set_title(f"CBF Safety Landscape | Holding: {safety_context.manipulated_object}",
                fontsize=13, fontweight='bold')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect('equal')

    path = save_path or "/tmp/vlm_cbf_landscape.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[VIZ] CBF landscape saved to {path}")
    return fig


# ============================================================================
# MAIN DEMO PIPELINE
# ============================================================================

def run_demo(use_vlm: bool = False, api_key: str = None, scene_image: str = None):
    """
    Run the full VLM → CBF → Safety Filter pipeline.

    Args:
        use_vlm: If True, uses Claude API for scene analysis
        api_key: Anthropic API key (if use_vlm=True)
        scene_image: Path to scene image (optional, for VLM image mode)
    """
    print("=" * 70)
    print("VLM-to-CBF Pipeline Demo")
    print("=" * 70)

    # --- Step 1: Scene Analysis via VLM ---
    print("\n[1/4] Analyzing scene with VLM...")
    analyzer = VLMSceneAnalyzer(api_key=api_key)

    scene_desc = (
        "A kitchen table with: a laptop (center-right), a book (left), "
        "a smartphone (center-front), and a cutting board (far left). "
        "The robot arm is mounted behind the table."
    )
    manipulated_object = "cup of water"

    if use_vlm and scene_image:
        safety_context = analyzer.analyze_scene_from_image(scene_image, manipulated_object)
    elif use_vlm:
        safety_context = analyzer.analyze_scene_from_text(scene_desc, manipulated_object)
    else:
        safety_context = analyzer.mock_analyze(scene_desc, manipulated_object)

    print(f"  Manipulated object: {safety_context.manipulated_object}")
    print(f"  Objects detected: {[o.name for o in safety_context.objects]}")
    print(f"  Spatial constraints: {len(safety_context.spatial_constraints)}")
    print(f"  Behavioral constraints: {len(safety_context.behavioral_constraints)}")
    print(f"  Pose constraint: {safety_context.pose_constraint}")
    print(f"  Reasoning: {safety_context.reasoning}")

    # --- Step 2: Construct CBFs ---
    print("\n[2/4] Constructing CBFs from semantic constraints...")
    constructor = CBFConstructor()
    cbf_data = constructor.build_cbfs(safety_context)

    print(f"  Built {len(cbf_data['spatial_cbfs'])} spatial CBFs:")
    for _, _, name, params in cbf_data["spatial_cbfs"]:
        print(f"    - {name}: center={np.array2string(params['center'], precision=3)}, axes={np.array2string(params['semi_axes'], precision=3)}")
    print(f"  Behavioral params: {cbf_data['behavioral_params']}")
    print(f"  Pose constrained: {cbf_data['pose_params']['constrained']}")

    # --- Step 3: Run simulation with safety filter ---
    print("\n[3/4] Running simulation with CBF safety filter...")
    sim = ManipulationSimulator2D(dt=0.02)
    safety_filter = CBFSafetyFilter(cbf_data, dt=0.02)

    # Generate a trajectory that would violate constraints without the filter
    commands = sim.generate_figure_eight(
        center=np.array([0.0, 0.0]),
        radius=0.35,
        speed=0.15,
        n_steps=600
    )

    info_history = []
    n_modified = 0

    for cmd in commands:
        u_cert, info = safety_filter.certify(sim.x_ee, cmd)
        sim.step(u_cert)
        info_history.append(info)
        if info["modified"]:
            n_modified += 1

    print(f"  Simulated {len(commands)} timesteps")
    print(f"  Safety filter modified {n_modified}/{len(commands)} commands ({100*n_modified/len(commands):.1f}%)")

    # Verify no constraint violations
    violations = 0
    for info in info_history:
        for name, h_val in info["cbf_values"]:
            if h_val < -0.01:
                violations += 1
                break
    print(f"  Constraint violations: {violations}")

    # --- Step 4: Visualize ---
    print("\n[4/4] Generating visualizations...")
    visualize_results(sim, safety_context, cbf_data, info_history,
                     save_path="/home/claude/vlm_cbf_trajectory.png")
    visualize_cbf_landscape(cbf_data, safety_context,
                           save_path="/home/claude/vlm_cbf_landscape.png")

    # --- Also run with "dry sponge" for comparison ---
    print("\n" + "=" * 70)
    print("Comparison: Running with 'dry sponge' (fewer constraints)")
    print("=" * 70)

    safety_context_sponge = analyzer.mock_analyze(scene_desc, "dry sponge")
    cbf_data_sponge = constructor.build_cbfs(safety_context_sponge)
    print(f"  Spatial constraints for sponge: {len(cbf_data_sponge['spatial_cbfs'])}")
    print(f"  Pose constrained: {cbf_data_sponge['pose_params']['constrained']}")

    sim_sponge = ManipulationSimulator2D(dt=0.02)
    filter_sponge = CBFSafetyFilter(cbf_data_sponge, dt=0.02)

    commands_sponge = sim_sponge.generate_figure_eight(
        center=np.array([0.0, 0.0]), radius=0.35, speed=0.15, n_steps=600
    )

    info_history_sponge = []
    for cmd in commands_sponge:
        u_cert, info = filter_sponge.certify(sim_sponge.x_ee, cmd)
        sim_sponge.step(u_cert)
        info_history_sponge.append(info)

    n_mod_sponge = sum(1 for i in info_history_sponge if i["modified"])
    print(f"  Safety filter modified {n_mod_sponge}/{len(commands_sponge)} commands for sponge")

    visualize_results(sim_sponge, safety_context_sponge, cbf_data_sponge, info_history_sponge,
                     save_path="/home/claude/vlm_cbf_trajectory_sponge.png")

    print("\n" + "=" * 70)
    print("Done! Key insight: same trajectory, different held object →")
    print("VLM produces different semantic constraints → different CBFs →")
    print("different safety-filtered behavior.")
    print("=" * 70)

    return safety_context, cbf_data, sim, info_history


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLM-to-CBF Pipeline Demo")
    parser.add_argument("--use-vlm", action="store_true", help="Use Claude API (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--api-key", type=str, default=None, help="Anthropic API key")
    parser.add_argument("--image", type=str, default=None, help="Path to scene image")
    parser.add_argument("--object", type=str, default="cup of water", help="Manipulated object name")
    args = parser.parse_args()

    import os
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    run_demo(use_vlm=args.use_vlm, api_key=api_key, scene_image=args.image)
