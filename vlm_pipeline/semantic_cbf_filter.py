"""
semantic_cbf_filter.py

Semantic CBF Safety Filter pipeline (Brunke et al. RA-L 2025 + SQ-CBF 2026).

Pipeline (each step logged):
  [Step 1/4] Camera Images → SAM Segmentation → Object Masks + Labels
  [Step 2/4] 3D Map Construction (superquadric fitting)
  [Step 3/4] VLM Semantic Constraint Synthesis  (Qwen in "qwen" conda env)
  [Step 4/4] CBF-QP Safety Filter Construction

Runtime:
  OpenVLA-OFT → u_nominal → CBF certify_action() → u_safe → env.step()

Environment setup:
  - SafeLIBERO + OpenVLA-OFT run in the main conda env
  - Qwen VLM runs in the "qwen" conda env (called via subprocess)
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

GRIPPER_SPHERE_CENTER_OFFSET = np.array([0.0, 0.0, -0.08], dtype=np.float64)


def offset_gripper_sphere_center(eef_pos, eef_quat=None, offset=GRIPPER_SPHERE_CENTER_OFFSET):
    """Place the gripper sphere using an offset in the end-effector frame."""
    center = np.asarray(eef_pos, dtype=np.float64).copy()
    offset = np.asarray(offset, dtype=np.float64)
    if eef_quat is not None:
        quat = np.asarray(eef_quat, dtype=np.float64)
        if quat.shape == (4,) and np.all(np.isfinite(quat)):
            offset = Rotation.from_quat(quat).apply(offset)
    center += offset
    return center


# ═══════════════════════════════════════════════════════════════════════
# 1. Data Structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SuperquadricParams:
    """SQ implicit: f = ((x/a1)^(2/e2)+(y/a2)^(2/e2))^(e2/e1)+(z/a3)^(2/e1)-1"""
    a1: float = 0.05; a2: float = 0.05; a3: float = 0.05
    e1: float = 1.0;  e2: float = 1.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

@dataclass
class SemanticObject:
    label: str; position: np.ndarray; point_cloud: np.ndarray
    superquadric: SuperquadricParams
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))

@dataclass
class SemanticConstraint:
    object_label: str; object_idx: int; constraint_type: str
    spatial_relation: str = ""; caution_level: str = "normal"

@dataclass
class SemanticContext:
    manipulated_object: str
    spatial_constraints: List[SemanticConstraint] = field(default_factory=list)
    behavioral_constraints: List[SemanticConstraint] = field(default_factory=list)
    pose_constraint: str = "free_rotation"


# ═══════════════════════════════════════════════════════════════════════
# 2. Perception: Segmentation → 3D Map
# ═══════════════════════════════════════════════════════════════════════

class PerceptionModule:
    def __init__(self, env=None):
        self.env = env
        self.objects: List[SemanticObject] = []

    def segment_and_build_map(self, obs, camera_images=None, depth_images=None):
        """[Step 1/4] Camera Images → Segmentation → Object Masks + Labels"""
        self.objects = []
        logger.info("=" * 70)
        logger.info("[Pipeline Step 1/4] CAMERA IMAGES → SEGMENTATION")
        logger.info("=" * 70)
        if camera_images:
            logger.info(f"  Input: {len(camera_images)} camera image(s): "
                         + ", ".join(str(im.shape) for im in camera_images))
        else:
            logger.info("  Input: No camera images (using sim state)")

        if self.env is not None:
            logger.info("  Source: MuJoCo sim ground-truth state")
            logger.info("  Segmentation: extracting poses from sim (SAM not needed)")
            self._build_from_sim(obs)
        elif camera_images and depth_images:
            logger.info("  Source: RGB-D → SAM → CLIP → labels")
            logger.warning("  RGB-D pipeline requires SAM+CLIP; using sim fallback")
        else:
            logger.warning("  No perception source available")

        logger.info(f"  Result: {len(self.objects)} objects detected")
        for i, o in enumerate(self.objects):
            logger.info(f"    [{i}] '{o.label}' pos=[{o.position[0]:.3f},"
                         f"{o.position[1]:.3f},{o.position[2]:.3f}] pts={o.point_cloud.shape[0]}")
        return self.objects

    def _build_from_sim(self, obs):
        names = set()
        for k in obs:
            if k.endswith("_pos") and "robot" not in k and "gripper" not in k:
                n = k[:-4]; p = obs[k]
                if isinstance(p, np.ndarray) and len(p)==3 and p[2]>-0.1 and abs(p[0])<1 and abs(p[1])<1:
                    names.add(n)
        logger.info(f"  Sim state: {len(names)} candidate objects")
        for n in names:
            pos = np.array(obs[f"{n}_pos"], dtype=np.float64)
            if pos[2] < -0.5: continue
            rot = Rotation.from_quat(obs[f"{n}_quat"]).as_matrix() if f"{n}_quat" in obs else np.eye(3)
            pc = self._synth_pc(pos, rot, n)
            sq = self._fit_sq(pc, pos, rot, n)
            lbl = n.replace("_"," ").strip()
            lbl = " ".join(p for p in lbl.split() if not p.isdigit()) or lbl
            self.objects.append(SemanticObject(
                label=lbl, position=pos, point_cloud=pc, superquadric=sq,
                bbox_min=pc.min(0), bbox_max=pc.max(0)))

    def _synth_pc(self, pos, rot, name):
        sz = np.array([0.04,0.04,0.06]) if "obstacle" in name.lower() else np.array([0.04,0.04,0.04])
        for kw,s in [("mug",[.04,.04,.05]),("cup",[.04,.04,.05]),("plate",[.08,.08,.02]),
                     ("can",[.03,.03,.08]),("bottle",[.03,.03,.08])]:
            if kw in name.lower(): sz = np.array(s); break
        pts = []
        for ax in range(3):
            for sg in [-1,1]:
                fp = np.random.uniform(-1,1,(50,3))*sz; fp[:,ax]=sg*sz[ax]; pts.append(fp)
        pc = np.vstack(pts)
        return (rot @ pc.T).T + pos

    def _fit_sq(self, pc, pos, rot, name):
        lp = (rot.T @ (pc - pos).T).T
        ext = np.max(np.abs(lp), axis=0)
        a1,a2,a3 = [max(e,0.01) for e in ext]
        nl = name.lower()
        if any(k in nl for k in ["box","cube","block","obstacle"]): e1,e2=0.3,0.3
        elif any(k in nl for k in ["can","bottle","mug","cup"]): e1,e2=0.5,1.0
        elif any(k in nl for k in ["ball","sphere"]): e1,e2=1.0,1.0
        else: e1,e2=0.5,0.5
        m=1.15
        sq = SuperquadricParams(a1=a1*m,a2=a2*m,a3=a3*m,e1=max(e1,.1),e2=max(e2,.1),
                                position=pos.copy(),rotation=rot.copy())
        logger.info(f"  SQ fit '{name}': a=[{sq.a1:.4f},{sq.a2:.4f},{sq.a3:.4f}] e=[{sq.e1:.2f},{sq.e2:.2f}]")
        return sq


# ═══════════════════════════════════════════════════════════════════════
# 3. Qwen VLM — Cross-Conda-Env Subprocess Bridge
# ═══════════════════════════════════════════════════════════════════════

class QwenVLMBridge:
    """Calls qwen_vlm_worker.py in the 'qwen' conda env via subprocess.

    Communication: JSON file IPC
        main env → writes queries.json → subprocess conda run -n qwen ...
                 ← reads results.json
    """
    def __init__(self, qwen_conda_env="qwen", model_name="qwen2.5-vl-7b",
                 worker_script="qwen_vlm_worker.py", device="auto",
                 load_in_4bit=False, max_new_tokens=256):
        self.qwen_conda_env = qwen_conda_env
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.worker_script = self._find_worker(worker_script)
        logger.info(f"  QwenVLMBridge: model={model_name} env={qwen_conda_env} worker={self.worker_script}")

    @staticmethod
    def _find_worker(script):
        for c in [script, os.path.join(os.path.dirname(__file__),script),
                  os.path.join(os.getcwd(),script)]:
            if os.path.isfile(c): return os.path.abspath(c)
        return script

    def batch_query(self, queries, scene_image=None):
        if not queries: return []
        tmp = tempfile.mkdtemp(prefix="cbf_vlm_")
        inp = os.path.join(tmp,"queries.json"); outp = os.path.join(tmp,"results.json")

        img_path = None
        if scene_image is not None:
            from PIL import Image as PILImage
            img_path = os.path.join(tmp,"scene.png")
            PILImage.fromarray(scene_image).save(img_path)
            logger.info(f"  Saved scene image: {img_path}")
        for q in queries: q["image_path"] = img_path

        with open(inp,"w") as f: json.dump({"model":self.model_name,"queries":queries},f,indent=2)
        logger.info(f"  Sending {len(queries)} queries to Qwen ({self.qwen_conda_env} env)")

        cmd = ["conda","run","-n",self.qwen_conda_env,"--no-capture-output",
               "python",self.worker_script,"--input_json",inp,"--output_json",outp,
               "--model",self.model_name,"--device",self.device,
               "--max_new_tokens",str(self.max_new_tokens)]
        if self.load_in_4bit: cmd.append("--load_in_4bit")

        t0 = time.time()
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if res.returncode != 0:
                logger.error(f"  Qwen worker FAILED (rc={res.returncode}): {res.stderr[-300:]}")
                return [{"id":q["id"],"response":"error"} for q in queries]
            if res.stdout:
                for ln in res.stdout.strip().split("\n")[-5:]:
                    logger.info(f"  [QwenWorker] {ln.strip()}")
        except subprocess.TimeoutExpired:
            logger.error("  Qwen worker TIMED OUT")
            return [{"id":q["id"],"response":"error"} for q in queries]
        except FileNotFoundError:
            logger.error(f"  'conda' not on PATH or env '{self.qwen_conda_env}' missing")
            return [{"id":q["id"],"response":"error"} for q in queries]

        logger.info(f"  Qwen done in {time.time()-t0:.1f}s")
        if not os.path.exists(outp):
            return [{"id":q["id"],"response":"error"} for q in queries]
        with open(outp) as f: out = json.load(f)
        results = out.get("results",[])
        logger.info(f"  Got {len(results)} responses")
        try:
            for fn in os.listdir(tmp): os.remove(os.path.join(tmp,fn))
            os.rmdir(tmp)
        except: pass
        return results


# ═══════════════════════════════════════════════════════════════════════
# 3b. Prompt Templates (Brunke et al. multi-prompt)
# ═══════════════════════════════════════════════════════════════════════

class SemanticPromptTemplates:
    @staticmethod
    def spatial(scene, manip, obj, rel):
        return (f"A robot is in: {scene}\nHolding '{manip}'. There is '{obj}'.\n"
                f"Is it safe to move '{manip}' {rel} '{obj}'?\nAnswer 'safe' or 'unsafe' + reason.")
    @staticmethod
    def behavioral(scene, manip, obj):
        return (f"A robot is in: {scene}\nHolding '{manip}'. There is '{obj}'.\n"
                f"Should the robot be cautious near '{obj}'?\nAnswer 'caution' or 'no_caution' + reason.")
    @staticmethod
    def pose(manip):
        return (f"Robot holds '{manip}'. Can it freely rotate it, or must orientation be constrained?\n"
                f"Answer 'constrained_rotation' or 'free_rotation' + reason.")
    @staticmethod
    def parse_spatial(r):
        r=r.lower()
        if any(k in r for k in ["unsafe","not safe","dangerous","should not"]): return True
        if r.startswith("safe") or "it is safe" in r: return False
        return True
    @staticmethod
    def parse_behavioral(r):
        return "caution" if any(k in r.lower() for k in ["caution","careful","slowly","yes"]) else "normal"
    @staticmethod
    def parse_pose(r):
        return "constrained_rotation" if any(k in r.lower() for k in ["constrained","upright","spill","pour"]) else "free_rotation"


# ═══════════════════════════════════════════════════════════════════════
# 3c. Constraint Synthesizer
# ═══════════════════════════════════════════════════════════════════════

class SemanticConstraintSynthesizer:
    LIQUID={"cup of water","mug","cup","bowl","water","soup"}
    FRAGILE={"glass","plate","vase","egg","cup"}
    ELECTRONICS={"laptop","computer","phone","tablet","monitor"}
    FLAMMABLE={"candle","paper","cloth","napkin"}

    def __init__(self, use_vlm=False, vlm_model="qwen2.5-vl-7b", vlm_conda_env="qwen",
                 vlm_device="auto", vlm_load_in_4bit=False, vlm_worker_script="qwen_vlm_worker.py",
                 num_votes=3):
        self.use_vlm=use_vlm; self.num_votes=num_votes; self._bridge=None
        if use_vlm:
            self._bridge = QwenVLMBridge(vlm_conda_env,vlm_model,vlm_worker_script,vlm_device,vlm_load_in_4bit)

    def synthesize_constraints(self, scene_objects, manipulated_object,
                                task_description="", camera_images=None):
        """[Step 3/4] Semantic constraint synthesis."""
        if self.use_vlm and self._bridge:
            logger.info("=" * 70)
            logger.info("[Pipeline Step 3/4] VLM SEMANTIC CONSTRAINT SYNTHESIS (Qwen)")
            logger.info("=" * 70)
            logger.info(f"  VLM: {self._bridge.model_name} | env: {self._bridge.qwen_conda_env}")
            logger.info(f"  Manipulated: '{manipulated_object}'")
            logger.info(f"  Scene: {[o.label for o in scene_objects]}")
            logger.info(f"  Votes: {self.num_votes} per query")
            try:
                return self._synth_vlm(scene_objects, manipulated_object, task_description, camera_images)
            except Exception as e:
                logger.warning(f"  Qwen failed ({e}), falling back to rules")
                return self._synth_rules(scene_objects, manipulated_object, task_description)
        else:
            logger.info("=" * 70)
            logger.info("[Pipeline Step 3/4] SEMANTIC CONSTRAINT SYNTHESIS (rule-based)")
            logger.info("=" * 70)
            logger.info(f"  Manipulated: '{manipulated_object}' | Scene: {[o.label for o in scene_objects]}")
            return self._synth_rules(scene_objects, manipulated_object, task_description)

    def _synth_vlm(self, objs, manip, desc, imgs):
        ctx = SemanticContext(manipulated_object=manip)
        T = SemanticPromptTemplates; sd = desc or "tabletop scene"
        scene_img = imgs[0] if imgs else None

        # Batch ALL queries
        queries=[]; meta=[]
        for idx,o in enumerate(objs):
            for rel in ["above","under","around"]:
                for v in range(self.num_votes):
                    qid=f"s_{idx}_{rel}_v{v}"
                    queries.append({"id":qid,"prompt":T.spatial(sd,manip,o.label,rel)})
                    meta.append(("spatial",idx,o.label,rel,v))
        for idx,o in enumerate(objs):
            for v in range(self.num_votes):
                qid=f"b_{idx}_v{v}"
                queries.append({"id":qid,"prompt":T.behavioral(sd,manip,o.label)})
                meta.append(("behavioral",idx,o.label,"",v))
        for v in range(self.num_votes):
            qid=f"p_v{v}"
            queries.append({"id":qid,"prompt":T.pose(manip)})
            meta.append(("pose",-1,"","",v))

        logger.info(f"  Total queries: {len(queries)}")
        results = self._bridge.batch_query(queries, scene_image=scene_img)
        rm = {r["id"]:r["response"] for r in results}

        # Parse spatial
        logger.info("  Parsing spatial constraints:")
        for idx,o in enumerate(objs):
            for rel in ["above","under","around"]:
                uv = sum(1 for v in range(self.num_votes) if T.parse_spatial(rm.get(f"s_{idx}_{rel}_v{v}","error")))
                if uv > self.num_votes//2:
                    ctx.spatial_constraints.append(SemanticConstraint(o.label,idx,"spatial",rel))
                    logger.info(f"    S_r: '{manip}' {rel} '{o.label}' → UNSAFE ({uv}/{self.num_votes})")

        # Parse behavioral
        logger.info("  Parsing behavioral constraints:")
        for idx,o in enumerate(objs):
            cv = sum(1 for v in range(self.num_votes) if T.parse_behavioral(rm.get(f"b_{idx}_v{v}",""))=="caution")
            if cv > self.num_votes//2:
                ctx.behavioral_constraints.append(SemanticConstraint(o.label,idx,"behavioral",caution_level="caution"))
                logger.info(f"    S_b: near '{o.label}' → caution ({cv}/{self.num_votes})")

        # Parse pose
        pv = sum(1 for v in range(self.num_votes) if T.parse_pose(rm.get(f"p_v{v}",""))=="constrained_rotation")
        ctx.pose_constraint = "constrained_rotation" if pv > self.num_votes//2 else "free_rotation"
        logger.info(f"  S_T: {ctx.pose_constraint} ({pv}/{self.num_votes})")

        # Ensure obstacles always have constraints
        for idx,o in enumerate(objs):
            if "obstacle" in o.label.lower():
                if not any(c.object_idx==idx for c in ctx.spatial_constraints):
                    ctx.spatial_constraints.append(SemanticConstraint(o.label,idx,"spatial","around"))
                if not any(c.object_idx==idx for c in ctx.behavioral_constraints):
                    ctx.behavioral_constraints.append(SemanticConstraint(o.label,idx,"behavioral",caution_level="caution"))

        logger.info(f"  Final: {len(ctx.spatial_constraints)} spatial, "
                     f"{len(ctx.behavioral_constraints)} behavioral, pose={ctx.pose_constraint}")
        return ctx

    def _synth_rules(self, objs, manip, desc):
        ctx = SemanticContext(manipulated_object=manip); ml = manip.lower()
        for idx,o in enumerate(objs):
            ol = o.label.lower()
            if any(k in ml for k in self.LIQUID):
                if any(k in ol for k in self.ELECTRONICS|{"book","paper"}):
                    ctx.spatial_constraints.append(SemanticConstraint(o.label,idx,"spatial","above"))
            if any(k in ml for k in self.FLAMMABLE):
                if any(k in ol for k in {"balloon","paper","cloth"}):
                    ctx.spatial_constraints.append(SemanticConstraint(o.label,idx,"spatial","around"))
            if "obstacle" in ol:
                ctx.spatial_constraints.append(SemanticConstraint(o.label,idx,"spatial","around"))
            if any(k in ol for k in self.FRAGILE|self.ELECTRONICS) or "obstacle" in ol:
                ctx.behavioral_constraints.append(SemanticConstraint(o.label,idx,"behavioral",caution_level="caution"))
        if any(k in ml for k in self.LIQUID|{"knife","sharp"}):
            ctx.pose_constraint = "constrained_rotation"

        logger.info(f"  Spatial ({len(ctx.spatial_constraints)}):")
        for c in ctx.spatial_constraints:
            logger.info(f"    S_r: '{manip}' {c.spatial_relation} '{c.object_label}' → UNSAFE")
        logger.info(f"  Behavioral ({len(ctx.behavioral_constraints)}):")
        for c in ctx.behavioral_constraints:
            logger.info(f"    S_b: near '{c.object_label}' → {c.caution_level}")
        logger.info(f"  Pose: S_T = {ctx.pose_constraint}")
        return ctx


# ═══════════════════════════════════════════════════════════════════════
# 4. Superquadric Geometry
# ═══════════════════════════════════════════════════════════════════════

def sq_implicit(pt, sq):
    p = sq.rotation.T @ (pt - sq.position)
    e1,e2 = max(sq.e1,.05), max(sq.e2,.05); eps=1e-12
    tx = np.abs(p[0]/sq.a1+eps)**(2/e2); ty = np.abs(p[1]/sq.a2+eps)**(2/e2)
    return (tx+ty+eps)**(e2/e1) + np.abs(p[2]/sq.a3+eps)**(2/e1) - 1.0

def sq_grad(pt, sq):
    eps=1e-6; f0=sq_implicit(pt,sq); g=np.zeros(3)
    for i in range(3):
        pp=pt.copy(); pp[i]+=eps; g[i]=(sq_implicit(pp,sq)-f0)/eps
    if np.linalg.norm(g)<1e-10:
        d=pt-sq.position; n=np.linalg.norm(d)
        g = d/n if n>1e-10 else np.array([0.,0.,1.])
    return g


def sq_radial_clearance(pt, sq):
    """Signed radial clearance from pt to the SQ surface along the center ray."""
    p = sq.rotation.T @ (pt - sq.position)
    dist = np.linalg.norm(p)
    if dist < 1e-12:
        return -min(sq.a1, sq.a2, sq.a3)

    e1, e2 = max(sq.e1, 0.05), max(sq.e2, 0.05)
    eps = 1e-12
    tx = np.abs(p[0] / sq.a1 + eps) ** (2.0 / e2)
    ty = np.abs(p[1] / sq.a2 + eps) ** (2.0 / e2)
    g = (tx + ty + eps) ** (e2 / e1) + np.abs(p[2] / sq.a3 + eps) ** (2.0 / e1)
    g = max(float(g), eps)
    surface_scale = g ** (-0.5 * e1)
    return dist * (1.0 - surface_scale)


def sq_margin_value(pt, sq, margin):
    return sq_radial_clearance(pt, sq) - max(float(margin), 0.0)


def sq_margin_grad(pt, sq, margin):
    eps = 1e-6
    g = np.zeros(3, dtype=np.float64)
    for i in range(3):
        pp = pt.copy(); pp[i] += eps
        pm = pt.copy(); pm[i] -= eps
        g[i] = (sq_margin_value(pp, sq, margin) - sq_margin_value(pm, sq, margin)) / (2.0 * eps)
    if np.linalg.norm(g) < 1e-10:
        d = pt - sq.position
        n = np.linalg.norm(d)
        g = d / n if n > 1e-10 else np.array([0.0, 0.0, 1.0])
    return g


# ═══════════════════════════════════════════════════════════════════════
# 5. Spatial CBF Envelope
# ═══════════════════════════════════════════════════════════════════════

def build_cbf_envelope(obj, relation, z_max=1.5, inflation=0.0):
    sq=obj.superquadric
    if relation=="above":
        a3=(z_max-sq.position[2])/2+sq.a3; c=sq.position.copy(); c[2]+=(z_max-sq.position[2])/2
        return SuperquadricParams(sq.a1, sq.a2, max(a3, sq.a3), sq.e1, sq.e2, c, sq.rotation.copy())
    elif relation=="under":
        a3=(sq.position[2]+0.1)/2+sq.a3; c=sq.position.copy(); c[2]-=(sq.position[2]+0.1)/2
        return SuperquadricParams(sq.a1, sq.a2, max(a3, sq.a3), sq.e1, sq.e2, c, sq.rotation.copy())
    elif relation=="around":
        return SuperquadricParams(sq.a1, sq.a2, sq.a3, sq.e1, sq.e2, sq.position.copy(), sq.rotation.copy())
    return SuperquadricParams(sq.a1, sq.a2, sq.a3, sq.e1, sq.e2, sq.position.copy(), sq.rotation.copy())


# ═══════════════════════════════════════════════════════════════════════
# 6. CBF-QP Safety Filter
# ═══════════════════════════════════════════════════════════════════════

class SemanticSafetyFilter:
    """CBF-QP: u_cert = argmin ||u-u_cmd||² + w_rot·L_rot  s.t. CBF constraints"""

    def __init__(self, dt=0.05, alpha_default=1.0, alpha_caution=0.25,
                 safety_margin=0.10, workspace_z_max=1.2,
                 action_pos_scale=None):
        self.dt=dt; self.alpha_default=alpha_default; self.alpha_caution=alpha_caution
        self.safety_margin=safety_margin; self.workspace_z_max=workspace_z_max
        self.semantic_envelopes: List[Tuple[SuperquadricParams,float,float]] = []
        self.collision_envelopes: List[Tuple[SuperquadricParams,float,float]] = []
        self.gripper_center_offset = GRIPPER_SPHERE_CENTER_OFFSET.copy()
        self.pose_constrained=False; self.desired_orientation=None
        # Maps normalized policy action [-1,1] to world-frame position delta [m].
        # Default matches robosuite OSC_POSE output_max=0.05. Override via env
        # CBF_ACTION_POS_SCALE or constructor arg.
        if action_pos_scale is None:
            action_pos_scale = float(os.environ.get("CBF_ACTION_POS_SCALE", 0.05))
        self.action_pos_scale = float(action_pos_scale)
        self._init=False; self._t=0; self.last_trace={}

    def initialize(self, objs, ctx):
        """[Step 4/4] Build CBF envelopes."""
        self.semantic_envelopes=[]; self.collision_envelopes=[]
        logger.info("  Building semantic CBF envelopes:")
        for c in ctx.spatial_constraints:
            if c.object_idx<len(objs):
                o=objs[c.object_idx]
                env=build_cbf_envelope(o, c.spatial_relation, self.workspace_z_max)
                a=self.alpha_default
                for bc in ctx.behavioral_constraints:
                    if bc.object_idx==c.object_idx and bc.caution_level=="caution": a=self.alpha_caution
                self.semantic_envelopes.append((env, a, self.safety_margin))
                logger.info(f"    h_sem[{len(self.semantic_envelopes)-1}]: '{o.label}' "
                             f"{c.spatial_relation} SQ(a=[{env.a1:.3f},{env.a2:.3f},{env.a3:.3f}]) α={a}")

        logger.info("  Building collision CBF envelopes:")
        for o in objs:
            sq=o.superquadric
            a=self.alpha_default
            for bc in ctx.behavioral_constraints:
                if bc.object_label==o.label and bc.caution_level=="caution": a=self.alpha_caution
            self.collision_envelopes.append((sq, a, self.safety_margin))
            logger.info(f"    h_env[{len(self.collision_envelopes)-1}]: '{o.label}' α={a} margin={self.safety_margin:.3f}")

        self.pose_constrained = ctx.pose_constraint=="constrained_rotation"
        self._init=True
        logger.info(f"  Summary: {len(self.semantic_envelopes)} sem, "
                     f"{len(self.collision_envelopes)} col, rot={self.pose_constrained}")

    def initialize_from_precomputed(self, semantic_envelopes,
                                    pose_constrained=False,
                                    collision_envelopes=None):
        """Initialize directly from precomputed superquadrics and explicit margins."""
        self.semantic_envelopes = list(semantic_envelopes)
        self.collision_envelopes = list(collision_envelopes or [])
        self.pose_constrained = pose_constrained
        self.desired_orientation = None
        self._init = True
        logger.info(f"  Summary: {len(self.semantic_envelopes)} sem, "
                     f"{len(self.collision_envelopes)} col, rot={self.pose_constrained}")

    def certify_action(self, u_cmd, ee_pos, ee_quat):
        """u_nominal → u_safe. Called every timestep."""
        if not self._init: return u_cmd.copy()
        self._t += 1
        u = u_cmd.copy(); dp_act = u[:3].copy(); dr = u[3:6].copy()
        grip = u[6] if len(u)>6 else 0.0
        gripper_center = offset_gripper_sphere_center(ee_pos, ee_quat, self.gripper_center_offset)

        # Convert policy action [-1,1] to world-frame position delta [m] so that
        # CBF linear-extrapolation (h(ee+dp)) uses a geometrically correct step.
        scale = self.action_pos_scale
        dp = dp_act * scale

        # Pre-filter per-constraint h (physical space)
        pre_h_sem = [
            float(sq_margin_value(gripper_center, env, margin))
            for (env, a, margin) in self.semantic_envelopes
        ]
        pre_h_col = [
            float(sq_margin_value(gripper_center, env, margin))
            for (env, a, margin) in self.collision_envelopes
        ]
        dp_cmd_act = dp_act.copy()       # original action-space delta
        dp_cmd_phys = dp.copy()           # nominal physical-space delta

        dp, si = self._cbf_sem(gripper_center, dp)
        dp, ci = self._cbf_col(gripper_center, dp)
        ri = False
        if self.pose_constrained:
            dr0=dr.copy(); dr=self._cbf_rot(ee_quat,dr); ri=np.linalg.norm(dr-dr0)>1e-6
        dp, wi = self._cbf_ws(gripper_center, dp)

        # Convert physical delta back to action space and clip to [-1,1].
        dp_act_safe = np.clip(dp / scale, -1.0, 1.0) if scale > 1e-9 else dp
        u[:3]=dp_act_safe; u[3:6]=dr
        if len(u)>6: u[6]=grip

        # Post-filter per-constraint h for diagnostic trace (physical)
        new_center = gripper_center + dp
        post_h_sem = [
            float(sq_margin_value(new_center, env, margin))
            for (env, a, margin) in self.semantic_envelopes
        ]
        post_h_col = [
            float(sq_margin_value(new_center, env, margin))
            for (env, a, margin) in self.collision_envelopes
        ]

        if si or ci or ri or wi:
            parts=[]
            if si: parts.append("semantic")
            if ci: parts.append("collision")
            if ri: parts.append("rotation")
            if wi: parts.append("workspace")
            logger.info(f"[CBF-QP t={self._t}] INTERVENTION: {'+'.join(parts)} "
                         f"ee=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}] "
                         f"sphere_center=[{gripper_center[0]:.3f},{gripper_center[1]:.3f},{gripper_center[2]:.3f}] "
                         f"|Δu|={np.linalg.norm(u[:6]-u_cmd[:6]):.5f}")
        elif self._t%50==0:
            logger.debug(f"[CBF-QP t={self._t}] no intervention")

        self.last_trace = {
            "filter_step": int(self._t),
            "intervened": bool(si or ci or ri or wi),
            "intervention_types": [name for flag, name in (
                (si, "semantic"),
                (ci, "collision"),
                (ri, "rotation"),
                (wi, "workspace"),
            ) if flag],
            "semantic_intervened": bool(si),
            "collision_intervened": bool(ci),
            "rotation_intervened": bool(ri),
            "workspace_intervened": bool(wi),
            "ee_pos": ee_pos.astype(float).tolist(),
            "gripper_center": gripper_center.astype(float).tolist(),
            "gripper_center_offset": self.gripper_center_offset.astype(float).tolist(),
            "ee_quat": ee_quat.astype(float).tolist(),
            "delta_norm": float(np.linalg.norm(u[:6] - u_cmd[:6])),
            "action_pos_scale": float(self.action_pos_scale),
            "dp_cmd_action": dp_cmd_act.astype(float).tolist(),
            "dp_cmd_phys": dp_cmd_phys.astype(float).tolist(),
            "dp_safe_action": np.asarray(dp_act_safe, dtype=float).tolist(),
            "dp_safe_phys": dp.astype(float).tolist(),
            "h_sem_pre": pre_h_sem,
            "h_sem_post": post_h_sem,
            "h_col_pre": pre_h_col,
            "h_col_post": post_h_col,
        }

        # Optional per-step dump to a jsonl file (enable with env var).
        debug_path = os.environ.get("CBF_TRACE_JSONL", "")
        if debug_path:
            try:
                with open(debug_path, "a") as f:
                    f.write(json.dumps(self.last_trace) + "\n")
            except Exception as e:  # pragma: no cover
                logger.warning(f"CBF trace write failed: {e}")

        return u

    def _cbf_sem(self, ee, dp) -> Tuple[np.ndarray,bool]:
        hit=False
        for i,(env,a,margin) in enumerate(self.semantic_envelopes):
            h=max(sq_margin_value(ee, env, margin), -0.5)
            hn=sq_margin_value(ee+dp, env, margin)
            dh=hn-h
            # Linear class-K: ah=a*h on BOTH sides (no floor).
            # h>0: ah>0 -> dh>=-ah (class-K throttle as we approach the boundary).
            # h<0: ah<0 -> dh>=|ah| (forced recovery, h must grow each step).
            ah=a*h
            if dh < -ah:
                g=sq_margin_grad(ee, env, margin); gn=g@g
                if gn>1e-10:
                    v=-ah-dh; c=v*g/gn; dp=dp+c; hit=True
                    logger.info(f"[CBF-QP t={self._t}] h_sem[{i}]: h={h:.4f} Δh={dh:.4f} "
                                 f"-α={-ah:.4f} |c|={np.linalg.norm(c):.5f}")
        return dp, hit

    def _cbf_col(self, ee, dp) -> Tuple[np.ndarray,bool]:
        hit=False
        for i,(env,a,margin) in enumerate(self.collision_envelopes):
            h=max(sq_margin_value(ee, env, margin), -0.5)
            hn=sq_margin_value(ee+dp, env, margin)
            dh=hn-h
            ah=a*h
            if dh < -ah:
                g=sq_margin_grad(ee, env, margin); gn=g@g
                if gn>1e-10:
                    v=-ah-dh; c=v*g/gn; dp=dp+c; hit=True
                    logger.info(f"[CBF-QP t={self._t}] h_env[{i}]: h={h:.4f} Δh={dh:.4f} "
                                 f"-α={-ah:.4f} |c|={np.linalg.norm(c):.5f}")
        return dp, hit

    def _cbf_rot(self, eq, dr):
        if self.desired_orientation is None:
            self.desired_orientation=eq.copy()
            logger.info(f"[CBF-QP] Desired orientation set: {eq}")
        try:
            Rd=Rotation.from_quat(self.desired_orientation).as_matrix()
            Rc=Rotation.from_quat(eq).as_matrix()
            ang=Rotation.from_matrix(Rd@Rc.T).magnitude()
            m=np.linalg.norm(dr)
            if m>0.05:
                dr=dr*(0.05/m)
                logger.info(f"[CBF-QP t={self._t}] rot clamped: {m:.4f}→0.05")
            if ang>0.3:
                s=max(0.1,1-(ang-0.3)/0.3); dr*=s
                logger.info(f"[CBF-QP t={self._t}] rot drift: {np.degrees(ang):.1f}° s={s:.3f}")
        except: pass
        return dr

    def _cbf_ws(self, ee, dp) -> Tuple[np.ndarray,bool]:
        d0=dp.copy(); en=ee+dp
        if en[2]<0: dp[2]=max(dp[2],-ee[2]+0.01)
        if en[2]>self.workspace_z_max: dp[2]=min(dp[2],self.workspace_z_max-ee[2]-0.01)
        for i in range(2):
            if abs(en[i])>0.8:
                dp[i]=np.clip(dp[i],-0.8-ee[i]+0.01,0.8-ee[i]-0.01)
        hit=np.linalg.norm(dp-d0)>1e-8
        if hit:
            ax=["xyz"[j] for j in range(3) if abs(dp[j]-d0[j])>1e-8]
            logger.info(f"[CBF-QP t={self._t}] workspace clamped: {','.join(ax)}")
        return dp, hit

    def reset(self):
        self.desired_orientation=None; self._t=0; self.last_trace={}


# ═══════════════════════════════════════════════════════════════════════
# 7. Full Pipeline
# ═══════════════════════════════════════════════════════════════════════

class SemanticCBFPipeline:
    """Setup (once/episode): [1/4]→[2/4]→[3/4]→[4/4]
       Runtime (every step):  OpenVLA u_nominal → certify → u_safe"""

    def __init__(self, env=None, use_vlm=False, vlm_model="qwen2.5-vl-7b",
                 vlm_conda_env="qwen", vlm_device="auto", vlm_load_in_4bit=False,
                 vlm_worker_script="qwen_vlm_worker.py", vlm_num_votes=3,
                 dt=0.05, alpha_default=1.0, alpha_caution=0.25,
                 safety_margin=0.10, workspace_z_max=1.2):
        self.perception = PerceptionModule(env=env)
        self.constraint_synth = SemanticConstraintSynthesizer(
            use_vlm=use_vlm, vlm_model=vlm_model, vlm_conda_env=vlm_conda_env,
            vlm_device=vlm_device, vlm_load_in_4bit=vlm_load_in_4bit,
            vlm_worker_script=vlm_worker_script, num_votes=vlm_num_votes)
        self.safety_filter = SemanticSafetyFilter(
            dt=dt, alpha_default=alpha_default, alpha_caution=alpha_caution,
            safety_margin=safety_margin, workspace_z_max=workspace_z_max)
        self.scene_objects=[]; self.semantic_context=None; self._built=False

    @staticmethod
    def _resolve_cbf_json_path(cbf_json_path):
        path = os.path.abspath(os.path.expanduser(cbf_json_path))
        if os.path.isdir(path):
            path = os.path.join(path, "cbf_params.json")
        return path

    def setup_from_cbf_json(self, cbf_json_path, task_description="", manipulated_object=""):
        """Load precomputed superquadric CBFs exported by cbf_superquadric.py."""
        path = self._resolve_cbf_json_path(cbf_json_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Precomputed CBF file not found: {path}")

        logger.info(""); logger.info("#"*70)
        logger.info("#  SEMANTIC CBF SAFETY FILTER — PRECOMPUTED LOAD")
        logger.info(f"#  Task: {task_description}")
        logger.info(f"#  Source: {path}")
        logger.info("#"*70)

        with open(path) as f:
            cbf_data = json.load(f)

        constraints = cbf_data.get("constraints", [])
        behavioral = cbf_data.get("behavioral", {})
        pose = cbf_data.get("pose", {})

        alpha = behavioral.get("alpha_scale", self.safety_filter.alpha_default)
        try:
            alpha = float(alpha)
        except (TypeError, ValueError):
            alpha = self.safety_filter.alpha_default
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = self.safety_filter.alpha_default

        gripper_cfg = cbf_data.get("gripper", {})
        center_offset = np.asarray(
            gripper_cfg.get("center_offset", GRIPPER_SPHERE_CENTER_OFFSET),
            dtype=np.float64,
        )
        if center_offset.shape != (3,) or not np.all(np.isfinite(center_offset)):
            center_offset = GRIPPER_SPHERE_CENTER_OFFSET.copy()
        self.safety_filter.gripper_center_offset = center_offset.copy()

        gripper_radius = gripper_cfg.get("radius", self.safety_filter.safety_margin)
        try:
            gripper_radius = float(gripper_radius)
        except (TypeError, ValueError):
            gripper_radius = self.safety_filter.safety_margin
        if not np.isfinite(gripper_radius) or gripper_radius < 0:
            gripper_radius = self.safety_filter.safety_margin

        semantic_envelopes = []
        collision_envelopes = []
        objects_by_name = {}
        self.scene_objects = []
        self.semantic_context = SemanticContext(
            manipulated_object=manipulated_object or "object",
            pose_constraint="constrained_rotation" if pose.get("rotation_lock", False) else "free_rotation",
        )

        for idx, entry in enumerate(constraints):
            params = entry.get("params", {})
            center = np.asarray(params.get("center", []), dtype=np.float64)
            scales = np.asarray(params.get("scales", []), dtype=np.float64)
            if center.shape != (3,) or scales.shape != (3,):
                raise ValueError(f"Invalid superquadric params at constraint {idx}: {entry}")

            sq = SuperquadricParams(
                a1=float(scales[0]),
                a2=float(scales[1]),
                a3=float(scales[2]),
                e1=float(params.get("epsilon1", 1.0)),
                e2=float(params.get("epsilon2", 1.0)),
                position=center.copy(),
                rotation=np.eye(3),
            )
            semantic_envelopes.append((sq, alpha, gripper_radius))

            obj_name = str(entry.get("object", f"object_{idx}"))
            relation = str(entry.get("relationship", "around"))
            if obj_name not in objects_by_name:
                obj_idx = len(self.scene_objects)
                objects_by_name[obj_name] = obj_idx
                self.scene_objects.append(SemanticObject(
                    label=obj_name,
                    position=center.copy(),
                    point_cloud=np.empty((0, 3), dtype=np.float64),
                    superquadric=sq,
                    bbox_min=center - scales,
                    bbox_max=center + scales,
                ))
                collision_envelopes.append((sq, alpha, gripper_radius))
                logger.info(f"  Loaded h_env[{len(collision_envelopes)-1}]: '{obj_name}' "
                            f"SQ(a=[{sq.a1:.3f},{sq.a2:.3f},{sq.a3:.3f}] "
                            f"e=[{sq.e1:.2f},{sq.e2:.2f}]) α={alpha:.3f} margin={gripper_radius:.3f}")
                if behavioral.get("caution", False):
                    self.semantic_context.behavioral_constraints.append(
                        SemanticConstraint(obj_name, obj_idx, "behavioral", caution_level="caution")
                    )
            self.semantic_context.spatial_constraints.append(
                SemanticConstraint(obj_name, objects_by_name[obj_name], "spatial", spatial_relation=relation)
            )
            logger.info(f"  Loaded h_sem[{idx}]: '{obj_name}' {relation} "
                        f"SQ(a=[{sq.a1:.3f},{sq.a2:.3f},{sq.a3:.3f}] "
                        f"e=[{sq.e1:.2f},{sq.e2:.2f}]) α={alpha:.3f}")

        self.safety_filter.initialize_from_precomputed(
            semantic_envelopes=semantic_envelopes,
            pose_constrained=bool(pose.get("rotation_lock", False)),
            collision_envelopes=collision_envelopes,
        )
        self._built = True

        logger.info(""); logger.info("#"*70)
        logger.info("#  PRECOMPUTED LOAD COMPLETE — Safety filter active")
        logger.info(f"#  Semantic CBFs: {len(self.safety_filter.semantic_envelopes)}")
        logger.info(f"#  Collision CBFs: {len(self.safety_filter.collision_envelopes)}")
        logger.info(f"#  Rotation constrained: {self.safety_filter.pose_constrained}")
        logger.info("#"*70); logger.info("")

    def setup_scene(self, obs, task_description="", manipulated_object="",
                    camera_images=None, depth_images=None):
        logger.info(""); logger.info("#"*70)
        logger.info("#  SEMANTIC CBF SAFETY FILTER — SCENE SETUP")
        logger.info(f"#  Task: {task_description}"); logger.info("#"*70)

        # [1/4] Perception
        self.scene_objects = self.perception.segment_and_build_map(obs,camera_images,depth_images)

        # [2/4] 3D Map
        logger.info("="*70)
        logger.info("[Pipeline Step 2/4] 3D MAP CONSTRUCTION (superquadric fitting)")
        logger.info("="*70)
        logger.info(f"  Objects: {len(self.scene_objects)}")
        for i,o in enumerate(self.scene_objects):
            sq=o.superquadric
            logger.info(f"  [{i}] '{o.label}': SQ(a=[{sq.a1:.3f},{sq.a2:.3f},{sq.a3:.3f}] "
                         f"e=[{sq.e1:.2f},{sq.e2:.2f}] pos=[{sq.position[0]:.3f},{sq.position[1]:.3f},{sq.position[2]:.3f}])")
        if not manipulated_object:
            manipulated_object = self._infer_obj(task_description)
            logger.info(f"  Inferred manipulated object: '{manipulated_object}'")
        else:
            logger.info(f"  Manipulated object: '{manipulated_object}'")

        # [3/4] VLM constraints
        self.semantic_context = self.constraint_synth.synthesize_constraints(
            self.scene_objects, manipulated_object, task_description, camera_images)

        # [4/4] CBF construction
        logger.info("="*70)
        logger.info("[Pipeline Step 4/4] CBF-QP SAFETY FILTER CONSTRUCTION")
        logger.info("="*70)
        self.safety_filter.initialize(self.scene_objects, self.semantic_context)
        self._built=True
        logger.info(""); logger.info("#"*70)
        logger.info("#  SCENE SETUP COMPLETE — Safety filter active")
        logger.info(f"#  Semantic CBFs: {len(self.safety_filter.semantic_envelopes)}")
        logger.info(f"#  Collision CBFs: {len(self.safety_filter.collision_envelopes)}")
        logger.info(f"#  Rotation constrained: {self.safety_filter.pose_constrained}")
        logger.info("#"*70); logger.info("")

    def filter_action(self, action, obs):
        """u_nominal (from OpenVLA-OFT) → u_safe (CBF certified)"""
        if not self._built: return action.copy()
        ee_pos = np.array(obs.get("robot0_eef_pos",np.zeros(3)), dtype=np.float64)
        ee_quat = np.array(obs.get("robot0_eef_quat",[0,0,0,1]), dtype=np.float64)
        return self.safety_filter.certify_action(action, ee_pos, ee_quat)

    def get_last_trace(self):
        return dict(self.safety_filter.last_trace)

    def reset(self): self.safety_filter.reset(); self._built=False

    @staticmethod
    def _infer_obj(desc):
        d=desc.lower()
        for p in ["pick up the ","pick the ","move the ","push the ",
                   "place the ","put the ","grasp the ","lift the "]:
            if p in d: return " ".join(d.split(p,1)[1].split()[:3]).rstrip(".,;:")
        return "object"
