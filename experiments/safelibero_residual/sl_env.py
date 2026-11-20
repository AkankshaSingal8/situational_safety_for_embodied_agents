"""SafeLIBERO-Spatial env helpers shared by eval, scene generation and the oracle.

Conventions match vlm_pipeline/run_safelibero_openvla_oft_eval.py where it matters for
comparability: env.seed(0), official init states, 10 settle steps with the dummy action,
300 max steps, collision = active obstacle displaced by more than 1 mm (sum of |dx|).
Images are used raw (no 180-degree flip) because the raw LIBERO hdf5 demos are raw too.
"""
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

# Canonical obstacle types. SafeLIBERO-Long uses "<type>_small_obstacle_1" variants of milk, moka pot and
# wine bottle; canon() maps them onto these six, so texts, residual rows and naming are suite independent.
OBSTACLES = ["moka_pot_obstacle_1", "white_storage_box_obstacle_1", "milk_obstacle_1",
             "wine_bottle_obstacle_1", "red_coffee_mug_obstacle_1", "yellow_book_obstacle_1"]
OBSTACLES_ALL = OBSTACLES + [n.replace("_obstacle_1", "_small_obstacle_1") for n in OBSTACLES]
NOUN = {"moka_pot_obstacle_1": "moka pot", "white_storage_box_obstacle_1": "white storage box",
        "milk_obstacle_1": "milk carton", "wine_bottle_obstacle_1": "wine bottle",
        "red_coffee_mug_obstacle_1": "red coffee mug", "yellow_book_obstacle_1": "yellow book"}
SUITES = ["safelibero_spatial", "safelibero_object", "safelibero_goal", "safelibero_long"]
SUITE = "safelibero_spatial"
MAX_STEPS_BY_SUITE = {"safelibero_long": 550}   # paper horizons; 300 for the other three
MAX_STEPS = 300
SETTLE = 10
ENV_SEED = 0  # eval_sl --env_seed / --settle override these per worker (pi0.5 driver: seed 7, 20 settle steps)
DUMMY = [0.0] * 6 + [-1.0]


def set_suite(name):
    """Select the SafeLIBERO suite for this process (call in every worker before make_env)."""
    global SUITE, MAX_STEPS
    assert name in SUITES, name
    SUITE = name
    MAX_STEPS = MAX_STEPS_BY_SUITE.get(name, 300)


def canon(name):
    """Scene obstacle name -> canonical type name (identity for non-small variants)."""
    return name.replace("_small_obstacle_1", "_obstacle_1")


def make_env(level, task_id, res=128):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()[SUITE](safety_level=level)
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    from robosuite.utils.errors import RandomizationError
    for attempt in range(10):  # the constructor's random placement occasionally fails; init states override it
        try:
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=["agentview", "robot0_eye_in_hand"],
                                     camera_heights=res, camera_widths=res, hard_reset=False)
            break
        except RandomizationError:
            if attempt == 9:
                raise
    env.seed(ENV_SEED)
    env.ooi = objects_of_interest(bddl)
    return env, task.language, suite.get_task_init_states(task_id)


def objects_of_interest(bddl_path):
    """(:obj_of_interest ...) of a BDDL file: the manipulated object first, then targets/regions."""
    import re
    m = re.search(r"\(:obj_of_interest(.*?)\)", open(bddl_path).read(), re.S)
    return m.group(1).split() if m else []


def scene_name(obs, name):
    """The scene's own name for an obstacle given by canonical or scene name (handles '_small_'), or None."""
    for n in (name, name.replace("_obstacle_1", "_small_obstacle_1"), canon(name)):
        if n + "_pos" in obs:
            return n
    return None


def proprio(obs):
    j = np.arctan2(obs["robot0_joint_pos_sin"], obs["robot0_joint_pos_cos"])
    return np.concatenate([j, obs["robot0_gripper_qpos"]]).astype(np.float32)


def on_table(p):
    """In the workspace: the reference drivers' test (run_safelibero_pi05_eval.py: z > 0, |x|, |y| < 0.5).
    Parked objects sit 10-20 m away in xy. The earlier z > 0.5 only held for the Spatial/Goal table
    (z ~ 0.9); Object is a floor scene (z ~ 0.07) and Long's table is at z ~ 0.45."""
    # z > -0.02, not > 0: GSO models (gso_objects.py) have their origin at the base, so on the Object floor
    # (z = 0) they rest at z ~ 0; every other object's origin is at least 5 cm above its support
    return p[2] > -0.02 and abs(p[0]) < 0.5 and abs(p[1]) < 0.5


def active_obstacles(obs):
    """Scene names (possibly '_small_' variants) of obstacles standing in the workspace."""
    return [n for n in OBSTACLES_ALL if n + "_pos" in obs and on_table(obs[n + "_pos"])]


def scene_obstacles(obs):
    """Scene names of every obstacle object this env defines (active or parked)."""
    return [n for n in OBSTACLES_ALL if n + "_pos" in obs]


def body_tree_ids(sim, root_name):
    """Bodies of one object: robosuite names them '<object>_main', '<object>_...'."""
    m = sim.model
    return [i for i in range(m.nbody) if m.body_id2name(i) and m.body_id2name(i).startswith(root_name)]


def object_aabb(sim, name):
    """World AABB of all geoms of object `name`: exact for boxes and meshes, bounding sphere otherwise."""
    m, d = sim.model, sim.data
    ids = set(body_tree_ids(sim, name))
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in ids:
            continue
        c, R = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
        if m.geom_type[g] == 6:  # box
            half = np.abs(R) @ m.geom_size[g]
            glo, ghi = c - half, c + half
        elif m.geom_type[g] == 7:  # mesh: transform its vertices
            mid = m.geom_dataid[g]
            a = m.mesh_vertadr[mid]
            w = c + m.mesh_vert[a:a + m.mesh_vertnum[mid]] @ R.T
            glo, ghi = w.min(0), w.max(0)
        else:
            glo, ghi = c - m.geom_rbound[g], c + m.geom_rbound[g]
        lo, hi = np.minimum(lo, glo), np.maximum(hi, ghi)
    return lo, hi


def support_of(sim, obs, name):
    """'box_base_1' if the obstacle stands on the platform (level II), else None."""
    if "box_base_1_pos" not in obs or not on_table(obs["box_base_1_pos"]) or not on_table(obs[name + "_pos"]):
        return None
    blo, bhi = object_aabb(sim, "box_base_1")
    p = obs[name + "_pos"]
    if blo[0] - 0.03 < p[0] < bhi[0] + 0.03 and blo[1] - 0.03 < p[1] < bhi[1] + 0.03:
        return "box_base_1"
    return None


def obstacle_free_joint(sim, name):
    return name + "_joint0"


def move_object(env, name, xyz, quat=None):
    """Teleport a free-joint object (used to clear obstacles or place new ones)."""
    sim = env.sim
    jn = obstacle_free_joint(sim, name)
    addr = sim.model.get_joint_qpos_addr(jn)
    q = sim.data.qpos.copy()
    q[addr[0]:addr[0] + 3] = xyz
    if quat is not None:
        q[addr[0] + 3:addr[0] + 7] = quat
    sim.data.qpos[:] = q
    vaddr = sim.model.get_joint_qvel_addr(jn)
    sim.data.qvel[vaddr[0]:vaddr[1]] = 0
    sim.forward()


def settle(env, n=None):
    n = SETTLE if n is None else n
    obs = None
    for _ in range(n):
        obs, _, _, _ = env.step(DUMMY)
    return obs


def collided(obs, init_pos, names):
    return [n for n in names if np.sum(np.abs(obs[n + "_pos"] - init_pos[n])) > 0.001]
