"""Novel-obstacle SafeLIBERO envs: each training obstacle is swapped for a similar object never seen in training.

    milk carton        -> orange juice carton / popcorn box         (SafeLIBERO obstacle assets)
    wine bottle        -> ketchup / salad dressing / bbq sauce bottle
    white storage box  -> basket
    red coffee mug     -> white and yellow mug  (LIBERO white_yellow_mug mesh, registered here as an obstacle)
    yellow book        -> black book            (LIBERO black_book mesh, registered here as an obstacle)
    moka pot           -> (no similar asset; left unchanged, such episodes are reported separately)

The official BDDL is rewritten (object types and instance names) and written to a temp file; object order is
unchanged, so the official init states (one flat qpos per episode) still place the swapped object where the
original stood. sl_env.make_env(..., novel=True) uses this.
"""
import os
import re
import shutil
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# Candidates per seen obstacle, most similar first. A scene gets the first candidate whose object type is not
# already in its BDDL (SafeLIBERO-Object/Long use basket, ketchup, orange juice, mugs ... as task objects);
# if every candidate is taken the original obstacle stays and the episode is reported as not swapped.
CANDIDATES = {"milk": ["orange_juice", "popcorn"], "wine_bottle": ["ketchup", "new_salad_dressing", "bbq_sauce"],
              "white_storage_box": ["basket"], "red_coffee_mug": ["white_yellow_mug"], "yellow_book": ["black_book"]}
ALIASES = {"new_salad_dressing": {"salad_dressing", "new_salad_dressing"}}   # same bottle under two type names
NOUN = {"orange_juice_obstacle_1": "orange juice carton", "popcorn_obstacle_1": "popcorn box",
        "ketchup_obstacle_1": "ketchup bottle", "new_salad_dressing_obstacle_1": "salad dressing bottle",
        "bbq_sauce_obstacle_1": "bbq sauce bottle", "basket_obstacle_1": "basket",
        "white_yellow_mug_obstacle_1": "white and yellow mug", "black_book_obstacle_1": "black book"}
SEEN_OF = {"orange_juice_obstacle_1": "milk_obstacle_1", "popcorn_obstacle_1": "milk_obstacle_1",
           "ketchup_obstacle_1": "wine_bottle_obstacle_1", "new_salad_dressing_obstacle_1": "wine_bottle_obstacle_1",
           "bbq_sauce_obstacle_1": "wine_bottle_obstacle_1", "basket_obstacle_1": "white_storage_box_obstacle_1",
           "white_yellow_mug_obstacle_1": "red_coffee_mug_obstacle_1", "black_book_obstacle_1": "yellow_book_obstacle_1"}
NOVEL_OBSTACLES = list(NOUN)
_NEW_ASSETS = {"white_yellow_mug": "turbosquid_objects/white_yellow_mug", "black_book": "turbosquid_objects/black_book"}
_registered = False


def _asset_root():
    from libero.libero import get_libero_path
    return os.path.join(get_libero_path("benchmark_root"), "assets")


def register():
    """Register WhiteYellowMugObstacle and BlackBookObstacle (copies of the LIBERO meshes, obstacle xml layout)."""
    global _registered
    if _registered:
        return
    from libero.libero.envs.base_object import register_object
    from libero.libero.envs.objects.obstacle_objects import ObstacleObject
    from robosuite.models.objects import MujocoXMLObject

    root = os.path.join(HERE, "novel_assets")
    for name, src in _NEW_ASSETS.items():
        dst = os.path.join(root, f"{name}_obstacle")
        if not os.path.exists(os.path.join(dst, f"{name}_obstacle.xml")):
            tmp = dst + f".tmp{os.getpid()}"
            shutil.copytree(os.path.join(_asset_root(), src), tmp)
            os.rename(os.path.join(tmp, f"{name}.xml"), os.path.join(tmp, f"{name}_obstacle.xml"))
            try:
                os.rename(tmp, dst)
            except OSError:          # another worker got there first
                shutil.rmtree(tmp, ignore_errors=True)

    def make(cls_name, obj):
        def __init__(self, name=f"{obj}_obstacle", obj_name=f"{obj}_obstacle"):
            MujocoXMLObject.__init__(self, os.path.join(root, f"{obj}_obstacle", f"{obj}_obstacle.xml"), name=name,
                                     joints=[dict(type="free", damping="0.0005")], obj_type="all",
                                     duplicate_collision_geoms=False)
            self.category_name = f"{obj}_obstacle"
            import numpy as np
            self.rotation = (np.pi / 2, np.pi / 2)
            self.rotation_axis = "x"
            self.object_properties = {"vis_site_names": {}}
        return type(cls_name, (ObstacleObject,), {"__init__": __init__})

    register_object(make("WhiteYellowMugObstacle", "white_yellow_mug"))
    register_object(make("BlackBookObstacle", "black_book"))
    _registered = True


def scene_types(txt):
    """Non-obstacle object types declared in a BDDL (the scene's task objects and furniture)."""
    objs = re.search(r"\(:objects(.*?)\)", txt, re.S)
    return {t for t in re.findall(r"- (\w+)", objs.group(1) if objs else "") if not t.endswith("_obstacle")}


def swaps_for(txt):
    """{seen base type (incl. _small) -> novel base type} for this BDDL."""
    present = scene_types(txt)
    out = {}
    for seen, cands in CANDIDATES.items():
        for c in cands:
            if not (ALIASES.get(c, {c}) & present):
                out[seen] = c
                out[seen + "_small"] = c
                break
    return out


def novel_bddl(path):
    """Write a copy of the BDDL with every swappable obstacle replaced; returns the new path."""
    txt = open(path).read()
    sw = swaps_for(txt)
    for old in sorted(sw, key=len, reverse=True):          # *_small_* before the plain name
        new = sw[old]
        txt = re.sub(rf"\b{old}_obstacle_1 - {old}_obstacle\b", f"{new}_obstacle_1 - {new}_obstacle", txt)
        txt = re.sub(rf"\b{old}_obstacle_1\b", f"{new}_obstacle_1", txt)
    d = tempfile.mkdtemp(prefix="novel_bddl_")
    out = os.path.join(d, os.path.basename(path))
    open(out, "w").write(txt)
    return out


def enable():
    """Switch this process's sl_env to the novel-obstacle envs (call in every worker, after sl_env.set_suite).
    Adds the novel names to the obstacle lists (collision checks, active obstacles) and makes make_env load the
    rewritten BDDL with the official init states."""
    import sl_env
    register()
    for n in NOVEL_OBSTACLES:
        if n not in sl_env.OBSTACLES_ALL:
            sl_env.OBSTACLES_ALL.append(n)
    sl_env.NOUN.update(NOUN)
    import texts                      # eval naming: "true" names the novel noun, "held" uses held wordings of it
    texts.NOUN_TRAIN.update(NOUN)
    texts.NOUN_HELD.update(NOUN)
    if getattr(sl_env, "_novel", False):
        return

    def make_env(level, task_id, res=128):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from robosuite.utils.errors import RandomizationError
        suite = benchmark.get_benchmark_dict()[sl_env.SUITE](safety_level=level)
        task = suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        nb = novel_bddl(bddl)
        for attempt in range(10):
            try:
                env = OffScreenRenderEnv(bddl_file_name=nb, camera_names=["agentview", "robot0_eye_in_hand"],
                                         camera_heights=res, camera_widths=res, hard_reset=False)
                break
            except RandomizationError:
                if attempt == 9:
                    raise
        env.seed(sl_env.ENV_SEED)
        env.ooi = sl_env.objects_of_interest(nb)
        return env, task.language, suite.get_task_init_states(task_id)

    sl_env.make_env = make_env
    sl_env._novel = True
