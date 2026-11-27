"""Google Scanned Objects (MuJoCo models, kevinzakka/mujoco_scanned_objects) as SafeLIBERO obstacles.

    python gso_objects.py index      # scan the tar once: sizes, nouns, train / held-out split -> gso_index.json
    python gso_objects.py rescale    # size-rejected TRAIN models rescaled into the limits -> gso_index_rs.json

Assets: /data/asingal2/assets/gso_models.tar (1030 models; git archive of the repo). Jobs unpack it once per node
into $GSO_LOCAL (default /tmp/gso_<user>), because NFS is too slow for 36k small files.

One obstacle slot of the official BDDL (the first <type>_obstacle_1 it declares) is rewritten to
"gso_obstacle_1 - gso_obstacle"; GsoObstacle loads whichever model was selected with select(name) before the env
is built. Object order is unchanged, so official init states (flat qpos) still load; scenes.py then clears the
other obstacles and places the GSO object upright at a sampled spot.

Split (fixed, hash of the name): ~85% train, ~15% held out (never in training data; used for the novel-object
test, together with the SafeLIBERO look-alikes of novel_objects.py).
"""
import hashlib
import json
import os
import re
import tarfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAR = "/data/asingal2/assets/gso_models.tar"
INDEX = os.path.join(HERE, "gso_index.json")
NAME = "gso_obstacle_1"
_CUR = {"name": None}

# size limits for a tabletop obstacle (metres): a few cm to about a shoebox
MIN_H, MAX_H = 0.025, 0.35   # v2 (r2 data); v1 was 0.035-0.30 (gso_index_v1_487.json)
MIN_W, MAX_W = 0.025, 0.28   # v1 0.035-0.24

# keyword -> category noun (first match wins; order = specific before generic)
CATS = [
    (r"cartridge|toner", "ink cartridge"), (r"coffee_?maker", "coffee maker"), (r"kettle", "kettle"), (r"teapot", "teapot"),
    (r"skillet|frypan|frying", "pan"), (r"mug", "mug"), (r"cup\b|_cup_|cups", "cup"), (r"bowl", "bowl"),
    (r"plate\b|plates|dish\b|dish_", "dish"), (r"basket", "basket"), (r"bucket|pail", "bucket"),
    (r"mouse\b|mouse_", "computer mouse"), (r"keyboard", "keyboard"), (r"headphone|earbud", "headphones"),
    (r"camera", "camera"), (r"phone", "phone"), (r"lamp", "lamp"), (r"candle", "candle"), (r"clock", "clock"),
    (r"speaker", "speaker"), (r"scissors", "scissors"), (r"tape", "tape roll"), (r"sponge", "sponge"), (r"brush", "brush"),
    (r"spatula|utensil|spoon|fork|knife|whisk|ladle", "utensil"), (r"boot", "boot"), (r"sandal", "sandal"),
    (r"slipper", "slipper"),
    (r"loafer|moc\b|moccasin|flats|tieks|oxford|wingtip|chukka|shoe|sneaker|runner|reebok|asics|nike|adidas|adizero|vans_|puma|d_rose|womens_|mens_|kids_", "shoe"),
    (r"hat\b|cap\b", "hat"), (r"backpack|purse|cooler|lunch|bag\b", "bag"), (r"towel|sheet|cloth|bib", "towel"),
    (r"flower_pot|garden_pot|plant|planter", "plant pot"), (r"\bpot\b|^pot_|cooking_pot|saucepan", "pot"),
    (r"vitamin|protein|supplement|capsule|tablet|coq10|omega|probiotic|whey|softgel", "supplement bottle"),
    (r"bottle|bottes|spray|lotion|shampoo|conditioner|syrup|cleanser", "bottle"), (r"jar\b|jar_", "jar"),
    (r"cream|balm|moisturizer|serum", "cream jar"), (r"\bcan\b|_can_|soda|pepsi|cola", "can"), (r"book", "book"),
    (r"pencil|\bpen\b|marker|crayon|crayola", "pencil box"),
    (r"lego|duplo|blocks|block|puzzle|toy|figure|robot|train|\bcar\b|truck|helicopter|dinosaur|animal|puppy|dog|\bcat\b|bear|doll|dollhouse|game|board|sorting|stacking|roller|racer|lion|tricycle|nintendo|vegetable_set|steak_set", "toy"),
    (r"organizer|sorter|tray|caddy|holder|rack|bin\b|cube|container", "organizer"),
    (r"cereal|crackers|cookies|snack|food|meal|box|pack|count|oz|ounce|tea\b|tampons|pads", "box"),
]


def category(name):
    s = name.lower()
    for pat, noun in CATS:
        if re.search(pat, s):
            return noun
    return "object"


def specific_noun(name, cat):
    """Readable name: drop size codes, ids and counts; keep at most 4 words; ensure the category word appears."""
    words = []
    for w in name.split("_"):
        if not w or re.search(r"\d", w) or (len(w) > 6 and re.search(r"[a-z][A-Z]|[A-Z]{2}[a-z]", w) and not w.isupper()):
            continue
        words.append(w.lower())
    words = [w for w in words if w not in {"x", "in", "the", "and", "with", "of", "for", "by", "oz", "fl", "count", "pack"}]
    s = " ".join(words[-4:]) if words else cat
    if cat != "object" and cat.split()[-1] not in s:
        s = f"{s} {cat}" if len(s.split()) < 3 else cat
    return s


def split_of(name):
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100
    return "test" if h < 15 else "train"


def obj_bounds(fobj):
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    buf = []
    for line in fobj:
        if line.startswith(b"v "):
            buf.append(line[2:].split()[:3])
            if len(buf) >= 20000:
                v = np.array(buf, float); lo, hi = np.minimum(lo, v.min(0)), np.maximum(hi, v.max(0)); buf = []
    if buf:
        v = np.array(buf, float); lo, hi = np.minimum(lo, v.min(0)), np.maximum(hi, v.max(0))
    return lo, hi


def build_index():
    out = {}
    ncoll = {}
    with tarfile.open(TAR) as tf:
        for m in tf:
            p = m.name.split("/")
            if len(p) != 3 or not m.isfile():
                continue
            name, fn = p[1], p[2]
            if fn.startswith("model_collision_"):
                ncoll[name] = ncoll.get(name, 0) + 1
            elif fn == "model.obj":
                lo, hi = obj_bounds(tf.extractfile(m))
                out[name] = dict(lo=lo.round(4).tolist(), hi=hi.round(4).tolist())
    keep = 0
    for name, d in out.items():
        ext = np.array(d["hi"]) - np.array(d["lo"])
        d["n_coll"] = ncoll.get(name, 0)
        d["size"] = ext.round(3).tolist()
        d["category"] = category(name)
        d["noun"] = specific_noun(name, d["category"])
        d["split"] = split_of(name)
        w = max(ext[0], ext[1])
        d["ok"] = bool(MIN_H <= ext[2] <= MAX_H and MIN_W <= w <= MAX_W and min(ext[0], ext[1]) >= 0.02 and d["n_coll"] > 0)
        keep += d["ok"]
    json.dump(out, open(INDEX, "w"), indent=0)
    cats = {}
    for d in out.values():
        if d["ok"]:
            cats[d["category"]] = cats.get(d["category"], 0) + 1
    print(f"{len(out)} models, {keep} within size limits "
          f"(train {sum(d['ok'] and d['split'] == 'train' for d in out.values())}, "
          f"test {sum(d['ok'] and d['split'] == 'test' for d in out.values())})")
    print("categories:", dict(sorted(cats.items(), key=lambda x: -x[1])))


def renoun():
    """Recompute category / noun / ok from the stored sizes (after editing CATS or the limits), no re-scan."""
    out = load_index()
    for name, d in out.items():
        ext = np.array(d["size"])
        d["category"] = category(name)
        d["noun"] = specific_noun(name, d["category"])
        w = max(ext[0], ext[1])
        d["ok"] = bool(MIN_H <= ext[2] <= MAX_H and MIN_W <= w <= MAX_W and min(ext[0], ext[1]) >= 0.02 and d["n_coll"] > 0)
    json.dump(out, open(INDEX, "w"), indent=0)


NAMES = os.path.join(HERE, "gso_names.json")   # audited {model: {noun, category}} (from runs_v2/gso_sheets by eye)


def load_index():
    """The index, with noun / category replaced by the audited names where gso_names.json has them."""
    idx = json.load(open(INDEX))
    if os.path.exists(NAMES):
        for n, d in json.load(open(NAMES)).items():
            if n in idx:
                idx[n].update(noun=d["noun"], category=d["category"])
    if os.environ.get("GSO_RESCALED") and os.path.exists(INDEX_RS):
        for n, d in json.load(open(INDEX_RS)).items():
            b = idx[d["base"]]
            idx[n] = dict(d, noun=b["noun"], category=b["category"])
    return idx


INDEX_RS = os.path.join(HERE, "gso_index_rs.json")
RS = "__rs"   # name suffix of a rescaled variant


def rescale_index():
    """Training-split models the size filter rejected, uniformly rescaled into the limits -> gso_index_rs.json.
    Only split == "train" models (held-out / test models are never touched) with collision meshes, and only when
    one factor puts height, width and thickness all inside the limits. The factor is drawn log-uniformly inside
    the feasible range from a hash of the name, so it is fixed. Used only when GSO_RESCALED=1 (load_index)."""
    idx = json.load(open(INDEX))
    out = {}
    for name, d in idx.items():
        if d["ok"] or d["split"] != "train" or d["n_coll"] == 0:
            continue
        x = np.array(d["size"], float)
        w, t = max(x[0], x[1]), min(x[0], x[1])
        lo = max(MIN_H / x[2], MIN_W / w, 0.02 / t)
        hi = min(MAX_H / x[2], MAX_W / w)
        if not (0 < lo <= hi):
            continue
        u = int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000 / 999
        f = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
        out[name + RS] = dict(base=name, scale=round(f, 4), lo=(np.array(d["lo"]) * f).round(5).tolist(),
                              hi=(np.array(d["hi"]) * f).round(5).tolist(), size=(x * f).round(3).tolist(),
                              n_coll=d["n_coll"], split="train", ok=True)
    json.dump(out, open(INDEX_RS, "w"), indent=0)
    print(f"{len(out)} rescaled training variants -> {INDEX_RS}")


def objects(split):
    return sorted(n for n, d in load_index().items() if d["ok"] and d["split"] == split)


# ---------------------------------------------------------------- assets on the node
def local_root():
    return os.environ.get("GSO_LOCAL", f"/tmp/gso_{os.environ.get('USER', 'user')}")


def ensure_local():
    """Unpack the tar once per node (file lock, so parallel workers wait for the first)."""
    import fcntl
    root = local_root()
    os.makedirs(root, exist_ok=True)
    done = os.path.join(root, ".unpacked")
    if os.path.exists(done):
        return root
    with open(os.path.join(root, ".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not os.path.exists(done):
            with tarfile.open(TAR) as tf:
                tf.extractall(root)
            open(done, "w").close()
    return root


def libero_xml(name):
    """Write (once) a LIBERO obstacle-layout xml for one model; returns its path."""
    root = ensure_local()
    idx = load_index()[name]
    base, f = idx.get("base", name), idx.get("scale")
    d = os.path.join(root, "models", base)
    path = os.path.join(d, "gso_obstacle.xml" if f is None else f"gso_obstacle_s{f}.xml")
    if os.path.exists(path):
        return path
    sc = "" if f is None else f' scale="{f} {f} {f}"'
    lo, hi = np.array(idx["lo"]), np.array(idx["hi"])
    cols = sorted(f for f in os.listdir(d) if f.startswith("model_collision_") and f.endswith(".obj"))
    prop = 'solimp="0.998 0.998 0.001" solref="0.001 1" density="100" friction="0.95 0.3 0.1"'
    assets = [f'<texture file="{d}/texture.png" name="tex" type="2d"/>',
              '<material name="mat" reflectance="0.3" texrepeat="1 1" texture="tex" texuniform="false"/>',
              f'<mesh file="{d}/model.obj" name="vis"{sc}/>']
    geoms = [f'<geom {prop} type="mesh" mesh="vis" conaffinity="0" contype="0" group="1" material="mat"/>']
    for i, c in enumerate(cols):
        assets.append(f'<mesh file="{d}/{c}" name="col{i}"{sc}/>')
        geoms.append(f'<geom {prop} type="mesh" mesh="col{i}" group="0" rgba="0.8 0.8 0.8 0.3"/>')
    c = (lo + hi) / 2
    r = 0.5 * max(hi[0] - lo[0], hi[1] - lo[1])
    xml = f"""<mujoco model="gso_obstacle">
  <asset>
    {chr(10).join('    ' + a for a in assets).strip()}
  </asset>
  <worldbody>
    <body>
      <body name="object">
        {chr(10).join('        ' + g for g in geoms).strip()}
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="{c[0]:.4f} {c[1]:.4f} {lo[2]:.4f}" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="{c[0]:.4f} {c[1]:.4f} {hi[2]:.4f}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="{c[0] + r:.4f} {c[1]:.4f} {c[2]:.4f}" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""
    tmp = path + f".tmp{os.getpid()}"
    open(tmp, "w").write(xml)
    os.replace(tmp, path)
    return path


_registered = False


def register():
    global _registered
    if _registered:
        return
    import numpy as np  # noqa: F811
    from libero.libero.envs.base_object import register_object
    from libero.libero.envs.objects.obstacle_objects import ObstacleObject
    from robosuite.models.objects import MujocoXMLObject

    class GsoObstacle(ObstacleObject):
        def __init__(self, name="gso_obstacle", obj_name="gso_obstacle"):
            MujocoXMLObject.__init__(self, libero_xml(_CUR["name"]), name=name,
                                     joints=[dict(type="free", damping="0.0005")], obj_type="all",
                                     duplicate_collision_geoms=False)
            self.category_name = "gso_obstacle"
            self.rotation = (0.0, 0.0)
            self.rotation_axis = "z"
            self.object_properties = {"vis_site_names": {}}

    register_object(GsoObstacle)
    _registered = True


def enable():
    """Make sl_env / texts know the GSO obstacle (call once per worker, after sl_env.set_suite)."""
    import sl_env
    import texts
    register()
    if NAME not in sl_env.OBSTACLES_ALL:
        sl_env.OBSTACLES_ALL.append(NAME)
    sl_env.NOUN.setdefault(NAME, "object")
    texts.NOUN_TRAIN.setdefault(NAME, "object")
    texts.NOUN_HELD.setdefault(NAME, "object")


def select(name):
    """Choose the GSO model for the next env and point the sentence nouns at it."""
    import sl_env
    import texts
    _CUR["name"] = name
    noun = load_index()[name]["noun"] if name else "object"
    sl_env.NOUN[NAME] = texts.NOUN_TRAIN[NAME] = texts.NOUN_HELD[NAME] = noun


def gso_bddl(path):
    """Official BDDL with its first obstacle slot replaced by the GSO obstacle."""
    import tempfile
    txt = open(path).read()
    m = re.search(r"\b(\w+_obstacle_1) - (\w+_obstacle)\b", txt)
    if not m:
        raise ValueError(f"no obstacle slot in {path}")
    old = m.group(1)
    txt = txt.replace(m.group(0), f"{NAME} - gso_obstacle")
    txt = re.sub(rf"\b{old}\b", NAME, txt)
    d = tempfile.mkdtemp(prefix="gso_bddl_")
    out = os.path.join(d, os.path.basename(path))
    open(out, "w").write(txt)
    return out


def make_env(level, task_id, name, res=128):
    """SafeLIBERO env of (level, task) with GSO model `name` in the first obstacle slot."""
    import sl_env
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.errors import RandomizationError
    enable()
    select(name)
    suite = benchmark.get_benchmark_dict()[sl_env.SUITE](safety_level=level)
    task = suite.get_task(task_id)
    bddl = gso_bddl(os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file))
    for attempt in range(10):
        try:
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=["agentview", "robot0_eye_in_hand"],
                                     camera_heights=res, camera_widths=res, hard_reset=False,
                                     horizon=max(1000, sl_env.MAX_STEPS + sl_env.SETTLE + 50))   # robosuite default is 1000; longer only for --max_steps_mult
            break
        except RandomizationError:
            if attempt == 9:
                raise
    env.seed(sl_env.ENV_SEED)
    env.ooi = sl_env.objects_of_interest(bddl)
    return env, task.language, suite.get_task_init_states(task_id)


def yaw_quat(rng):
    """Upright (GSO models are z-up) with a random yaw; MuJoCo qpos order wxyz."""
    a = rng.uniform(0, 2 * np.pi)
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["index"]:
        build_index()
    elif sys.argv[1:] == ["renoun"]:
        renoun()
    elif sys.argv[1:] == ["rescale"]:
        rescale_index()
