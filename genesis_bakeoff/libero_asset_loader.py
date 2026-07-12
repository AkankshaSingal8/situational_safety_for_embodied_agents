"""
Loads LIBERO's existing MJCF object-asset library (SafeLIBERO/safelibero/libero/
libero/assets/) into Genesis scenes.

Root-cause finding: LIBERO's per-object XML files (e.g. akita_black_bowl.xml)
define a movable body with NO <joint> element -- in the real LIBERO/robosuite
pipeline, robosuite's MujocoXMLObject wrapper injects a free joint in Python
when compositing the object into a task scene. Loading the raw XML directly
via gs.morphs.MJCF() skips that step, so the body comes in welded/fixed to
the world (confirmed empirically: object position was bit-identical before
and after 100 settle steps under gravity, and Genesis's own FileMorph
docstring describes "basic rigid objects" as needing "only a root free
joint"). This module patches a free joint into a body-less copy of each XML
before handing it to gs.morphs.MJCF(), so objects fall/settle/get grasped
like real rigid bodies.
"""
import os
import re
import tempfile

# Bump this when _inject_free_joint's logic changes, so stale patched copies
# (keyed by source mtime, which doesn't change when only this file does)
# don't silently get reused.
_LOADER_VERSION = "v2"
_PATCH_CACHE_DIR = os.path.join(tempfile.gettempdir(), f"genesis_libero_asset_patched_{_LOADER_VERSION}")


def _inject_free_joint(xml_text, joint_name="root_free_joint"):
    """
    Inserts <joint type="free" name="..."/> as the first child of the
    TOP-LEVEL <body> -- i.e. the first <body ...> that is a direct child of
    <worldbody>, NOT the inner named body LIBERO wraps it in (its XMLs use
    <worldbody><body><body name="object" ...> ... </body></body></worldbody>,
    and a free joint is only legal on the outermost body -- MuJoCo/Genesis
    both reject "free joint can only be used on top level" if placed on the
    nested one, which is what naively matching name="object" would do).
    Idempotent: does nothing if a <joint already exists inside that body.
    """
    wb_match = re.search(r"<worldbody[^>]*>", xml_text)
    if wb_match is None:
        raise ValueError("No <worldbody> element found")

    body_match = re.search(r"<body\b[^>]*>", xml_text[wb_match.end():])
    if body_match is None:
        raise ValueError("Could not find a top-level <body> element under <worldbody>")

    insert_at = wb_match.end() + body_match.end()
    already_has_joint = re.search(r"<joint\b", xml_text[insert_at:insert_at + 200])
    if already_has_joint:
        return xml_text

    joint_tag = f'\n      <joint type="free" name="{joint_name}" damping="0.0005"/>'
    return xml_text[:insert_at] + joint_tag + xml_text[insert_at:]


def get_loadable_asset_path(libero_xml_path):
    """
    Returns a path to a version of libero_xml_path that's safe to pass to
    gs.morphs.MJCF() -- i.e. has a free joint. Patched copies are cached
    (keyed by source path + mtime) alongside their original asset folder's
    referenced mesh/texture files via a symlinked directory, since MJCF
    <mesh file="..."> / <texture file="..."> references are relative to the
    XML's own directory.
    """
    os.makedirs(_PATCH_CACHE_DIR, exist_ok=True)
    src_dir = os.path.dirname(os.path.abspath(libero_xml_path))
    src_name = os.path.basename(libero_xml_path)
    mtime = os.path.getmtime(libero_xml_path)
    patch_dir = os.path.join(_PATCH_CACHE_DIR, f"{os.path.basename(src_dir)}_{int(mtime)}")
    patched_xml_path = os.path.join(patch_dir, src_name)

    if os.path.exists(patched_xml_path):
        return patched_xml_path

    os.makedirs(patch_dir, exist_ok=True)
    # Symlink every sibling file/dir (meshes, textures, visual/ subfolder) so
    # the relative <mesh file="..."> / <texture file="..."> references in the
    # XML still resolve from the patched copy's location.
    for entry in os.listdir(src_dir):
        if entry == src_name:
            continue
        src_entry = os.path.join(src_dir, entry)
        dst_entry = os.path.join(patch_dir, entry)
        if not os.path.exists(dst_entry):
            os.symlink(src_entry, dst_entry)

    with open(libero_xml_path) as f:
        xml_text = f.read()
    patched = _inject_free_joint(xml_text)
    with open(patched_xml_path, "w") as f:
        f.write(patched)

    return patched_xml_path


def load_libero_object(scene, libero_xml_path, pos=(0.0, 0.0, 0.0), **mjcf_kwargs):
    """Convenience wrapper: patches the free joint and adds the entity to scene."""
    import genesis as gs

    loadable_path = get_loadable_asset_path(libero_xml_path)
    return scene.add_entity(gs.morphs.MJCF(file=loadable_path, pos=pos, **mjcf_kwargs))
