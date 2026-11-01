"""Build the rollout-viewer page from render_v2.py's manifest and analyze_v2.py's results.
    python make_artifact_v2.py --art runs_v2/artifact --results runs_v2/results_v2.json
writes <art>/index.html (videos are referenced as clips/*.mp4 next to it)
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--art", default="runs_v2/artifact")
ap.add_argument("--results", default="runs_v2/results_v2.json")
a = ap.parse_args()

M = json.load(open(os.path.join(a.art, "manifest.json")))
base_paths = {(m["split"], m["scene"]): m["path"] for m in M if m["group"] == "base"}
for m in M:
    m["base_path"] = base_paths.get((m["split"], m["scene"]))
R = json.load(open(a.results)) if os.path.exists(a.results) else {}


def pick(split, *keys):
    x = R.get(split)
    for k in keys:
        if x is None:
            return None
        x = x.get(k)
    return None if x is None else x["k"] / x["n"]


ROWS = [
    ("Base trajectory: passes over the laptop", ("summary", "base", "viol_A")),
    ("Base trajectory: tilts the mug past 15°", ("summary", "base", "viol_B")),
    ("A text: laptop avoided, task done", ("summary", "main|A", "A_ok&succ")),
    ("A text: tilt left alone (still over 15°)", ("summary", "main|A", "viol_B")),
    ("B text: mug upright, task done", ("summary", "main|B", "B_ok&succ")),
    ("B text: path left alone (still over laptop)", ("summary", "main|B", "viol_A")),
    ("No-text model: laptop avoided, task done", ("summary", "notext", "A_ok&succ")),
    ("Unseen laptop paraphrases, set 1 (8)", ("exp2_heldout_A",)),
    ("Unseen laptop paraphrases, set 2 (8)", ("exp2_heldout2_A",)),
    ("Unseen upright paraphrases, set 1 (8)", ("exp2_heldout_B",)),
    ("Unseen upright paraphrases, set 2 (8)", ("exp2_heldout2_B",)),
    ("Combined sentence, no combined training", ("exp3", "main|combined", "AB_ok&succ")),
    ("Combined, no training: clauses in turn A→B", ("exp3", "stack_AB|combined", "AB_ok&succ")),
    ("Combined sentence, with combined training", ("exp3", "main_ab|combined", "AB_ok&succ")),
    ("Far-worded combined sentence, no combined training", ("exp3", "main|far", "AB_ok&succ")),
    ("Far-worded combined, no training: clauses in turn A→B", ("exp3", "stack_AB|far", "AB_ok&succ")),
    ("Far-worded combined sentence, with combined training", ("exp3", "main_ab|far", "AB_ok&succ")),
]
table = [dict(label=l, train=pick("train", *k), test=pick("test", *k)) for l, k in ROWS]

GROUPS = [
    ("base", "Base"), ("A", "Constraint A"), ("B", "Constraint B"),
    ("ab_direct", "A+B, no training"), ("ab_stack", "A+B, clauses in turn"), ("ab_trained", "A+B, trained"),
    ("ab_trained_far", "A+B, trained, far wording"),
]
html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifact_template.html")).read()
html = html.replace("/*__DATA__*/", "const CLIPS = " + json.dumps(M, separators=(",", ":")) + ";\n"
                    "const TABLE = " + json.dumps(table) + ";\n"
                    "const GROUPS = " + json.dumps(GROUPS) + ";")
open(os.path.join(a.art, "index.html"), "w").write(html)
print("wrote", os.path.join(a.art, "index.html"), len(M), "clips", f"{len(html) / 1e6:.2f} MB")
