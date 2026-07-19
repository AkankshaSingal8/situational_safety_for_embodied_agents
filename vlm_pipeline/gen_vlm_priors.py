"""Generate the VLM hazard-prior JSON consumed by symbolic_identity
(load_vlm_priors). One offline API pass over the LIBERO object vocabulary —
runtime never calls the API (compute nodes have no egress); unseen names fall
back to default+floor.

Usage:
  python gen_vlm_priors.py --backend anthropic:claude-haiku-4-5-20251001 \
      --prompt anchored --out ../results_tables/vlm_hazard_priors.json
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from vlm_slot_bench import load_episodes, make_backend, rate_name  # noqa: E402

# LIBERO / SafeLIBERO object vocabulary (clean names). Union of capture names,
# OBSTACLE_RADII families, and common LIBERO-90 objects; unseen runtime names
# fall back to symbolic_identity's default+floor.
VOCAB = [
    "moka pot", "wine bottle", "milk", "white storage box", "yellow book",
    "red coffee mug", "akita black bowl", "plate", "cookies",
    "glazed rim porcelain ramekin", "butter", "chocolate pudding",
    "cream cheese", "ketchup", "tomato sauce", "bbq sauce", "orange juice",
    "salad dressing", "alphabet soup", "basket", "wooden cabinet",
    "flat stove", "wine rack", "microwave", "frying pan", "kettle",
    "white bowl", "wooden tray", "dish rack", "book", "mug", "carton",
    "storage box", "bottle", "black bowl",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="anthropic:claude-haiku-4-5-20251001")
    ap.add_argument("--prompt", default="anchored")
    ap.add_argument("--n_votes", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    backend = make_backend(args.backend)
    # extend vocab with any names present in saved captures
    names = set(VOCAB)
    for e in load_episodes(["safelibero_spatial"], 50):
        for n in e["cands"]:
            clean = n.lower().replace("_", " ")
            clean = "".join(c for c in clean if not c.isdigit()).strip()
            names.add(clean.replace(" obstacle", ""))

    out = {}
    for name in sorted(names):
        w, detail = rate_name(backend, name, args.n_votes, args.prompt)
        out[name] = round(w, 3)
        print(f"{name}: {w:.2f}")
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out)} priors -> {args.out}")


if __name__ == "__main__":
    main()
